"""Response-construction and cache-writing helpers for the orchestrator.

Split out of agent/orchestrator.py — these functions either construct a
final AIMessage directly or write to the vector cache, as opposed to
orchestrator_routing.py's purely read-only decision helpers.
"""

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

from agent.orchestrator_routing import (
    _all_tool_results,
    _last_tool_call_args,
    _last_tool_result,
    _needs_credibility_scoring,
)
from agent.tools.vector_lookup_tool import store_verdict
from agent.verdict_completeness import check_verdict_completeness


def _gather_sources_for_scoring(state: MessagesState) -> list[dict[str, Any]]:
    """Build the sources list for credibility_scoring_tool from prior tool results.

    The model isn't reliable at manually re-copying earlier tool outputs into
    a new tool call's arguments — the orchestrator constructs this list
    itself instead of trusting the model's tool-call args.
    """
    sources: list[dict[str, Any]] = []

    fact_check_result = _last_tool_result(state, "fact_check_lookup_tool")
    if fact_check_result:
        for claim in fact_check_result.get("claims", []):
            sources.append(
                {
                    "url": claim.get("url", ""),
                    "content": (
                        f"{claim.get('publisher', 'Unknown publisher')} rated this "
                        f"\"{claim.get('rating', 'unrated')}\": {claim.get('claim_text', '')}"
                    ),
                }
            )

    # All rounds, not just the most recent — a retry search adds evidence
    # on top of the first round rather than replacing it (see
    # _all_tool_results).
    for search_result in _all_tool_results(state, "web_search_tool"):
        for result in search_result.get("sources", []):
            sources.append({"url": result.get("url", ""), "content": result.get("snippet", "")})

    return sources


def _build_cache_hit_response(cache_result: dict[str, Any]) -> AIMessage:
    """Build a final answer directly from a cache hit, with no further model call.

    A cache hit already has everything a normal turn would produce (a
    verdict summary, confidence, sources) — reformatting it through another
    Gemini call would just add latency and cost for no real benefit, so the
    response is assembled directly from the cached fields instead.
    """
    summary = cache_result.get("verdict_summary", "This claim has already been checked.")
    sources = cache_result.get("sources", [])
    sources_text = "\n".join(f"- {url}" for url in sources) if sources else "No sources recorded."

    text = (
        "This claim (or a close rewording of it) has already been checked "
        f"previously.\n\n{summary}\n\n**Sources:**\n{sources_text}"
    )
    return AIMessage(content=text)


def _extract_verdict_text(content: str | list) -> str:
    """Normalize a final AIMessage's content into plain text for the
    verdict-completeness check.

    Duplicated from main.py's _extract_text rather than imported — main.py
    is the outermost layer (the FastAPI app), so importing from it here
    would invert that layering. Small enough that duplicating a handful of
    lines is less risk than restructuring an already-tested shared helper.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _check_and_store_verdict(state: MessagesState, response: AIMessage) -> bool:
    """Check this turn's final answer for completeness, and only cache it
    for future reuse if it passes.

    An incomplete verdict shouldn't be cached and silently reused for
    every future semantically-similar claim — that would multiply the
    harm of one weak answer instead of containing it to a single turn.
    Runs regardless of whether this turn resolved via
    fact_check_lookup_tool or credibility_scoring_tool.

    Returns True if there was nothing to check (no claim on record, e.g.
    an edge case where vector_lookup_tool was never actually called) or if
    the verdict was judged complete; False otherwise. Callers that care
    about escalating incomplete verdicts should check this return value.
    """
    vector_lookup_args = _last_tool_call_args(state, "vector_lookup_tool")
    claim = vector_lookup_args.get("claim", "") if vector_lookup_args else ""
    if not claim:
        return True

    sources = _gather_sources_for_scoring(state)
    verdict_text = _extract_verdict_text(response.content)
    is_complete = check_verdict_completeness(claim, verdict_text, sources)

    if is_complete:
        _store_verdict_if_new(state)

    return is_complete


def _store_verdict_if_new(state: MessagesState) -> None:
    """Store a freshly resolved claim's verdict for future cache lookups.

    Only called when this turn's final answer did not come from a cache
    hit (a hit returns early in call_model and never reaches this).

    Uses the same claim text vector_lookup_tool was called with, not the
    raw user message — otherwise lookup-time and store-time embeddings
    drift apart (the model can paraphrase the user's message into a
    cleaner claim before calling vector_lookup_tool), undermining the
    cache's own consistency.
    """
    vector_lookup_args = _last_tool_call_args(state, "vector_lookup_tool")
    claim = vector_lookup_args.get("claim", "") if vector_lookup_args else ""
    if not claim:
        return

    credibility_result = _last_tool_result(state, "credibility_scoring_tool")
    fact_check_result = _last_tool_result(state, "fact_check_lookup_tool")

    if credibility_result and not credibility_result.get("error"):
        store_verdict(
            claim=claim,
            confidence=credibility_result.get("overall_confidence", 0.0),
            sources=[s.get("url", "") for s in credibility_result.get("source_scores", [])],
            verdict_summary=credibility_result.get("verdict_summary", ""),
            resolved_by="credibility_scoring_tool",
        )
    elif fact_check_result and not _needs_credibility_scoring(fact_check_result):
        claim_entry = fact_check_result.get("claims", [{}])[0]
        store_verdict(
            claim=claim,
            confidence=1.0,
            sources=[claim_entry.get("url", "")],
            verdict_summary=(
                f"{claim_entry.get('publisher', 'A fact-checker')} rated this "
                f"\"{claim_entry.get('rating', 'unrated')}\": {claim_entry.get('claim_text', '')}"
            ),
            resolved_by="fact_check_lookup_tool",
        )
