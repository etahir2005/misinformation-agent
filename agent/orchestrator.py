"""The orchestrator: a single tool-calling agent built on LangGraph."""

import json
import logging
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import (
    LOW_CONFIDENCE_THRESHOLD,
    MAX_MESSAGES_BEFORE_SUMMARY,
    MODEL_NAME,
    SYSTEM_PROMPT,
)
from agent.summarizer import summarize
from agent.tools.credibility_scoring_tool import credibility_scoring_tool
from agent.tools.fact_check_tool import fact_check_lookup_tool
from agent.tools.source_retrieval_tool import source_retrieval_tool
from agent.tools.vector_lookup_tool import store_verdict, vector_lookup_tool
from agent.tools.web_search_tool import web_search_tool

logger = logging.getLogger(__name__)

TOOLS = [
    vector_lookup_tool,
    fact_check_lookup_tool,
    web_search_tool,
    source_retrieval_tool,
    credibility_scoring_tool,
]

_EVIDENCE_TOOLS = [fact_check_lookup_tool, web_search_tool, source_retrieval_tool]

# Real fact-checker rating text is inconsistent across publishers (Snopes,
# PolitiFact, Reuters, etc. all use their own wording), so this can't be an
# exhaustive list — it's deliberately narrow. Only ratings that are fully
# unambiguous go here. Hedged/partial ratings (PolitiFact's "Mostly True" /
# "Half True" / "Mostly False", Snopes's "Mixture", etc.) are intentionally
# excluded even though they lean one way — presenting a hedged rating as a
# clean pass-through verdict would be misleading, so those still go through
# credibility_scoring_tool for a fair, nuanced summary instead.
_CLEAN_RATINGS = {
    "true",
    "false",
    "correct",
    "incorrect",
    "accurate",
    "inaccurate",
    "pants on fire",
    "pants on fire!",
}

# Fixed id so the stored summary message can be found, replaced, and
# excluded from what actually gets sent to Gemini (see
# _existing_summary_text and call_model below) — never rely on position.
_SUMMARY_MESSAGE_ID = "conversation-summary"


def _current_turn_messages(state: MessagesState) -> list:
    """Return only the messages from the most recent HumanMessage onward.

    state["messages"] holds the entire thread's history across every turn
    (persisted per thread_id by the checkpointer), not just the current
    claim. Every routing decision below needs to reason about only the
    current turn's tool activity — otherwise a second claim asked in the
    same thread would look like it's already been cache-checked or searched
    just because an earlier claim's tool results are still sitting in
    history (caught in PR review: this was previously unscoped, and a
    cache hit from an earlier claim would incorrectly short-circuit every
    later claim in the same thread).

    Caveat: on a thread's first summarization, the new summary SystemMessage
    can end up merged in after the current turn's HumanMessage rather than
    before it (see the note in summarize_node), so this function's result
    can briefly include that stray message too. Harmless today since every
    caller below filters by concrete message type rather than trusting this
    function to return purely current-turn activity — but worth keeping in
    mind before adding a new turn-scoped helper that doesn't.
    """
    messages = state["messages"]
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def _messages_before_current_turn(state: MessagesState) -> list:
    """Return every message before the most recent HumanMessage.

    The complement of _current_turn_messages — used only for summarization,
    which must never touch the turn currently in progress (the model still
    needs that context intact to finish resolving the current claim).
    """
    messages = state["messages"]
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[:index]
    return []


def _existing_summary_text(messages: list) -> str | None:
    """Return the stored running summary's text, if one exists in this message list."""
    for message in messages:
        if getattr(message, "id", None) == _SUMMARY_MESSAGE_ID:
            return message.content
    return None


def _format_messages_for_summary(messages: list) -> str:
    """Render messages as plain text for the summarization prompt.

    Deliberately simple (role + text content only) — the summarization
    model needs what was asked and concluded, not tool-call machinery.
    Skips the stored summary placeholder itself (passed separately as
    existing_summary), raw ToolMessage results (the JSON payloads tools
    return — source URLs, snippets, confidence scores — are exactly the
    "machinery" this function exists to leave out, not substance worth
    summarizing), and any message with empty content.
    """
    lines = []
    for message in messages:
        if getattr(message, "id", None) == _SUMMARY_MESSAGE_ID:
            continue
        if isinstance(message, ToolMessage):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        if not content.strip():
            continue
        role = message.__class__.__name__.replace("Message", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _last_tool_result(state: MessagesState, tool_name: str) -> dict | None:
    """Return the parsed dict from the most recent ToolMessage for a given tool."""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.name == tool_name:
            try:
                return json.loads(message.content)
            except (TypeError, ValueError):
                return None
    return None


def _all_tool_results(state: MessagesState, tool_name: str) -> list[dict]:
    """Return every parsed ToolMessage result for a given tool, in call order.

    Unlike _last_tool_result, this doesn't discard earlier calls. Needed
    for web_search_tool specifically: the retry-search loop can call it
    twice in one turn, and the system prompt frames the retry as gathering
    an *additional* round of evidence, not replacing the first round.
    """
    results = []
    for message in state["messages"]:
        if isinstance(message, ToolMessage) and message.name == tool_name:
            try:
                results.append(json.loads(message.content))
            except (TypeError, ValueError):
                continue
    return results


def _last_tool_message_index(state: MessagesState, tool_name: str) -> int | None:
    """Return the index of the most recent ToolMessage for a given tool, or None."""
    for index in range(len(state["messages"]) - 1, -1, -1):
        message = state["messages"][index]
        if isinstance(message, ToolMessage) and message.name == tool_name:
            return index
    return None


def _last_tool_call_args(state: MessagesState, tool_name: str) -> dict | None:
    """Return the args dict from the most recent tool call made to a given tool."""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call["name"] == tool_name:
                    return tool_call["args"]
    return None


def _credibility_scoring_call_count(state: MessagesState) -> int:
    """Count how many times credibility_scoring_tool has been called so far."""
    return sum(
        1
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.name == "credibility_scoring_tool"
    )


def _needs_credibility_scoring(fact_check_result: dict) -> bool:
    """True unless there's exactly one claim with a clean True/False-equivalent rating."""
    claims = fact_check_result.get("claims", [])
    if len(claims) != 1:
        return True
    rating = claims[0].get("rating", "").strip().lower()
    return rating not in _CLEAN_RATINGS


def _has_gathered_evidence(state: MessagesState) -> bool:
    """True once fact_check_lookup_tool, web_search_tool, or source_retrieval_tool has run."""
    return (
        _last_tool_result(state, "fact_check_lookup_tool") is not None
        or _last_tool_result(state, "web_search_tool") is not None
        or _last_tool_result(state, "source_retrieval_tool") is not None
    )


def _has_evidence_to_score(state: MessagesState) -> bool:
    """True once at least one non-empty piece of evidence has been gathered."""
    fact_check_result = _last_tool_result(state, "fact_check_lookup_tool")
    has_fact_check_claims = bool(fact_check_result and fact_check_result.get("claims"))
    has_web_search = _last_tool_result(state, "web_search_tool") is not None
    return has_fact_check_claims or has_web_search


def _should_force_credibility_scoring(state: MessagesState) -> bool:
    """Decide whether this turn must call credibility_scoring_tool.

    Fires once evidence exists and it doesn't already amount to a single
    clean True/False ruling. Never fires if the user gave a specific URL
    (source_retrieval_tool's direct-read flow doesn't need this layer).

    Re-fires exactly once more after a retry search triggered by weak
    evidence (see _should_force_retry_search) — but never a third time, and
    only once a new web_search_tool call has actually happened since the
    last scoring pass, so it doesn't re-fire on the same stale evidence.
    """
    if _last_tool_result(state, "source_retrieval_tool") is not None:
        return False
    if not _has_evidence_to_score(state):
        return False

    scoring_calls = _credibility_scoring_call_count(state)
    if scoring_calls >= 1:
        if scoring_calls >= 2:
            return False
        scoring_index = _last_tool_message_index(state, "credibility_scoring_tool")
        search_index = _last_tool_message_index(state, "web_search_tool")
        if search_index is None or search_index < scoring_index:
            return False

    fact_check_result = _last_tool_result(state, "fact_check_lookup_tool")
    if fact_check_result is not None and not _needs_credibility_scoring(fact_check_result):
        return False

    return True


def _evidence_is_weak(state: MessagesState) -> bool:
    """True if the last credibility_scoring_tool result was low-confidence or conflicting."""
    result = _last_tool_result(state, "credibility_scoring_tool")
    if result is None:
        return False
    confidence = result.get("overall_confidence", 1.0)
    return confidence < LOW_CONFIDENCE_THRESHOLD or result.get("sources_conflict", False)


def _should_force_retry_search(state: MessagesState) -> bool:
    """Decide whether this turn must call web_search_tool again for weak evidence.

    Capped at exactly one retry: only forces a search when there's been
    exactly one scoring pass so far, its result was weak, and no search has
    already happened since that scoring call.
    """
    if _credibility_scoring_call_count(state) != 1:
        return False
    if not _evidence_is_weak(state):
        return False

    scoring_index = _last_tool_message_index(state, "credibility_scoring_tool")
    search_index = _last_tool_message_index(state, "web_search_tool")
    if search_index is not None and search_index > scoring_index:
        return False

    return True


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


def _needs_summary(state: MessagesState) -> Literal["summarize", "orchestrator"]:
    """Route to the summarize node only once older history crosses the threshold.

    Checked once at the very start of each turn (see build_orchestrator's
    START edge). Self-limiting: once summarize_node runs, the older-message
    count drops back down to just the one summary placeholder, so this
    won't fire again until enough new messages accumulate.
    """
    older_messages = _messages_before_current_turn(state)
    if len(older_messages) >= MAX_MESSAGES_BEFORE_SUMMARY:
        return "summarize"
    return "orchestrator"


def summarize_node(state: MessagesState) -> dict:
    """Compress older, fully-resolved turns into a single running summary message.

    Only reached when _needs_summary routes here. Removes every message
    before the current turn (including any prior summary placeholder) and
    replaces them with one updated SystemMessage holding the new summary —
    never touches the turn currently in progress.
    """
    older_messages = _messages_before_current_turn(state)
    existing_summary = _existing_summary_text(older_messages)
    conversation_text = _format_messages_for_summary(older_messages)

    new_summary_text = summarize(existing_summary, conversation_text)
    if new_summary_text is None:
        # Summarization failed — leave history untouched this turn rather
        # than delete messages with nothing to replace them with.
        return {"messages": []}

    # Note: on this thread's *first* summarization there's no existing
    # message with _SUMMARY_MESSAGE_ID for add_messages to replace in
    # place, so this new summary_message gets appended to the end of the
    # merged list rather than positioned before the current turn's
    # HumanMessage — see the caveat on _current_turn_messages. Later
    # summarizations don't have this issue: replacing an existing id keeps
    # its original index.
    removals = [RemoveMessage(id=message.id) for message in older_messages]
    summary_message = SystemMessage(content=new_summary_text, id=_SUMMARY_MESSAGE_ID)
    return {"messages": removals + [summary_message]}


def build_orchestrator(checkpointer: BaseCheckpointSaver) -> StateGraph:
    """Build and compile the single-agent orchestrator graph.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. the Postgres-backed one
            from agent.checkpointer.build_checkpointer, or InMemorySaver in
            tests) — injected rather than constructed here so this module
            stays decoupled from where/how state is actually persisted.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    model = init_chat_model(f"google_genai:{MODEL_NAME}", temperature=0)
    model_forced_cache_check = model.bind_tools(TOOLS, tool_choice="vector_lookup_tool")
    model_forced_search = model.bind_tools(_EVIDENCE_TOOLS, tool_choice="any")
    model_forced_retry_search = model.bind_tools(TOOLS, tool_choice="web_search_tool")
    model_forced_credibility = model.bind_tools(TOOLS, tool_choice="credibility_scoring_tool")
    model_auto = model.bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        """Invoke the LLM with the current conversation state."""
        # Every routing decision below must look only at the current turn's
        # tool activity, not the whole thread — see _current_turn_messages.
        turn_state = {"messages": _current_turn_messages(state)}
        cache_result = _last_tool_result(turn_state, "vector_lookup_tool")

        if cache_result is not None and cache_result.get("hit"):
            return {"messages": [_build_cache_hit_response(cache_result)]}

        # Gemini itself still sees the full thread history, not just the
        # current turn — that's what gives it multi-turn conversational
        # context. Only the deterministic routing above is turn-scoped.
        # If older history has been summarized, fold that summary into the
        # system prompt (one combined SystemMessage, not two separate
        # ones — safer across chat-model integrations) and exclude the raw
        # stored placeholder from what's actually sent.
        summary_text = _existing_summary_text(state["messages"])
        system_prompt = SYSTEM_PROMPT
        if summary_text:
            system_prompt = (
                f"{SYSTEM_PROMPT}\n\nSummary of earlier claims already "
                f"discussed in this thread:\n{summary_text}"
            )
        sendable_messages = [
            message
            for message in state["messages"]
            if getattr(message, "id", None) != _SUMMARY_MESSAGE_ID
        ]
        messages = [SystemMessage(content=system_prompt)] + sendable_messages

        if cache_result is None:
            model_to_use = model_forced_cache_check
        elif not _has_gathered_evidence(turn_state):
            model_to_use = model_forced_search
        elif _should_force_retry_search(turn_state):
            model_to_use = model_forced_retry_search
        elif _should_force_credibility_scoring(turn_state):
            model_to_use = model_forced_credibility
        else:
            model_to_use = model_auto

        response = model_to_use.invoke(messages)

        for tool_call in response.tool_calls:
            if tool_call["name"] == "credibility_scoring_tool":
                tool_call["args"]["sources"] = _gather_sources_for_scoring(turn_state)

        if not response.tool_calls:
            _store_verdict_if_new(turn_state)

        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("summarize", summarize_node)
    builder.add_node("orchestrator", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_conditional_edges(
        START, _needs_summary, {"summarize": "summarize", "orchestrator": "orchestrator"}
    )
    builder.add_edge("summarize", "orchestrator")
    builder.add_conditional_edges("orchestrator", tools_condition)
    builder.add_edge("tools", "orchestrator")

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Orchestrator graph compiled with %d tool(s).", len(TOOLS))
    return graph
