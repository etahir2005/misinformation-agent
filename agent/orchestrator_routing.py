"""Pure routing/decision helpers for the orchestrator — every function here
reads only from the `state` dict passed to it and returns a bool or data,
with no side effects. Kept separate from orchestrator_responses.py (which
builds response content / has side effects) and orchestrator.py (graph
wiring) so each concern lives in one obvious place instead of one large
file mixing all three.
"""

import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState

from agent.config import LOW_CONFIDENCE_THRESHOLD, MAX_MESSAGES_BEFORE_SUMMARY

# Fixed id so the stored summary message can be found, replaced, and
# excluded from what actually gets sent to Gemini — never rely on position.
_SUMMARY_MESSAGE_ID = "conversation-summary"

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


def _has_evidence_tool_run(state: MessagesState) -> bool:
    """True once fact_check_lookup_tool, web_search_tool, or source_retrieval_tool has run.

    Renamed from _has_gathered_evidence during the Day 8 restructuring — the
    old name was easily confused with _has_usable_evidence (below), which
    checks something different (non-empty results, not just "did it run").
    """
    return (
        _last_tool_result(state, "fact_check_lookup_tool") is not None
        or _last_tool_result(state, "web_search_tool") is not None
        or _last_tool_result(state, "source_retrieval_tool") is not None
    )


def _has_usable_evidence(state: MessagesState) -> bool:
    """True once at least one non-empty piece of evidence has been gathered.

    Renamed from _has_evidence_to_score during the Day 8 restructuring — see
    _has_evidence_tool_run's docstring for why the pair was renamed.
    """
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
    if not _has_usable_evidence(state):
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


def _needs_summary(state: MessagesState) -> Literal["summarize", "orchestrator"]:
    """Route to the summarize node only once older history crosses the threshold.

    Checked once at the very start of each turn, right after the guardrail
    (see route_after_guardrail). Self-limiting: once summarize_node runs,
    the older-message count drops back down to just the one summary
    placeholder, so this won't fire again until enough new messages
    accumulate.
    """
    older_messages = _messages_before_current_turn(state)
    if len(older_messages) >= MAX_MESSAGES_BEFORE_SUMMARY:
        return "summarize"
    return "orchestrator"
