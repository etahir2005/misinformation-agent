"""The orchestrator: a single tool-calling agent built on LangGraph."""

import json
import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import LOW_CONFIDENCE_THRESHOLD, MODEL_NAME, SYSTEM_PROMPT
from agent.tools.credibility_scoring_tool import credibility_scoring_tool
from agent.tools.fact_check_tool import fact_check_lookup_tool
from agent.tools.source_retrieval_tool import source_retrieval_tool
from agent.tools.web_search_tool import web_search_tool

logger = logging.getLogger(__name__)

TOOLS = [fact_check_lookup_tool, web_search_tool, source_retrieval_tool, credibility_scoring_tool]

_CLEAN_RATINGS = {"true", "false", "correct", "incorrect", "accurate", "inaccurate"}


def _last_tool_result(state: MessagesState, tool_name: str) -> dict | None:
    """Return the parsed dict from the most recent ToolMessage for a given tool."""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.name == tool_name:
            try:
                return json.loads(message.content)
            except (TypeError, ValueError):
                return None
    return None


def _last_tool_message_index(state: MessagesState, tool_name: str) -> int | None:
    """Return the index of the most recent ToolMessage for a given tool, or None."""
    for index in range(len(state["messages"]) - 1, -1, -1):
        message = state["messages"][index]
        if isinstance(message, ToolMessage) and message.name == tool_name:
            return index
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
    already happened since that scoring call. Without that last check, this
    would keep firing on every turn until a rescore updates the stale
    result — which never happens if this function keeps winning priority
    over _should_force_credibility_scoring.
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
    a new tool call's arguments — observed it passing an empty placeholder for
    `sources` in testing. The orchestrator constructs this list itself
    instead of trusting the model's tool-call args.
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

    search_result = _last_tool_result(state, "web_search_tool")
    if search_result:
        for result in search_result.get("sources", []):
            sources.append({"url": result.get("url", ""), "content": result.get("snippet", "")})

    return sources


def build_orchestrator() -> StateGraph:
    """Build and compile the single-agent orchestrator graph.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    model = init_chat_model(f"google_genai:{MODEL_NAME}", temperature=0)
    model_forced_search = model.bind_tools(TOOLS, tool_choice="any")
    model_forced_retry_search = model.bind_tools(TOOLS, tool_choice="web_search_tool")
    model_forced_credibility = model.bind_tools(TOOLS, tool_choice="credibility_scoring_tool")
    model_auto = model.bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        """Invoke the LLM with the current conversation state."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        has_searched = any(isinstance(m, ToolMessage) for m in state["messages"])

        if not has_searched:
            model_to_use = model_forced_search
        elif _should_force_retry_search(state):
            model_to_use = model_forced_retry_search
        elif _should_force_credibility_scoring(state):
            model_to_use = model_forced_credibility
        else:
            model_to_use = model_auto

        response = model_to_use.invoke(messages)

        for tool_call in response.tool_calls:
            if tool_call["name"] == "credibility_scoring_tool":
                tool_call["args"]["sources"] = _gather_sources_for_scoring(state)

        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("orchestrator", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", tools_condition)
    builder.add_edge("tools", "orchestrator")

    graph = builder.compile(checkpointer=InMemorySaver())
    logger.info("Orchestrator graph compiled with %d tool(s).", len(TOOLS))
    return graph