"""The orchestrator: a single tool-calling agent built on LangGraph."""

import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import MODEL_NAME, SYSTEM_PROMPT
from agent.guardrail import classify_message_intent
from agent.orchestrator_responses import (
    _build_cache_hit_response,
    _gather_sources_for_scoring,
    _store_verdict_if_new,
)
from agent.orchestrator_routing import (
    _SUMMARY_MESSAGE_ID,
    _current_turn_messages,
    _existing_summary_text,
    _format_messages_for_summary,
    _has_evidence_tool_run,
    _last_tool_result,
    _messages_before_current_turn,
    _needs_summary,
    _should_force_credibility_scoring,
    _should_force_retry_search,
)
from agent.summarizer import summarize
from agent.tools.credibility_scoring_tool import credibility_scoring_tool
from agent.tools.fact_check_tool import fact_check_lookup_tool
from agent.tools.source_retrieval_tool import source_retrieval_tool
from agent.tools.vector_lookup_tool import vector_lookup_tool
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

_GREETING_RESPONSE = AIMessage(
    content="Hi! Send me a claim you'd like fact-checked, or a link to an "
    "article, and I'll look into it."
)
_OUT_OF_SCOPE_RESPONSE = AIMessage(
    content="I'm built specifically for fact-checking claims — I can't help "
    "with that, but send me something to verify and I'll get on it."
)


class OrchestratorState(MessagesState):
    """Adds the guardrail's classification result to the standard message-list state.

    Kept as an explicit state field rather than inferring the guardrail's
    decision from message content — a dedicated field is unambiguous and
    survives checkpointing cleanly, whereas comparing message identity or
    counting message-list length would be fragile.
    """

    intent_category: Literal["greeting", "claim", "out_of_scope"] | None


def guardrail_node(state: OrchestratorState) -> dict:
    """Classify the current turn's message before it reaches the orchestrator.

    Runs once, ahead of everything else in the graph — see agent/guardrail.py
    for the classification logic itself. Greetings and out-of-scope messages
    get a canned reply here and never enter the forced tool-calling pipeline.
    Genuine claims fall through with state otherwise unchanged — call_model
    behaves exactly as it did before this node existed.
    """
    latest_message = _current_turn_messages(state)[0]
    category = classify_message_intent(latest_message.content)

    if category == "greeting":
        return {"intent_category": category, "messages": [_GREETING_RESPONSE]}
    if category == "out_of_scope":
        return {"intent_category": category, "messages": [_OUT_OF_SCOPE_RESPONSE]}
    return {"intent_category": category}


def route_after_guardrail(state: OrchestratorState) -> str:
    """Send greetings/out-of-scope straight to END; claims continue on to
    either the summarize node (if older history has crossed the threshold)
    or straight to the orchestrator.
    """
    if state["intent_category"] != "claim":
        return END
    return _needs_summary(state)


def summarize_node(state: MessagesState) -> dict:
    """Compress older, fully-resolved turns into a single running summary message.

    Only reached when _needs_summary (via route_after_guardrail) routes
    here. Removes every message before the current turn (including any
    prior summary placeholder) and replaces them with one updated
    SystemMessage holding the new summary — never touches the turn
    currently in progress.
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
    # HumanMessage — see the caveat on _current_turn_messages in
    # orchestrator_routing.py. Later summarizations don't have this issue:
    # replacing an existing id keeps its original index.
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
        elif not _has_evidence_tool_run(turn_state):
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

    builder = StateGraph(OrchestratorState)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("orchestrator", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges("guardrail", route_after_guardrail)
    builder.add_edge("summarize", "orchestrator")
    builder.add_conditional_edges("orchestrator", tools_condition)
    builder.add_edge("tools", "orchestrator")

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Orchestrator graph compiled with %d tool(s) + guardrail.", len(TOOLS))
    return graph
