"""The orchestrator: a single tool-calling agent built on LangGraph."""

import logging
from typing import Literal

from langchain.agents.middleware import PIIMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent.config import MODEL_NAME, SYSTEM_PROMPT
from agent.guardrail import classify_message_intent
from agent.orchestrator_responses import (
    _build_cache_hit_response,
    _check_and_store_verdict,
    _gather_sources_for_scoring,
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

# One PIIMiddleware instance per PII type — each instance only knows how to
# detect/redact its own type, so scrubbing several types means chaining
# several instances (see pii_scrub_node). Deliberately excludes "url": users
# legitimately submit article URLs for source_retrieval_tool to fetch, and
# redacting those before the model ever sees them would silently break that
# feature entirely rather than protecting anyone's privacy.
_PII_MIDDLEWARES = [
    PIIMiddleware("email", strategy="redact"),
    PIIMiddleware("credit_card", strategy="redact"),
    PIIMiddleware("ip", strategy="redact"),
    PIIMiddleware("mac_address", strategy="redact"),
]


def _apply_pii_middlewares(messages: list) -> tuple[list, bool]:
    """Chain every configured PIIMiddleware instance over a message list.

    Each instance only detects/redacts its own PII type, so scrubbing
    several types means running them in sequence — each one needs to see
    the previous one's redacted output, otherwise only the last-run type's
    redaction would survive (e.g. an email redaction would be silently lost
    if a later credit-card check reconstructed the message from the
    original, unredacted text).

    Shared by scrub_pii() (the authoritative, pre-invoke scrub) and
    pii_scrub_node() (an in-graph defense-in-depth backstop) so both stay in
    sync with exactly one implementation of the chaining logic.

    Returns (possibly-redacted messages, whether anything was changed) —
    the bool lets callers skip returning a no-op state update when nothing
    was found, same as before_model() itself returning None for "no PII
    here."
    """
    state = {"messages": messages}
    any_modified = False
    for middleware in _PII_MIDDLEWARES:
        result = middleware.before_model(state, runtime=None)
        if result is not None:
            state = {**state, **result}
            any_modified = True
    return state["messages"], any_modified


def scrub_pii(claim: str) -> str:
    """Redact PII out of a raw claim string before it ever reaches the graph.

    This — not pii_scrub_node — is the authoritative PII scrub. LangGraph
    checkpoints the exact payload passed to graph.invoke() before running
    any node, including the graph's own first node, so placing redaction
    inside the graph isn't actually early enough to keep raw PII out of
    Postgres. This was confirmed empirically, not assumed from graph
    structure: a test walking graph.get_state_history() found the raw claim
    still sitting in the earliest checkpoint snapshot even with
    pii_scrub_node running first in the graph.

    Called from graph.run_claim() on every turn's claim, before the
    invoke() payload is ever built — the one place both main.py and cli.py
    funnel every real request through, so scrubbing here covers every
    actual caller in this codebase.
    """
    redacted_messages, _ = _apply_pii_middlewares([HumanMessage(content=claim)])
    return redacted_messages[-1].content


_GREETING_TEXT = (
    "Hi! Send me a claim you'd like fact-checked, or a link to an "
    "article, and I'll look into it."
)
_OUT_OF_SCOPE_TEXT = (
    "I'm built specifically for fact-checking claims — I can't help "
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
    # Set once per turn, only when this turn produced a final answer (no
    # tool_calls) — see _check_and_store_verdict. None on every other turn
    # (cache hits, greetings/out-of-scope, or mid-tool-loop steps), not
    # False — a future escalation-routing node needs to tell "this verdict
    # was judged incomplete" apart from "no verdict was produced to judge
    # this turn at all."
    verdict_is_complete: bool | None


def pii_scrub_node(state: OrchestratorState) -> dict:
    """In-graph defense-in-depth backstop — not the primary PII defense.

    scrub_pii() (agent/orchestrator.py, defined above) is the authoritative
    scrub, called from graph.run_claim() on the raw claim string before
    graph.invoke() is ever called. This node exists only to protect any
    caller that invokes the compiled graph directly, bypassing run_claim()
    — there's no such caller in this codebase today (both main.py and
    cli.py go through run_claim()), but nothing at the graph level enforces
    that, so this stays as cheap insurance rather than relying on every
    future caller remembering to scrub first.

    Note this cannot, by itself, keep raw PII out of Postgres for a direct
    graph.invoke() call — LangGraph checkpoints the exact invoke() payload
    before running any node, this one included (confirmed empirically via
    get_state_history(), see scrub_pii()'s docstring). Running first in the
    graph still matters for what Gemini, tools, and Langfuse traces see
    within a single turn, just not for the very first checkpoint.

    runtime=None is safe here even though PIIMiddleware.before_model()'s
    signature expects a LangGraph Runtime object — this project builds a
    hand-rolled StateGraph rather than using langchain's create_agent(),
    which is what PIIMiddleware is designed to plug into automatically.
    Reading its actual implementation confirmed before_model() never
    touches the runtime argument, only state["messages"], so calling it
    directly as a plain function (via _apply_pii_middlewares) is safe.
    """
    redacted_messages, any_modified = _apply_pii_middlewares(state["messages"])
    if any_modified:
        return {"messages": redacted_messages}
    return {}


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
        return {"intent_category": category, "messages": [AIMessage(content=_GREETING_TEXT)]}
    if category == "out_of_scope":
        return {
            "intent_category": category,
            "messages": [AIMessage(content=_OUT_OF_SCOPE_TEXT)],
        }
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


def human_review_node(state: OrchestratorState) -> dict:
    """Pause the graph for human review when a verdict was judged incomplete.

    Only reached via route_after_orchestrator when verdict_is_complete is
    False. interrupt() pauses execution here and surfaces the payload below
    to whatever's driving the graph — nothing consumes it yet. Phase 4
    (Slack escalation tool) and Phase 5 (Streamlit approval UI) are what
    will actually notify a human and let them respond with
    Command(resume=...); this node only proves the pause/resume mechanics
    work, it doesn't yet decide what a human's response should change.

    IMPORTANT for whoever builds Phase 4's Slack notification: LangGraph
    re-executes a node's logic from the top on every resume, not just the
    interrupt() call itself. A one-time side effect like sending a Slack
    message placed *before* this interrupt() call would fire again on
    every resume unless it's made idempotent or split into its own node
    that runs once, ahead of this one. Do not add the Slack call directly
    into this node as currently written.

    The resume value's shape isn't finalized — Phase 5 decides what a human
    actually submits. Resuming with any value (or none) simply lets the
    graph finish with the existing answer for now; nothing reads the
    resume value yet.
    """
    latest_human_message = _current_turn_messages(state)[0]
    final_answer = state["messages"][-1]
    interrupt(
        {
            "reason": "verdict_incomplete",
            "claim": latest_human_message.content,
            "verdict": final_answer.content,
        }
    )
    return {}


def route_after_orchestrator(state: OrchestratorState) -> str:
    """Route after the orchestrator's response: continue the tool loop if
    there are more tool calls to make; otherwise decide whether this
    turn's final answer needs human review before the graph finishes.

    Replaces langgraph.prebuilt.tools_condition (rather than composing
    with it) since tools_condition has no hook for a second check after
    "no tool calls" — this reimplements its tool-call check directly, plus
    the new verdict_is_complete branch. A final answer whose
    verdict_is_complete is False (set by _check_and_store_verdict inside
    call_model) routes to human_review instead of ending the graph
    outright; True or None (cache hits, greetings, and anything that
    doesn't set it) end normally, same as before this feature existed.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    if state.get("verdict_is_complete") is False:
        return "human_review"
    return END


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

        verdict_is_complete = None
        if not response.tool_calls:
            verdict_is_complete = _check_and_store_verdict(turn_state, response)

        return {"messages": [response], "verdict_is_complete": verdict_is_complete}

    builder = StateGraph(OrchestratorState)
    builder.add_node("pii_scrub", pii_scrub_node)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("orchestrator", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "pii_scrub")
    builder.add_edge("pii_scrub", "guardrail")
    builder.add_conditional_edges("guardrail", route_after_guardrail)
    builder.add_edge("summarize", "orchestrator")
    builder.add_conditional_edges("orchestrator", route_after_orchestrator)
    builder.add_edge("tools", "orchestrator")
    builder.add_edge("human_review", END)

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Orchestrator graph compiled with %d tool(s) + guardrail.", len(TOOLS))
    return graph
