"""Tests for graph.py's build_graph() and run_claim() helpers."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agent.guardrail import MessageIntent
from graph import build_graph, resume_review, run_claim


@patch("agent.guardrail._get_intent_model")
@patch("agent.orchestrator.init_chat_model")
def test_run_claim_returns_final_answer(
    mock_init_chat_model: MagicMock, mock_get_intent_model: MagicMock
) -> None:
    """A normal claim should resolve to a final answer with no recursion issues."""
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Final answer.")
    mock_init_chat_model.return_value = mock_model
    mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")

    graph = build_graph(InMemorySaver())
    result = run_claim(graph, "Is the sky blue?", "test-thread")

    assert result["recursion_limit_hit"] is False
    assert result["messages"][-1].content == "Final answer."


@patch("agent.guardrail._get_intent_model")
@patch("agent.orchestrator.init_chat_model")
def test_run_claim_never_checkpoints_raw_pii(
    mock_init_chat_model: MagicMock, mock_get_intent_model: MagicMock
) -> None:
    """run_claim() — not pii_scrub_node — is the actual authoritative PII
    defense, because it scrubs the claim string before graph.invoke() is
    ever called. This matters because LangGraph checkpoints the exact
    invoke() payload before running any node, including the graph's own
    first node — confirmed empirically the hard way: an earlier version of
    this test called graph.invoke() directly (bypassing run_claim()) and
    found the raw claim still present in the earliest checkpoint snapshot
    even with pii_scrub_node running first in the graph.

    Walks graph.get_state_history() (every checkpoint ever written for the
    thread, oldest included) rather than just graph.get_state() (the final
    checkpoint only) — the whole point is confirming no snapshot, not just
    the last one, ever held the raw claim.
    """
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Answer.")
    mock_init_chat_model.return_value = mock_model
    mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")

    graph = build_graph(InMemorySaver())
    run_claim(graph, "Email me at jane@example.com about this claim.", "pii-history-thread")

    for snapshot in graph.get_state_history({"configurable": {"thread_id": "pii-history-thread"}}):
        for message in snapshot.values.get("messages", []):
            if isinstance(message, HumanMessage):
                assert "jane@example.com" not in message.content


@patch("agent.guardrail._get_intent_model")
@patch("agent.orchestrator.init_chat_model")
def test_run_claim_returns_the_scrubbed_claim_not_the_raw_input(
    mock_init_chat_model: MagicMock, mock_get_intent_model: MagicMock
) -> None:
    """Regression test: callers (main.py's derive_title() in particular)
    must be able to get the *scrubbed* claim text back from run_claim(),
    not just have it redacted internally — otherwise a caller that derives
    something display- or storage-bound from the original claim string
    (e.g. a conversation title) ends up with raw PII in it even though the
    claim itself was correctly redacted everywhere else. Caught via a live
    Streamlit smoke test, not the mocked test suite.
    """
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Answer.")
    mock_init_chat_model.return_value = mock_model
    mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")

    graph = build_graph(InMemorySaver())
    result = run_claim(graph, "Email me at jane@example.com about this claim.", "test-thread")

    assert "jane@example.com" not in result["claim"]
    assert "[REDACTED_EMAIL]" in result["claim"]


@patch("graph.build_orchestrator")
def test_run_claim_handles_recursion_limit_gracefully(mock_build_orchestrator: MagicMock) -> None:
    """A GraphRecursionError should return partial progress, not raise into the caller."""
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = GraphRecursionError("too many steps")
    mock_graph.get_state.return_value.values = {
        "messages": [HumanMessage(content="a claim"), AIMessage(content="partial progress")]
    }
    mock_build_orchestrator.return_value = mock_graph

    graph = build_graph(InMemorySaver())
    result = run_claim(graph, "a hard claim", "test-thread")

    assert result["recursion_limit_hit"] is True
    assert result["messages"][-1].content == "partial progress"


def test_resume_review_invokes_with_command_resume_scoped_to_thread() -> None:
    """resume_review() should hand LangGraph's own Command(resume=...)
    wrapper to graph.invoke(), scoped to the given thread_id via
    config["configurable"] — this is the actual mechanism that unblocks a
    thread paused at human_review_node (agent/orchestrator.py).
    """
    mock_graph = MagicMock()
    mock_graph.get_state.return_value.metadata = {"user_id": "user-abc"}

    resume_review(mock_graph, "some-thread-id", "approve")

    mock_graph.invoke.assert_called_once()
    args, kwargs = mock_graph.invoke.call_args
    resume_command = args[0]
    assert isinstance(resume_command, Command)
    assert resume_command.resume == "approve"
    assert kwargs["config"]["configurable"]["thread_id"] == "some-thread-id"


def test_resume_review_preserves_thread_ownership_metadata() -> None:
    """Regression test for a live bug: LangGraph checkpoint metadata isn't
    cumulative across invokes — each call's metadata only applies to the
    checkpoint(s) that call writes, it doesn't carry forward from the
    thread's prior checkpoints automatically. run_claim() always sets
    metadata={"user_id": ...}; resume_review() originally invoked with no
    metadata at all, so the checkpoint(s) written by the admin's
    approve/reject ended up with no user_id, and main.py's
    _get_thread_owner() (which reads the *latest* checkpoint's metadata)
    would then see the thread as ownerless — silently breaking that user's
    ability to continue, view, or delete their own conversation. Caught
    live: after an admin resolved an escalation, the original user got a
    403 "belongs to a different account" on their own thread, and a later
    message on that thread hit a duplicate-key error trying to re-create
    its sidebar conversation row (main.py treated it as brand new).

    Uses a real graph + InMemorySaver (not a mock) so this actually proves
    the fix against LangGraph's real metadata behavior, not an assumption
    about it.
    """
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    class _State(TypedDict):
        value: str

    def _pausing_node(state: _State) -> dict:
        interrupt({"reason": "test_pause"})
        return {"value": "resumed"}

    builder = StateGraph(_State)
    builder.add_node("pause", _pausing_node)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    thread_id = "ownership-metadata-thread"
    config = {"configurable": {"thread_id": thread_id}, "metadata": {"user_id": "user-123"}}
    graph.invoke({"value": "start"}, config=config)

    resume_review(graph, thread_id, "approve")

    final_state = graph.get_state({"configurable": {"thread_id": thread_id}})
    assert final_state.metadata.get("user_id") == "user-123"
