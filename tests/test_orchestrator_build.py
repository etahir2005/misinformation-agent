"""Tests that build_orchestrator() itself is wired correctly, end to end.

test_orchestrator_routing.py only ever imports the private routing helper
functions directly, never build_orchestrator() itself — so the graph's
structure and, critically, the injected checkpointer's wiring into
builder.compile() have never actually been exercised by a test. These tests
close that gap: they build a real graph with a real (in-memory) checkpointer,
invoke it, and confirm conversation context is actually persisted and
correctly scoped per thread_id, exactly as the README/docstrings claim.

The chat model is mocked throughout — these tests are about graph wiring and
persistence, not about routing decisions (already covered in
test_orchestrator_routing.py) or real LLM behavior.
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.guardrail import MessageIntent
from agent.orchestrator import build_orchestrator


def _mock_model_returning(text: str) -> MagicMock:
    """A mock chat model that always answers immediately with no tool calls.

    bind_tools() returns the same mock regardless of args, so every one of
    build_orchestrator()'s forced tool_choice variants resolves to this one
    mock — fine here, since these tests exercise graph/checkpointer wiring,
    not which tool_choice gets picked.
    """
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content=text)
    return mock_model


@patch("agent.guardrail._get_intent_model")
@patch("agent.orchestrator.init_chat_model")
def test_build_orchestrator_compiles_and_runs_with_injected_checkpointer(
    mock_init_chat_model: MagicMock,
    mock_get_intent_model: MagicMock,
) -> None:
    """build_orchestrator() should compile and produce an answer using the injected checkpointer."""
    mock_init_chat_model.return_value = _mock_model_returning("Final answer.")
    mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")

    graph = build_orchestrator(InMemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="Is the sky blue?")]}, config=config
    )

    assert result["messages"][-1].content == "Final answer."


@patch("agent.guardrail._get_intent_model")
@patch("agent.orchestrator.init_chat_model")
def test_conversation_context_persists_across_turns(
    mock_init_chat_model: MagicMock, mock_get_intent_model: MagicMock
) -> None:
    """A second turn on the same thread_id should see the first turn's history."""
    mock_init_chat_model.return_value = _mock_model_returning("Answer.")
    mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")

    graph = build_orchestrator(InMemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}

    graph.invoke({"messages": [HumanMessage(content="First claim.")]}, config=config)
    graph.invoke({"messages": [HumanMessage(content="Second claim.")]}, config=config)

    state = graph.get_state(config)
    human_messages = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]

    assert len(human_messages) == 2
    assert human_messages[0].content == "First claim."
    assert human_messages[1].content == "Second claim."


@patch("agent.guardrail._get_intent_model")
@patch("agent.orchestrator.init_chat_model")
def test_different_thread_ids_have_independent_history(
    mock_init_chat_model: MagicMock, mock_get_intent_model: MagicMock
) -> None:
    """Two different thread_ids should never see each other's conversation history."""
    mock_init_chat_model.return_value = _mock_model_returning("Answer.")
    mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")

    graph = build_orchestrator(InMemorySaver())

    graph.invoke(
        {"messages": [HumanMessage(content="Thread A claim.")]},
        config={"configurable": {"thread_id": "thread-a"}},
    )
    graph.invoke(
        {"messages": [HumanMessage(content="Thread B claim.")]},
        config={"configurable": {"thread_id": "thread-b"}},
    )

    state_b = graph.get_state({"configurable": {"thread_id": "thread-b"}})
    human_messages_b = [m for m in state_b.values["messages"] if isinstance(m, HumanMessage)]

    assert len(human_messages_b) == 1
    assert human_messages_b[0].content == "Thread B claim."


@patch("agent.guardrail._get_intent_model")
@patch("agent.orchestrator.init_chat_model")
def test_pii_is_redacted_before_reaching_checkpointed_state(
    mock_init_chat_model: MagicMock, mock_get_intent_model: MagicMock
) -> None:
    """pii_scrub_node runs first in the compiled graph — confirms its
    backstop redaction end to end via a direct graph.invoke() call, not just
    as a standalone function call (see tests/test_pii_scrub.py for the
    node's own unit tests). Note this only covers the *final* checkpointed
    state: see tests/test_graph.py's PII test for why the graph node alone
    can't keep a raw claim out of the *earliest* checkpoint, and why
    scrub_pii() in graph.run_claim() is the actual authoritative defense.
    """
    mock_init_chat_model.return_value = _mock_model_returning("Answer.")
    mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")

    graph = build_orchestrator(InMemorySaver())
    config = {"configurable": {"thread_id": "pii-thread"}}

    graph.invoke(
        {"messages": [HumanMessage(content="Email me at jane@example.com about this claim.")]},
        config=config,
    )

    state = graph.get_state(config)
    human_messages = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]

    assert "jane@example.com" not in human_messages[0].content
    assert "[REDACTED_EMAIL]" in human_messages[0].content


@patch("agent.guardrail._get_intent_model")
@patch("agent.orchestrator.summarize")
@patch("agent.orchestrator.init_chat_model")
def test_long_thread_triggers_summarization(
    mock_init_chat_model: MagicMock,
    mock_summarize: MagicMock,
    mock_get_intent_model: MagicMock,
) -> None:
    """Once a thread's older history crosses the threshold, it gets summarized and shrunk."""
    from agent.config import MAX_MESSAGES_BEFORE_SUMMARY

    mock_init_chat_model.return_value = _mock_model_returning("Answer.")
    mock_summarize.return_value = "Summary of everything discussed so far."
    mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")

    graph = build_orchestrator(InMemorySaver())
    config = {"configurable": {"thread_id": "long-thread"}}

    # Each invoke() adds one HumanMessage + one AIMessage (no tool calls,
    # per the mocked model) — enough turns to push the older-message count
    # well past MAX_MESSAGES_BEFORE_SUMMARY. Deliberately generous (not just
    # enough to clear the threshold on the last iteration) — LangGraph's
    # exact timing of when a turn's new message becomes visible to the
    # START routing check vs. downstream nodes isn't something this test
    # should be tightly coupled to.
    num_turns = MAX_MESSAGES_BEFORE_SUMMARY + 6
    for i in range(num_turns):
        graph.invoke({"messages": [HumanMessage(content=f"claim {i}")]}, config=config)

    mock_summarize.assert_called()

    state = graph.get_state(config)
    messages = state.values["messages"]
    summary_messages = [m for m in messages if getattr(m, "id", None) == "conversation-summary"]

    assert len(summary_messages) == 1
    assert summary_messages[0].content == "Summary of everything discussed so far."
    # Older raw messages should have been removed, not just accumulated.
    assert len(messages) < num_turns * 2
