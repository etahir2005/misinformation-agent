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
