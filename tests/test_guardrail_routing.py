"""Tests for the guardrail's graph-level wiring in agent/orchestrator.py."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.guardrail import MessageIntent
from agent.orchestrator import build_orchestrator


def _build_graph_with_mocked_model():
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Final answer.")
    with patch("agent.orchestrator.init_chat_model", return_value=mock_model):
        return build_orchestrator(InMemorySaver())


def test_greeting_short_circuits_before_orchestrator() -> None:
    """A greeting should never reach any of the five tools."""
    mock_intent_model = MagicMock()
    mock_intent_model.invoke.return_value = MessageIntent(category="greeting")

    with patch("agent.guardrail._get_intent_model", return_value=mock_intent_model):
        graph = _build_graph_with_mocked_model()
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "hi there"}]},
            config={"configurable": {"thread_id": "test-thread"}},
        )

    tool_messages = [m for m in result["messages"] if getattr(m, "name", None) is not None]
    assert tool_messages == []
    assert "fact-checked" in result["messages"][-1].content.lower()


def test_out_of_scope_short_circuits_before_orchestrator() -> None:
    """An out-of-scope message should never reach any of the five tools."""
    mock_intent_model = MagicMock()
    mock_intent_model.invoke.return_value = MessageIntent(category="out_of_scope")

    with patch("agent.guardrail._get_intent_model", return_value=mock_intent_model):
        graph = _build_graph_with_mocked_model()
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "write me a poem"}]},
            config={"configurable": {"thread_id": "test-thread"}},
        )

    tool_messages = [m for m in result["messages"] if getattr(m, "name", None) is not None]
    assert tool_messages == []
    assert "fact-checking" in result["messages"][-1].content.lower()


def test_claim_proceeds_past_guardrail_into_orchestrator() -> None:
    """A genuine claim should reach the orchestrator and trigger the forced pipeline."""
    mock_intent_model = MagicMock()
    mock_intent_model.invoke.return_value = MessageIntent(category="claim")

    with patch("agent.guardrail._get_intent_model", return_value=mock_intent_model):
        graph = _build_graph_with_mocked_model()
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Is the sky blue?"}]},
            config={"configurable": {"thread_id": "test-thread"}},
        )

    assert result["messages"][-1].content == "Final answer."
