"""Tests for graph.py's build_graph() and run_claim() helpers."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from agent.guardrail import MessageIntent
from graph import build_graph, run_claim


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
