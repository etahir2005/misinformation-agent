"""Tests for the escalation-routing pieces of Phase 3's human-in-the-loop
work: route_after_orchestrator, human_review_node, and the underlying
LangGraph interrupt() mechanics that graph.py's run_claim() depends on.

The last test doesn't touch agent/orchestrator.py's real graph at all — it
uses a minimal standalone graph, almost identical to LangGraph's own
interrupt() documentation example, specifically to empirically confirm
what graph.invoke() actually returns when a node pauses (a "__interrupt__"
key holding a tuple of Interrupt objects). That return shape is what
graph.py's run_claim() reads (result.get("__interrupt__")) — this proves
the assumption rather than trusting it from reading docs alone.
"""

from typing import TypedDict
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.orchestrator import human_review_node, route_after_orchestrator


def test_route_after_orchestrator_continues_tool_loop_when_tool_calls_present() -> None:
    state = {
        "messages": [
            HumanMessage(content="claim"),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search_tool", "args": {}, "id": "call_1"}],
            ),
        ],
        "verdict_is_complete": None,
    }
    assert route_after_orchestrator(state) == "tools"


def test_route_after_orchestrator_escalates_when_verdict_incomplete() -> None:
    state = {
        "messages": [HumanMessage(content="claim"), AIMessage(content="A vague answer.")],
        "verdict_is_complete": False,
    }
    assert route_after_orchestrator(state) == "human_review"


def test_route_after_orchestrator_ends_when_verdict_complete() -> None:
    state = {
        "messages": [HumanMessage(content="claim"), AIMessage(content="A complete answer.")],
        "verdict_is_complete": True,
    }
    assert route_after_orchestrator(state) == END


def test_route_after_orchestrator_ends_when_verdict_is_complete_unset() -> None:
    """Greetings, out-of-scope replies, and cache hits never set
    verdict_is_complete — None must still end normally, not be mistaken
    for an incomplete verdict.
    """
    state = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="Hi! Send me a claim.")],
        "verdict_is_complete": None,
    }
    assert route_after_orchestrator(state) == END


@patch("agent.orchestrator._store_verdict_if_new")
@patch("agent.orchestrator.interrupt")
def test_human_review_node_calls_interrupt_with_claim_and_verdict(
    mock_interrupt: MagicMock, mock_store: MagicMock
) -> None:
    mock_interrupt.return_value = None
    state = {
        "messages": [
            HumanMessage(content="Is the sky green?"),
            AIMessage(content="The evidence is unclear."),
        ],
        "verdict_is_complete": False,
    }

    result = human_review_node(state)

    assert result == {}
    mock_interrupt.assert_called_once()
    payload = mock_interrupt.call_args[0][0]
    assert payload["reason"] == "verdict_incomplete"
    assert payload["claim"] == "Is the sky green?"
    assert payload["verdict"] == "The evidence is unclear."
    mock_store.assert_not_called()


@patch("agent.orchestrator._store_verdict_if_new")
@patch("agent.orchestrator.interrupt")
def test_human_review_node_stores_verdict_when_admin_approves(
    mock_interrupt: MagicMock, mock_store: MagicMock
) -> None:
    """A resume value of "approve" (Command(resume="approve") on the real
    graph) should cache the verdict — this is the whole point of the
    admin's approval, see human_review_node's docstring on what the
    review is actually gating (the shared cache, not the answer already
    shown to the asker).
    """
    mock_interrupt.return_value = "approve"
    state = {
        "messages": [
            HumanMessage(content="Is the sky green?"),
            AIMessage(content="The evidence is unclear."),
        ],
        "verdict_is_complete": False,
    }

    result = human_review_node(state)

    assert result == {}
    mock_store.assert_called_once()


@patch("agent.orchestrator._store_verdict_if_new")
@patch("agent.orchestrator.interrupt")
def test_human_review_node_does_not_store_verdict_when_admin_rejects(
    mock_interrupt: MagicMock, mock_store: MagicMock
) -> None:
    mock_interrupt.return_value = "reject"
    state = {
        "messages": [
            HumanMessage(content="Is the sky green?"),
            AIMessage(content="The evidence is unclear."),
        ],
        "verdict_is_complete": False,
    }

    result = human_review_node(state)

    assert result == {}
    mock_store.assert_not_called()


class _MinimalState(TypedDict):
    value: str


def _pausing_node(state: _MinimalState) -> dict:
    interrupt({"reason": "test_pause"})
    return {}


def test_langgraph_invoke_surfaces_pending_interrupt_in_returned_dict() -> None:
    """Empirically confirms (not just reads from docs) what graph.invoke()
    returns when a node calls interrupt(): a "__interrupt__" key holding a
    tuple of Interrupt objects, each with a .value attribute matching what
    was passed to interrupt(). This is exactly what graph.py's run_claim()
    reads via result.get("__interrupt__") and interrupts[0].value.
    """
    builder = StateGraph(_MinimalState)
    builder.add_node("pause", _pausing_node)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "interrupt-test-thread"}}
    result = graph.invoke({"value": "start"}, config=config)

    assert "__interrupt__" in result
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    assert interrupts[0].value == {"reason": "test_pause"}
