"""Builds the compiled orchestrator graph and provides a clean way to invoke it.

Kept separate from main.py (the FastAPI layer) and app.py (the Streamlit UI)
so there's one obvious, shared place that knows how to build and run the
graph. main.py imports this directly (it's the FastAPI process itself).
app.py never touches this file at all — it talks to main.py over HTTP
instead, the same way any other client would.
"""

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError

from agent.orchestrator import build_orchestrator
from agent.tracing import get_langfuse_handler

logger = logging.getLogger(__name__)

_MAX_TOOL_LOOP_STEPS = 16


def build_graph(checkpointer: BaseCheckpointSaver):
    """Build the compiled orchestrator graph using the given checkpointer.

    Thin wrapper around agent.orchestrator.build_orchestrator — kept here so
    callers (main.py, cli.py, tests) have one obvious place to get a ready
    graph from, without needing to know anything about agent/ internals.
    """
    return build_orchestrator(checkpointer)


def run_claim(graph, claim: str, thread_id: str, user_id: str | None = None) -> dict:
    """Run a single claim through the graph and return the resulting state.

    Args:
        graph: A compiled graph from build_graph().
        claim: The claim text to check.
        thread_id: Which conversation thread this claim belongs to.
        user_id: Which user owns this thread, if any. Recorded in the
            checkpoint's own metadata (not just passed through) so
            main.py can later confirm thread ownership via
            graph.get_state(...).metadata without a separate table.
            Optional — cli.py has no auth layer and calls this with no
            user_id, which is fine; the thread simply has no owner on
            record.

    Returns:
        A dict with "messages" (the full resulting message list) and
        "recursion_limit_hit" (bool) — True if the graph had to be cut off
        before reaching a final answer, in which case "messages" holds
        whatever partial progress was made rather than a complete answer.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": _MAX_TOOL_LOOP_STEPS,
        "metadata": {"user_id": user_id} if user_id else {},
    }
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]
        config["metadata"]["langfuse_session_id"] = thread_id
    try:
        result = graph.invoke({"messages": [{"role": "user", "content": claim}]}, config=config)
        return {"messages": result["messages"], "recursion_limit_hit": False}
    except GraphRecursionError:
        logger.warning(
            "Recursion limit (%d) hit for thread %s — returning partial progress.",
            _MAX_TOOL_LOOP_STEPS,
            thread_id,
        )
        partial_state = graph.get_state(config)
        return {
            "messages": partial_state.values.get("messages", []),
            "recursion_limit_hit": True,
        }
