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
from langgraph.types import Command

from agent.orchestrator import build_orchestrator, scrub_pii
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
        A dict with "messages" (the full resulting message list),
        "recursion_limit_hit" (bool) — True if the graph had to be cut off
        before reaching a final answer, in which case "messages" holds
        whatever partial progress was made rather than a complete answer —
        "pending_review" (dict | None) — set when the graph paused at
        human_review_node (agent/orchestrator.py) because
        verdict_is_complete was False for this turn. "messages" is still
        populated in that case (the not-yet-reviewed answer), so callers
        that don't check pending_review keep working exactly as before;
        this key is additive, not a breaking change to the return shape.
        Nothing resumes the graph yet — that's Phase 4/5's job, once a
        human actually has a way to submit a decision — and "claim" (str),
        the *scrubbed* version of the input claim. Callers that derive
        anything display- or storage-bound from the claim text (e.g.
        main.py's derive_title() for the sidebar conversation list) must
        use this, not their own original input — scrub_pii() only redacts
        the copy passed into graph.invoke(); it was never handed back to
        the caller before, which is exactly how a raw email ended up
        stored in a conversation's title despite this same claim's content
        being correctly redacted everywhere else. Caught via a live
        Streamlit smoke test, not by the (fully mocked) test suite.
    """
    claim = scrub_pii(claim)
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
        interrupts = result.get("__interrupt__")
        pending_review = interrupts[0].value if interrupts else None
        return {
            "messages": result.get("messages", []),
            "recursion_limit_hit": False,
            "pending_review": pending_review,
            "claim": claim,
        }
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
            "pending_review": None,
            "claim": claim,
        }


def resume_review(graph, thread_id: str, decision: str) -> None:
    """Resume a thread paused at human_review_node with an admin's decision.

    decision is "approve" or "reject" (see main.py's ResolveEscalationRequest
    and agent/orchestrator.py's human_review_node) — passed straight through
    as the interrupt()'s resume value. No return value: the original asker
    already has their answer (it was shown before the pause ever happened,
    see human_review_node's docstring), so resuming here only affects
    whether the verdict gets written into the shared semantic cache. Callers
    don't need anything back beyond confirmation that the resume itself
    didn't raise.

    No PII-scrub or recursion-limit handling needed here unlike run_claim()
    — this doesn't process new user input, it only unblocks a node that's
    already paused mid-graph, so neither concern applies.

    Re-attaches the thread's existing owner as metadata before resuming.
    LangGraph checkpoint metadata isn't cumulative across invokes — each
    call's metadata only applies to the checkpoint(s) that call writes, it
    doesn't carry forward from the thread's prior checkpoints on its own.
    run_claim() always sets metadata={"user_id": ...} on the checkpoint it
    creates; without re-passing that here, the checkpoint(s) written by
    this resumed run would end up with no user_id at all, and
    main.py's _get_thread_owner() (which reads the *latest* checkpoint's
    metadata) would then see the thread as ownerless — silently breaking
    that user's ability to continue, view, or delete their own
    conversation the moment an admin resolves its escalation. Confirmed
    empirically with a standalone interrupt/resume graph: a resume with no
    metadata drops user_id from the next checkpoint entirely; re-passing
    the looked-up owner_id here keeps it intact.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    owner_id = state.metadata.get("user_id") if state and state.metadata else None
    resume_config = {"configurable": {"thread_id": thread_id}}
    if owner_id:
        resume_config["metadata"] = {"user_id": owner_id}
    graph.invoke(Command(resume=decision), config=resume_config)
