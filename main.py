"""Entry point for manually testing the orchestrator end to end."""

import sys
import uuid

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langgraph.errors import GraphRecursionError  # noqa: E402

from agent.orchestrator import build_orchestrator  # noqa: E402

_MAX_TOOL_LOOP_STEPS = 14


def run_claim(claim: str) -> None:
    """Run a single claim through the orchestrator and print the result.

    Args:
        claim: The claim text to fact-check.
    """
    graph = build_orchestrator()
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": _MAX_TOOL_LOOP_STEPS,
    }

    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": claim}]},
            config=config,
        )
    except GraphRecursionError:
        print(
            "Couldn't reach a confident answer within the tool-call limit "
            f"({_MAX_TOOL_LOOP_STEPS} steps) — the evidence may be unusually "
            "thin or conflicting for this claim. Partial progress:\n"
        )
        partial_state = graph.get_state(config)
        for message in partial_state.values.get("messages", []):
            message.pretty_print()
        return

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    test_claim = "Video games cause violent behavior in teenagers."
    run_claim(test_claim)
