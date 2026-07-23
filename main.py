"""Entry point for manually testing the orchestrator end to end."""

import uuid

from agent.orchestrator import build_orchestrator


def run_claim(claim: str) -> None:
    """Run a single claim through the orchestrator and print the result.

    Args:
        claim: The claim text to fact-check.
    """
    graph = build_orchestrator()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(
        {"messages": [{"role": "user", "content": claim}]},
        config=config,
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    test_claim = "Is it true that the Great Wall of China is visible from space?"
    run_claim(test_claim)
