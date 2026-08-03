"""Manual CLI entry point for testing one claim end to end.

Superseded main.py's old role once main.py became the FastAPI app. Kept as
its own script since it's still useful for quick manual testing without
needing the FastAPI server or Streamlit running.
"""

import sys
import uuid

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from agent.checkpointer import build_checkpointer  # noqa: E402
from graph import build_graph, run_claim  # noqa: E402


def main() -> None:
    claim = "Is it true that the Great Wall of China is visible from space?"
    thread_id = str(uuid.uuid4())

    with build_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        result = run_claim(graph, claim, thread_id)

        if result["recursion_limit_hit"]:
            print(
                "Couldn't reach a confident answer within the tool-call limit "
                "— evidence may be unusually thin or conflicting. Partial progress:\n"
            )
        for message in result["messages"]:
            message.pretty_print()


if __name__ == "__main__":
    main()
