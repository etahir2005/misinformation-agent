"""Langfuse tracing setup for the orchestrator graph.

Kept as its own module, same lazy-singleton pattern used elsewhere in this
codebase (agent/guardrail.py's _get_intent_model(),
agent/tools/vector_lookup_tool.py's _get_embedding_model()) — avoids doing
client setup at import time, which would otherwise fail test collection
whenever Langfuse credentials aren't configured.
"""

import logging

from agent.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

logger = logging.getLogger(__name__)

_langfuse_handler = None
_checked = False


def get_langfuse_handler():
    """Lazily build the Langfuse callback handler, or None if not configured.

    Returns None rather than raising when Langfuse credentials aren't set,
    so tracing stays opt-in observability, not a hard dependency — the
    graph still runs untraced if Langfuse isn't set up yet.
    """
    global _langfuse_handler, _checked
    if _checked:
        return _langfuse_handler
    _checked = True
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        logger.info("Langfuse credentials not set — running without tracing.")
        return None
    from langfuse.langchain import CallbackHandler

    _langfuse_handler = CallbackHandler()
    return _langfuse_handler
