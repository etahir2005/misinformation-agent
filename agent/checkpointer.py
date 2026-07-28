"""Postgres-backed checkpointer factory (Neon).

Kept as its own module so agent/orchestrator.py never has to know where
conversation state is actually persisted. build_orchestrator() takes a
checkpointer as a constructor argument (dependency injection) instead of
instantiating one itself — tests pass InMemorySaver, main.py and the future
Streamlit app pass the real Postgres-backed one built here.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver

from agent.config import POSTGRES_CONNECTION_STRING

logger = logging.getLogger(__name__)


@contextmanager
def build_checkpointer() -> Iterator[PostgresSaver]:
    """Open a Postgres-backed checkpointer for the lifetime of the `with` block.

    Calls `.setup()` on every open. That runs the checkpointer's own
    migrations (creating the checkpoints/checkpoint_blobs/checkpoint_writes
    tables and tracking schema version) and is idempotent, so it's safe to
    call on every process start instead of relying on a separate one-off
    setup step that's easy to forget before a fresh deploy or a new
    developer's first run.

    Note: Neon's free tier scales compute to zero after a few minutes of
    inactivity and takes a moment to wake on the next connection — the
    first checkpoint read/write after a period of inactivity may be
    noticeably slower than the rest, not an application bug.

    Yields:
        A PostgresSaver ready to pass to build_orchestrator(). The
        underlying connection pool is closed automatically when the `with`
        block exits.
    """
    with PostgresSaver.from_conn_string(POSTGRES_CONNECTION_STRING) as checkpointer:
        checkpointer.setup()
        logger.info("Postgres checkpointer ready.")
        yield checkpointer
