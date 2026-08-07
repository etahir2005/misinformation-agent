"""Postgres-backed checkpointer factory (Neon), backed by a shared connection pool.

Kept as its own module so agent/orchestrator.py never has to know where
conversation state is actually persisted. build_orchestrator() takes a
checkpointer as a constructor argument (dependency injection) instead of
instantiating one itself — tests pass InMemorySaver, main.py and cli.py
pass the real Postgres-backed one built here.

Uses a psycopg_pool.ConnectionPool rather than a single held-open
connection (the original design here). A single connection goes stale
whenever Neon's free tier auto-suspends its compute after a period of
inactivity — the connection gets forcibly killed server-side
(psycopg.errors.AdminShutdown), and every query against it then fails
until the app is restarted. A pool alone only half-fixes this: by default
it discards a dead connection reactively, after a caller's query has
already failed against it — so it heals for the *next* request but the
one that found the dead connection still errors. The check=
ConnectionPool.check_connection callback below closes that gap: it makes
the pool verify a connection is alive before ever handing it out, so a
dead one is replaced transparently and no caller sees the failure at all.
It also means concurrent requests use separate connections rather than
serializing on one, which matters once multi-user traffic is real.
agent/users_db.py shares this same pool rather than opening its own
short-lived connections per call.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from agent.config import POSTGRES_CONNECTION_STRING

logger = logging.getLogger(__name__)

# Small pool, not a large one — this is a single small app, not a
# high-traffic service, and Neon's own direct-connection ceiling is itself
# limited on the free tier. A handful of connections is enough to avoid
# serializing every request on one connection while staying well under it.
_POOL_MIN_SIZE = 2
_POOL_MAX_SIZE = 10


@contextmanager
def build_connection_pool() -> Iterator[ConnectionPool]:
    """Open a Postgres connection pool for the lifetime of the `with` block.

    autocommit=True is required by PostgresSaver, which manages its own
    transaction boundaries per operation rather than across a session — and
    is also why agent/users_db.py, sharing this pool, no longer needs
    explicit conn.commit() calls. prepare_threshold=0 disables server-side
    prepared-statement caching, which nothing here relies on and which can
    behave oddly across pooled/recycled connections. check=
    ConnectionPool.check_connection makes the pool test each connection's
    liveness before handing it to a caller — without this, the pool only
    notices a connection is dead when a real query fails against it, which
    means that one request still errors even though later ones would
    succeed. This costs a small amount of latency per checkout (a trivial
    validation query) in exchange for callers never seeing a dead
    connection at all.

    Yields:
        An opened ConnectionPool, closed automatically when the `with`
        block exits. Pass this to build_checkpointer() and share it with
        agent/users_db.py rather than opening a second connection source.
    """
    pool = ConnectionPool(
        conninfo=POSTGRES_CONNECTION_STRING,
        min_size=_POOL_MIN_SIZE,
        max_size=_POOL_MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        check=ConnectionPool.check_connection,
        open=False,
    )
    pool.open(wait=True)
    logger.info(
        "Postgres connection pool ready (min=%d, max=%d).", _POOL_MIN_SIZE, _POOL_MAX_SIZE
    )
    try:
        yield pool
    finally:
        pool.close()


def build_checkpointer(pool: ConnectionPool) -> PostgresSaver:
    """Build a Postgres-backed checkpointer using the given connection pool.

    Calls `.setup()`, which runs the checkpointer's own migrations
    (creating the checkpoints/checkpoint_blobs/checkpoint_writes tables and
    tracking schema version) and is idempotent, so it's safe to call on
    every process start instead of relying on a separate one-off setup
    step that's easy to forget before a fresh deploy or a new developer's
    first run.

    Note: Neon's free tier scales compute to zero after a few minutes of
    inactivity and takes a moment to wake on the next connection — the
    first checkpoint read/write after a period of inactivity may be
    noticeably slower than the rest (the pool's check callback validating
    and replacing the stale connection), not an application bug. Unlike
    the single-connection design this replaced, a connection killed during
    that suspend/resume cycle is caught and replaced before any caller's
    query runs against it, rather than causing that query to fail.

    Returns:
        A PostgresSaver ready to pass to build_orchestrator().
    """
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    logger.info("Postgres checkpointer ready.")
    return checkpointer
