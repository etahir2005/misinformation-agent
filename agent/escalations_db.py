"""Postgres-backed storage for human-in-the-loop review escalations.

One row per turn that got paused at agent/orchestrator.py's
human_review_node — deliberately its own table, not a column bolted onto
conversations (agent/conversations_db.py), since an escalation is a
review-queue item scoped to a single flagged verdict, not a property of
the whole conversation thread. A thread could in principle be flagged more
than once across different claims, so thread_id is a plain (non-unique)
column here, not the primary key.

Shares the same psycopg_pool.ConnectionPool as the other *_db.py modules
(see agent/checkpointer.py) rather than opening a separate connection
source.
"""

import logging
import uuid

from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


def setup_escalations_table(pool: ConnectionPool) -> None:
    """Create the escalations table and its lookup index if missing.

    Idempotent (CREATE TABLE/INDEX IF NOT EXISTS), same reasoning as the
    other *_db.py setup functions — safe to call on every app startup.
    """
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS escalations (
                id UUID PRIMARY KEY,
                thread_id UUID NOT NULL,
                claim TEXT NOT NULL,
                verdict TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                resolved_at TIMESTAMPTZ
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_escalations_status_created
                ON escalations (status, created_at DESC)
            """
        )
    logger.info("Escalations table ready.")


def create_escalation(
    pool: ConnectionPool, thread_id: str, claim: str, verdict: str, reason: str
) -> None:
    """Insert a new pending escalation row.

    Called from main.py's /chat handler when run_claim() reports a
    pending_review for this turn — see graph.py's run_claim() docstring.
    """
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO escalations (id, thread_id, claim, verdict, reason) "
            "VALUES (%s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), thread_id, claim, verdict, reason),
        )


def list_pending_escalations(pool: ConnectionPool) -> list[dict]:
    """Return every still-pending escalation, oldest first.

    Oldest first (not most-recent-first, unlike conversations_db's
    recency-sorted list) — a review queue should surface what's been
    waiting longest, not bury it under newer items.
    """
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT thread_id, claim, verdict, reason, created_at "
            "FROM escalations WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
    return [
        {
            "thread_id": str(row[0]),
            "claim": row[1],
            "verdict": row[2],
            "reason": row[3],
            "created_at": row[4].isoformat(),
        }
        for row in rows
    ]


def get_pending_escalation(pool: ConnectionPool, thread_id: str) -> dict | None:
    """Fetch this thread's current pending escalation, or None if there isn't one.

    A thread could in principle have more than one *resolved* escalation
    in its history, but should only ever have at most one still-pending
    at a time (the graph stays paused on that turn until it's resumed) —
    LIMIT 1 with the most recent pending row is a safe, simple read
    regardless.
    """
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT thread_id, claim, verdict, reason, status "
            "FROM escalations WHERE thread_id = %s AND status = 'pending' "
            "ORDER BY created_at DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "thread_id": str(row[0]),
        "claim": row[1],
        "verdict": row[2],
        "reason": row[3],
        "status": row[4],
    }


def resolve_escalation(pool: ConnectionPool, thread_id: str, status: str) -> None:
    """Mark this thread's pending escalation resolved.

    status is usually "approved"/"rejected" (the admin's real decision via
    main.py's /admin/escalations/{thread_id}/resolve), but main.py's
    conversation-delete endpoint also calls this with "cancelled" when a
    user deletes a conversation that still has a pending review — the
    checkpoint data a later resume would need is gone at that point, so the
    escalation is cleared rather than left to error confusingly if an admin
    later tries to act on it.

    Only updates rows still in 'pending' status — resolving an
    already-resolved (or nonexistent) escalation is a silent no-op rather
    than an error, same convention as conversations_db.delete_conversation.
    """
    with pool.connection() as conn:
        conn.execute(
            "UPDATE escalations SET status = %s, resolved_at = now() "
            "WHERE thread_id = %s AND status = 'pending'",
            (status, thread_id),
        )
