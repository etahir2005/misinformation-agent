"""Postgres-backed conversation listing for the sidebar/resume feature.

Stores one row per conversation thread — title and recency, for listing
purposes only. This is deliberately NOT a second copy of conversation
content: the actual messages and full graph state live entirely in
LangGraph's own checkpoint tables, keyed by the same thread_id. This
table exists only because those checkpoint tables aren't shaped to
answer "list this user's conversations, most recent first" efficiently
on their own (see README/schema notes). Shares the same
psycopg_pool.ConnectionPool as agent/checkpointer.py and
agent/users_db.py rather than opening a separate connection source.
"""

import logging

from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_TITLE_MAX_LENGTH = 60


def setup_conversations_table(pool: ConnectionPool) -> None:
    """Create the conversations table and its lookup index if missing.

    Idempotent (CREATE TABLE/INDEX IF NOT EXISTS), same reasoning as
    setup_users_table — safe to call on every app startup.
    """
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id UUID PRIMARY KEY,
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New conversation',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations (user_id, updated_at DESC)
            """
        )
    logger.info("Conversations table ready.")


def derive_title(claim: str) -> str:
    """Build a short, human-readable sidebar title from a claim's text.

    Deliberately simple truncation rather than an LLM-generated title —
    an extra model call on every new conversation isn't worth the added
    cost/latency just to produce a label, and truncation is good enough
    for a scannable sidebar list.
    """
    claim = claim.strip()
    if len(claim) <= _TITLE_MAX_LENGTH:
        return claim
    return claim[:_TITLE_MAX_LENGTH].rstrip() + "…"


def create_conversation(pool: ConnectionPool, thread_id: str, user_id: str, title: str) -> None:
    """Insert a new conversation row. Called once, the first time a thread is used."""
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO conversations (thread_id, user_id, title) VALUES (%s, %s, %s)",
            (thread_id, user_id, title),
        )


def touch_conversation(pool: ConnectionPool, thread_id: str) -> None:
    """Bump a conversation's updated_at to now().

    Called on every later message in an already-existing thread, so the
    sidebar can sort by most-recently-active rather than just creation time.
    """
    with pool.connection() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = now() WHERE thread_id = %s",
            (thread_id,),
        )


def list_conversations(pool: ConnectionPool, user_id: str) -> list[dict]:
    """Return this user's conversations, most recently active first."""
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at "
            "FROM conversations WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [
        {
            "thread_id": str(row[0]),
            "title": row[1],
            "created_at": row[2].isoformat(),
            "updated_at": row[3].isoformat(),
        }
        for row in rows
    ]
