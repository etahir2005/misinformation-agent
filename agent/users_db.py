"""Postgres-backed user storage for authentication.

Kept separate from agent/checkpointer.py — that module owns LangGraph's own
checkpoint tables (created and migrated by PostgresSaver.setup()); this one
owns a single app-specific table (`users`) that LangGraph knows nothing
about. Mixing the two would make it unclear which code is responsible for
which schema.

Shares the same psycopg_pool.ConnectionPool as the checkpointer (built in
agent/checkpointer.py, opened once by main.py's lifespan) rather than
opening its own connections — one right-sized, resilient pool for the
whole app, not two separate connection strategies. The pool's connections
are autocommit=True (set where the pool is built), so no explicit
conn.commit() calls are needed here.
"""

import logging
import uuid

from psycopg.errors import UniqueViolation
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class EmailAlreadyRegisteredError(Exception):
    """Raised when signup is attempted with an email that's already in use."""


def setup_users_table(pool: ConnectionPool) -> None:
    """Create the users table if it doesn't exist yet.

    Idempotent and safe to call on every app startup — same reasoning as
    PostgresSaver.setup() in agent/checkpointer.py: a new developer or a
    fresh deploy should never need a separate, easy-to-forget migration
    step run by hand.
    """
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    logger.info("Users table ready.")


def create_user(pool: ConnectionPool, email: str, password_hash: str) -> str:
    """Insert a new user and return their generated UUID as a string.

    Raises:
        EmailAlreadyRegisteredError: translated from Postgres's raw
        UniqueViolation so callers (the signup endpoint) don't need to know
        this is backed by a SQL unique constraint.
    """
    user_id = str(uuid.uuid4())
    try:
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash) VALUES (%s, %s, %s)",
                (user_id, email, password_hash),
            )
        return user_id
    except UniqueViolation as exc:
        raise EmailAlreadyRegisteredError(email) from exc


def get_user_by_email(pool: ConnectionPool, email: str) -> dict | None:
    """Fetch a user by email, or None if no such user exists."""
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s",
            (email,),
        ).fetchone()
    if row is None:
        return None
    return {"id": str(row[0]), "email": row[1], "password_hash": row[2]}
