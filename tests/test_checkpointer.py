"""Tests for the Postgres checkpointer factory and connection pool."""

from unittest.mock import MagicMock, patch

from psycopg_pool import ConnectionPool

from agent.checkpointer import build_checkpointer, build_connection_pool
from agent.config import POSTGRES_CONNECTION_STRING


@patch("agent.checkpointer.ConnectionPool")
def test_build_connection_pool_opens_and_closes(mock_pool_cls: MagicMock) -> None:
    """The pool should be explicitly opened on entry and closed on exit."""
    mock_pool = MagicMock()
    mock_pool_cls.return_value = mock_pool

    with build_connection_pool() as pool:
        assert pool is mock_pool
        mock_pool.open.assert_called_once_with(wait=True)
        mock_pool.close.assert_not_called()

    mock_pool.close.assert_called_once()


@patch("agent.checkpointer.ConnectionPool")
def test_build_connection_pool_uses_configured_connection_string(
    mock_pool_cls: MagicMock,
) -> None:
    """The pool should connect using POSTGRES_CONNECTION_STRING, with autocommit on."""
    mock_pool_cls.return_value = MagicMock()

    with build_connection_pool():
        pass

    _, kwargs = mock_pool_cls.call_args
    assert kwargs["conninfo"] == POSTGRES_CONNECTION_STRING
    assert kwargs["kwargs"]["autocommit"] is True


@patch("agent.checkpointer.ConnectionPool")
def test_build_connection_pool_validates_connections_before_handing_them_out(
    mock_pool_cls: MagicMock,
) -> None:
    """The pool must use check=ConnectionPool.check_connection.

    Without this, a connection killed by Neon's auto-suspend is only
    discovered *after* a caller's query fails against it — this check
    callback is what makes the pool catch it first instead.
    """
    mock_pool_cls.return_value = MagicMock()
    mock_pool_cls.check_connection = ConnectionPool.check_connection

    with build_connection_pool():
        pass

    _, kwargs = mock_pool_cls.call_args
    assert kwargs["check"] is ConnectionPool.check_connection


@patch("agent.checkpointer.PostgresSaver")
def test_build_checkpointer_calls_setup_and_returns_saver(mock_saver_cls: MagicMock) -> None:
    """build_checkpointer should build a PostgresSaver from the given pool and call .setup()."""
    mock_checkpointer = MagicMock()
    mock_saver_cls.return_value = mock_checkpointer
    mock_pool = MagicMock()

    checkpointer = build_checkpointer(mock_pool)

    assert checkpointer is mock_checkpointer
    mock_saver_cls.assert_called_once_with(mock_pool)
    mock_checkpointer.setup.assert_called_once()
