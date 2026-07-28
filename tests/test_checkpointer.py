"""Tests for the Postgres checkpointer factory."""

from unittest.mock import MagicMock, patch

from agent.checkpointer import build_checkpointer


@patch("agent.checkpointer.PostgresSaver")
def test_build_checkpointer_calls_setup_and_yields_saver(mock_saver_cls: MagicMock) -> None:
    """build_checkpointer should call .setup() once and yield the checkpointer."""
    mock_checkpointer = MagicMock()
    mock_saver_cls.from_conn_string.return_value.__enter__.return_value = mock_checkpointer

    with build_checkpointer() as checkpointer:
        assert checkpointer is mock_checkpointer

    mock_checkpointer.setup.assert_called_once()


@patch("agent.checkpointer.PostgresSaver")
def test_build_checkpointer_uses_configured_connection_string(mock_saver_cls: MagicMock) -> None:
    """build_checkpointer should open the connection using POSTGRES_CONNECTION_STRING."""
    from agent.config import POSTGRES_CONNECTION_STRING

    mock_saver_cls.from_conn_string.return_value.__enter__.return_value = MagicMock()

    with build_checkpointer():
        pass

    mock_saver_cls.from_conn_string.assert_called_once_with(POSTGRES_CONNECTION_STRING)
