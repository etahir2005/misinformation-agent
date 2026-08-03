"""Tests for agent/guardrail.py's classify_message_intent()."""

from unittest.mock import MagicMock, patch

import pytest

from agent.guardrail import MessageIntent, classify_message_intent


@pytest.mark.parametrize("greeting", ["hi", "Hello", "HEY", "good morning", "howdy!"])
def test_classify_message_intent_catches_greetings_via_keyword(greeting: str) -> None:
    """Obvious greetings should never reach the model — no call should happen."""
    with patch("agent.guardrail._get_intent_model") as mock_get_model:
        result = classify_message_intent(greeting)

    assert result == "greeting"
    mock_get_model.assert_not_called()


def test_classify_message_intent_calls_model_for_ambiguous_message() -> None:
    """Anything not matching the keyword list goes through the classifier call."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = MessageIntent(category="claim")

    with patch("agent.guardrail._get_intent_model", return_value=mock_model):
        result = classify_message_intent("Is the sky blue?")

    assert result == "claim"
    mock_model.invoke.assert_called_once()


def test_classify_message_intent_returns_out_of_scope_for_unrelated_request() -> None:
    """A clearly off-topic, non-greeting request should classify as out_of_scope."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = MessageIntent(category="out_of_scope")

    with patch("agent.guardrail._get_intent_model", return_value=mock_model):
        result = classify_message_intent("Can you write me a poem about autumn?")

    assert result == "out_of_scope"
