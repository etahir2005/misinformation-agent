"""Tests for the conversation summarization module."""

from unittest.mock import MagicMock, patch

from agent.summarizer import summarize


@patch("agent.summarizer._get_summarization_model")
def test_summarize_returns_new_summary_text(mock_get_model: MagicMock) -> None:
    """summarize() should return the model's response content."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="Updated summary text.")
    mock_get_model.return_value = mock_model

    result = summarize(existing_summary=None, conversation_text="Human: claim 1\nAI: false")

    assert result == "Updated summary text."


@patch("agent.summarizer._get_summarization_model")
def test_summarize_includes_existing_summary_in_prompt(mock_get_model: MagicMock) -> None:
    """The prior summary and new conversation text should both reach the model."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="Combined summary.")
    mock_get_model.return_value = mock_model

    summarize(existing_summary="Prior summary.", conversation_text="Human: claim 2")

    sent_prompt = mock_model.invoke.call_args[0][0]
    assert "Prior summary." in sent_prompt
    assert "Human: claim 2" in sent_prompt


@patch("agent.summarizer._get_summarization_model")
def test_summarize_handles_failure_gracefully(mock_get_model: MagicMock) -> None:
    """A failed Gemini call should return None, not raise."""
    mock_model = MagicMock()
    mock_model.invoke.side_effect = RuntimeError("API error")
    mock_get_model.return_value = mock_model

    result = summarize(existing_summary=None, conversation_text="Human: claim")

    assert result is None


@patch("agent.summarizer._get_summarization_model")
def test_summarize_normalizes_list_content_response(mock_get_model: MagicMock) -> None:
    """A Gemini response with list-of-blocks content should collapse to plain text.

    Regression test: caught via live verification (verify_summarization.py)
    against the real Gemini API, which returned response.content as
    [{"type": "text", "text": "...", "extras": {...}}] instead of a plain
    string. summarize() used to return that list as-is, so it ended up
    embedded — as a raw Python list repr, signature/extras and all — in the
    next summarization prompt and in the orchestrator's system prompt.
    """
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(
        content=[
            {"type": "text", "text": "Part one. ", "extras": {"signature": "abc123"}},
            {"type": "text", "text": "Part two."},
        ]
    )
    mock_get_model.return_value = mock_model

    result = summarize(existing_summary=None, conversation_text="Human: claim")

    assert result == "Part one. Part two."
