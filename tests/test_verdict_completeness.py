"""Tests for the verdict-completeness check (agent/verdict_completeness.py).

_get_judge_model() is a lazy singleton, same pattern as
agent/guardrail.py's _get_intent_model() — patched directly (not
init_chat_model, which a patch wouldn't retroactively affect once the real
model's already been built and cached) via its return_value, same
convention as tests/test_guardrail.py.
"""

from unittest.mock import MagicMock, patch

from agent.config import VERDICT_MIN_WORD_COUNT
from agent.verdict_completeness import _CompletenessJudgment, check_verdict_completeness

_LONG_VERDICT = (
    "This claim is false. Multiple fact-checking organizations reviewed the "
    "underlying evidence and found no credible support for it, citing "
    "several independent studies that directly contradict the claim."
)
_SOURCES = [{"url": "https://example.com", "content": "some evidence text"}]


def _judgment(is_complete: bool) -> _CompletenessJudgment:
    return _CompletenessJudgment(is_complete=is_complete, reasoning="test reasoning")


def test_fails_deterministic_check_when_no_sources() -> None:
    assert check_verdict_completeness("a claim", _LONG_VERDICT, sources=[]) is False


def test_fails_deterministic_check_when_too_short() -> None:
    short_verdict = " ".join(["word"] * (VERDICT_MIN_WORD_COUNT - 1))
    assert check_verdict_completeness("a claim", short_verdict, _SOURCES) is False


@patch("agent.verdict_completeness._get_judge_model")
def test_deterministic_checks_passing_does_not_skip_the_judge(
    mock_get_judge_model: MagicMock,
) -> None:
    """A long, sourced verdict still needs the judge — length/sources alone
    can't confirm completeness, only rule out the obvious failures.
    """
    mock_get_judge_model.return_value.invoke.return_value = _judgment(True)

    result = check_verdict_completeness("a claim", _LONG_VERDICT, _SOURCES)

    assert result is True
    assert mock_get_judge_model.return_value.invoke.called


@patch("agent.verdict_completeness._get_judge_model")
def test_majority_vote_true_wins(mock_get_judge_model: MagicMock) -> None:
    mock_get_judge_model.return_value.invoke.side_effect = [
        _judgment(True),
        _judgment(True),
        _judgment(False),
    ]

    assert check_verdict_completeness("a claim", _LONG_VERDICT, _SOURCES) is True


@patch("agent.verdict_completeness._get_judge_model")
def test_majority_vote_false_wins(mock_get_judge_model: MagicMock) -> None:
    mock_get_judge_model.return_value.invoke.side_effect = [
        _judgment(False),
        _judgment(False),
        _judgment(True),
    ]

    assert check_verdict_completeness("a claim", _LONG_VERDICT, _SOURCES) is False


@patch("agent.verdict_completeness._get_judge_model")
def test_tie_fails_closed(mock_get_judge_model: MagicMock) -> None:
    """A 1-1 split among successful votes (one call failed) should not be
    treated as complete — ties fail closed, same as any other ambiguous case.
    """
    mock_get_judge_model.return_value.invoke.side_effect = [
        _judgment(True),
        _judgment(False),
        RuntimeError("API error"),
    ]

    assert check_verdict_completeness("a claim", _LONG_VERDICT, _SOURCES) is False


@patch("agent.verdict_completeness._get_judge_model")
def test_fails_closed_when_too_many_judge_calls_error(mock_get_judge_model: MagicMock) -> None:
    """2 of 3 calls erroring means only 1 real vote exists — not enough to
    trust a majority either way, even though that one vote said "complete."
    """
    mock_get_judge_model.return_value.invoke.side_effect = [
        RuntimeError("API error"),
        _judgment(True),
        RuntimeError("API error"),
    ]

    assert check_verdict_completeness("a claim", _LONG_VERDICT, _SOURCES) is False


@patch("agent.verdict_completeness._get_judge_model")
def test_tolerates_a_single_judge_call_failure_if_majority_still_stands(
    mock_get_judge_model: MagicMock,
) -> None:
    """1 of 3 calls erroring still leaves 2 successful votes — a real
    majority, so the verdict shouldn't be penalized for one infra hiccup.
    """
    mock_get_judge_model.return_value.invoke.side_effect = [
        RuntimeError("API error"),
        _judgment(True),
        _judgment(True),
    ]

    assert check_verdict_completeness("a claim", _LONG_VERDICT, _SOURCES) is True
