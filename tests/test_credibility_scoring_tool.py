"""Tests for the credibility scoring tool."""

from unittest.mock import MagicMock, patch

from agent.tools.credibility_scoring_tool import CredibilityAssessment, credibility_scoring_tool


@patch("agent.tools.credibility_scoring_tool._credibility_model")
def test_credibility_scoring_tool_returns_structured_assessment(mock_model: MagicMock) -> None:
    """credibility_scoring_tool should return a structured assessment dict."""
    mock_model.invoke.return_value = CredibilityAssessment(
        source_scores=[
            {"url": "https://example.com", "reliability": "high", "reasoning": "Reputable outlet."}
        ],
        overall_confidence=0.85,
        sources_conflict=False,
        verdict_summary="Evidence supports the claim.",
    )

    result = credibility_scoring_tool.invoke(
        {"claim": "test claim", "sources": [{"url": "https://example.com", "snippet": "..."}]}
    )

    assert result["overall_confidence"] == 0.85
    assert result["sources_conflict"] is False
    assert len(result["source_scores"]) == 1
    assert result["verdict_summary"] == "Evidence supports the claim."


@patch("agent.tools.credibility_scoring_tool._credibility_model")
def test_credibility_scoring_tool_handles_failure(mock_model: MagicMock) -> None:
    """credibility_scoring_tool should return an error, not raise, if the model call fails."""
    mock_model.invoke.side_effect = RuntimeError("API error")

    result = credibility_scoring_tool.invoke({"claim": "test claim", "sources": []})

    assert result["error"] == "credibility_scoring_failed"
    assert result["overall_confidence"] == 0.0
