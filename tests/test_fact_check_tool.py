"""Tests for the fact-check lookup tool."""

from unittest.mock import MagicMock, patch

from agent.tools.fact_check_tool import fact_check_lookup_tool


@patch("agent.tools.fact_check_tool.requests.get")
def test_fact_check_lookup_tool_returns_structured_claims(mock_get: MagicMock) -> None:
    """fact_check_lookup_tool should return a structured claims list."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "claims": [
            {
                "text": "The moon is made of cheese",
                "claimant": "Someone online",
                "claimReview": [
                    {
                        "publisher": {"name": "PolitiFact", "site": "politifact.com"},
                        "url": "https://politifact.com/example",
                        "textualRating": "False",
                        "reviewDate": "2024-01-01T00:00:00Z",
                    }
                ],
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fact_check_lookup_tool.invoke({"query": "moon made of cheese"})

    assert "claims" in result
    assert len(result["claims"]) == 1
    assert result["claims"][0]["rating"] == "False"
    assert result["claims"][0]["publisher"] == "PolitiFact"


@patch("agent.tools.fact_check_tool.requests.get")
def test_fact_check_lookup_tool_handles_malformed_json_response(mock_get: MagicMock) -> None:
    """A 200 response with a non-JSON body should return an error, not raise.

    Regression test (caught in PR review): response.json() used to be
    called outside the try/except, so a malformed body would raise
    uncaught instead of returning the documented error dict.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = ValueError("not valid json")
    mock_get.return_value = mock_response

    result = fact_check_lookup_tool.invoke({"query": "moon made of cheese"})

    assert result["claims"] == []
    assert result["error"] == "fact_check_lookup_failed"
