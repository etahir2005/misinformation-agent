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
