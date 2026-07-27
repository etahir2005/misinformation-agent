"""Tests for the web search tool."""

from unittest.mock import MagicMock, patch

from agent.tools.web_search_tool import web_search_tool


@patch("agent.tools.web_search_tool.tavily_client")
def test_web_search_tool_returns_structured_sources(mock_client: MagicMock) -> None:
    """web_search_tool should return a structured sources list, not raw text."""
    mock_client.search.return_value = {
        "results": [
            {
                "url": "https://example.com/article",
                "title": "Example Article",
                "content": "Some snippet text.",
            }
        ]
    }

    result = web_search_tool.invoke({"query": "test claim"})

    assert "sources" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["url"] == "https://example.com/article"
    assert result["sources"][0]["domain"] == "example.com"


@patch("agent.tools.web_search_tool.tavily_client")
def test_web_search_tool_handles_non_quota_tavily_failure(mock_client: MagicMock) -> None:
    """A Tavily failure other than quota exhaustion should return an error, not raise.

    Regression test (caught in PR review): only UsageLimitExceededError was
    being caught, so a bad API key, invalid query, timeout, or generic
    TavilyError would crash the tool instead of degrading gracefully.
    """
    mock_client.search.side_effect = RuntimeError("simulated Tavily failure")

    result = web_search_tool.invoke({"query": "test claim"})

    assert result["sources"] == []
    assert result["error"] == "search_failed"
