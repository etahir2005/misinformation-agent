"""Tests for the web search tool."""

from unittest.mock import MagicMock, patch

from agent.tools.search_tool import web_search_tool


@patch("agent.tools.search_tool._client")
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
