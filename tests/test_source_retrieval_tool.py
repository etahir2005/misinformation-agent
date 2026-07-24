"""Tests for the source retrieval tool."""

from unittest.mock import MagicMock, patch

from agent.tools.source_retrieval_tool import source_retrieval_tool


@patch("agent.tools.source_retrieval_tool.tavily_client")
def test_source_retrieval_tool_returns_extracted_content(mock_client: MagicMock) -> None:
    """source_retrieval_tool should return the extracted article text."""
    mock_client.extract.return_value = {
        "results": [
            {"url": "https://example.com/article", "raw_content": "Full article text here."}
        ],
        "failed_results": [],
    }

    result = source_retrieval_tool.invoke({"url": "https://example.com/article"})

    assert result["content"] == "Full article text here."
    assert result["url"] == "https://example.com/article"


@patch("agent.tools.source_retrieval_tool.tavily_client")
def test_source_retrieval_tool_handles_extraction_failure(mock_client: MagicMock) -> None:
    """source_retrieval_tool should return an error, not raise, if extraction fails."""
    mock_client.extract.return_value = {"results": [], "failed_results": [{"url": "bad"}]}

    result = source_retrieval_tool.invoke({"url": "https://example.com/bad-article"})

    assert result["content"] == ""
    assert result["error"] == "extraction_failed"
