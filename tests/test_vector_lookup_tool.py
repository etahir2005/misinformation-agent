"""Tests for the vector claim-cache tool."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tools.vector_lookup_tool import store_verdict, vector_lookup_tool


@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_returns_hit_above_threshold(
    mock_index: MagicMock, mock_embed: MagicMock
) -> None:
    """A close, confident match should be returned as a cache hit."""
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(
                score=0.95,
                metadata={
                    "claim_text": "Vaccines cause autism.",
                    "confidence": 0.9,
                    "sources": ["https://example.com/a"],
                    "verdict_summary": "No credible evidence supports this claim.",
                },
            )
        ]
    )

    result = vector_lookup_tool.invoke({"claim": "Do vaccines cause autism?"})

    assert result["hit"] is True
    assert result["confidence"] == 0.9
    assert result["verdict_summary"] == "No credible evidence supports this claim."


@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_returns_miss_below_similarity_threshold(
    mock_index: MagicMock, mock_embed: MagicMock
) -> None:
    """A match that doesn't clear the similarity threshold should be a miss."""
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[SimpleNamespace(score=0.5, metadata={"confidence": 0.9})]
    )

    result = vector_lookup_tool.invoke({"claim": "Some unrelated claim."})

    assert result["hit"] is False


@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_returns_miss_for_low_original_confidence(
    mock_index: MagicMock, mock_embed: MagicMock
) -> None:
    """A similar match shouldn't be reused if its original verdict was shaky."""
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[SimpleNamespace(score=0.95, metadata={"confidence": 0.3})]
    )

    result = vector_lookup_tool.invoke({"claim": "Some claim."})

    assert result["hit"] is False


@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_returns_miss_when_no_matches(
    mock_index: MagicMock, mock_embed: MagicMock
) -> None:
    """An empty Pinecone index (or no matches) should be a clean miss, not an error."""
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(matches=[])

    result = vector_lookup_tool.invoke({"claim": "A brand new claim."})

    assert result["hit"] is False
    assert "error" not in result


@patch("agent.tools.vector_lookup_tool._embed")
def test_vector_lookup_tool_handles_query_failure(mock_embed: MagicMock) -> None:
    """A Pinecone failure should return a miss with an error, not raise."""
    mock_embed.side_effect = RuntimeError("embedding failed")

    result = vector_lookup_tool.invoke({"claim": "Some claim."})

    assert result["hit"] is False
    assert result["error"] == "vector_lookup_failed"


@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_store_verdict_upserts_expected_metadata(
    mock_index: MagicMock, mock_embed: MagicMock
) -> None:
    """store_verdict should upsert one vector with the expected metadata fields."""
    mock_embed.return_value = [0.2] * 768

    store_verdict(
        claim="The sky is green.",
        confidence=0.95,
        sources=["https://example.com/sky"],
        verdict_summary="The sky is blue, not green.",
        resolved_by="credibility_scoring_tool",
    )

    mock_index.upsert.assert_called_once()
    upserted_vector = mock_index.upsert.call_args.kwargs["vectors"][0]
    assert upserted_vector["metadata"]["claim_text"] == "The sky is green."
    assert upserted_vector["metadata"]["confidence"] == 0.95
    assert upserted_vector["metadata"]["resolved_by"] == "credibility_scoring_tool"


@patch("agent.tools.vector_lookup_tool._embed")
def test_store_verdict_handles_failure_gracefully(mock_embed: MagicMock) -> None:
    """store_verdict should log and swallow failures, not raise into the caller."""
    mock_embed.side_effect = RuntimeError("embedding failed")

    store_verdict(
        claim="Some claim.",
        confidence=0.9,
        sources=[],
        verdict_summary="summary",
        resolved_by="fact_check_lookup_tool",
    )  # should not raise
