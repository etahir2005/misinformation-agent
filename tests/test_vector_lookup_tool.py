"""Tests for the vector claim-cache tool."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tools.vector_lookup_tool import store_verdict, vector_lookup_tool


@patch("agent.tools.vector_lookup_tool._reranker_model")
@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_returns_hit_for_confident_unambiguous_match(
    mock_index: MagicMock, mock_embed: MagicMock, mock_reranker: MagicMock
) -> None:
    """A single, confident, well-separated match should be a cache hit."""
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
    mock_reranker.predict.return_value = [0.9]

    result = vector_lookup_tool.invoke({"claim": "Do vaccines cause autism?"})

    assert result["hit"] is True
    assert result["confidence"] == 0.9
    assert result["verdict_summary"] == "No credible evidence supports this claim."


@patch("agent.tools.vector_lookup_tool._reranker_model")
@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_reranking_can_override_pinecones_top_cosine_match(
    mock_index: MagicMock, mock_embed: MagicMock, mock_reranker: MagicMock
) -> None:
    """The winning match should be whichever one the reranker scores highest,
    not necessarily whichever Pinecone ranked first by raw cosine similarity
    — that's the entire point of adding a reranking stage.
    """
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(
                score=0.90,  # Pinecone's top pick by cosine similarity...
                metadata={
                    "claim_text": "The earth is round.",
                    "confidence": 0.9,
                    "sources": [],
                    "verdict_summary": "Wrong topic entirely.",
                },
            ),
            SimpleNamespace(
                score=0.85,  # ...but this one is the true semantic match.
                metadata={
                    "claim_text": "Vaccines cause autism.",
                    "confidence": 0.9,
                    "sources": [],
                    "verdict_summary": "No credible evidence supports this claim.",
                },
            ),
        ]
    )
    # Reranker scores in the opposite order of Pinecone's cosine ranking.
    mock_reranker.predict.return_value = [0.1, 0.9]

    result = vector_lookup_tool.invoke({"claim": "Do vaccines cause autism?"})

    assert result["hit"] is True
    assert result["verdict_summary"] == "No credible evidence supports this claim."


@patch("agent.tools.vector_lookup_tool._reranker_model")
@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_rejects_match_below_rerank_score_floor(
    mock_index: MagicMock, mock_embed: MagicMock, mock_reranker: MagicMock
) -> None:
    """A match that doesn't clear RERANK_MIN_SCORE should be a miss."""
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[SimpleNamespace(score=0.5, metadata={"confidence": 0.9})]
    )
    mock_reranker.predict.return_value = [-1.0]

    result = vector_lookup_tool.invoke({"claim": "Some unrelated claim."})

    assert result["hit"] is False


@patch("agent.tools.vector_lookup_tool._reranker_model")
@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_rejects_ambiguous_match_with_thin_margin(
    mock_index: MagicMock, mock_embed: MagicMock, mock_reranker: MagicMock
) -> None:
    """Two close-scoring candidates should be treated as ambiguous and
    rejected, even though the top one alone clears the score floor —
    a close runner-up means genuine uncertainty about which is the real match.
    """
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(score=0.9, metadata={"claim_text": "Claim A", "confidence": 0.9}),
            SimpleNamespace(score=0.88, metadata={"claim_text": "Claim B", "confidence": 0.9}),
        ]
    )
    # Both clear RERANK_MIN_SCORE, but the gap between them (0.05) is
    # below the default RERANK_MIN_MARGIN (0.15).
    mock_reranker.predict.return_value = [0.9, 0.85]

    result = vector_lookup_tool.invoke({"claim": "Some claim."})

    assert result["hit"] is False


@patch("agent.tools.vector_lookup_tool._reranker_model")
@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_skips_margin_check_for_a_single_candidate(
    mock_index: MagicMock, mock_embed: MagicMock, mock_reranker: MagicMock
) -> None:
    """A single confident match has no runner-up to be ambiguous against —
    only the score floor should apply, not the margin check.
    """
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(
                score=0.9,
                metadata={
                    "claim_text": "Claim A",
                    "confidence": 0.9,
                    "sources": [],
                    "verdict_summary": "summary",
                },
            )
        ]
    )
    mock_reranker.predict.return_value = [0.9]

    result = vector_lookup_tool.invoke({"claim": "Some claim."})

    assert result["hit"] is True


@patch("agent.tools.vector_lookup_tool._reranker_model")
@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_returns_miss_for_low_original_confidence(
    mock_index: MagicMock, mock_embed: MagicMock, mock_reranker: MagicMock
) -> None:
    """A similar match shouldn't be reused if its original verdict was shaky,
    even after it clears the rerank score/margin checks.
    """
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[SimpleNamespace(score=0.95, metadata={"confidence": 0.3})]
    )
    mock_reranker.predict.return_value = [0.9]

    result = vector_lookup_tool.invoke({"claim": "Some claim."})

    assert result["hit"] is False


@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_returns_miss_when_no_matches(
    mock_index: MagicMock, mock_embed: MagicMock
) -> None:
    """An empty Pinecone index (or no matches) should be a clean miss, not an
    error — and shouldn't even attempt to call the reranker, since there's
    nothing to rerank.
    """
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


@patch("agent.tools.vector_lookup_tool._reranker_model")
@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_handles_reranker_failure(
    mock_index: MagicMock, mock_embed: MagicMock, mock_reranker: MagicMock
) -> None:
    """A reranker failure (model load error, bad prediction, etc.) should
    return a miss with an error, not raise — the same graceful-degradation
    guarantee the embedding and Pinecone-query steps already have.
    """
    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(score=0.9, metadata={"claim_text": "Claim A", "confidence": 0.9})
        ]
    )
    mock_reranker.predict.side_effect = RuntimeError("reranker failed")

    result = vector_lookup_tool.invoke({"claim": "Some claim."})

    assert result["hit"] is False
    assert result["error"] == "vector_lookup_failed"


@patch("agent.tools.vector_lookup_tool._embed")
@patch("agent.tools.vector_lookup_tool._index")
def test_vector_lookup_tool_queries_with_configured_top_k_and_recency_filter(
    mock_index: MagicMock, mock_embed: MagicMock
) -> None:
    """The Pinecone query should ask for VECTOR_SEARCH_TOP_K candidates and
    filter out anything older than CLAIM_CACHE_MAX_AGE_DAYS.
    """
    from agent.config import VECTOR_SEARCH_TOP_K

    mock_embed.return_value = [0.1] * 768
    mock_index.query.return_value = SimpleNamespace(matches=[])

    vector_lookup_tool.invoke({"claim": "Some claim."})

    call_kwargs = mock_index.query.call_args.kwargs
    assert call_kwargs["top_k"] == VECTOR_SEARCH_TOP_K
    assert "checked_at_ts" in call_kwargs["filter"]
    assert "$gte" in call_kwargs["filter"]["checked_at_ts"]


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
    assert isinstance(upserted_vector["metadata"]["checked_at_ts"], float)


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
