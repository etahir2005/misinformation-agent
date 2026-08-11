"""Semantic claim-cache tool, backed by Pinecone.

Checks whether a semantically similar claim has already been resolved, so
the pipeline can reuse that verdict instead of re-running fact-checking
from scratch. Also exposes store_verdict(), a plain function (not a
model-callable tool) that the orchestrator calls directly once a claim is
actually resolved — storing a verdict is deterministic bookkeeping, not a
judgment call that should be left to the model.

Two-stage retrieval: Pinecone's cosine similarity does a cheap first pass
over VECTOR_SEARCH_TOP_K candidates, then a CrossEncoder reranker re-scores
that smaller set more precisely before a match is accepted or rejected.
Raw cosine similarity on a single nearest neighbor (the old top_k=1
approach) is vulnerable to the "curse of dimensionality" — in a
high-dimensional embedding space, the single closest vector isn't always
the true best semantic match. Reranking is only feasible against a small
candidate set, not the whole index, which is why the wider top_k retrieval
and the reranking step go together rather than being independent fixes.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain.tools import tool
from pinecone import Pinecone
from sentence_transformers import CrossEncoder, SentenceTransformer

from agent.config import (
    CLAIM_CACHE_MAX_AGE_DAYS,
    LOW_CONFIDENCE_THRESHOLD,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    RERANK_MIN_MARGIN,
    RERANK_MIN_SCORE,
    RERANKER_MODEL_NAME,
    VECTOR_SEARCH_TOP_K,
)

logger = logging.getLogger(__name__)

# Constructed lazily (see _get_embedding_model / _get_reranker_model /
# _get_index below), not at import time. Building these eagerly at module
# import made just importing this module — which orchestrator.py, main.py,
# and test_orchestrator_routing.py all do transitively — require live
# Pinecone credentials, a pre-provisioned index, and network access, even
# for tests that only exercise pure routing logic and never touch Pinecone
# at all.
_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None
_pinecone_client: Pinecone | None = None
_index = None


def _get_embedding_model() -> SentenceTransformer:
    """Construct the embedding model on first use, then reuse it."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _embedding_model


def _get_reranker_model() -> CrossEncoder:
    """Construct the CrossEncoder reranker on first use, then reuse it.

    Same lazy pattern as the embedding model, and for the same reason —
    importing this module shouldn't require downloading a second model.
    """
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model


def _get_index():
    """Construct the Pinecone client/index on first use, then reuse it."""
    global _pinecone_client, _index
    if _index is None:
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pinecone_client.Index(PINECONE_INDEX_NAME)
    return _index


def _embed(text: str) -> list[float]:
    """Embed a single piece of text into a vector for Pinecone."""
    return _get_embedding_model().encode(text, normalize_embeddings=True).tolist()


def _rerank(claim: str, matches: list) -> list[tuple[float, Any]]:
    """Re-score Pinecone's candidate matches with the more accurate CrossEncoder.

    Pinecone's cosine similarity is a cheap first pass across the whole
    index; the CrossEncoder is far more accurate at judging whether two
    claims genuinely mean the same thing, but too slow to run against
    every stored claim — only feasible against this small candidate set.

    Returns (score, match) pairs sorted best-first. The winner here is not
    guaranteed to be Pinecone's own top-ranked candidate (matches[0]) —
    that's the entire point of this second stage.
    """
    pairs = [(claim, (m.metadata or {}).get("claim_text", "")) for m in matches]
    scores = _get_reranker_model().predict(pairs)
    return sorted(zip(scores, matches), key=lambda pair: pair[0], reverse=True)


@tool
def vector_lookup_tool(claim: str) -> dict[str, Any]:
    """Check whether a semantically similar claim has already been resolved.

    Use this first, before any other tool — if a close match is found, its
    verdict is reused instead of re-running the full fact-checking pipeline.

    Args:
        claim: The claim to check against previously resolved claims.

    Returns:
        A dict with "hit" (bool). If True, also includes "confidence",
        "sources", and "verdict_summary" from the cached entry. If the
        lookup itself fails, "hit" is False and an "error" key explains why.
    """
    logger.info("Checking claim cache for: %s", claim)

    cutoff_timestamp = time.time() - (CLAIM_CACHE_MAX_AGE_DAYS * 86400)

    try:
        query_vector = _embed(claim)
        result = _get_index().query(
            vector=query_vector,
            top_k=VECTOR_SEARCH_TOP_K,
            include_metadata=True,
            filter={"checked_at_ts": {"$gte": cutoff_timestamp}},
        )
        matches = result.matches
        if not matches:
            logger.info("No cached claims found.")
            return {"hit": False}
        ranked = _rerank(claim, matches)
    except Exception:
        logger.exception("Claim cache lookup failed for: %s", claim)
        return {"hit": False, "error": "vector_lookup_failed"}

    best_score, best_match = ranked[0]
    metadata = best_match.metadata or {}

    if best_score < RERANK_MIN_SCORE:
        logger.info("Best reranked match scored %.3f, below floor — no reuse.", best_score)
        return {"hit": False}

    # Only meaningful with more than one candidate — a single match has no
    # runner-up to be ambiguous against, so the floor check above is the
    # only bar it needs to clear.
    if len(ranked) > 1:
        runner_up_score = ranked[1][0]
        margin = best_score - runner_up_score
        if margin < RERANK_MIN_MARGIN:
            logger.info(
                "Best match's margin over runner-up (%.3f) too thin — ambiguous, no reuse.",
                margin,
            )
            return {"hit": False}

    if metadata.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD:
        logger.info("Best match's original confidence was too low to reuse.")
        return {"hit": False}

    logger.info(
        "Cache hit (rerank_score=%.3f) for claim: %s", best_score, metadata.get("claim_text", "")
    )
    return {
        "hit": True,
        "confidence": metadata.get("confidence", 0.0),
        "sources": metadata.get("sources", []),
        "verdict_summary": metadata.get("verdict_summary", ""),
    }


def store_verdict(
    claim: str,
    confidence: float,
    sources: list[str],
    verdict_summary: str,
    resolved_by: str,
) -> None:
    """Embed and store a resolved claim's verdict for future cache lookups.

    Not a model-callable tool on purpose — the orchestrator calls this
    directly once a claim is actually resolved.
    """
    try:
        vector = _embed(claim)
        now = datetime.now(timezone.utc)
        _get_index().upsert(
            vectors=[
                {
                    "id": str(uuid.uuid4()),
                    "values": vector,
                    "metadata": {
                        "claim_text": claim,
                        "confidence": confidence,
                        "sources": sources,
                        "verdict_summary": verdict_summary,
                        "resolved_by": resolved_by,
                        "checked_at": now.isoformat(),
                        # Numeric mirror of checked_at, used for the
                        # recency metadata filter in vector_lookup_tool —
                        # Pinecone's filter comparisons ($gte etc.) work
                        # against numbers, not ISO date strings.
                        "checked_at_ts": now.timestamp(),
                    },
                }
            ]
        )
        logger.info("Stored verdict for future cache lookups: %s", claim)
    except Exception:
        logger.exception("Failed to store verdict for claim: %s", claim)
