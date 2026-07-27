"""Semantic claim-cache tool, backed by Pinecone.

Checks whether a semantically similar claim has already been resolved, so
the pipeline can reuse that verdict instead of re-running fact-checking
from scratch. Also exposes store_verdict(), a plain function (not a
model-callable tool) that the orchestrator calls directly once a claim is
actually resolved — storing a verdict is deterministic bookkeeping, not a
judgment call that should be left to the model.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain.tools import tool
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from agent.config import (
    CLAIM_SIMILARITY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

logger = logging.getLogger(__name__)

_embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
_pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
_index = _pinecone_client.Index(PINECONE_INDEX_NAME)


def _embed(text: str) -> list[float]:
    """Embed a single piece of text into a vector for Pinecone."""
    return _embedding_model.encode(text, normalize_embeddings=True).tolist()


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

    try:
        query_vector = _embed(claim)
        result = _index.query(vector=query_vector, top_k=1, include_metadata=True)
    except Exception:
        logger.exception("Claim cache lookup failed for: %s", claim)
        return {"hit": False, "error": "vector_lookup_failed"}

    matches = result.matches
    if not matches:
        logger.info("No cached claims found.")
        return {"hit": False}

    best_match = matches[0]
    score = best_match.score
    metadata = best_match.metadata or {}

    if score < CLAIM_SIMILARITY_THRESHOLD:
        logger.info("Best match scored %.3f, below threshold — no reuse.", score)
        return {"hit": False}

    if metadata.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD:
        logger.info("Best match's original confidence was too low to reuse.")
        return {"hit": False}

    logger.info(
        "Cache hit (similarity=%.3f) for claim: %s", score, metadata.get("claim_text", "")
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
        _index.upsert(
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
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            ]
        )
        logger.info("Stored verdict for future cache lookups: %s", claim)
    except Exception:
        logger.exception("Failed to store verdict for claim: %s", claim)
