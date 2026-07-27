"""Fact-check database lookup tool, backed by Google's Fact Check Tools API."""

import logging
from typing import Any

import requests
from langchain.tools import tool

from agent.config import GOOGLE_FACT_CHECK_API_KEY

logger = logging.getLogger(__name__)

_FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


@tool
def fact_check_lookup_tool(query: str) -> dict[str, Any]:
    """Check whether a professional fact-checking organization has already ruled on this claim.

    Use this tool first, before web_search_tool, to see if outlets like
    PolitiFact, Snopes, or Reuters Fact Check have already investigated and
    rated this exact or a very similar claim. If this returns no claims,
    fall back to web_search_tool to gather general evidence instead.

    Args:
        query: A focused search phrase describing the claim to look up.

    Returns:
        A dict with a "claims" key containing matching fact-checks, each with
        claim_text, claimant, rating, publisher, url, and review_date. If the
        lookup fails, "claims" is empty and an "error" key explains why.
    """
    logger.info("Checking fact-check database for: %s", query)

    try:
        response = requests.get(
            _FACT_CHECK_API_URL,
            params={"query": query, "key": GOOGLE_FACT_CHECK_API_KEY, "languageCode": "en"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        # ValueError catches response.json() failing on a malformed body —
        # a 200 response with a non-JSON payload shouldn't crash the tool
        # any more than a network error should.
        logger.error("Fact Check Tools API request failed for query %r: %s", query, exc)
        return {
            "claims": [],
            "query_used": query,
            "error": "fact_check_lookup_failed",
        }

    claims = [
        {
            "claim_text": claim.get("text", ""),
            "claimant": claim.get("claimant", ""),
            "rating": review.get("textualRating", ""),
            "publisher": review.get("publisher", {}).get("name", ""),
            "url": review.get("url", ""),
            "review_date": review.get("reviewDate", ""),
        }
        for claim in data.get("claims", [])
        for review in claim.get("claimReview", [])
    ]

    logger.info("Found %d fact-check(s) for query: %s", len(claims), query)
    return {"claims": claims, "query_used": query}
