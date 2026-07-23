"""Web search tool for gathering evidence on a claim, backed by Tavily."""

import logging
from typing import Any
from urllib.parse import urlparse

from langchain.tools import tool
from tavily import UsageLimitExceededError

from agent.tools._clients import tavily_client

logger = logging.getLogger(__name__)


@tool
def web_search_tool(query: str) -> dict[str, Any]:
    """Search the web for general evidence relevant to a claim.

    Use this if fact_check_lookup_tool found no existing ruling on the claim.
    Gathers sources that could support or contradict a claim that needs to be
    fact-checked.

    Args:
        query: A focused search query describing the claim to investigate.

    Returns:
        A dict with a "sources" key containing matching sources, each with
        url, title, snippet, and domain. If the search quota is exhausted,
        "sources" is empty and an "error" key explains why.
    """
    logger.info("Searching for: %s", query)

    try:
        response = tavily_client.search(query=query, max_results=5, include_raw_content=False)
    except UsageLimitExceededError:
        logger.error("Tavily usage limit exceeded — search skipped for query: %s", query)
        return {
            "sources": [],
            "query_used": query,
            "error": "search_quota_exhausted",
        }

    sources = [
        {
            "url": result.get("url", ""),
            "title": result.get("title", ""),
            "snippet": result.get("content", ""),
            "domain": urlparse(result.get("url", "")).netloc,
        }
        for result in response.get("results", [])
    ]

    logger.info("Found %d sources for query: %s", len(sources), query)
    return {"sources": sources, "query_used": query}
