"""Full-article-text retrieval tool, backed by Tavily's Extract endpoint."""

import logging
from typing import Any

from langchain.tools import tool
from tavily import UsageLimitExceededError

from agent.tools._clients import tavily_client

logger = logging.getLogger(__name__)


@tool
def source_retrieval_tool(url: str) -> dict[str, Any]:
    """Fetch and extract the full text content of a specific article URL.

    Use this tool when the user has given a specific link and wants its
    content read and checked, rather than a general web search. Do not use
    this for a plain-text claim with no URL — use fact_check_lookup_tool or
    web_search_tool instead.

    Args:
        url: The article URL to fetch and read.

    Returns:
        A dict with "content" (the extracted article text) and "url". If
        extraction fails, "content" is empty and an "error" key explains why.
    """
    logger.info("Retrieving source content for: %s", url)

    try:
        response = tavily_client.extract(urls=[url], extract_depth="basic")
    except UsageLimitExceededError:
        logger.error("Tavily usage limit exceeded — extraction skipped for url: %s", url)
        return {
            "content": "",
            "url": url,
            "error": "search_quota_exhausted",
        }

    results = response.get("results", [])
    if not results:
        logger.warning("No content extracted for url: %s", url)
        return {
            "content": "",
            "url": url,
            "error": "extraction_failed",
        }

    result = results[0]
    content = result.get("raw_content", "")
    logger.info("Retrieved %d characters from: %s", len(content), url)
    return {"content": content, "url": result.get("url", url)}
