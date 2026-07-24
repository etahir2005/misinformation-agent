"""Configuration and environment variable loading for the fact-checking agent."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _require_env(var_name: str) -> str:
    """Fetch a required environment variable or raise a clear error.

    Args:
        var_name: Name of the environment variable to fetch.

    Returns:
        The environment variable's value.

    Raises:
        RuntimeError: If the environment variable is not set.
    """
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. "
            "Copy .env.example to .env and fill in your keys."
        )
    return value


GOOGLE_API_KEY: str = _require_env("GOOGLE_API_KEY")
TAVILY_API_KEY: str = _require_env("TAVILY_API_KEY")
GOOGLE_FACT_CHECK_API_KEY: str = _require_env("GOOGLE_FACT_CHECK_API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
SYSTEM_PROMPT: str = (
    "You are a fact-checking assistant. Given a claim, first use "
    "fact_check_lookup_tool to check whether a professional fact-checking "
    "organization has already ruled on it. If that returns no useful claims, "
    "use web_search_tool to gather general evidence instead. If the user "
    "provides a specific article URL, use source_retrieval_tool to read its "
    "full content before evaluating the claim. Cite the sources you used. "
    "If the evidence is thin or conflicting, say so explicitly rather than "
    "guessing. If a tool result contains an \"error\" field, do not retry "
    "that tool — tell the user plainly that you're temporarily unable to "
    "check the claim right now."
)
