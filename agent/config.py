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
