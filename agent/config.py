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

LOW_CONFIDENCE_THRESHOLD = 0.5

PINECONE_API_KEY: str = _require_env("PINECONE_API_KEY")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "misinformation-agent-claims")
CLAIM_SIMILARITY_THRESHOLD = 0.82

# Once the messages preceding the current turn reach this count, they get
# compressed into a running summary (see agent/summarizer.py) instead of
# being resent to Gemini in full on every turn forever. Deliberately a
# message count, not a token count — simple, deterministic, and easy to
# test, matching the other thresholds in this file. A typical resolved claim
# is roughly 4-8 messages, so this triggers after a handful of claims in the
# same thread, not on every turn.
MAX_MESSAGES_BEFORE_SUMMARY = 20

# Neon Postgres connection string, e.g.
# postgresql://user:password@ep-xxxx.region.aws.neon.tech/dbname?sslmode=require
# Use Neon's *direct* connection string, not the pooled/PgBouncer one (the
# one with "-pooler" in the hostname) — the checkpointer needs
# session-level Postgres features a transaction pooler can break.
POSTGRES_CONNECTION_STRING: str = _require_env("POSTGRES_CONNECTION_STRING")

SYSTEM_PROMPT: str = (
    "You are a fact-checking assistant. Given a claim, first use "
    "vector_lookup_tool to check whether a semantically similar claim has "
    "already been resolved — if there's a match, its verdict is reused "
    "automatically. If not, use fact_check_lookup_tool to check whether a "
    "professional fact-checking organization has already ruled on it. If "
    "that returns no useful claims, use web_search_tool to gather general "
    "evidence instead. If the user provides a specific article URL, use "
    "source_retrieval_tool to read its full content before evaluating the "
    "claim. "
    "If there is no single, clean True/False ruling from fact_check_lookup_tool, "
    "you will be required to call credibility_scoring_tool to form your own "
    "judgment from the evidence gathered so far — pass it the claim and the "
    "sources you have. "
    f"If overall_confidence comes back below {LOW_CONFIDENCE_THRESHOLD} or "
    "sources_conflict is true, you will be required to gather one more round "
    "of evidence with web_search_tool and then be scored again before "
    "finalizing your answer. "
    "Cite the sources you used. If the evidence is still thin or conflicting "
    "after that second pass, say so explicitly rather than guessing. If a "
    "tool result contains an \"error\" field, do not retry that tool — tell "
    "the user plainly that you're temporarily unable to check the claim "
    "right now."
)

CREDIBILITY_SCORING_PROMPT: str = (
    "You are judging the credibility of evidence gathered about a claim. "
    "Given the claim and a list of sources (each with a URL and some text "
    "content), assess how reliable each source is, whether they conflict "
    "with each other, and how confident a fact-checker should be in a "
    "verdict based on this evidence. Judge reliability using domain "
    "reputation, specificity, and consistency with other sources — not "
    "whether the source happens to agree with what you'd expect."
)

SUMMARIZATION_PROMPT: str = (
    "You maintain a running summary of an ongoing fact-checking conversation "
    "so it can continue across many claims without resending the entire "
    "history. Given the existing summary (if any) and a new chunk of "
    "conversation to fold in, produce one updated, concise summary. "
    "Preserve every distinct claim discussed and its verdict — do not "
    "generalize specific claims away or drop any of them. Do not include "
    "tool mechanics (which tool was called, raw source URLs, confidence "
    "scores) — only the substance of what was asked and what was concluded, "
    "in plain language.\n\n"
    "Existing summary:\n{existing_summary}\n\n"
    "New conversation to fold in:\n{conversation_text}"
)
