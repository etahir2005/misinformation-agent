"""Source-credibility judgment tool.

Along with the orchestrator itself, this is the only other piece of the
system that calls Gemini directly — everything else in agent/tools/ is a
plain API call to a third-party service.
"""

import logging
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from pydantic import BaseModel, Field

from agent.config import CREDIBILITY_SCORING_PROMPT, MODEL_NAME

logger = logging.getLogger(__name__)


class SourceScore(BaseModel):
    """Credibility judgment for a single source."""

    url: str = Field(description="The source's URL.")
    reliability: Literal["high", "medium", "low"] = Field(
        description="How reliable this source is."
    )
    reasoning: str = Field(description="One-sentence justification for the rating.")


class CredibilityAssessment(BaseModel):
    """Structured output for the credibility-scoring judgment call."""

    source_scores: list[SourceScore] = Field(
        description="A reliability judgment for each source provided."
    )
    overall_confidence: float = Field(
        description="Confidence (0.0-1.0) in a verdict based on this evidence."
    )
    sources_conflict: bool = Field(
        description="True if the sources meaningfully disagree with each other."
    )
    verdict_summary: str = Field(
        description="A concise, plain-language summary of what the evidence shows."
    )


_credibility_model = init_chat_model(
    f"google_genai:{MODEL_NAME}", temperature=0
).with_structured_output(CredibilityAssessment)


@tool
def credibility_scoring_tool(claim: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge the reliability and sufficiency of gathered evidence for a claim.

    Use this when there is no clean existing True/False ruling to rely on
    and you need to form your own judgment about whether the evidence
    supports or contradicts the claim.

    Args:
        claim: The claim being evaluated.
        sources: Evidence gathered so far (fact-check claims and/or web
            search results), each with at least a url and some text.

    Returns:
        A dict with source_scores, overall_confidence (0.0-1.0),
        sources_conflict (bool), and verdict_summary. If the judgment call
        itself fails, all fields are empty/zeroed and an "error" key
        explains why.
    """
    logger.info("Scoring credibility of %d source(s) for claim: %s", len(sources), claim)

    try:
        assessment = _credibility_model.invoke(
            [
                {"role": "system", "content": CREDIBILITY_SCORING_PROMPT},
                {"role": "user", "content": f"Claim: {claim}\n\nSources:\n{sources}"},
            ]
        )
    except Exception:
        # Broad on purpose: unlike Tavily's UsageLimitExceededError or
        # requests' RequestException, there's no single well-known
        # exception type for a structured-output LLM call failing (API
        # error, rate limit, or the model returning something that fails
        # schema validation all land here).
        logger.exception("Credibility scoring failed for claim: %s", claim)
        return {
            "source_scores": [],
            "overall_confidence": 0.0,
            "sources_conflict": False,
            "verdict_summary": "",
            "error": "credibility_scoring_failed",
        }

    logger.info(
        "Credibility assessment: confidence=%.2f conflict=%s",
        assessment.overall_confidence,
        assessment.sources_conflict,
    )
    return assessment.model_dump()
