"""Verdict-completeness check — the last line of defense before a claim's
final answer is cached for future reuse.

Two layers, cheapest first: deterministic hard-fails catch the easy,
unambiguous cases (no sources at all, or a suspiciously short answer) with
zero LLM cost and zero hallucination risk. Anything that clears those goes
to an LLM judge instead — a word count or an empty sources list can prove a
verdict is *definitely* incomplete, but it can't prove one is complete: a
long, sourced answer can still be vague or evasive, and only a judge that
actually reads it can catch that half of the failure mode.

The judge itself is called multiple times (majority-vote consensus) rather
than trusted on a single pass, since an LLM judge is still an LLM reasoning
probabilistically about another LLM's output — it inherits the same
failure modes it's meant to catch, just at one remove. Judge-call errors
fail closed: if the judge can't be trusted to answer, the safer default is
"incomplete," never "complete."
"""

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from agent.config import (
    JUDGE_CONSENSUS_CALLS,
    JUDGE_MODEL_NAME,
    VERDICT_COMPLETENESS_JUDGE_PROMPT,
    VERDICT_MIN_WORD_COUNT,
)

logger = logging.getLogger(__name__)


class _CompletenessJudgment(BaseModel):
    """Structured output for a single verdict-completeness judge call."""

    is_complete: bool = Field(
        description="True if the verdict gives a real, reasoned answer grounded in the sources."
    )
    reasoning: str = Field(description="One-sentence justification for the judgment.")


_judge_model = None


def _get_judge_model():
    """Lazily build the structured-output-bound judge model.

    Lazy singleton, same pattern as agent/guardrail.py's _get_intent_model()
    and agent/tools/vector_lookup_tool.py's _get_embedding_model() — avoids
    doing model setup at import time. A separate model from MODEL_NAME (see
    agent/config.py's JUDGE_MODEL_NAME) — a second, independent pass is more
    useful for catching what the main answer-generating pass missed than
    reusing the exact same model/config that produced the verdict being
    checked.
    """
    global _judge_model
    if _judge_model is None:
        _judge_model = init_chat_model(
            f"google_genai:{JUDGE_MODEL_NAME}", temperature=0
        ).with_structured_output(_CompletenessJudgment)
    return _judge_model


def _judge_once(claim: str, verdict: str, sources: list[dict[str, Any]]) -> bool | None:
    """A single judge call. Returns None (not False) on failure so the
    caller can tell "the judge said incomplete" apart from "the judge call
    itself broke" — conflating the two would make a run of infrastructure
    errors look identical to a real majority vote against completeness.
    """
    try:
        judgment = _get_judge_model().invoke(
            [
                {"role": "system", "content": VERDICT_COMPLETENESS_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Claim: {claim}\n\nVerdict given: {verdict}\n\nSources used: {sources}"
                    ),
                },
            ]
        )
    except Exception:
        # Broad on purpose, same reasoning as credibility_scoring_tool: no
        # single well-known exception type covers every way a
        # structured-output LLM call can fail (API error, rate limit, or a
        # response that fails schema validation all land here).
        logger.exception("Verdict-completeness judge call failed for claim: %s", claim)
        return None
    return judgment.is_complete


def check_verdict_completeness(claim: str, verdict: str, sources: list[dict[str, Any]]) -> bool:
    """Decide whether a claim's final verdict is complete enough to cache and reuse.

    Args:
        claim: The claim being verified (same text used for the vector cache).
        verdict: The final answer's text, as shown to the user.
        sources: Evidence gathered for this claim (same shape credibility_scoring_tool uses).

    Returns:
        True only if the verdict clears both deterministic checks and wins
        a strict majority of successful judge votes. False for anything
        incomplete, vague, or where too many judge calls failed to trust a
        majority either way (fail closed).
    """
    if not sources:
        logger.info("Verdict completeness: no sources — incomplete, skipping the judge.")
        return False
    if len(verdict.split()) < VERDICT_MIN_WORD_COUNT:
        logger.info(
            "Verdict completeness: below %d-word floor — incomplete, skipping the judge.",
            VERDICT_MIN_WORD_COUNT,
        )
        return False

    votes = [_judge_once(claim, verdict, sources) for _ in range(JUDGE_CONSENSUS_CALLS)]
    successful_votes = [vote for vote in votes if vote is not None]

    # Require a strict majority of *intended* calls to have actually
    # succeeded before trusting any vote among them — two calls succeeding
    # out of three still means a real majority voted; one out of three
    # succeeding does not, even if that one vote said "complete."
    if len(successful_votes) <= JUDGE_CONSENSUS_CALLS // 2:
        logger.warning(
            "Verdict completeness: only %d/%d judge calls succeeded — failing closed as "
            "incomplete.",
            len(successful_votes),
            JUDGE_CONSENSUS_CALLS,
        )
        return False

    complete_votes = sum(1 for vote in successful_votes if vote)
    is_complete = complete_votes > len(successful_votes) / 2
    logger.info(
        "Verdict completeness: judge consensus %d/%d complete — verdict is %s.",
        complete_votes,
        len(successful_votes),
        "complete" if is_complete else "incomplete",
    )
    return is_complete
