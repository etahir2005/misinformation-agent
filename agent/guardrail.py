"""Classifies an incoming message as a greeting, an out-of-scope request, or
an actual claim to fact-check — runs once, ahead of the main orchestrator
loop.

Kept as its own module, same reasoning as
agent/tools/credibility_scoring_tool.py: this is a distinct structured-output
judgment, not a tool the main model calls, so it gets its own lazily
initialized model instance rather than being mixed into the tool-calling
model.

This has to be a separate step ahead of the orchestrator, not text folded
into its prompt — the orchestrator's first call is hard-forced to
vector_lookup_tool (tool_choice="vector_lookup_tool" in
agent/orchestrator.py), which cannot be overridden by prompt instructions.
A prompt-only guardrail would never actually stop that forced call from
firing on a genuine greeting.
"""

import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agent.config import GUARDRAIL_PROMPT, MODEL_NAME

logger = logging.getLogger(__name__)

# Deliberately narrow and lowercase-normalized — this is a free fast-path for
# unambiguous greetings only. Anything not caught here (including the entire
# open-ended off-topic space, which can't be reliably matched by keywords)
# falls through to the classification call below.
_GREETING_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "yo",
    "howdy",
    "good morning",
    "good afternoon",
    "good evening",
}

_intent_model = None


class MessageIntent(BaseModel):
    """Structured classification result for an incoming message."""

    category: Literal["greeting", "claim", "out_of_scope"]


def _get_intent_model():
    """Lazily build the structured-output-bound classification model.

    Lazy singleton, same pattern as vector_lookup_tool.py's
    _get_embedding_model()/_get_index() — avoids doing model setup at import
    time, which would otherwise break test collection whenever real
    credentials aren't configured.
    """
    global _intent_model
    if _intent_model is None:
        _intent_model = init_chat_model(
            f"google_genai:{MODEL_NAME}", temperature=0
        ).with_structured_output(MessageIntent)
    return _intent_model


def classify_message_intent(text: str) -> str:
    """Classify a message as "greeting", "claim", or "out_of_scope".

    A cheap keyword pre-check catches obvious greetings for free, without
    spending a model call. Everything else goes through one dedicated
    structured-output call — necessary because unlike greetings,
    off-topic messages have no fixed pattern to match against (see
    GUARDRAIL_PROMPT for the full reasoning on why "claim" is preferred
    when the classification is ambiguous).

    Args:
        text: The user's raw message text for this turn.

    Returns:
        One of "greeting", "claim", "out_of_scope".
    """
    normalized = text.strip().lower().rstrip("!.,")
    if normalized in _GREETING_KEYWORDS:
        return "greeting"

    result = _get_intent_model().invoke(
        [SystemMessage(content=GUARDRAIL_PROMPT), HumanMessage(content=text)]
    )
    logger.info("Classified message intent as: %s", result.category)
    return result.category
