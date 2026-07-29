"""Conversation summarization for long-running threads.

Deliberately knows nothing about LangGraph state or message objects — it
takes plain text in, returns plain text (or None on failure) out. All
message/state manipulation (which messages count as "older", building
RemoveMessage entries, storing the result back into state) stays in
orchestrator.py, which already owns every other piece of state-reading
logic in this project.

The model here is lazy-initialized (built on first actual call, not at
import time), mirroring vector_lookup_tool.py's _get_index() /
_get_embedding_model() pattern — importing this module for a routing test
should never require live Gemini credentials.
"""

import logging

from langchain.chat_models import init_chat_model

from agent.config import MODEL_NAME, SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)

_summarization_model = None


def _get_summarization_model():
    """Construct the summarization model on first use, then reuse it."""
    global _summarization_model
    if _summarization_model is None:
        _summarization_model = init_chat_model(f"google_genai:{MODEL_NAME}", temperature=0)
    return _summarization_model


def _content_to_text(content: str | list) -> str:
    """Normalize a chat model response's content into a plain string.

    Some Gemini responses come back as a list of content blocks (e.g.
    [{"type": "text", "text": "...", "extras": {...}}]) instead of a plain
    string — join just the text portions so callers (and the next
    summarization prompt this gets fed back into) always see clean text,
    never a raw Python list repr with signature/extras metadata baked in.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
        else:
            parts.append(str(block))
    return "".join(parts)


def summarize(existing_summary: str | None, conversation_text: str) -> str | None:
    """Produce an updated running summary that folds conversation_text into existing_summary.

    Args:
        existing_summary: The prior running summary's text, or None if this
            is the first time summarization has run for this thread.
        conversation_text: Plain-text rendition of the older messages being
            folded in (see orchestrator._format_messages_for_summary).

    Returns:
        The new summary text, or None if the Gemini call itself failed —
        callers should leave the history unsummarized this turn rather than
        lose data on a transient failure.
    """
    prompt = SUMMARIZATION_PROMPT.format(
        existing_summary=existing_summary or "(none yet)",
        conversation_text=conversation_text,
    )
    try:
        response = _get_summarization_model().invoke(prompt)
    except Exception:
        # Broad on purpose — same reasoning as credibility_scoring_tool.py:
        # no single well-known exception type for an LLM call failing (API
        # error, rate limit, etc.) to catch narrowly.
        logger.exception("Summarization failed — leaving message history unsummarized this turn.")
        return None

    summary_text = _content_to_text(response.content)
    logger.info("Produced updated conversation summary (%d chars).", len(summary_text))
    return summary_text
