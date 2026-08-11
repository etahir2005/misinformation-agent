"""Unit tests for the orchestrator's response-construction and cache-writing helpers."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.orchestrator_responses import (
    _build_cache_hit_response,
    _check_and_store_verdict,
    _extract_verdict_text,
    _gather_sources_for_scoring,
    _store_verdict_if_new,
)


def _tool_message(tool_name: str, content: dict) -> ToolMessage:
    """Build a ToolMessage the way the graph would produce one."""
    return ToolMessage(name=tool_name, content=json.dumps(content), tool_call_id="fake-id")


def test_gather_sources_for_scoring_combines_both_evidence_types() -> None:
    """_gather_sources_for_scoring should pull from both fact-check and search results."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message(
                "fact_check_lookup_tool",
                {
                    "claims": [
                        {
                            "url": "https://factcheck.example/1",
                            "publisher": "ExamplePublisher",
                            "rating": "Misleading",
                            "claim_text": "Some claim text.",
                        }
                    ],
                    "query_used": "x",
                },
            ),
            _tool_message(
                "web_search_tool",
                {
                    "sources": [
                        {
                            "url": "https://search.example/2",
                            "snippet": "Some snippet.",
                            "title": "T",
                        }
                    ],
                    "query_used": "x",
                },
            ),
        ]
    }

    sources = _gather_sources_for_scoring(state)

    assert len(sources) == 2
    assert sources[0]["url"] == "https://factcheck.example/1"
    assert "ExamplePublisher" in sources[0]["content"]
    assert "Misleading" in sources[0]["content"]
    assert sources[1]["url"] == "https://search.example/2"
    assert sources[1]["content"] == "Some snippet."


def test_gather_sources_for_scoring_empty_when_no_evidence() -> None:
    """_gather_sources_for_scoring should return an empty list with no prior tool results."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _gather_sources_for_scoring(state) == []


def test_gather_sources_for_scoring_merges_both_search_rounds() -> None:
    """A retry search shouldn't discard the first round's sources.

    Regression test (caught in PR review): _gather_sources_for_scoring used
    to only look at the most recent web_search_tool call, so when weak
    evidence triggered a retry search, the second scoring pass would only
    see the retry's sources — the original round's evidence was silently
    dropped instead of being combined with it.
    """
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message(
                "web_search_tool",
                {"sources": [{"url": "https://round-one.example", "snippet": "First round."}]},
            ),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
            _tool_message(
                "web_search_tool",
                {"sources": [{"url": "https://round-two.example", "snippet": "Second round."}]},
            ),
        ]
    }

    sources = _gather_sources_for_scoring(state)

    urls = {s["url"] for s in sources}
    assert urls == {"https://round-one.example", "https://round-two.example"}


def test_build_cache_hit_response_includes_summary_and_sources() -> None:
    """_build_cache_hit_response should surface the cached verdict summary and sources."""
    cache_result = {
        "hit": True,
        "confidence": 0.9,
        "sources": ["https://example.com/a", "https://example.com/b"],
        "verdict_summary": "This claim is false.",
    }

    response = _build_cache_hit_response(cache_result)

    assert "This claim is false." in response.content
    assert "https://example.com/a" in response.content
    assert "https://example.com/b" in response.content


@patch("agent.orchestrator_responses.store_verdict")
def test_store_verdict_if_new_uses_vector_lookup_claim_text(
    mock_store_verdict: MagicMock,
) -> None:
    """Storage should embed the same claim text vector_lookup_tool used, not the raw message."""
    state = {
        "messages": [
            HumanMessage(content="Is it true that the sky is green?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "vector_lookup_tool",
                        "args": {"claim": "The sky is green"},
                        "id": "call_1",
                    }
                ],
            ),
            _tool_message("vector_lookup_tool", {"hit": False}),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fact_check_lookup_tool",
                        "args": {"query": "sky green"},
                        "id": "call_2",
                    }
                ],
            ),
            _tool_message(
                "fact_check_lookup_tool",
                {
                    "claims": [
                        {
                            "rating": "False",
                            "url": "https://a.com",
                            "publisher": "X",
                            "claim_text": "The sky is green",
                        }
                    ],
                    "query_used": "sky green",
                },
            ),
        ]
    }

    _store_verdict_if_new(state)

    mock_store_verdict.assert_called_once()
    assert mock_store_verdict.call_args.kwargs["claim"] == "The sky is green"


@patch("agent.orchestrator_responses.store_verdict")
def test_store_verdict_if_new_skips_when_vector_lookup_never_called(
    mock_store_verdict: MagicMock,
) -> None:
    """No claim text to store should mean no store call at all."""
    state = {"messages": [HumanMessage(content="claim")]}
    _store_verdict_if_new(state)
    mock_store_verdict.assert_not_called()


def _state_with_vector_lookup_claim(claim: str) -> dict:
    """Minimal state with a recorded vector_lookup_tool call, same shape
    _check_and_store_verdict / _store_verdict_if_new read claim text from.
    """
    return {
        "messages": [
            HumanMessage(content=claim),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "vector_lookup_tool", "args": {"claim": claim}, "id": "call_1"}
                ],
            ),
            _tool_message("vector_lookup_tool", {"hit": False}),
        ]
    }


def test_extract_verdict_text_handles_plain_string() -> None:
    assert _extract_verdict_text("A plain string answer.") == "A plain string answer."


def test_extract_verdict_text_handles_list_content_blocks() -> None:
    """Some Gemini responses come back as a list of content blocks instead
    of a plain string — same shape main.py's _extract_text normalizes.
    """
    content = [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]
    assert _extract_verdict_text(content) == "Part one. Part two."


@patch("agent.orchestrator_responses.check_verdict_completeness")
@patch("agent.orchestrator_responses.store_verdict")
def test_check_and_store_verdict_stores_when_complete(
    mock_store_verdict: MagicMock, mock_check_completeness: MagicMock
) -> None:
    """A verdict judged complete should still only actually get cached if
    _store_verdict_if_new itself finds a resolvable result to store — so
    this state needs a real fact_check_lookup_tool result on record, same
    as test_store_verdict_if_new_uses_vector_lookup_claim_text, not just a
    bare vector_lookup_tool miss.
    """
    mock_check_completeness.return_value = True
    claim = "The sky is green"
    state = {
        "messages": [
            HumanMessage(content=claim),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "vector_lookup_tool", "args": {"claim": claim}, "id": "call_1"}
                ],
            ),
            _tool_message("vector_lookup_tool", {"hit": False}),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fact_check_lookup_tool", "args": {"query": claim}, "id": "call_2"}
                ],
            ),
            _tool_message(
                "fact_check_lookup_tool",
                {
                    "claims": [
                        {
                            "rating": "False",
                            "url": "https://a.com",
                            "publisher": "X",
                            "claim_text": claim,
                        }
                    ],
                    "query_used": claim,
                },
            ),
        ]
    }
    response = AIMessage(content="False. The sky is blue due to Rayleigh scattering.")

    result = _check_and_store_verdict(state, response)

    assert result is True
    mock_store_verdict.assert_called_once()


@patch("agent.orchestrator_responses.check_verdict_completeness")
@patch("agent.orchestrator_responses.store_verdict")
def test_check_and_store_verdict_skips_storage_when_incomplete(
    mock_store_verdict: MagicMock, mock_check_completeness: MagicMock
) -> None:
    """An incomplete verdict must never be cached — it would otherwise get
    reused for every future semantically-similar claim.
    """
    mock_check_completeness.return_value = False
    state = _state_with_vector_lookup_claim("The sky is green")
    response = AIMessage(content="Unclear.")

    result = _check_and_store_verdict(state, response)

    assert result is False
    mock_store_verdict.assert_not_called()


@patch("agent.orchestrator_responses.check_verdict_completeness")
@patch("agent.orchestrator_responses.store_verdict")
def test_check_and_store_verdict_skips_check_when_no_claim_on_record(
    mock_store_verdict: MagicMock, mock_check_completeness: MagicMock
) -> None:
    """No vector_lookup_tool call on record means nothing to check or store
    — same edge case _store_verdict_if_new already guards against.
    """
    state = {"messages": [HumanMessage(content="claim")]}
    response = AIMessage(content="Some answer.")

    result = _check_and_store_verdict(state, response)

    assert result is True
    mock_check_completeness.assert_not_called()
    mock_store_verdict.assert_not_called()
