"""Unit tests for the orchestrator's deterministic routing logic."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.orchestrator import (
    _build_cache_hit_response,
    _credibility_scoring_call_count,
    _current_turn_messages,
    _evidence_is_weak,
    _gather_sources_for_scoring,
    _has_evidence_to_score,
    _has_gathered_evidence,
    _last_tool_call_args,
    _last_tool_message_index,
    _last_tool_result,
    _needs_credibility_scoring,
    _should_force_credibility_scoring,
    _should_force_retry_search,
    _store_verdict_if_new,
)


def _tool_message(tool_name: str, content: dict) -> ToolMessage:
    """Build a ToolMessage the way the graph would produce one."""
    return ToolMessage(name=tool_name, content=json.dumps(content), tool_call_id="fake-id")


def test_last_tool_result_returns_most_recent_match() -> None:
    """_last_tool_result should return the latest ToolMessage for the given tool."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "old"}),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "new"}),
        ]
    }
    result = _last_tool_result(state, "fact_check_lookup_tool")
    assert result == {"claims": [], "query_used": "new"}


def test_last_tool_result_returns_none_when_absent() -> None:
    """_last_tool_result should return None if the tool was never called."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _last_tool_result(state, "fact_check_lookup_tool") is None


def test_needs_credibility_scoring_false_for_single_clean_rating() -> None:
    """A single claim with a clean True/False-equivalent rating needs no scoring."""
    result = {"claims": [{"rating": "False"}]}
    assert _needs_credibility_scoring(result) is False


def test_needs_credibility_scoring_true_for_multiple_claims() -> None:
    """More than one claim always needs scoring, regardless of ratings."""
    result = {"claims": [{"rating": "False"}, {"rating": "True"}]}
    assert _needs_credibility_scoring(result) is True


def test_needs_credibility_scoring_true_for_messy_rating() -> None:
    """A single claim with a non-clean rating (e.g. 'Misleading') needs scoring."""
    result = {"claims": [{"rating": "Misleading"}]}
    assert _needs_credibility_scoring(result) is True


def test_needs_credibility_scoring_true_for_zero_claims() -> None:
    """Zero claims needs scoring (there's nothing clean to rely on)."""
    result = {"claims": []}
    assert _needs_credibility_scoring(result) is True


def test_has_evidence_to_score_false_with_no_tool_calls() -> None:
    """No evidence gathered yet means nothing to score."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _has_evidence_to_score(state) is False


def test_has_evidence_to_score_false_with_empty_fact_check() -> None:
    """An empty fact-check result alone isn't evidence to score."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
        ]
    }
    assert _has_evidence_to_score(state) is False


def test_has_evidence_to_score_true_with_web_search_results() -> None:
    """Web search results count as evidence, even with no fact-check claims."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
            _tool_message("web_search_tool", {"sources": [{"url": "a"}], "query_used": "x"}),
        ]
    }
    assert _has_evidence_to_score(state) is True


def test_should_force_credibility_scoring_true_when_evidence_is_messy() -> None:
    """Should force scoring once real evidence exists and isn't a clean ruling."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message(
                "fact_check_lookup_tool", {"claims": [{"rating": "Misleading"}], "query_used": "x"}
            ),
        ]
    }
    assert _should_force_credibility_scoring(state) is True


def test_should_force_credibility_scoring_false_when_already_scored() -> None:
    """Should not force scoring again with no new evidence since the last score."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message(
                "fact_check_lookup_tool", {"claims": [{"rating": "Misleading"}], "query_used": "x"}
            ),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.5, "sources_conflict": False},
            ),
        ]
    }
    assert _should_force_credibility_scoring(state) is False


def test_should_force_credibility_scoring_false_when_url_given() -> None:
    """Should not force scoring when the user gave a URL (source_retrieval_tool flow)."""
    state = {
        "messages": [
            HumanMessage(content="https://example.com/article"),
            _tool_message("source_retrieval_tool", {"content": "text", "url": "x"}),
        ]
    }
    assert _should_force_credibility_scoring(state) is False


def test_should_force_credibility_scoring_true_on_retry_rescoring() -> None:
    """Should force a second scoring pass once a retry search follows a weak first score."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
            _tool_message("web_search_tool", {"sources": [{"url": "a"}], "query_used": "x"}),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
            _tool_message("web_search_tool", {"sources": [{"url": "b"}], "query_used": "y"}),
        ]
    }
    assert _should_force_credibility_scoring(state) is True


def test_should_force_credibility_scoring_false_after_second_score() -> None:
    """Should not force a third scoring pass — the one retry cycle is already used up."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
            _tool_message("web_search_tool", {"sources": [{"url": "a"}], "query_used": "x"}),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
            _tool_message("web_search_tool", {"sources": [{"url": "b"}], "query_used": "y"}),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
        ]
    }
    assert _should_force_credibility_scoring(state) is False


def test_evidence_is_weak_false_when_no_score_exists() -> None:
    """_evidence_is_weak should be False if credibility_scoring_tool hasn't run yet."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _evidence_is_weak(state) is False


def test_evidence_is_weak_true_for_low_confidence() -> None:
    """_evidence_is_weak should be True when confidence is below the threshold."""
    state = {
        "messages": [
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
        ]
    }
    assert _evidence_is_weak(state) is True


def test_evidence_is_weak_true_for_conflicting_sources() -> None:
    """_evidence_is_weak should be True on conflict alone, even with high confidence."""
    state = {
        "messages": [
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.95, "sources_conflict": True},
            ),
        ]
    }
    assert _evidence_is_weak(state) is True


def test_should_force_retry_search_true_after_weak_first_score() -> None:
    """A single weak/conflicting scoring result should force one retry search."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("web_search_tool", {"sources": [{"url": "a"}], "query_used": "x"}),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
        ]
    }
    assert _should_force_retry_search(state) is True


def test_should_force_retry_search_false_for_strong_evidence() -> None:
    """A confident, non-conflicting score should not trigger a retry."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("web_search_tool", {"sources": [{"url": "a"}], "query_used": "x"}),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.9, "sources_conflict": False},
            ),
        ]
    }
    assert _should_force_retry_search(state) is False


def test_should_force_retry_search_false_after_retry_already_used() -> None:
    """A second scoring call should never trigger another retry, even if still weak."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("web_search_tool", {"sources": [{"url": "a"}], "query_used": "x"}),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
            _tool_message("web_search_tool", {"sources": [{"url": "b"}], "query_used": "y"}),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
        ]
    }
    assert _should_force_retry_search(state) is False


def test_should_force_retry_search_false_once_retry_search_has_happened() -> None:
    """Should stop forcing retry search once the one retry search already ran.

    Regression test: the first version of this function only checked the
    scoring call count, not whether a search had already happened since
    that score — which caused it to force search forever instead of
    handing off to rescoring (confirmed live via a recursion-limit hang on
    the "video games cause violent behavior" test claim).
    """
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("web_search_tool", {"sources": [{"url": "a"}], "query_used": "x"}),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.7, "sources_conflict": True},
            ),
            _tool_message("web_search_tool", {"sources": [{"url": "b"}], "query_used": "y"}),
        ]
    }
    assert _should_force_retry_search(state) is False


def test_credibility_scoring_call_count_counts_all_calls() -> None:
    """_credibility_scoring_call_count should count every matching ToolMessage."""
    state = {
        "messages": [
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.3, "sources_conflict": False},
            ),
            _tool_message(
                "credibility_scoring_tool",
                {"source_scores": [], "overall_confidence": 0.8, "sources_conflict": False},
            ),
        ]
    }
    assert _credibility_scoring_call_count(state) == 2


def test_last_tool_message_index_returns_latest_position() -> None:
    """_last_tool_message_index should return the index of the most recent match."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("web_search_tool", {"sources": [], "query_used": "x"}),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
            _tool_message("web_search_tool", {"sources": [], "query_used": "y"}),
        ]
    }
    assert _last_tool_message_index(state, "web_search_tool") == 3


def test_last_tool_message_index_none_when_absent() -> None:
    """_last_tool_message_index should return None if the tool was never called."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _last_tool_message_index(state, "web_search_tool") is None


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


def test_has_gathered_evidence_false_before_any_evidence_tool() -> None:
    """_has_gathered_evidence should be False before fact-check/search/retrieval run."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _has_gathered_evidence(state) is False


def test_has_gathered_evidence_true_after_fact_check() -> None:
    """_has_gathered_evidence should be True once fact_check_lookup_tool has run."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
        ]
    }
    assert _has_gathered_evidence(state) is True


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


def test_last_tool_call_args_returns_args_for_matching_tool_call() -> None:
    """_last_tool_call_args should return the args dict from the matching tool call."""
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
        ]
    }
    assert _last_tool_call_args(state, "vector_lookup_tool") == {"claim": "The sky is green"}


def test_last_tool_call_args_none_when_tool_not_called() -> None:
    """_last_tool_call_args should return None if the tool was never called."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _last_tool_call_args(state, "vector_lookup_tool") is None


@patch("agent.orchestrator.store_verdict")
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


@patch("agent.orchestrator.store_verdict")
def test_store_verdict_if_new_skips_when_vector_lookup_never_called(
    mock_store_verdict: MagicMock,
) -> None:
    """No claim text to store should mean no store call at all."""
    state = {"messages": [HumanMessage(content="claim")]}
    _store_verdict_if_new(state)
    mock_store_verdict.assert_not_called()


def test_current_turn_messages_excludes_prior_turns() -> None:
    """_current_turn_messages should only include messages from the latest HumanMessage on.

    Regression test (caught in PR review): without this scoping, a second
    claim asked in the same thread would inherit an earlier claim's tool
    results from history, making the orchestrator think the current claim
    was already cache-checked or searched when it wasn't.
    """
    state = {
        "messages": [
            HumanMessage(content="first claim"),
            _tool_message("vector_lookup_tool", {"hit": False}),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
            HumanMessage(content="second claim"),
        ]
    }
    turn_messages = _current_turn_messages(state)
    assert len(turn_messages) == 1
    assert turn_messages[0].content == "second claim"


def test_current_turn_messages_returns_all_when_no_human_message() -> None:
    """_current_turn_messages should fall back to the full list if no HumanMessage exists."""
    state = {"messages": [_tool_message("web_search_tool", {"sources": [], "query_used": "x"})]}
    assert _current_turn_messages(state) == state["messages"]
