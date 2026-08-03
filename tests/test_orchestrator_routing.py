"""Unit tests for the orchestrator's deterministic routing logic."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.config import MAX_MESSAGES_BEFORE_SUMMARY
from agent.orchestrator_routing import (
    _credibility_scoring_call_count,
    _current_turn_messages,
    _evidence_is_weak,
    _existing_summary_text,
    _format_messages_for_summary,
    _has_evidence_tool_run,
    _has_usable_evidence,
    _last_tool_call_args,
    _last_tool_message_index,
    _last_tool_result,
    _messages_before_current_turn,
    _needs_credibility_scoring,
    _needs_summary,
    _should_force_credibility_scoring,
    _should_force_retry_search,
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


def test_has_usable_evidence_false_with_no_tool_calls() -> None:
    """No evidence gathered yet means nothing to score."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _has_usable_evidence(state) is False


def test_has_usable_evidence_false_with_empty_fact_check() -> None:
    """An empty fact-check result alone isn't evidence to score."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
        ]
    }
    assert _has_usable_evidence(state) is False


def test_has_usable_evidence_true_with_web_search_results() -> None:
    """Web search results count as evidence, even with no fact-check claims."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
            _tool_message("web_search_tool", {"sources": [{"url": "a"}], "query_used": "x"}),
        ]
    }
    assert _has_usable_evidence(state) is True


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


def test_has_evidence_tool_run_false_before_any_evidence_tool() -> None:
    """_has_evidence_tool_run should be False before fact-check/search/retrieval run."""
    state = {"messages": [HumanMessage(content="claim")]}
    assert _has_evidence_tool_run(state) is False


def test_has_evidence_tool_run_true_after_fact_check() -> None:
    """_has_evidence_tool_run should be True once fact_check_lookup_tool has run."""
    state = {
        "messages": [
            HumanMessage(content="claim"),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
        ]
    }
    assert _has_evidence_tool_run(state) is True


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


def test_messages_before_current_turn_excludes_current_turn() -> None:
    """_messages_before_current_turn should be the exact complement of _current_turn_messages."""
    state = {
        "messages": [
            HumanMessage(content="first claim"),
            _tool_message("vector_lookup_tool", {"hit": False}),
            _tool_message("fact_check_lookup_tool", {"claims": [], "query_used": "x"}),
            HumanMessage(content="second claim"),
        ]
    }
    older_messages = _messages_before_current_turn(state)
    assert len(older_messages) == 3
    assert older_messages[0].content == "first claim"


def test_messages_before_current_turn_empty_on_first_turn() -> None:
    """With only one HumanMessage in the thread, there's nothing older to summarize yet."""
    state = {"messages": [HumanMessage(content="only claim so far")]}
    assert _messages_before_current_turn(state) == []


def test_existing_summary_text_finds_stored_summary() -> None:
    """_existing_summary_text should return the stored summary's content, found by fixed id."""
    messages = [
        SystemMessage(content="Earlier claims summarized here.", id="conversation-summary"),
        HumanMessage(content="a new claim"),
    ]
    assert _existing_summary_text(messages) == "Earlier claims summarized here."


def test_existing_summary_text_none_when_absent() -> None:
    """_existing_summary_text should return None when no summary has been stored yet."""
    messages = [HumanMessage(content="a claim"), _tool_message("web_search_tool", {"sources": []})]
    assert _existing_summary_text(messages) is None


def test_format_messages_for_summary_skips_summary_placeholder() -> None:
    """The summary placeholder itself and empty-content messages should be excluded.

    The placeholder is passed separately as existing_summary — including it
    again in conversation_text would duplicate it in the summarization
    prompt. Tool-call-only AIMessages (empty content) add no readable
    substance either.
    """
    messages = [
        SystemMessage(content="Old summary text.", id="conversation-summary"),
        HumanMessage(content="Is the sky blue?"),
        AIMessage(content="", tool_calls=[{"name": "vector_lookup_tool", "args": {}, "id": "c1"}]),
        AIMessage(content="Yes, the sky is blue."),
    ]
    formatted = _format_messages_for_summary(messages)
    assert "Old summary text." not in formatted
    assert "Human: Is the sky blue?" in formatted
    assert "AI: Yes, the sky is blue." in formatted


def test_format_messages_for_summary_skips_tool_messages() -> None:
    """Raw ToolMessage results should never reach the summarization prompt.

    Regression test for a bug caught in review: the function's docstring
    claimed tool-call machinery was excluded, but nothing actually filtered
    ToolMessage — only the summary placeholder and empty-content messages
    were skipped, so a real tool result's JSON payload (source URLs,
    snippets, confidence scores) rendered straight into the output.
    """
    messages = [
        HumanMessage(content="Is the sky blue?"),
        AIMessage(content="", tool_calls=[{"name": "web_search_tool", "args": {}, "id": "c1"}]),
        _tool_message(
            "web_search_tool",
            {"sources": [{"url": "https://example.com/sky", "snippet": "The sky is blue."}]},
        ),
        AIMessage(content="Yes, the sky is blue."),
    ]
    formatted = _format_messages_for_summary(messages)
    assert "https://example.com/sky" not in formatted
    assert "Tool:" not in formatted
    assert "Human: Is the sky blue?" in formatted
    assert "AI: Yes, the sky is blue." in formatted


def test_needs_summary_false_below_threshold() -> None:
    """_needs_summary should route to the orchestrator when older history is short."""
    state = {
        "messages": [
            HumanMessage(content="first claim"),
            AIMessage(content="answer"),
            HumanMessage(content="second claim"),
        ]
    }
    assert _needs_summary(state) == "orchestrator"


def test_needs_summary_true_at_threshold() -> None:
    """_needs_summary should route to summarize once older history hits the configured max."""
    older_messages = [
        HumanMessage(content=f"claim {i}") for i in range(MAX_MESSAGES_BEFORE_SUMMARY)
    ]
    state = {"messages": older_messages + [HumanMessage(content="current claim")]}
    assert _needs_summary(state) == "summarize"
