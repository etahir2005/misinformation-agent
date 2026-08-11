"""Tests for pii_scrub_node — the graph's first node, which redacts PII out of
the current turn's message before anything else (guardrail, Gemini, Pinecone,
Postgres checkpointing) ever sees it.

Calls pii_scrub_node directly with plain state dicts, same style as
test_orchestrator_routing.py — no need to build the full graph just to
exercise this one node's behavior. See test_orchestrator_build.py for the
end-to-end wiring test confirming this node is actually reached first.
"""

from langchain_core.messages import HumanMessage

from agent.orchestrator import pii_scrub_node, scrub_pii


def test_pii_scrub_node_redacts_email() -> None:
    state = {"messages": [HumanMessage(content="Contact me at jane@example.com about this claim.")]}

    result = pii_scrub_node(state)

    assert result != {}
    assert "jane@example.com" not in result["messages"][-1].content
    assert "[REDACTED_EMAIL]" in result["messages"][-1].content


def test_pii_scrub_node_redacts_credit_card() -> None:
    # 4111111111111111 is a standard Luhn-valid Visa test number.
    state = {"messages": [HumanMessage(content="My card is 4111111111111111, is this claim true?")]}

    result = pii_scrub_node(state)

    assert result != {}
    assert "4111111111111111" not in result["messages"][-1].content
    assert "[REDACTED_CREDIT_CARD]" in result["messages"][-1].content


def test_pii_scrub_node_redacts_ip_address() -> None:
    state = {"messages": [HumanMessage(content="This came from 192.168.1.1, can you verify it?")]}

    result = pii_scrub_node(state)

    assert result != {}
    assert "192.168.1.1" not in result["messages"][-1].content
    assert "[REDACTED_IP]" in result["messages"][-1].content


def test_pii_scrub_node_redacts_mac_address() -> None:
    state = {"messages": [HumanMessage(content="Device 00:1A:2B:3C:4D:5E sent this claim.")]}

    result = pii_scrub_node(state)

    assert result != {}
    assert "00:1A:2B:3C:4D:5E" not in result["messages"][-1].content
    assert "[REDACTED_MAC_ADDRESS]" in result["messages"][-1].content


def test_pii_scrub_node_chains_all_pii_types_in_one_message() -> None:
    """Every middleware instance must see the previous one's redacted output,
    not the original message — otherwise only the last-run type would survive.
    """
    content = "Reach me at jane@example.com or from 192.168.1.1, card 4111111111111111."
    state = {"messages": [HumanMessage(content=content)]}

    result = pii_scrub_node(state)

    redacted = result["messages"][-1].content
    assert "jane@example.com" not in redacted
    assert "192.168.1.1" not in redacted
    assert "4111111111111111" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_IP]" in redacted
    assert "[REDACTED_CREDIT_CARD]" in redacted


def test_pii_scrub_node_leaves_clean_message_unchanged() -> None:
    """No PII detected — should return {} (no state update), not an
    unnecessary no-op messages replacement.
    """
    state = {"messages": [HumanMessage(content="Vaccines cause autism.")]}

    result = pii_scrub_node(state)

    assert result == {}


def test_pii_scrub_node_never_redacts_urls() -> None:
    """Deliberately excluded from _PII_MIDDLEWARES — redacting article URLs
    here would silently break source_retrieval_tool, since the model never
    sees the message before this node runs.
    """
    state = {
        "messages": [
            HumanMessage(content="Is this article true? https://example.com/some-article")
        ]
    }

    result = pii_scrub_node(state)

    assert result == {}


def test_pii_scrub_node_handles_empty_messages_gracefully() -> None:
    state = {"messages": []}

    result = pii_scrub_node(state)

    assert result == {}


# scrub_pii() — the authoritative pre-invoke scrub called from
# graph.run_claim(), not pii_scrub_node itself. See that function's
# docstring in agent/orchestrator.py for why the graph-node placement
# alone isn't early enough to keep raw PII out of Postgres.


def test_scrub_pii_redacts_email_in_a_raw_claim_string() -> None:
    redacted = scrub_pii("Contact me at jane@example.com about this claim.")

    assert "jane@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted


def test_scrub_pii_leaves_a_clean_claim_unchanged() -> None:
    assert scrub_pii("Vaccines cause autism.") == "Vaccines cause autism."


def test_scrub_pii_never_redacts_urls() -> None:
    claim = "Is this article true? https://example.com/some-article"

    assert scrub_pii(claim) == claim
