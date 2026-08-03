"""Tests for the FastAPI app — exercises real endpoint wiring, not just graph.py in isolation."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import main
from agent.guardrail import MessageIntent


@pytest.fixture(autouse=True)
def disable_api_key(monkeypatch):
    """Run tests with API key auth disabled by default."""
    monkeypatch.setattr(main, "API_ACCESS_KEY", None)


@pytest.fixture
def client():
    """A TestClient wired to a real graph (InMemorySaver), chat model mocked.

    Patches init_chat_model (no real Gemini call), the guardrail's own
    separate classification model (no real Gemini call there either — every
    request goes through guardrail_node before anything else now), and
    main.build_checkpointer (no real Postgres connection) — patched where
    they're used (in main's, orchestrator's, and guardrail's namespaces),
    same convention already used throughout this codebase's tests. Without
    the guardrail patch, every /chat request in these tests would silently
    fall through to a real, unmocked Gemini call for intent classification.
    """
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Final answer.")

    with patch(
        "agent.orchestrator.init_chat_model", return_value=mock_model
    ), patch("agent.guardrail._get_intent_model") as mock_get_intent_model, patch(
        "main.build_checkpointer"
    ) as mock_build_checkpointer:
        mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")
        mock_build_checkpointer.return_value.__enter__.return_value = InMemorySaver()
        mock_build_checkpointer.return_value.__exit__.return_value = False

        with TestClient(main.app) as test_client:
            yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_generates_thread_id(client: TestClient) -> None:
    response = client.post("/chat", json={"claim": "Is the sky blue?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Final answer."
    assert body["thread_id"]
    assert body["recursion_limit_hit"] is False


def test_chat_reuses_provided_thread_id(client: TestClient) -> None:
    response = client.post("/chat", json={"claim": "claim one", "thread_id": "my-thread"})
    assert response.json()["thread_id"] == "my-thread"


def test_chat_rejects_empty_claim(client: TestClient) -> None:
    response = client.post("/chat", json={"claim": "   "})
    assert response.status_code == 400


def test_chat_flattens_list_style_message_content() -> None:
    """Some Gemini responses return content as a list of blocks instead of
    a plain string — the endpoint must flatten this into ChatResponse's
    required str field rather than raising a validation error.
    """
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(
        content=[{"type": "text", "text": "Final answer."}]
    )

    with patch(
        "agent.orchestrator.init_chat_model", return_value=mock_model
    ), patch("agent.guardrail._get_intent_model") as mock_get_intent_model, patch(
        "main.build_checkpointer"
    ) as mock_build_checkpointer, patch.object(main, "API_ACCESS_KEY", None):
        mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")
        mock_build_checkpointer.return_value.__enter__.return_value = InMemorySaver()
        mock_build_checkpointer.return_value.__exit__.return_value = False

        with TestClient(main.app) as test_client:
            response = test_client.post("/chat", json={"claim": "Is the sky blue?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Final answer."


def test_chat_requires_api_key_when_configured(client: TestClient, monkeypatch) -> None:
    """When API_ACCESS_KEY is set, requests must include a matching header."""
    monkeypatch.setattr(main, "API_ACCESS_KEY", "test-secret-key")

    no_header_response = client.post("/chat", json={"claim": "Is the sky blue?"})
    assert no_header_response.status_code == 401

    wrong_key_response = client.post(
        "/chat",
        json={"claim": "Is the sky blue?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert wrong_key_response.status_code == 401

    correct_key_response = client.post(
        "/chat",
        json={"claim": "Is the sky blue?"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert correct_key_response.status_code == 200
