"""Tests for the FastAPI app — exercises real endpoint wiring, not just graph.py in isolation."""

from unittest.mock import ANY, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import main
from agent.auth import create_access_token
from agent.guardrail import MessageIntent
from agent.users_db import EmailAlreadyRegisteredError


@pytest.fixture
def client():
    """A TestClient wired to a real graph (InMemorySaver), chat model mocked.

    Patches init_chat_model (no real Gemini call), the guardrail's own
    separate classification model (no real Gemini call there either — every
    request goes through guardrail_node before anything else now),
    main.build_connection_pool (no real Postgres pool), and
    main.build_checkpointer (returns InMemorySaver instead of a real
    Postgres-backed one) — patched where they're used (in main's,
    orchestrator's, and guardrail's namespaces), same convention already
    used throughout this codebase's tests. Without the guardrail patch,
    every /chat request in these tests would silently fall through to a
    real, unmocked Gemini call for intent classification.

    main.setup_users_table and main.setup_conversations_table are also
    patched out — the lifespan calls both on every startup, and these
    tests should never need a live Postgres connection just to create
    those tables. Tests that exercise the conversation-list behavior
    itself (creating/touching/listing rows) patch the specific
    agent.conversations_db functions used by main.py instead, following
    the same "patch where used" convention as main.create_user etc.
    """
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Final answer.")

    with patch(
        "agent.orchestrator.init_chat_model", return_value=mock_model
    ), patch("agent.guardrail._get_intent_model") as mock_get_intent_model, patch(
        "main.build_connection_pool"
    ) as mock_build_pool, patch(
        "main.build_checkpointer", return_value=InMemorySaver()
    ), patch("main.setup_users_table"), patch("main.setup_conversations_table"), patch(
        "main.create_conversation"
    ), patch("main.touch_conversation"):
        mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")
        mock_build_pool.return_value.__enter__.return_value = MagicMock()
        mock_build_pool.return_value.__exit__.return_value = False

        with TestClient(main.app) as test_client:
            yield test_client


def _auth_header(user_id: str = "user-1", email: str = "person@example.com") -> dict:
    """Build an Authorization header carrying a valid JWT for the given user."""
    token = create_access_token(user_id, email)
    return {"Authorization": f"Bearer {token}"}


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /auth/signup -------------------------------------------------------------


def test_signup_returns_token(client: TestClient) -> None:
    with patch("main.create_user", return_value="new-user-id"):
        response = client.post(
            "/auth/signup",
            json={"email": "new@example.com", "password": "a-long-enough-password"},
        )
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_signup_rejects_short_password(client: TestClient) -> None:
    response = client.post(
        "/auth/signup", json={"email": "new@example.com", "password": "short"}
    )
    assert response.status_code == 400


def test_signup_rejects_duplicate_email(client: TestClient) -> None:
    with patch("main.create_user", side_effect=EmailAlreadyRegisteredError("dup@example.com")):
        response = client.post(
            "/auth/signup",
            json={"email": "dup@example.com", "password": "a-long-enough-password"},
        )
    assert response.status_code == 409


# --- /auth/login --------------------------------------------------------------


def test_login_returns_token_for_correct_credentials(client: TestClient) -> None:
    stored_user = {
        "id": "user-1",
        "email": "person@example.com",
        "password_hash": "irrelevant-because-verify_password-is-mocked",
    }
    with patch("main.get_user_by_email", return_value=stored_user), patch(
        "main.verify_password", return_value=True
    ):
        response = client.post(
            "/auth/login", json={"email": "person@example.com", "password": "correct-password"}
        )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_rejects_wrong_password(client: TestClient) -> None:
    stored_user = {"id": "user-1", "email": "person@example.com", "password_hash": "hash"}
    with patch("main.get_user_by_email", return_value=stored_user), patch(
        "main.verify_password", return_value=False
    ):
        response = client.post(
            "/auth/login", json={"email": "person@example.com", "password": "wrong-password"}
        )
    assert response.status_code == 401


def test_login_rejects_unknown_email(client: TestClient) -> None:
    with patch("main.get_user_by_email", return_value=None):
        response = client.post(
            "/auth/login", json={"email": "nobody@example.com", "password": "whatever"}
        )
    assert response.status_code == 401


# --- /chat auth -----------------------------------------------------------------


def test_chat_requires_a_token(client: TestClient) -> None:
    response = client.post("/chat", json={"claim": "Is the sky blue?"})
    assert response.status_code == 401


def test_chat_rejects_garbage_token(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"claim": "Is the sky blue?"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401


def test_chat_returns_answer_and_generates_thread_id(client: TestClient) -> None:
    response = client.post("/chat", json={"claim": "Is the sky blue?"}, headers=_auth_header())
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Final answer."
    assert body["thread_id"]
    assert body["recursion_limit_hit"] is False


def test_chat_reuses_provided_thread_id_for_the_same_user(client: TestClient) -> None:
    headers = _auth_header(user_id="user-1")

    first = client.post(
        "/chat", json={"claim": "claim one", "thread_id": "my-thread"}, headers=headers
    )
    assert first.json()["thread_id"] == "my-thread"

    second = client.post(
        "/chat", json={"claim": "claim two", "thread_id": "my-thread"}, headers=headers
    )
    assert second.status_code == 200
    assert second.json()["thread_id"] == "my-thread"


def test_chat_rejects_empty_claim(client: TestClient) -> None:
    response = client.post("/chat", json={"claim": "   "}, headers=_auth_header())
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
        "main.build_connection_pool"
    ) as mock_build_pool, patch(
        "main.build_checkpointer", return_value=InMemorySaver()
    ), patch("main.setup_users_table"), patch("main.setup_conversations_table"), patch(
        "main.create_conversation"
    ), patch("main.touch_conversation"):
        mock_get_intent_model.return_value.invoke.return_value = MessageIntent(category="claim")
        mock_build_pool.return_value.__enter__.return_value = MagicMock()
        mock_build_pool.return_value.__exit__.return_value = False

        with TestClient(main.app) as test_client:
            response = test_client.post(
                "/chat", json={"claim": "Is the sky blue?"}, headers=_auth_header()
            )

    assert response.status_code == 200
    assert response.json()["answer"] == "Final answer."


# --- thread ownership -------------------------------------------------------------


def test_chat_rejects_continuing_another_users_thread(client: TestClient) -> None:
    """User A starts a thread; user B must not be able to continue it."""
    user_a_headers = _auth_header(user_id="user-a", email="a@example.com")
    user_b_headers = _auth_header(user_id="user-b", email="b@example.com")

    created = client.post(
        "/chat",
        json={"claim": "claim one", "thread_id": "shared-thread-id"},
        headers=user_a_headers,
    )
    assert created.status_code == 200

    hijack_attempt = client.post(
        "/chat",
        json={"claim": "claim two", "thread_id": "shared-thread-id"},
        headers=user_b_headers,
    )
    assert hijack_attempt.status_code == 403


# --- /conversations -------------------------------------------------------------


def test_get_conversations_requires_token(client: TestClient) -> None:
    response = client.get("/conversations")
    assert response.status_code == 401


def test_get_conversations_returns_list(client: TestClient) -> None:
    fake_rows = [
        {
            "thread_id": "11111111-1111-1111-1111-111111111111",
            "title": "Is the sky blue?",
            "created_at": "2026-08-07T00:00:00+00:00",
            "updated_at": "2026-08-07T00:00:00+00:00",
        }
    ]
    with patch("main.list_conversations", return_value=fake_rows):
        response = client.get("/conversations", headers=_auth_header())
    assert response.status_code == 200
    assert response.json() == fake_rows


def test_chat_creates_conversation_for_new_thread(client: TestClient) -> None:
    """The very first message on a brand-new thread should record a sidebar entry."""
    with patch("main.create_conversation") as mock_create:
        response = client.post(
            "/chat", json={"claim": "Is the sky blue?"}, headers=_auth_header(user_id="user-1")
        )
    assert response.status_code == 200
    mock_create.assert_called_once()
    _, thread_id, user_id, title = mock_create.call_args.args
    assert thread_id == response.json()["thread_id"]
    assert user_id == "user-1"
    assert title == "Is the sky blue?"


def test_chat_touches_conversation_for_existing_thread(client: TestClient) -> None:
    """A second message on an already-existing thread should bump recency, not re-create it."""
    headers = _auth_header(user_id="user-1")
    client.post("/chat", json={"claim": "claim one", "thread_id": "my-thread"}, headers=headers)

    with patch("main.touch_conversation") as mock_touch:
        response = client.post(
            "/chat", json={"claim": "claim two", "thread_id": "my-thread"}, headers=headers
        )
    assert response.status_code == 200
    mock_touch.assert_called_once_with(ANY, "my-thread")


# --- /conversations/{thread_id}/messages -----------------------------------------


def test_chat_creates_conversation_for_client_provided_fresh_thread_id(
    client: TestClient,
) -> None:
    """app.py always sends a thread_id it generated itself, even for a brand-new
    conversation's first message — this must still be recorded as new (regression
    test: previously any provided thread_id was treated as already-existing, so
    Streamlit conversations never appeared in the sidebar).
    """
    with patch("main.create_conversation") as mock_create, patch(
        "main.touch_conversation"
    ) as mock_touch:
        response = client.post(
            "/chat",
            json={"claim": "Is the sky blue?", "thread_id": "client-generated-thread"},
            headers=_auth_header(user_id="user-1"),
        )
    assert response.status_code == 200
    mock_touch.assert_not_called()
    mock_create.assert_called_once()
    _, thread_id, user_id, title = mock_create.call_args.args
    assert thread_id == "client-generated-thread"
    assert user_id == "user-1"
    assert title == "Is the sky blue?"


def test_get_conversation_messages_requires_token(client: TestClient) -> None:
    response = client.get("/conversations/some-thread/messages")
    assert response.status_code == 401


def test_get_conversation_messages_rejects_unknown_thread(client: TestClient) -> None:
    """A thread_id with no recorded owner (never used) is rejected, not empty-but-valid."""
    response = client.get("/conversations/never-existed/messages", headers=_auth_header())
    assert response.status_code == 403


def test_get_conversation_messages_rejects_other_users_thread(client: TestClient) -> None:
    user_a_headers = _auth_header(user_id="user-a", email="a@example.com")
    user_b_headers = _auth_header(user_id="user-b", email="b@example.com")
    client.post(
        "/chat",
        json={"claim": "claim one", "thread_id": "shared-thread-id"},
        headers=user_a_headers,
    )

    response = client.get("/conversations/shared-thread-id/messages", headers=user_b_headers)
    assert response.status_code == 403


def test_get_conversation_messages_returns_history(client: TestClient) -> None:
    """History should contain the real user turn and final answer, in order."""
    headers = _auth_header(user_id="user-1")
    client.post(
        "/chat", json={"claim": "Is the sky blue?", "thread_id": "history-thread"}, headers=headers
    )

    response = client.get("/conversations/history-thread/messages", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "history-thread"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "Is the sky blue?"
    assert body["messages"][1]["content"] == "Final answer."
