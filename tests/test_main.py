"""Tests for the FastAPI app — exercises real endpoint wiring, not just graph.py in isolation."""

from unittest.mock import ANY, MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import main
from agent.auth import create_access_token, decode_access_token
from agent.guardrail import MessageIntent
from agent.users_db import EmailAlreadyRegisteredError

# Fixed, named UUIDs for tests that need a real, well-formed thread_id
# (the /chat endpoint now rejects anything else — see _is_valid_uuid in
# main.py) but don't care about its specific value. Named rather than
# inlined so tests stay short/readable and every occurrence of "the same
# thread" is obviously the same constant, not four different-looking
# 36-character literals that happen to match by coincidence.
_THREAD_ID = "33333333-3333-3333-3333-333333333333"
_SHARED_THREAD_ID = "22222222-2222-2222-2222-222222222222"
_CLIENT_GENERATED_THREAD_ID = "44444444-4444-4444-4444-444444444444"
_HISTORY_THREAD_ID = "55555555-5555-5555-5555-555555555555"


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

    main.purge_stale_conversations is also patched to return an empty
    list — the lifespan calls it on every startup too, and it would
    otherwise run its DELETE ... RETURNING query against the mocked pool
    (a MagicMock, not iterable) and blow up before any test body runs.
    Tests that specifically exercise the retention purge build their own
    TestClient instead of using this fixture, same pattern already used
    by test_chat_flattens_list_style_message_content below.

    main.get_pending_escalation is patched to return None by default for
    the same reason as purge_stale_conversations above, but with a sharper
    failure mode if left unpatched: /chat calls it on every message to an
    existing thread (see the pending-review guard in main.py's chat()), and
    an unpatched real call against the mocked pool returns a MagicMock
    result — which is truthy and not None, so the guard would wrongly
    reject every second message on any thread as "pending review" rather
    than simply erroring loudly. Caught by actually running the full suite,
    not by ruff or import-time checks alone. Tests that specifically
    exercise the pending-review guard override this default explicitly.
    """
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    # A fresh AIMessage per call, not one shared object reused across every
    # invocation — LangGraph's add_messages reducer assigns a message a
    # random id the first time it sees one with id=None, mutating that
    # object in place. Reusing the same object across multiple turns means
    # its id gets "reused" too, so add_messages sees it as an update to an
    # already-known message and overwrites it in place instead of
    # appending a new one — silently corrupting message order on any
    # second-or-later turn. A real Gemini call always returns a brand-new
    # object, so this only ever bites mocks; a side_effect callable
    # sidesteps it by constructing a new instance on every call.
    mock_model.invoke.side_effect = lambda *args, **kwargs: AIMessage(content="Final answer.")

    with patch(
        "agent.orchestrator.init_chat_model", return_value=mock_model
    ), patch("agent.guardrail._get_intent_model") as mock_get_intent_model, patch(
        "main.build_connection_pool"
    ) as mock_build_pool, patch(
        "main.build_checkpointer", return_value=InMemorySaver()
    ), patch("main.setup_users_table"), patch("main.setup_conversations_table"), patch(
        "main.setup_escalations_table"
    ), patch(
        "main.create_conversation"
    ), patch("main.touch_conversation"), patch(
        "main.get_pending_escalation", return_value=None
    ), patch(
        "main.purge_stale_conversations", return_value=[]
    ):
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
    assert "X-New-Token" not in response.headers


def test_chat_rejects_garbage_token(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"claim": "Is the sky blue?"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401
    assert "X-New-Token" not in response.headers


def test_chat_returns_answer_and_generates_thread_id(client: TestClient) -> None:
    response = client.post("/chat", json={"claim": "Is the sky blue?"}, headers=_auth_header())
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Final answer."
    assert body["thread_id"]
    assert body["recursion_limit_hit"] is False


def test_chat_response_includes_a_refreshed_token(client: TestClient) -> None:
    """Every authenticated call issues a fresh token via X-New-Token — the
    sliding-session mechanism, not just one fixed-expiry token handed out
    at login. See get_current_user() in main.py.
    """
    response = client.post(
        "/chat",
        json={"claim": "Is the sky blue?"},
        headers=_auth_header(user_id="user-1", email="person@example.com"),
    )
    assert response.status_code == 200
    new_token = response.headers.get("X-New-Token")
    assert new_token
    payload = decode_access_token(new_token)
    assert payload is not None
    assert payload["sub"] == "user-1"
    assert payload["email"] == "person@example.com"


def test_chat_reuses_provided_thread_id_for_the_same_user(client: TestClient) -> None:
    headers = _auth_header(user_id="user-1")

    first = client.post(
        "/chat", json={"claim": "claim one", "thread_id": _THREAD_ID}, headers=headers
    )
    assert first.json()["thread_id"] == _THREAD_ID

    second = client.post(
        "/chat", json={"claim": "claim two", "thread_id": _THREAD_ID}, headers=headers
    )
    assert second.status_code == 200
    assert second.json()["thread_id"] == _THREAD_ID


def test_chat_rejects_empty_claim(client: TestClient) -> None:
    response = client.post("/chat", json={"claim": "   "}, headers=_auth_header())
    assert response.status_code == 400


def test_chat_rejects_malformed_thread_id(client: TestClient) -> None:
    """A non-UUID thread_id must be rejected immediately with 400, before
    any real claim processing happens — the conversations table's
    thread_id column requires a real UUID (see agent/conversations_db.py),
    so silently accepting one here would fully process a claim only to
    create an orphaned checkpoint thread that can never be recorded in,
    listed from, or deleted through the sidebar.
    """
    response = client.post(
        "/chat",
        json={"claim": "Is the sky blue?", "thread_id": "not-a-real-uuid"},
        headers=_auth_header(),
    )
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
        "main.setup_escalations_table"
    ), patch(
        "main.create_conversation"
    ), patch("main.touch_conversation"), patch(
        "main.get_pending_escalation", return_value=None
    ), patch(
        "main.purge_stale_conversations", return_value=[]
    ):
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
        json={"claim": "claim one", "thread_id": _SHARED_THREAD_ID},
        headers=user_a_headers,
    )
    assert created.status_code == 200

    hijack_attempt = client.post(
        "/chat",
        json={"claim": "claim two", "thread_id": _SHARED_THREAD_ID},
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


def test_chat_creates_conversation_title_from_scrubbed_claim_not_raw_request(
    client: TestClient,
) -> None:
    """Regression test: the sidebar conversation title must be derived from
    the *scrubbed* claim (result["claim"], from graph.run_claim()), not the
    raw request.claim — otherwise PII redacted everywhere else in the
    pipeline still ends up stored in the conversations table's title
    column. Caught via a live Streamlit smoke test, not the (fully mocked)
    test suite that existed before this test.
    """
    with patch("main.create_conversation") as mock_create:
        response = client.post(
            "/chat",
            json={"claim": "Email me at jane@example.com about this claim."},
            headers=_auth_header(user_id="user-1"),
        )
    assert response.status_code == 200
    mock_create.assert_called_once()
    _, _, _, title = mock_create.call_args.args
    assert "jane@example.com" not in title
    assert "[REDACTED_EMAIL]" in title


def test_chat_touches_conversation_for_existing_thread(client: TestClient) -> None:
    """A second message on an already-existing thread should bump recency, not re-create it."""
    headers = _auth_header(user_id="user-1")
    client.post("/chat", json={"claim": "claim one", "thread_id": _THREAD_ID}, headers=headers)

    with patch("main.touch_conversation") as mock_touch:
        response = client.post(
            "/chat", json={"claim": "claim two", "thread_id": _THREAD_ID}, headers=headers
        )
    assert response.status_code == 200
    mock_touch.assert_called_once_with(ANY, _THREAD_ID)


def test_chat_still_returns_answer_if_create_conversation_fails(client: TestClient) -> None:
    """A DB error while recording a brand-new thread in the sidebar list
    (a malformed thread_id, a transient hiccup) must not discard an
    already-successfully-processed answer or turn into a 500 — the claim
    was genuinely answered by this point, regardless of what happens to
    the sidebar bookkeeping.
    """
    with patch("main.create_conversation", side_effect=RuntimeError("db exploded")):
        response = client.post(
            "/chat", json={"claim": "Is the sky blue?"}, headers=_auth_header()
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "Final answer."


def test_chat_still_returns_answer_if_touch_conversation_fails(client: TestClient) -> None:
    """Same guarantee on the existing-thread path (touch_conversation)."""
    headers = _auth_header(user_id="user-1")
    client.post("/chat", json={"claim": "claim one", "thread_id": _THREAD_ID}, headers=headers)

    with patch("main.touch_conversation", side_effect=RuntimeError("db exploded")):
        response = client.post(
            "/chat", json={"claim": "claim two", "thread_id": _THREAD_ID}, headers=headers
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "Final answer."


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
            json={"claim": "Is the sky blue?", "thread_id": _CLIENT_GENERATED_THREAD_ID},
            headers=_auth_header(user_id="user-1"),
        )
    assert response.status_code == 200
    mock_touch.assert_not_called()
    mock_create.assert_called_once()
    _, thread_id, user_id, title = mock_create.call_args.args
    assert thread_id == _CLIENT_GENERATED_THREAD_ID
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
        json={"claim": "claim one", "thread_id": _SHARED_THREAD_ID},
        headers=user_a_headers,
    )

    response = client.get(f"/conversations/{_SHARED_THREAD_ID}/messages", headers=user_b_headers)
    assert response.status_code == 403


def test_get_conversation_messages_returns_history(client: TestClient) -> None:
    """History should contain the real user turn and final answer, in order."""
    headers = _auth_header(user_id="user-1")
    client.post(
        "/chat",
        json={"claim": "Is the sky blue?", "thread_id": _HISTORY_THREAD_ID},
        headers=headers,
    )

    response = client.get(f"/conversations/{_HISTORY_THREAD_ID}/messages", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == _HISTORY_THREAD_ID
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "Is the sky blue?"
    assert body["messages"][1]["content"] == "Final answer."


# --- DELETE /conversations/{thread_id} -------------------------------------------


def test_delete_conversation_requires_token(client: TestClient) -> None:
    response = client.delete("/conversations/some-thread")
    assert response.status_code == 401


def test_delete_conversation_rejects_unknown_thread(client: TestClient) -> None:
    """A thread_id with no recorded owner (never used) is rejected, same as
    the read-side /conversations/{thread_id}/messages behavior — not
    silently treated as "nothing to delete"."""
    response = client.delete("/conversations/never-existed", headers=_auth_header())
    assert response.status_code == 403


def test_delete_conversation_rejects_other_users_thread(client: TestClient) -> None:
    user_a_headers = _auth_header(user_id="user-a", email="a@example.com")
    user_b_headers = _auth_header(user_id="user-b", email="b@example.com")
    client.post(
        "/chat",
        json={"claim": "claim one", "thread_id": _SHARED_THREAD_ID},
        headers=user_a_headers,
    )

    response = client.delete(f"/conversations/{_SHARED_THREAD_ID}", headers=user_b_headers)
    assert response.status_code == 403


def test_delete_conversation_succeeds_for_owner(client: TestClient) -> None:
    """Deleting an owned conversation returns 204 with no body, removes the
    sidebar row via delete_conversation(), and actually clears the
    underlying checkpoint data too — not just the sidebar row.
    """
    headers = _auth_header(user_id="user-1")
    client.post("/chat", json={"claim": "claim one", "thread_id": _THREAD_ID}, headers=headers)

    with patch("main.delete_conversation") as mock_delete:
        response = client.delete(f"/conversations/{_THREAD_ID}", headers=headers)

    assert response.status_code == 204
    assert response.content == b""
    mock_delete.assert_called_once_with(ANY, _THREAD_ID)

    # Confirm the checkpoint data is really gone, not just the sidebar row:
    # the thread's owner metadata should no longer be findable, same as a
    # thread that never existed.
    follow_up = client.get(f"/conversations/{_THREAD_ID}/messages", headers=headers)
    assert follow_up.status_code == 403


def test_delete_conversation_returns_500_on_unexpected_error(client: TestClient) -> None:
    headers = _auth_header(user_id="user-1")
    client.post("/chat", json={"claim": "claim one", "thread_id": _THREAD_ID}, headers=headers)

    with patch("main.delete_conversation", side_effect=RuntimeError("db exploded")):
        response = client.delete(f"/conversations/{_THREAD_ID}", headers=headers)

    assert response.status_code == 500


def test_delete_conversation_cancels_any_pending_escalation(client: TestClient) -> None:
    """A pending escalation for this thread (agent/escalations_db.py) would
    otherwise be orphaned by the delete — still showing in the admin's
    queue but pointing at checkpoint data that no longer exists. It should
    be marked "cancelled" instead, not left dangling.
    """
    headers = _auth_header(user_id="user-1")
    client.post("/chat", json={"claim": "claim one", "thread_id": _THREAD_ID}, headers=headers)

    with patch("main.resolve_escalation") as mock_resolve:
        response = client.delete(f"/conversations/{_THREAD_ID}", headers=headers)

    assert response.status_code == 204
    mock_resolve.assert_called_once_with(ANY, _THREAD_ID, "cancelled")


def test_delete_conversation_succeeds_even_if_cancelling_escalation_fails(
    client: TestClient,
) -> None:
    """The conversation delete itself is the real, must-succeed operation;
    cancelling a pending escalation is secondary bookkeeping. A failure
    there shouldn't turn an otherwise-successful delete into a 500 — same
    fail-soft convention used throughout this codebase's secondary
    bookkeeping (e.g. /chat's conversation-row and escalation-row writes).
    """
    headers = _auth_header(user_id="user-1")
    client.post("/chat", json={"claim": "claim one", "thread_id": _THREAD_ID}, headers=headers)

    with patch("main.resolve_escalation", side_effect=Exception("db hiccup")):
        response = client.delete(f"/conversations/{_THREAD_ID}", headers=headers)

    assert response.status_code == 204


# --- retention purge on startup ---------------------------------------------------


def test_lifespan_purges_stale_conversations_and_their_checkpoints() -> None:
    """On startup, conversations past the retention window should be purged
    from both the sidebar table and the checkpointer — deleting only one
    of the two would either leave an orphaned sidebar entry pointing at
    nothing, or checkpoint data with no way to find/resume it.

    Uses a real InMemorySaver rather than a bare MagicMock for the
    checkpointer — LangGraph's builder.compile() validates that the
    checkpointer it's given is an actual BaseCheckpointSaver instance and
    raises TypeError otherwise, so a fully-mocked checkpointer can't be
    passed through build_graph() at all. Only delete_thread() itself is
    replaced with a spy (wraps the real method), which is enough to
    assert on its call arguments without breaking that isinstance check.
    """
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Final answer.")
    real_checkpointer = InMemorySaver()
    original_delete_thread = real_checkpointer.delete_thread

    with patch(
        "agent.orchestrator.init_chat_model", return_value=mock_model
    ), patch("agent.guardrail._get_intent_model"), patch(
        "main.build_connection_pool"
    ) as mock_build_pool, patch(
        "main.build_checkpointer", return_value=real_checkpointer
    ), patch("main.setup_users_table"), patch("main.setup_conversations_table"), patch(
        "main.setup_escalations_table"
    ), patch(
        "main.purge_stale_conversations",
        return_value=["stale-thread-1", "stale-thread-2"],
    ) as mock_purge, patch.object(
        real_checkpointer, "delete_thread", wraps=original_delete_thread
    ) as mock_delete_thread:
        mock_build_pool.return_value.__enter__.return_value = MagicMock()
        mock_build_pool.return_value.__exit__.return_value = False

        with TestClient(main.app):
            pass

    mock_purge.assert_called_once_with(ANY, main.CONVERSATION_RETENTION_DAYS)
    assert mock_delete_thread.call_args_list == [
        call("stale-thread-1"),
        call("stale-thread-2"),
    ]


# --- /chat's pending-review guard -----------------------------------------------


def test_chat_rejects_new_claim_on_thread_with_pending_review(client: TestClient) -> None:
    """A thread with a still-pending escalation has a paused interrupt in
    the graph (agent/orchestrator.py's human_review_node) — running a
    fresh claim on it would silently abandon that pause rather than error
    (confirmed empirically that a plain invoke on an interrupted thread
    just starts a new turn instead of raising), which would later make the
    admin's approve/reject on the orphaned escalation a silent no-op. This
    is rejected up front instead.
    """
    with patch("main._get_thread_owner", return_value="user-1"), patch(
        "main.get_pending_escalation", return_value={"status": "pending"}
    ):
        response = client.post(
            "/chat",
            json={"claim": "Another claim", "thread_id": _THREAD_ID},
            headers=_auth_header(user_id="user-1"),
        )
    assert response.status_code == 409


def test_chat_proceeds_normally_when_no_pending_review(client: TestClient) -> None:
    with patch("main._get_thread_owner", return_value="user-1"), patch(
        "main.get_pending_escalation", return_value=None
    ):
        response = client.post(
            "/chat",
            json={"claim": "Another claim", "thread_id": _THREAD_ID},
            headers=_auth_header(user_id="user-1"),
        )
    assert response.status_code == 200


def test_chat_skips_pending_review_check_for_a_brand_new_thread(client: TestClient) -> None:
    """A brand-new thread (no thread_id sent at all) can't have a prior
    escalation — the guard should be skipped entirely, not just pass
    because get_pending_escalation happens to return None.
    """
    with patch("main.get_pending_escalation") as mock_get_pending:
        response = client.post(
            "/chat", json={"claim": "A brand new claim"}, headers=_auth_header(user_id="user-1")
        )
    assert response.status_code == 200
    mock_get_pending.assert_not_called()


# --- /admin/escalations --------------------------------------------------------

_ADMIN_EMAIL = "admin@example.com"


def test_get_escalations_rejects_non_admin_account(client: TestClient) -> None:
    """A regular signed-in user — not just an anonymous request — must still
    be rejected. Thread ownership isn't the gate here; require_admin is.
    """
    with patch("main.ADMIN_EMAIL", _ADMIN_EMAIL):
        response = client.get(
            "/admin/escalations", headers=_auth_header(email="regular@example.com")
        )
    assert response.status_code == 403


def test_get_escalations_rejects_everyone_when_admin_email_unset(client: TestClient) -> None:
    """Fails closed: an unset ADMIN_EMAIL means no account can reach this
    endpoint, not that anyone can — see main.py's require_admin.
    """
    with patch("main.ADMIN_EMAIL", None):
        response = client.get(
            "/admin/escalations", headers=_auth_header(email=_ADMIN_EMAIL)
        )
    assert response.status_code == 403


def test_get_escalations_returns_pending_list_for_admin(client: TestClient) -> None:
    pending = [
        {
            "thread_id": "11111111-1111-1111-1111-111111111111",
            "claim": "The sky is green.",
            "verdict": "The evidence is unclear.",
            "reason": "verdict_incomplete",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    with patch("main.ADMIN_EMAIL", _ADMIN_EMAIL), patch(
        "main.list_pending_escalations", return_value=pending
    ):
        response = client.get(
            "/admin/escalations", headers=_auth_header(email=_ADMIN_EMAIL)
        )
    assert response.status_code == 200
    assert response.json() == pending


def test_resolve_escalation_rejects_non_admin_account(client: TestClient) -> None:
    with patch("main.ADMIN_EMAIL", _ADMIN_EMAIL):
        response = client.post(
            f"/admin/escalations/{_THREAD_ID}/resolve",
            json={"decision": "approve"},
            headers=_auth_header(email="regular@example.com"),
        )
    assert response.status_code == 403


def test_resolve_escalation_rejects_invalid_uuid(client: TestClient) -> None:
    with patch("main.ADMIN_EMAIL", _ADMIN_EMAIL):
        response = client.post(
            "/admin/escalations/not-a-uuid/resolve",
            json={"decision": "approve"},
            headers=_auth_header(email=_ADMIN_EMAIL),
        )
    assert response.status_code == 400


def test_resolve_escalation_returns_404_when_nothing_pending(client: TestClient) -> None:
    with patch("main.ADMIN_EMAIL", _ADMIN_EMAIL), patch(
        "main.get_pending_escalation", return_value=None
    ):
        response = client.post(
            f"/admin/escalations/{_THREAD_ID}/resolve",
            json={"decision": "approve"},
            headers=_auth_header(email=_ADMIN_EMAIL),
        )
    assert response.status_code == 404


def test_resolve_escalation_resumes_graph_and_marks_resolved(client: TestClient) -> None:
    escalation = {
        "thread_id": _THREAD_ID,
        "claim": "The sky is green.",
        "verdict": "The evidence is unclear.",
        "reason": "verdict_incomplete",
        "status": "pending",
    }
    with patch("main.ADMIN_EMAIL", _ADMIN_EMAIL), patch(
        "main.get_pending_escalation", return_value=escalation
    ), patch("main.resume_review") as mock_resume, patch(
        "main.resolve_escalation"
    ) as mock_resolve:
        response = client.post(
            f"/admin/escalations/{_THREAD_ID}/resolve",
            json={"decision": "approve"},
            headers=_auth_header(email=_ADMIN_EMAIL),
        )
    assert response.status_code == 204
    mock_resume.assert_called_once_with(ANY, _THREAD_ID, "approve")
    mock_resolve.assert_called_once_with(ANY, _THREAD_ID, "approve")


def test_resolve_escalation_succeeds_even_if_marking_resolved_fails(
    client: TestClient,
) -> None:
    """resume_review() is the real side effect (it actually unblocks the
    paused graph); resolve_escalation() is secondary bookkeeping. A failure
    in the bookkeeping step, after the graph was already successfully
    resumed, shouldn't turn into a 500 for the admin — same fail-soft
    convention as /chat's conversation/escalation bookkeeping.
    """
    escalation = {
        "thread_id": _THREAD_ID,
        "claim": "The sky is green.",
        "verdict": "The evidence is unclear.",
        "reason": "verdict_incomplete",
        "status": "pending",
    }
    with patch("main.ADMIN_EMAIL", _ADMIN_EMAIL), patch(
        "main.get_pending_escalation", return_value=escalation
    ), patch("main.resume_review") as mock_resume, patch(
        "main.resolve_escalation", side_effect=Exception("db hiccup")
    ):
        response = client.post(
            f"/admin/escalations/{_THREAD_ID}/resolve",
            json={"decision": "approve"},
            headers=_auth_header(email=_ADMIN_EMAIL),
        )
    assert response.status_code == 204
    mock_resume.assert_called_once_with(ANY, _THREAD_ID, "approve")


def test_resolve_escalation_rejects_invalid_decision_value(client: TestClient) -> None:
    """decision is a Literal["approve", "reject"] on ResolveEscalationRequest
    — anything else should fail Pydantic validation (422), not silently
    fall through to resume_review with an unexpected value.
    """
    with patch("main.ADMIN_EMAIL", _ADMIN_EMAIL):
        response = client.post(
            f"/admin/escalations/{_THREAD_ID}/resolve",
            json={"decision": "maybe"},
            headers=_auth_header(email=_ADMIN_EMAIL),
        )
    assert response.status_code == 422


# --- /chat's escalation bookkeeping ---------------------------------------------


def test_chat_records_escalation_when_verdict_incomplete(client: TestClient) -> None:
    """When a turn's verdict is judged incomplete (agent/verdict_completeness.py),
    chat() should record it in the escalation review queue so the admin
    account can see and resolve it later — see main.py's chat() handler and
    agent/orchestrator.py's human_review_node.

    Patches agent.orchestrator._check_and_store_verdict directly rather
    than driving a real multi-step tool loop through the mocked model —
    the fixture's default mock always returns a tool_call-free final
    answer on the very first call regardless of tool_choice binding, so
    there's no way to make it naturally call vector_lookup_tool first;
    forcing the completeness result directly is the more precise, more
    maintainable way to reach this branch.
    """
    with patch("agent.orchestrator._check_and_store_verdict", return_value=False), patch(
        "main.create_escalation"
    ) as mock_create_escalation:
        response = client.post(
            "/chat", json={"claim": "Is the sky green?"}, headers=_auth_header()
        )
    assert response.status_code == 200
    thread_id = response.json()["thread_id"]

    mock_create_escalation.assert_called_once()
    args, kwargs = mock_create_escalation.call_args
    assert args[1] == thread_id
    assert kwargs["claim"] == "Is the sky green?"
    assert kwargs["verdict"] == "Final answer."
    assert kwargs["reason"] == "verdict_incomplete"


def test_chat_does_not_record_escalation_when_verdict_complete(client: TestClient) -> None:
    with patch("agent.orchestrator._check_and_store_verdict", return_value=True), patch(
        "main.create_escalation"
    ) as mock_create_escalation:
        response = client.post(
            "/chat", json={"claim": "Is the sky green?"}, headers=_auth_header()
        )
    assert response.status_code == 200
    mock_create_escalation.assert_not_called()


def test_chat_never_stores_raw_pii_in_an_escalation_row(client: TestClient) -> None:
    """Regression test, same shape as test_graph.py's
    test_run_claim_never_checkpoints_raw_pii and
    test_run_claim_returns_the_scrubbed_claim_not_the_raw_input — the
    escalations table (agent/escalations_db.py) is a newer, separate place
    PII could in principle leak into if it ever read from raw request text
    instead of the already-scrubbed claim, the same class of bug as the
    earlier conversation-title leak. It doesn't (create_escalation() is
    called with result["pending_review"]["claim"], which is read from
    state["messages"] after scrub_pii() already ran in graph.run_claim()),
    but that's worth proving with a real assertion, not just reasoning
    about the code.
    """
    with patch("agent.orchestrator._check_and_store_verdict", return_value=False), patch(
        "main.create_escalation"
    ) as mock_create_escalation:
        response = client.post(
            "/chat",
            json={"claim": "Email me at jane@example.com about this claim."},
            headers=_auth_header(),
        )
    assert response.status_code == 200

    mock_create_escalation.assert_called_once()
    _, kwargs = mock_create_escalation.call_args
    assert "jane@example.com" not in kwargs["claim"]
    assert "[REDACTED_EMAIL]" in kwargs["claim"]
