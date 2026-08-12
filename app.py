"""Streamlit chat UI for the misinformation agent — talks to the FastAPI backend over HTTP."""

import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Defaults to localhost for local (non-Docker) runs. Inside docker-compose,
# containers can't reach each other via "localhost" — the ui container needs
# the api container's *service name* instead — so docker-compose.yml
# overrides this via the API_URL environment variable, pointing it at
# http://api:8000.
API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 120

# Mirrors main.py's ADMIN_EMAIL (agent/config.py) — compared against the
# signed-in user's own email (set on login/signup below) to decide whether
# to show the Pending Reviews section at all. This is purely a UI-level
# convenience: the real access control is main.py's require_admin, which
# checks the same value server-side on every /admin/* call regardless of
# what this UI shows or hides.
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

st.set_page_config(
    page_title="Misinformation Agent",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Dark "investigation desk" palette, plain CSS injected via st.markdown — no
# extra dependencies (streamlit-extras, etc.) required.
_THEME_CSS = """
<style>
:root {
    --bg-deep: #0b1120;
    --bg-panel: #141b2d;
    --bg-card: #1b2338;
    --border: #2a3350;
    --accent: #2dd4bf;
    --accent-soft: rgba(45, 212, 191, 0.12);
    --warn: #f59e0b;
    --warn-soft: rgba(245, 158, 11, 0.12);
    --danger: #f87171;
    --danger-soft: rgba(248, 113, 113, 0.12);
    --text-main: #e2e8f0;
    --text-dim: #8b98b8;
}

.stApp {
    background: radial-gradient(circle at 20% 0%, #101935 0%, var(--bg-deep) 55%);
    color: var(--text-main);
}

section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
}

.agent-header {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.4rem;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-panel) 100%);
    border: 1px solid var(--border);
}
.agent-header .icon {
    font-size: 2rem;
    line-height: 1;
    filter: drop-shadow(0 0 8px rgba(45, 212, 191, 0.45));
}
.agent-header .titles h1 {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-main);
}
.agent-header .titles p {
    margin: 0.15rem 0 0 0;
    font-size: 0.85rem;
    color: var(--text-dim);
}

.sidebar-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 0.9rem;
    margin-bottom: 0.9rem;
    font-size: 0.85rem;
    color: var(--text-dim);
}
.sidebar-card .label {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.68rem;
    color: var(--accent);
    margin-bottom: 0.25rem;
    font-weight: 600;
}
.sidebar-card code {
    color: var(--text-main);
    background: rgba(255, 255, 255, 0.05);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
}

div[data-testid="stChatMessage"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.6rem;
}

.verdict-banner {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.verdict-banner.ok {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(45, 212, 191, 0.35);
}
.verdict-banner.partial {
    background: var(--warn-soft);
    color: var(--warn);
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.verdict-banner.error {
    background: var(--danger-soft);
    color: var(--danger);
    border: 1px solid rgba(248, 113, 113, 0.35);
}

footer, header[data-testid="stHeader"] {
    background: transparent;
}
</style>
"""

st.markdown(_THEME_CSS, unsafe_allow_html=True)


def _auth_request(endpoint: str, email: str, password: str) -> str | None:
    """Call /auth/signup or /auth/login and return the access token, or None on failure.

    Renders its own st.error on any failure (connection, timeout, or a 4xx
    from the backend) so callers just check the return value instead of
    handling three different error shapes themselves.
    """
    try:
        response = requests.post(
            f"{API_URL}/auth/{endpoint}",
            json={"email": email, "password": password},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException:
        st.error("Couldn't reach the backend — is the API running?")
        return None
    if response.status_code >= 400:
        st.error(response.json().get("detail", "Something went wrong."))
        return None
    return response.json()["access_token"]


def _apply_refreshed_token(response: requests.Response) -> None:
    """Pick up a sliding-session refresh token if the backend issued one.

    main.py's get_current_user() reissues a fresh, short-lived token on
    every successful authenticated call (see JWT_EXPIRY_MINUTES) instead of
    handing out one long-lived one — this is the client-side half of that:
    whenever a response carries a new token, swap it into session state so
    the *next* request uses the fresh one instead of the older, closer-to-
    expiring one. Without this, sessions would hard-expire after
    JWT_EXPIRY_MINUTES regardless of how active the user actually was.
    """
    new_token = response.headers.get("X-New-Token")
    if new_token:
        st.session_state.access_token = new_token


def _fetch_conversations() -> list[dict]:
    """Fetch the signed-in user's conversation list for the sidebar.

    Fails soft — an unreachable backend just means an empty sidebar list
    for this render, not a crash, since this runs on every rerun.
    """
    try:
        response = requests.get(
            f"{API_URL}/conversations",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        _apply_refreshed_token(response)
        return response.json()
    except requests.exceptions.RequestException:
        return []


def _load_conversation(thread_id: str) -> None:
    """Switch the active thread and pull its real message history from the backend.

    Without this, clicking a past conversation would just swap thread_id
    and show an empty chat window — the backend has the full history, but
    st.session_state.messages is purely client-side and has no memory of
    it until fetched explicitly.
    """
    try:
        response = requests.get(
            f"{API_URL}/conversations/{thread_id}/messages",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        _apply_refreshed_token(response)
        data = response.json()
        st.session_state.thread_id = thread_id
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]} for m in data["messages"]
        ]
    except requests.exceptions.RequestException:
        st.error("Couldn't load that conversation — is the API running?")


def _delete_conversation(thread_id: str) -> bool:
    """Delete a conversation via the backend. Returns True on success.

    Renders its own st.error on failure, same convention as the other
    HTTP helpers above (_auth_request, _fetch_conversations,
    _load_conversation) — callers just check the return value instead of
    handling the request/response themselves.
    """
    try:
        response = requests.delete(
            f"{API_URL}/conversations/{thread_id}",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        _apply_refreshed_token(response)
        return True
    except requests.exceptions.RequestException:
        st.error("Couldn't delete that conversation — is the API running?")
        return False


def _fetch_pending_escalations() -> list[dict]:
    """Fetch the admin-only review queue. Fails soft, same convention as
    _fetch_conversations — an unreachable backend or a non-admin account
    (403) just means an empty list for this render, not a crash.
    """
    try:
        response = requests.get(
            f"{API_URL}/admin/escalations",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        _apply_refreshed_token(response)
        return response.json()
    except requests.exceptions.RequestException:
        return []


def _resolve_escalation(thread_id: str, decision: str) -> bool:
    """Submit an approve/reject decision for a pending escalation. Returns True on success."""
    try:
        response = requests.post(
            f"{API_URL}/admin/escalations/{thread_id}/resolve",
            json={"decision": decision},
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        _apply_refreshed_token(response)
        return True
    except requests.exceptions.RequestException:
        st.error("Couldn't resolve that escalation — is the API running?")
        return False


if "access_token" not in st.session_state:
    st.session_state.access_token = None

if st.session_state.access_token is None:
    st.markdown("## 🔎 Misinformation Agent — Sign in")
    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in") and email and password:
                token = _auth_request("login", email, password)
                if token:
                    st.session_state.access_token = token
                    # Reset per-session state on every (re)login — otherwise a
                    # second person logging in on the same browser tab would
                    # inherit the previous user's thread_id (tripping the
                    # backend's ownership check) or see their leftover chat
                    # bubbles.
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    # Only used to decide whether to show the Pending
                    # Reviews section — see ADMIN_EMAIL above. Not a
                    # security boundary on its own; main.py's require_admin
                    # re-checks this server-side on every /admin/* call.
                    st.session_state.user_email = email
                    st.rerun()

    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input(
                "Password (min 8 characters)", type="password", key="signup_password"
            )
            if st.form_submit_button("Sign up") and email and password:
                token = _auth_request("signup", email, password)
                if token:
                    st.session_state.access_token = token
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.session_state.user_email = email
                    st.rerun()

    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 🔎 Session")

    short_id = st.session_state.thread_id.split("-")[0]
    st.markdown(
        f"""<div class="sidebar-card">
            <div class="label">Thread ID</div>
            <code>{short_id}…</code>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="sidebar-card">
            <div class="label">Claims checked</div>
            {len([m for m in st.session_state.messages if m["role"] == "user"])}
        </div>""",
        unsafe_allow_html=True,
    )

    if st.button("🗑️  New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    conversations = _fetch_conversations()
    if conversations:
        st.markdown("### Conversations")
        for convo in conversations:
            is_active = convo["thread_id"] == st.session_state.thread_id
            label = ("→ " if is_active else "") + convo["title"]
            row_col, delete_col = st.columns([5, 1])
            with row_col:
                if st.button(
                    label, key=f"convo_{convo['thread_id']}", use_container_width=True
                ):
                    _load_conversation(convo["thread_id"])
                    st.rerun()
            with delete_col:
                if st.button(
                    "🗑️", key=f"delete_{convo['thread_id']}", use_container_width=True
                ):
                    if _delete_conversation(convo["thread_id"]):
                        # If the conversation being deleted is the one
                        # currently open, reset to a fresh session — same
                        # as clicking "New conversation" — so the chat
                        # window doesn't keep showing messages for a
                        # thread that no longer exists on the backend.
                        if is_active:
                            st.session_state.thread_id = str(uuid.uuid4())
                            st.session_state.messages = []
                        st.rerun()

    st.divider()

    if st.button("🚪  Log out", use_container_width=True):
        # Clearing access_token drops back to the login gate on rerun.
        # thread_id/messages are also reset so that if the same or a
        # different person logs back in on this tab, they start clean
        # instead of inheriting this session's leftover state — same
        # reasoning as the reset already done on login/signup above.
        st.session_state.access_token = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.user_email = None
        st.rerun()

    if ADMIN_EMAIL and st.session_state.get("user_email") == ADMIN_EMAIL:
        st.divider()
        pending = _fetch_pending_escalations()
        st.markdown(f"### 🛡️ Pending Reviews ({len(pending)})")
        for item in pending:
            preview = item["claim"][:60] + ("…" if len(item["claim"]) > 60 else "")
            with st.expander(preview):
                st.markdown(f"**Verdict:** {item['verdict']}")
                st.markdown(f"**Flagged because:** {item['reason']}")
                approve_col, reject_col = st.columns(2)
                with approve_col:
                    if st.button(
                        "✅ Approve",
                        key=f"approve_{item['thread_id']}",
                        use_container_width=True,
                    ):
                        if _resolve_escalation(item["thread_id"], "approve"):
                            st.rerun()
                with reject_col:
                    if st.button(
                        "❌ Reject",
                        key=f"reject_{item['thread_id']}",
                        use_container_width=True,
                    ):
                        if _resolve_escalation(item["thread_id"], "reject"):
                            st.rerun()

    with st.expander("How this works"):
        st.markdown(
            "Every claim runs through a fact-check cache lookup, then a "
            "search/fact-check API pass, then credibility scoring — with a "
            "retry loop if evidence is thin or conflicting. Sources are "
            "cited in the final answer."
        )

st.markdown(
    """<div class="agent-header">
        <div class="icon">🔎</div>
        <div class="titles">
            <h1>Misinformation Agent</h1>
            <p>Ask about a claim — it'll check fact-check databases, search
            the web, and score source credibility.</p>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

_AVATARS = {"user": "🧑", "assistant": "🔎"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=_AVATARS.get(message["role"])):
        if message["role"] == "assistant" and message.get("banner"):
            banner_class, banner_text = message["banner"]
            st.markdown(
                f'<span class="verdict-banner {banner_class}">{banner_text}</span>',
                unsafe_allow_html=True,
            )
        st.write(message["content"])

claim = st.chat_input("Enter a claim to fact-check...")

_SPINNER_MESSAGES = [
    "Checking the cache for a prior verdict...",
    "Searching fact-check databases...",
    "Scoring source credibility...",
]

if claim:
    st.session_state.messages.append({"role": "user", "content": claim})
    with st.chat_message("user", avatar=_AVATARS["user"]):
        st.write(claim)

    with st.chat_message("assistant", avatar=_AVATARS["assistant"]):
        banner = None
        with st.spinner(_SPINNER_MESSAGES[0]):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"claim": claim, "thread_id": st.session_state.thread_id},
                    headers={"Authorization": f"Bearer {st.session_state.access_token}"},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                _apply_refreshed_token(response)
                data = response.json()
                answer = data["answer"]
                if data.get("recursion_limit_hit"):
                    banner = ("partial", "⚠️ Partial result — step limit reached")
                else:
                    banner = ("ok", "✅ Verdict reached")
            except requests.exceptions.ConnectionError:
                answer = "Couldn't reach the fact-checking service — is the API running?"
                banner = ("error", "⛔ Connection failed")
            except requests.exceptions.Timeout:
                answer = "The request took too long and timed out. Please try again."
                banner = ("error", "⛔ Timed out")
            except requests.exceptions.HTTPError:
                if response.status_code in (401, 403):
                    st.session_state.access_token = None
                    st.rerun()
                if response.status_code == 409:
                    # A claim rejected because this thread has a review
                    # still pending (main.py's /chat) — the backend's own
                    # detail message explains this clearly; a generic
                    # "please try again" would be actively misleading here,
                    # since retrying the same message won't help until the
                    # pending review is resolved.
                    answer = response.json().get("detail", "This conversation is on hold.")
                    banner = ("partial", "⏸️ Awaiting review")
                else:
                    answer = "The request failed — please try again."
                    banner = ("error", "⛔ Request failed")
            except requests.exceptions.RequestException as exc:
                answer = f"Something went wrong: {exc}"
                banner = ("error", "⛔ Request failed")

        if banner:
            st.markdown(
                f'<span class="verdict-banner {banner[0]}">{banner[1]}</span>',
                unsafe_allow_html=True,
            )
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "banner": banner}
    )
    st.rerun()
