"""Streamlit chat UI for the misinformation agent — talks to the FastAPI backend over HTTP."""

import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 120

st.set_page_config(
    page_title="Misinformation Agent",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
)

API_ACCESS_KEY = os.getenv("API_ACCESS_KEY")
if not API_ACCESS_KEY:
    st.error(
        "This app is not configured correctly (missing API_ACCESS_KEY) and "
        "cannot reach the backend right now."
    )
    st.stop()

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
                    headers={"X-API-Key": API_ACCESS_KEY},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
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
