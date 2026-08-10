# Misinformation Agent

A LangGraph-based fact-checking agent that takes a claim (or article link) and produces a sourced verdict with a credibility score — built for ordinary social media users who want a quick way to check a claim before believing or resharing it, not for professional fact-checkers.

## Status

Early build in progress. Currently implemented:

- Single-agent orchestrator (LangGraph `StateGraph`) with a tool-calling loop
- `fact_check_lookup_tool` (Google Fact Check Tools API) — checks whether a professional fact-checker has already ruled on the claim
- `web_search_tool` (Tavily) — general web search for evidence when no existing ruling is found
- `source_retrieval_tool` (Tavily Extract) — fetches full article text from a specific URL
- `credibility_scoring_tool` — judges source reliability and produces a confidence score when there's no clean existing ruling to rely on
- `vector_lookup_tool` — semantic claim cache (Pinecone + local embeddings) that reuses a prior verdict when a claim is a close rewording of one already checked, instead of re-running the full pipeline. Two-stage retrieval: Pinecone's cosine similarity does a cheap first pass over `VECTOR_SEARCH_TOP_K` candidates, then a CrossEncoder reranker (`BAAI/bge-reranker-base` by default) re-scores that smaller set more accurately — mitigates the "curse of dimensionality," where the single nearest neighbor by raw cosine similarity isn't always the true best match in a high-dimensional embedding space. A match is only reused if the reranked score clears `RERANK_MIN_SCORE` and — when more than one candidate exists — beats the runner-up by at least `RERANK_MIN_MARGIN`, replacing the old flat `CLAIM_SIMILARITY_THRESHOLD` cutoff. The Pinecone query also filters out anything older than `CLAIM_CACHE_MAX_AGE_DAYS` via a `checked_at_ts` metadata filter, so a stale verdict on a fast-moving topic can't be reused indefinitely
- Deterministic routing: cache check first, then fact-check database, web search as fallback, credibility scoring forced whenever the evidence gathered doesn't already amount to a single clean True/False ruling
- First-turn tool use forced (`tool_choice="any"`) so the agent always checks the cache and gathers evidence before answering
- A 16-step recursion limit on the tool-calling loop, with a graceful partial-progress fallback instead of a crash
- Postgres-backed persistence (Neon) via `agent/checkpointer.py` — conversation state survives restarts, keyed by thread_id. The checkpointer is injected into `build_orchestrator()` rather than hardcoded, so tests still use `InMemorySaver`. Backed by a shared `psycopg_pool.ConnectionPool` (also used by `agent/users_db.py`) rather than a single held-open connection — Neon's free tier kills idle connections when it auto-suspends, and a pool detects and replaces them instead of every later query failing until the app is restarted
- `graph.py` — shared graph-building and invocation logic, used by both the FastAPI app and the CLI script
- `agent/tracing.py` — optional Langfuse tracing for the orchestrator graph. Lazily builds a callback handler on first use (same pattern as `agent/guardrail.py`'s intent model and `vector_lookup_tool.py`'s embedding model), and returns `None` if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` aren't set — tracing is opt-in observability, never a hard dependency. Wired into every `run_claim()` call in `graph.py`; each trace is tagged with `langfuse_session_id` set to the thread_id, so a whole conversation's turns group into one session in the Langfuse dashboard instead of showing up as disconnected calls
- `main.py` — FastAPI app wrapping the graph, with `/chat`, `/auth/signup`, `/auth/login`, and a `/health` check. Checkpointer is opened once at startup via `lifespan`, not per request. `/chat` requires a `Bearer` JWT (via `Authorization` header) identifying the requesting user — `/auth/signup` and `/auth/login` are necessarily open, and `/health` stays open for uptime monitors
- `agent/auth.py` — password hashing (argon2id) and JWT session-token creation/validation for multi-user authentication
- `agent/users_db.py` — Postgres-backed user storage (a `users` table, separate from LangGraph's own checkpoint tables), used by the `/auth/*` endpoints
- Per-user thread ownership scoping — each `/chat` call records the requesting user's id in the thread's own checkpoint metadata; continuing an existing `thread_id` that belongs to a different user is rejected (403) instead of silently resuming their conversation
- `agent/conversations_db.py` — a thin `conversations` table (title + recency only, not a second copy of message content) backing a sidebar conversation list. `/chat` records a new row on a thread's first message and bumps `updated_at` on later ones; `GET /conversations` lists a user's threads most-recently-active first, `GET /conversations/{thread_id}/messages` returns a thread's real history (read back from the checkpointer, filtered down to user turns and final answers) so resuming a conversation shows its actual content rather than an empty chat. Basic scope only — list and resume to latest, no branching/forking of a conversation from an earlier point
- `app.py` — Streamlit chat UI with a login/signup gate, a sidebar conversation list (click to resume, active conversation marked), and a sidebar logout button, calling the FastAPI backend over HTTP with the signed-in user's bearer token. Note: a browser refresh also currently drops the session (the token lives only in `st.session_state`, not in a cookie or local storage), so it behaves like a second logout path today, not something guarded against
- `agent/guardrail.py` — classifies incoming messages as greeting/claim/out-of-scope before the main pipeline runs, using a keyword fast-path for obvious greetings and one dedicated structured-output call for everything else. Wired into `agent/orchestrator.py` as a `"guardrail"` node ahead of the tool-calling loop, so greetings and off-topic messages short-circuit to a canned reply and never trigger the forced tool pipeline
- `agent/orchestrator_routing.py` — pure routing/decision helpers, split out of `orchestrator.py` so deterministic decision logic is easy to test in isolation
- `agent/orchestrator_responses.py` — response-construction and cache-writing helpers (cache-hit responses, source gathering for scoring, verdict storage), also split out of `orchestrator.py`
- `agent/summarizer.py` — compresses older, fully-resolved turns into a running summary once a thread's history crosses `MAX_MESSAGES_BEFORE_SUMMARY`, so long-running threads don't exceed Gemini's context window. Never touches the turn currently in progress. Wired in as a `"summarize"` graph node, reached via `route_after_guardrail` once older history crosses the threshold — runs after the guardrail check (so greetings/off-topic messages never trigger it) and before the orchestrator. Model is lazy-initialized, same pattern as `vector_lookup_tool.py`
- Tests for all five tools, the orchestrator's routing logic, the response-construction helpers, the guardrail classification and its graph wiring, the checkpointer, the shared graph helpers, the summarizer, and the FastAPI endpoints

Planned next: human-in-the-loop review, Docker Compose deployment.

## Architecture

- **Orchestrator** — a single LangGraph agent, not a multi-agent supervisor setup. One model, multiple tools underneath.
- **Tools** — each tool returns a structured dict, not raw text, so the orchestrator can reason over results reliably.
- **Persistence** — Postgres (Neon), via `agent/checkpointer.py`. `build_orchestrator()` takes the checkpointer as an injected argument rather than constructing one itself, so tests can pass `InMemorySaver` without touching a real database. A single shared connection pool (`build_connection_pool()`) backs the checkpointer, `agent/users_db.py`, and `agent/conversations_db.py`, opened once in `main.py`'s `lifespan` and closed on shutdown. `conversations` is a thin index table for the sidebar (title + recency) — it has no enforced foreign key into LangGraph's own checkpoint tables (those are managed entirely by `langgraph-checkpoint-postgres`'s own migrations, not ours), only a real FK into `users`.
- **Serving layer** — `graph.py` holds the shared build+invoke logic; `main.py` (FastAPI) and `cli.py` both use it directly; `app.py` (Streamlit) talks to `main.py` over HTTP instead of importing the graph logic directly.
- **Context management** — `agent/summarizer.py`, wired in as a `"summarize"` graph node reached via `route_after_guardrail` once older history crosses `MAX_MESSAGES_BEFORE_SUMMARY`. Self-limiting: once it runs, older-message count drops back below threshold until enough new messages accumulate again.
- **Observability** — `agent/tracing.py` (Langfuse), attached as a LangGraph callback in `graph.py`'s `run_claim()`. Opt-in — the app runs identically whether or not it's configured — and each conversation's traces are grouped into one Langfuse session via `langfuse_session_id` = thread_id.

## Setup

1. Install Miniconda from anaconda.com/download, then create and activate the environment:

```
conda create -n misinformation-agent python=3.13
conda activate misinformation-agent
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Set up a Pinecone index for the semantic claim cache:

   - Sign up at pinecone.io and create a serverless index named to match `PINECONE_INDEX_NAME` (default `misinformation-agent-claims`)
   - Set dimensions to `768` and metric to `cosine` — this must match the output of the `BAAI/bge-base-en-v1.5` embedding model used by `vector_lookup_tool.py`, which normalizes its embeddings for cosine comparison

4. Set up a Neon Postgres database:

   - Sign up at neon.com and create a project (free tier is enough for development)
   - From the project dashboard, click **Connect** and copy the **direct** connection string (the hostname should *not* contain `-pooler` — the checkpointer needs session-level Postgres features a transaction pooler can break)

5. Copy `.env.example` to `.env` and fill in your API keys:

```
GOOGLE_API_KEY=
TAVILY_API_KEY=
GOOGLE_FACT_CHECK_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
VECTOR_SEARCH_TOP_K=5
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
RERANK_MIN_SCORE=0.5
RERANK_MIN_MARGIN=0.15
CLAIM_CACHE_MAX_AGE_DAYS=180
MODEL_NAME=gemini-3.1-flash-lite
POSTGRES_CONNECTION_STRING=
JWT_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

Generate a real value for `JWT_SECRET_KEY` — don't leave it as the placeholder:

```
python -c "import secrets; print(secrets.token_hex(32))"
```

`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are optional — sign up at langfuse.com and create a project if you want tracing. Leave both blank to run without it; `agent/tracing.py` falls back to no-op tracing automatically rather than failing.

6. Run a test claim through the agent directly (no server needed):

```
python cli.py
```

7. Or run the full FastAPI + Streamlit stack (two terminals):

```
# Terminal 1
uvicorn main:app --reload --port 8000

# Terminal 2
streamlit run app.py
```

## Testing

```
python -m pytest
python -m ruff check .
```

## Tech stack

- LangGraph / LangChain — agent orchestration
- Google Gemini — LLM
- Tavily — web search and article extraction
- Google Fact Check Tools API — existing fact-check lookups
- Pinecone — vector database for the semantic claim cache
- sentence-transformers — local embedding model (BAAI/bge-base-en-v1.5) and CrossEncoder reranker (BAAI/bge-reranker-base) for the claim cache's two-stage retrieval
- Postgres (Neon) via `langgraph-checkpoint-postgres` / `psycopg` — conversation state persistence
- argon2-cffi — password hashing (argon2id) for multi-user authentication
- PyJWT — signed session tokens for multi-user authentication
- Langfuse — optional LLM/agent tracing and observability
- FastAPI / uvicorn — HTTP API layer
- Streamlit — chat UI
- pytest, ruff — testing and linting

## Project structure

```
graph.py                       # shared graph build + invoke logic
main.py                        # FastAPI app wrapping the graph
app.py                         # Streamlit chat UI, calls main.py over HTTP
cli.py                         # manual CLI entry point for one-off testing
agent/
  config.py                       # env var loading, logging setup
  checkpointer.py                 # Postgres (Neon) checkpointer factory
  tracing.py                      # optional Langfuse tracing for the orchestrator graph
  auth.py                         # password hashing (argon2id) + JWT session tokens
  users_db.py                     # Postgres-backed user storage for authentication
  conversations_db.py             # Postgres-backed conversation listing for the sidebar
  orchestrator.py                 # LangGraph StateGraph: graph wiring, guardrail node, tool-calling loop
  orchestrator_routing.py         # pure routing/decision helpers
  orchestrator_responses.py       # response-construction and cache-writing helpers
  guardrail.py                    # message intent classification (greeting/claim/out-of-scope)
  summarizer.py                   # conversation summarization for long-running threads
  tools/
    _clients.py                    # shared third-party API clients
    credibility_scoring_tool.py    # Gemini-backed source-reliability judgment
    fact_check_tool.py             # Google Fact Check Tools API lookup
    source_retrieval_tool.py       # Tavily Extract-backed full-article retrieval
    vector_lookup_tool.py          # Pinecone-backed semantic claim cache
    web_search_tool.py             # Tavily-backed web search tool
tests/
  test_auth.py
  test_checkpointer.py
  test_credibility_scoring_tool.py
  test_fact_check_tool.py
  test_graph.py
  test_guardrail.py
  test_guardrail_routing.py
  test_main.py
  test_orchestrator_build.py
  test_orchestrator_responses.py
  test_orchestrator_routing.py
  test_source_retrieval_tool.py
  test_summarizer.py
  test_vector_lookup_tool.py
  test_web_search_tool.py
```
