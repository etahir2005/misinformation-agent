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
- `main.py` — FastAPI app wrapping the graph, with `/chat`, `/auth/signup`, `/auth/login`, `/conversations`, `/conversations/{thread_id}/messages`, `DELETE /conversations/{thread_id}`, and a `/health` check. Checkpointer is opened once at startup via `lifespan`, not per request. Every endpoint except `/auth/signup`, `/auth/login`, and `/health` requires a `Bearer` JWT (via `Authorization` header) identifying the requesting user
- `agent/auth.py` — password hashing (argon2id) and JWT session-token creation/validation for multi-user authentication. Sessions slide rather than have a single fixed expiry: `get_current_user()` reissues a fresh `JWT_EXPIRY_MINUTES`-minute token (default 30) on every authenticated request and returns it via an `X-New-Token` response header, so an active session keeps extending itself and only a genuinely idle one actually expires
- `agent/users_db.py` — Postgres-backed user storage (a `users` table, separate from LangGraph's own checkpoint tables), used by the `/auth/*` endpoints
- Per-user thread ownership scoping — each `/chat` call records the requesting user's id in the thread's own checkpoint metadata; continuing an existing `thread_id` that belongs to a different user is rejected (403) instead of silently resuming their conversation
- `agent/conversations_db.py` — a thin `conversations` table (title + recency only, not a second copy of message content) backing a sidebar conversation list. `/chat` records a new row on a thread's first message and bumps `updated_at` on later ones; `GET /conversations` lists a user's threads most-recently-active first, `GET /conversations/{thread_id}/messages` returns a thread's real history (read back from the checkpointer, filtered down to user turns and final answers) so resuming a conversation shows its actual content rather than an empty chat. `DELETE /conversations/{thread_id}` removes a conversation entirely — both its sidebar row (`delete_conversation()`) and its actual checkpoint data (via the checkpointer's own `delete_thread()`), ownership-checked the same way as the other endpoints. Conversations idle longer than `CONVERSATION_RETENTION_DAYS` (default 10) are also purged automatically on every app startup, via `purge_stale_conversations()` in `main.py`'s `lifespan`. Basic scope only — list, resume to latest, and delete whole conversations, no branching/forking of a conversation from an earlier point
- `app.py` — Streamlit chat UI with a login/signup gate, a sidebar conversation list (click to resume, active conversation marked, a 🗑️ button to delete), and a sidebar logout button, calling the FastAPI backend over HTTP with the signed-in user's bearer token. Note: a browser refresh also currently drops the session (the token lives only in `st.session_state`, not in a cookie or local storage), so it behaves like a second logout path today, not something guarded against
- `agent/guardrail.py` — classifies incoming messages as greeting/claim/out-of-scope before the main pipeline runs, using a keyword fast-path for obvious greetings and one dedicated structured-output call for everything else. Wired into `agent/orchestrator.py` as a `"guardrail"` node ahead of the tool-calling loop, so greetings and off-topic messages short-circuit to a canned reply and never trigger the forced tool pipeline
- PII scrubbing (`agent/orchestrator.py`) — redacts email, credit card, IP, and MAC addresses out of every claim before it's processed, using LangChain's built-in `PIIMiddleware` (regex/algorithmic detection, no LLM call). Two layers: `scrub_pii()` is the authoritative scrub, called from `graph.py`'s `run_claim()` on the raw claim string before it's ever passed to `graph.invoke()` — necessary because LangGraph checkpoints the exact `invoke()` payload before running any node, so redacting inside the graph alone isn't early enough to keep raw PII out of Postgres (confirmed empirically via `graph.get_state_history()`, not assumed from graph structure). `pii_scrub_node` is a defense-in-depth backstop wired as the graph's first node (`START -> pii_scrub -> guardrail -> ...`) — it keeps what Gemini/tools/Langfuse see within a turn clean, and would cover any future caller that invokes the graph directly instead of going through `run_claim()`. Deliberately excludes the `url` PII type — users legitimately submit article URLs for `source_retrieval_tool` to fetch, and redacting those before the model ever sees them would silently break that feature. `run_claim()` also returns the scrubbed claim text itself (`result["claim"]`) — callers that derive anything storage- or display-bound from the claim (e.g. `main.py`'s `derive_title()` for the sidebar conversation list) must use this, not their own original input, or PII redacted everywhere else still ends up stored in a conversation's title. Caught via a live Streamlit smoke test, not the (fully mocked) test suite — a reminder that mocked tests alone don't catch every real code path
- `agent/orchestrator_routing.py` — pure routing/decision helpers, split out of `orchestrator.py` so deterministic decision logic is easy to test in isolation
- `agent/orchestrator_responses.py` — response-construction and cache-writing helpers (cache-hit responses, source gathering for scoring, verdict storage), also split out of `orchestrator.py`
- `agent/verdict_completeness.py` — checks whether a claim's final verdict is complete enough to cache, gating `_check_and_store_verdict()` in `orchestrator_responses.py`: an incomplete verdict is never stored for future reuse, since caching a weak answer would multiply its harm across every future semantically-similar claim instead of containing it to one turn. Two layers, cheapest first: deterministic hard-fails (`VERDICT_MIN_WORD_COUNT`, no sources at all) catch the easy cases with zero LLM cost, since they can prove a verdict is *definitely* incomplete but can't prove one is complete on their own — a long, sourced answer can still be vague. Anything that clears those goes to an LLM judge (`JUDGE_MODEL_NAME`, default `gemini-3.5-flash-lite` — a separate model from `MODEL_NAME`, not the non-lite Flash tiers, which cap free-tier usage at 20 requests/day, nowhere near enough for consensus voting), called `JUDGE_CONSENSUS_CALLS` times (default 3) with a strict majority of *successful* calls required to agree "complete" — a single call would inherit the judge's own hallucination risk with no redundancy check. Judge-call errors fail closed (treated as "incomplete," never "complete"), and too many failed calls to trust any majority also fails closed rather than trusting a lone successful vote. Model is lazy-initialized, same pattern as `guardrail.py`/`vector_lookup_tool.py`
- `agent/summarizer.py` — compresses older, fully-resolved turns into a running summary once a thread's history crosses `MAX_MESSAGES_BEFORE_SUMMARY`, so long-running threads don't exceed Gemini's context window. Never touches the turn currently in progress. Wired in as a `"summarize"` graph node, reached via `route_after_guardrail` once older history crosses the threshold — runs after the guardrail check (so greetings/off-topic messages never trigger it) and before the orchestrator. Model is lazy-initialized, same pattern as `vector_lookup_tool.py`
- Human-in-the-loop escalation (`agent/orchestrator.py`, `agent/escalations_db.py`, `main.py`, `app.py`) — a verdict judged incomplete doesn't get silently discarded or silently cached; it's queued for a human decision on whether it's trustworthy enough to write into the *shared* semantic cache (the original asker already has their answer regardless — see `human_review_node`'s docstring for why the pause isn't about withholding it). `route_after_orchestrator` replaces `langgraph.prebuilt.tools_condition` as the graph's post-orchestrator routing: continues the tool loop as before when there are more tool calls, but now also checks `verdict_is_complete` (set by `_check_and_store_verdict`, see `agent/verdict_completeness.py` above) — a final answer judged incomplete routes to a new `"human_review"` node instead of ending the graph outright; complete (or unset, e.g. greetings/cache hits) ends normally. `human_review_node` calls LangGraph's `interrupt()`, which pauses graph execution there (confirmed empirically, not just from documentation, that `graph.invoke()` returns a `"__interrupt__"` key holding the paused node's payload rather than raising or hanging) and requires the checkpointer this project already has. `graph.py`'s `run_claim()` surfaces this as an additive `"pending_review"` key in its return dict (`None` when nothing's pending) — existing callers that don't check it are unaffected. `main.py`'s `/chat` records a row in a new `escalations` table (`agent/escalations_db.py`, thin index table, same pattern as `conversations_db.py`) whenever a turn produces a pending review. There's no separate reviewer role in this project's auth system — the reviewer is a single designated admin account, identified purely by matching the signed-in user's email against `ADMIN_EMAIL` (a config value, not a database role column — a one-line `.env` change if the reviewer account ever needs to swap), gating a new `require_admin` FastAPI dependency used by `GET /admin/escalations` (the pending queue) and `POST /admin/escalations/{thread_id}/resolve` (submits "approve"/"reject", calling `graph.py`'s `resume_review()` to resume the paused thread via `Command(resume=...)`). This is a deliberate, narrow carve-out — the admin endpoints are the *only* place normal per-user thread ownership doesn't apply, since an admin resolving someone else's flagged claim is the entire point; every other endpoint's ownership check is untouched. `app.py` shows a "Pending Reviews" section in the sidebar, visible only when the signed-in user's email matches `ADMIN_EMAIL`, with Approve/Reject buttons per item — this is a UI-level convenience only, not the real access control (`require_admin` re-checks server-side on every call regardless of what the UI shows). `/chat` also rejects a new claim on a thread that still has a pending review (409) rather than silently letting it through — confirmed empirically that a plain `graph.invoke()` on a thread with a paused interrupt doesn't error, it just starts a new turn and orphans the old pause, which would otherwise make a later admin approve/reject on that escalation a silent no-op. `DELETE /conversations/{thread_id}` also cancels any pending escalation for that thread (status `"cancelled"`, not `"approved"`/`"rejected"` — neither actually happened) rather than leaving it orphaned in the admin's queue — confirmed empirically that resuming a thread whose checkpoint was deleted doesn't error cleanly either, it silently restarts the graph with no real input, which would surface as an opaque 500 with no indication the actual cause was a deleted conversation
- Tests for all five tools, the orchestrator's routing logic, the response-construction helpers, the guardrail classification and its graph wiring, the checkpointer, the shared graph helpers, the summarizer, and the FastAPI endpoints

- Docker Compose deployment (`docker/api.Dockerfile`, `docker/streamlit.Dockerfile`, `docker-compose.yml`) — mirrors the security pattern already established in the sibling `medical-triage-assistant` project: multi-stage builds (build tools discarded from the final image), base image pinned by SHA256 digest rather than a mutable tag, secrets injected only via `env_file: .env` at container *runtime* (never baked into an image layer via `ARG`/`ENV`), a non-root `appuser` running both containers, and a `.dockerignore` that excludes `.env` from the build context entirely so it can never end up in an image even by accident. The `api` image pre-downloads both `sentence-transformers` models (`BAAI/bge-base-en-v1.5` and `BAAI/bge-reranker-base`) at build time with `HF_HUB_OFFLINE=1` as the runtime default, so the running container never needs a live Hugging Face connection. Each image installs only what it actually runs, not the full `requirements.txt` (which also covers local dev, so it includes `pytest`/`ruff`/`streamlit` together): the `api` image uses `requirements-api.txt` (excludes `streamlit` and its dependency tree, plus `pytest`/`ruff`), and the `ui` image uses `requirements-streamlit.txt` (`app.py` is a pure HTTP client with no LangGraph/ML dependency, unlike the reference project's UI). `app.py`'s `API_URL` is now read from an `API_URL` env var (default `http://localhost:8000` for local runs) so `docker-compose.yml` can point the `ui` container at the `api` container by service name (`http://api:8000`) instead of `localhost`, which wouldn't resolve across containers

Planned next: human-in-the-loop review (Slack notify + escalation tool, Streamlit approval UI).

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
JUDGE_MODEL_NAME=gemini-3.5-flash-lite
VERDICT_MIN_WORD_COUNT=15
JUDGE_CONSENSUS_CALLS=3
POSTGRES_CONNECTION_STRING=
JWT_SECRET_KEY=
JWT_EXPIRY_MINUTES=30
CONVERSATION_RETENTION_DAYS=10
ADMIN_EMAIL=
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

## Docker

Runs the same `.env` file as the non-Docker setup above — no separate secrets configuration. Nothing from `.env` is ever baked into an image: both Dockerfiles only read secrets via `env_file: .env` at container start, and `.dockerignore` excludes `.env` from the build context so it can't be copied in even by accident.

```
docker compose up --build
```

- API: http://localhost:8000
- Streamlit UI: http://localhost:8501

`docker compose down` to stop. The `api` image is larger and slower to build the first time — it pre-downloads two ML models at build time (see Status above) so the running container starts up without needing a live Hugging Face connection.

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
docker-compose.yml             # api + ui services, both env_file-fed from .env
docker/
  api.Dockerfile                  # FastAPI image, pre-bakes both ML models at build time
  streamlit.Dockerfile            # Streamlit image, uses requirements-streamlit.txt
requirements-api.txt           # api image deps only (no streamlit, no pytest/ruff)
requirements-streamlit.txt     # lightweight deps for the ui image (no ML/LangGraph)
.dockerignore                  # excludes .env, tests/, and other non-runtime files from build context
agent/
  config.py                       # env var loading, logging setup
  checkpointer.py                 # Postgres (Neon) checkpointer factory
  tracing.py                      # optional Langfuse tracing for the orchestrator graph
  auth.py                         # password hashing (argon2id) + JWT session tokens
  users_db.py                     # Postgres-backed user storage for authentication
  conversations_db.py             # Postgres-backed conversation listing for the sidebar
  escalations_db.py               # Postgres-backed human-review queue (pending/approved/rejected)
  orchestrator.py                 # LangGraph StateGraph: graph wiring, guardrail node, tool-calling loop
  orchestrator_routing.py         # pure routing/decision helpers
  orchestrator_responses.py       # response-construction and cache-writing helpers
  guardrail.py                    # message intent classification (greeting/claim/out-of-scope)
  verdict_completeness.py         # deterministic + LLM-judge check gating verdict caching
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
  test_orchestrator_escalation.py
  test_orchestrator_responses.py
  test_orchestrator_routing.py
  test_pii_scrub.py
  test_source_retrieval_tool.py
  test_summarizer.py
  test_vector_lookup_tool.py
  test_verdict_completeness.py
  test_web_search_tool.py
```
