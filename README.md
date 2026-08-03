# Misinformation Agent

A LangGraph-based fact-checking agent that takes a claim (or article link) and produces a sourced verdict with a credibility score — built for ordinary social media users who want a quick way to check a claim before believing or resharing it, not for professional fact-checkers.

## Status

Early build in progress. Currently implemented:

- Single-agent orchestrator (LangGraph `StateGraph`) with a tool-calling loop
- `fact_check_lookup_tool` (Google Fact Check Tools API) — checks whether a professional fact-checker has already ruled on the claim
- `web_search_tool` (Tavily) — general web search for evidence when no existing ruling is found
- `source_retrieval_tool` (Tavily Extract) — fetches full article text from a specific URL
- `credibility_scoring_tool` — judges source reliability and produces a confidence score when there's no clean existing ruling to rely on
- `vector_lookup_tool` — semantic claim cache (Pinecone + local embeddings) that reuses a prior verdict when a claim is a close rewording of one already checked, instead of re-running the full pipeline
- Deterministic routing: cache check first, then fact-check database, web search as fallback, credibility scoring forced whenever the evidence gathered doesn't already amount to a single clean True/False ruling
- First-turn tool use forced (`tool_choice="any"`) so the agent always checks the cache and gathers evidence before answering
- A 16-step recursion limit on the tool-calling loop, with a graceful partial-progress fallback instead of a crash
- Postgres-backed persistence (Neon) via `agent/checkpointer.py` — conversation state survives restarts, keyed by thread_id. The checkpointer is injected into `build_orchestrator()` rather than hardcoded, so tests still use `InMemorySaver`
- `agent/summarizer.py` — compresses older, fully-resolved turns into a running summary once a thread's history crosses `MAX_MESSAGES_BEFORE_SUMMARY`, so long-running threads don't exceed Gemini's context window. Never touches the turn currently in progress. Model is lazy-initialized, same pattern as `vector_lookup_tool.py`
- Tests for all five tools, the orchestrator's routing logic, the checkpointer, and the summarizer

Planned next: corrective re-search + authentic-source feature, human-in-the-loop review, a Streamlit UI, and a FastAPI layer.

## Architecture

- **Orchestrator** — a single LangGraph agent, not a multi-agent supervisor setup. One model, multiple tools underneath.
- **Tools** — each tool returns a structured dict, not raw text, so the orchestrator can reason over results reliably.
- **Persistence** — Postgres (Neon), via `agent/checkpointer.py`. `build_orchestrator()` takes the checkpointer as an injected argument rather than constructing one itself, so tests can pass `InMemorySaver` without touching a real database.
- **Context management** — `agent/summarizer.py`, wired in as a `"summarize"` graph node reached via a conditional edge off `START`. Self-limiting: once it runs, older-message count drops back below threshold until enough new messages accumulate again.

## Setup

1. Create a virtual environment:

```
   python -m venv venv
   venv\Scripts\activate
```

2. Install dependencies:

```
   venv\Scripts\python.exe -m pip install -r requirements.txt
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
   MODEL_NAME=gemini-3.1-flash-lite
   POSTGRES_CONNECTION_STRING=
```

6. Run a test claim through the agent:

```
   venv\Scripts\python.exe main.py
```

## Testing

```
venv\Scripts\python.exe -m pytest
venv\Scripts\python.exe -m ruff check .
```

## Tech stack

- LangGraph / LangChain — agent orchestration
- Google Gemini — LLM
- Tavily — web search and article extraction
- Google Fact Check Tools API — existing fact-check lookups
- Pinecone — vector database for the semantic claim cache
- sentence-transformers (BAAI/bge-base-en-v1.5) — local embedding model for the claim cache
- Postgres (Neon) via `langgraph-checkpoint-postgres` / `psycopg` — conversation state persistence
- pytest, ruff — testing and linting

## Project structure

```
agent/
  config.py                       # env var loading, logging setup
  checkpointer.py                 # Postgres (Neon) checkpointer factory
  orchestrator.py                 # LangGraph StateGraph, tool-calling loop, routing logic
  summarizer.py                   # conversation summarization for long-running threads
  tools/
    _clients.py                    # shared third-party API clients
    credibility_scoring_tool.py    # Gemini-backed source-reliability judgment
    fact_check_tool.py             # Google Fact Check Tools API lookup
    source_retrieval_tool.py       # Tavily Extract-backed full-article retrieval
    vector_lookup_tool.py          # Pinecone-backed semantic claim cache
    web_search_tool.py             # Tavily-backed web search tool
main.py                            # manual end-to-end test entry point
tests/
  test_checkpointer.py
  test_credibility_scoring_tool.py
  test_fact_check_tool.py
  test_orchestrator_build.py
  test_orchestrator_routing.py
  test_source_retrieval_tool.py
  test_summarizer.py
  test_vector_lookup_tool.py
  test_web_search_tool.py
```