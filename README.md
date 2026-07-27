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
- Tests for all five tools, plus the orchestrator's routing logic

Planned next: corrective re-search + authentic-source feature, Postgres-backed persistence, human-in-the-loop review, a Streamlit UI, and a FastAPI layer.

## Architecture

- **Orchestrator** — a single LangGraph agent, not a multi-agent supervisor setup. One model, multiple tools underneath.
- **Tools** — each tool returns a structured dict, not raw text, so the orchestrator can reason over results reliably.
- **Persistence** — currently in-memory (`InMemorySaver`); migrating to Postgres (Neon) once the core loop is stable.

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

3. Copy `.env.example` to `.env` and fill in your API keys:

```
   GOOGLE_API_KEY=
   TAVILY_API_KEY=
   GOOGLE_FACT_CHECK_API_KEY=
   MODEL_NAME=gemini-3.1-flash-lite
```

4. Run a test claim through the agent:

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
- pytest, ruff — testing and linting

## Project structure

```
agent/
  config.py                       # env var loading, logging setup
  orchestrator.py                 # LangGraph StateGraph, tool-calling loop, routing logic
  tools/
    _clients.py                    # shared third-party API clients
    credibility_scoring_tool.py    # Gemini-backed source-reliability judgment
    fact_check_tool.py             # Google Fact Check Tools API lookup
    source_retrieval_tool.py       # Tavily Extract-backed full-article retrieval
    vector_lookup_tool.py          # Pinecone-backed semantic claim cache
    web_search_tool.py             # Tavily-backed web search tool
main.py                            # manual end-to-end test entry point
tests/
  test_credibility_scoring_tool.py
  test_fact_check_tool.py
  test_orchestrator_routing.py
  test_source_retrieval_tool.py
  test_vector_lookup_tool.py
  test_web_search_tool.py
```