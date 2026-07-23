# Misinformation Agent

A LangGraph-based fact-checking agent that takes a claim (or article link) and produces a sourced verdict with a credibility score — built for ordinary social media users who want a quick way to check a claim before believing or resharing it, not for professional fact-checkers.

## Status

Early build in progress. Currently implemented:

- Single-agent orchestrator (LangGraph `StateGraph`) with a tool-calling loop
- `fact_check_lookup_tool` (Google Fact Check Tools API) — checks whether a professional fact-checker has already ruled on the claim
- `web_search_tool` (Tavily) — general web search for evidence when no existing ruling is found
- `source_retrieval_tool` (Tavily Extract) — fetches full article text from a specific URL
- Tool priority enforced via the system prompt: fact-check database first, then web search, with source retrieval used as needed for deeper detail
- First-turn tool use forced (`tool_choice="any"`) so the agent always gathers evidence before answering
- Tests for all three tools

Planned next: source-credibility scoring, a corrective re-search + authentic-source feature, Postgres-backed persistence, human-in-the-loop review, a Streamlit UI, and a FastAPI layer.

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
- pytest, ruff — testing and linting

## Project structure

```
agent/
  config.py                    # env var loading, logging setup
  orchestrator.py              # LangGraph StateGraph, tool-calling loop
  tools/
    _clients.py                 # shared third-party API clients
    fact_check_tool.py          # Google Fact Check Tools API lookup
    search_tool.py              # Tavily-backed web search tool
    source_retrieval_tool.py    # Tavily Extract-backed full-article retrieval
main.py                         # manual end-to-end test entry point
tests/
  test_fact_check_tool.py
  test_search_tool.py
  test_source_retrieval_tool.py
```