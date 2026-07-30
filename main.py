"""FastAPI app wrapping the fact-checking orchestrator graph."""

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from agent.checkpointer import build_checkpointer
from graph import build_graph, run_claim

logger = logging.getLogger(__name__)

API_ACCESS_KEY = os.getenv("API_ACCESS_KEY")
if not API_ACCESS_KEY:
    raise EnvironmentError(
        "API_ACCESS_KEY is not set. This API refuses to start without it — "
        "running without authentication would let anyone consume your "
        "Gemini/Tavily/Pinecone quota. Add it to your .env file."
    )


def _extract_text(content: str | list) -> str:
    """Normalize a message's content into a plain string.

    Some Gemini responses come back as a list of content blocks
    (e.g. [{"type": "text", "text": "..."}]) instead of a plain string —
    ChatResponse requires a str, so this flattens either shape into one.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    """Reject requests that don't carry the correct X-API-Key header.

    API_ACCESS_KEY is required at startup (see the check above), so this
    always enforces auth in a real run.

    Raises:
        HTTPException: 401 if the header is missing or incorrect.
    """
    if x_api_key != API_ACCESS_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the Postgres checkpointer once for the app's lifetime, not per request.

    A fresh connection per request would be wasteful and slow — this opens
    it once at startup and keeps it alive until the server shuts down,
    at which point the `with` block's __exit__ closes it cleanly.
    """
    with build_checkpointer() as checkpointer:
        app.state.graph = build_graph(checkpointer)
        logger.info("FastAPI startup complete — graph ready.")
        yield
    logger.info("FastAPI shutting down — checkpointer connection closed.")


app = FastAPI(title="Misinformation Agent API", lifespan=lifespan)


class ChatRequest(BaseModel):
    claim: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    recursion_limit_hit: bool


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(request: ChatRequest) -> ChatResponse:
    """Submit a claim (optionally continuing an existing thread) and get a verdict."""
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim must not be empty.")

    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        result = run_claim(app.state.graph, request.claim, thread_id)
    except Exception:
        logger.exception("Unexpected error running claim for thread %s.", thread_id)
        raise HTTPException(status_code=500, detail="Something went wrong processing this claim.")

    messages = result["messages"]
    answer = _extract_text(messages[-1].content) if messages else ""

    return ChatResponse(
        answer=answer,
        thread_id=thread_id,
        recursion_limit_hit=result["recursion_limit_hit"],
    )
