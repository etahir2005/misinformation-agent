"""FastAPI app wrapping the fact-checking orchestrator graph."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr

from agent.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from agent.checkpointer import build_checkpointer, build_connection_pool
from agent.users_db import (
    EmailAlreadyRegisteredError,
    create_user,
    get_user_by_email,
    setup_users_table,
)
from graph import build_graph, run_claim

logger = logging.getLogger(__name__)

MIN_PASSWORD_LENGTH = 8

_bearer_scheme = HTTPBearer(auto_error=False)


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


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """Resolve the authenticated user from a Bearer JWT, or reject the request.

    auto_error=False on HTTPBearer means a missing header falls through to
    here instead of FastAPI's generic 403 — every failure mode (missing
    header, malformed token, expired token, bad signature) ends up as the
    same 401 with a clear message, not several different error shapes.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token."
        )
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token."
        )
    return {"id": payload["sub"], "email": payload["email"]}


def _get_thread_owner(graph, thread_id: str) -> str | None:
    """Look up which user_id owns an existing thread, or None if unknown.

    Reads the thread's own checkpoint metadata rather than a separate
    table — LangGraph already persists whatever's passed as
    config["metadata"] on invoke (see graph.run_claim), so recording
    user_id there once is enough to answer "does this thread belong to
    this user" without a second source of truth to keep in sync.
    """
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not state or not state.metadata:
        return None
    return state.metadata.get("user_id")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open one shared Postgres connection pool for the app's lifetime.

    Both the checkpointer and agent/users_db.py draw from this single pool
    (see agent/checkpointer.py) rather than a single held-open connection
    or one connection per call — a pool detects and replaces connections
    that Neon's free tier kills when it auto-suspends from inactivity,
    which a single held-open connection can't recover from without an app
    restart. setup_users_table() is idempotent (CREATE TABLE IF NOT EXISTS)
    so it's safe to run on every startup too, same reasoning as the
    checkpointer's own .setup() call.
    """
    with build_connection_pool() as pool:
        app.state.pool = pool
        setup_users_table(pool)
        checkpointer = build_checkpointer(pool)
        app.state.graph = build_graph(checkpointer)
        logger.info("FastAPI startup complete — graph ready.")
        yield
    logger.info("FastAPI shutting down — connection pool closed.")


app = FastAPI(title="Misinformation Agent API", lifespan=lifespan)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


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


@app.post("/auth/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(request: SignupRequest) -> AuthResponse:
    """Create a new account and return a bearer token for it, already logged in."""
    if len(request.password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
        )
    password_hash = hash_password(request.password)
    try:
        user_id = create_user(app.state.pool, request.email, password_hash)
    except EmailAlreadyRegisteredError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    return AuthResponse(access_token=create_access_token(user_id, request.email))


@app.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest) -> AuthResponse:
    """Verify credentials and return a bearer token."""
    user = get_user_by_email(app.state.pool, request.email)
    # Deliberately the same error for "no such user" and "wrong password" —
    # distinguishing them would let an attacker enumerate which emails have
    # accounts, a well-known auth anti-pattern.
    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password."
    )
    if user is None or not verify_password(request.password, user["password_hash"]):
        raise invalid_credentials
    return AuthResponse(access_token=create_access_token(user["id"], user["email"]))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)) -> ChatResponse:
    """Submit a claim (optionally continuing an existing thread) and get a verdict.

    A provided thread_id must belong to the authenticated user — reusing
    someone else's thread_id is rejected with 403 rather than silently
    continuing their conversation, since thread_id alone is otherwise
    guessable/replayable across accounts.
    """
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim must not be empty.")

    if request.thread_id:
        owner_id = _get_thread_owner(app.state.graph, request.thread_id)
        if owner_id is not None and owner_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This conversation belongs to a different account.",
            )
        thread_id = request.thread_id
    else:
        thread_id = str(uuid.uuid4())

    try:
        result = run_claim(app.state.graph, request.claim, thread_id, user_id=current_user["id"])
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
