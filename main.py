"""FastAPI app wrapping the fact-checking orchestrator graph."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr

from agent.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from agent.checkpointer import build_checkpointer, build_connection_pool
from agent.config import CONVERSATION_RETENTION_DAYS
from agent.conversations_db import (
    create_conversation,
    delete_conversation,
    derive_title,
    list_conversations,
    purge_stale_conversations,
    setup_conversations_table,
    touch_conversation,
)
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


def _is_valid_uuid(value: str) -> bool:
    """Check whether a string is a well-formed UUID.

    Used to reject a malformed client-provided thread_id up front, before
    any real work happens — the conversations table's thread_id column is
    typed UUID (see agent/conversations_db.py), so a non-UUID thread_id
    would otherwise process a claim in full (a real Gemini/Tavily/Pinecone
    pass) only to end up as an orphaned checkpoint thread that can never
    be recorded in, listed from, or deleted through the sidebar — better
    to reject it immediately with a clear 400 than let that happen.
    """
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def get_current_user(
    response: Response,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """Resolve the authenticated user from a Bearer JWT, or reject the request.

    auto_error=False on HTTPBearer means a missing header falls through to
    here instead of FastAPI's generic 403 — every failure mode (missing
    header, malformed token, expired token, bad signature) ends up as the
    same 401 with a clear message, not several different error shapes.

    Also implements the sliding session: JWT_EXPIRY_MINUTES is short (30
    minutes), so every successful authenticated call reissues a brand-new
    token with a fresh expiry and attaches it via the X-New-Token response
    header. A JWT's own expiry can't be extended in place once signed — the
    only way to slide a session forward is to hand back a new token — so an
    active user's session keeps renewing itself on every request, while a
    genuinely idle session still hard-expires after JWT_EXPIRY_MINUTES with
    no activity. app.py is responsible for picking this header up and
    replacing its stored token; every endpoint that depends on this
    function gets the refresh for free, nothing extra needed per-endpoint.
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
    response.headers["X-New-Token"] = create_access_token(payload["sub"], payload["email"])
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

    Both the checkpointer, agent/users_db.py, and agent/conversations_db.py
    draw from this single pool (see agent/checkpointer.py) rather than a
    single held-open connection or one connection per call — a pool
    detects and replaces connections that Neon's free tier kills when it
    auto-suspends from inactivity, which a single held-open connection
    can't recover from without an app restart. setup_users_table() and
    setup_conversations_table() are both idempotent (CREATE TABLE IF NOT
    EXISTS) so it's safe to run them on every startup too, same reasoning
    as the checkpointer's own .setup() call.
    """
    with build_connection_pool() as pool:
        app.state.pool = pool
        setup_users_table(pool)
        setup_conversations_table(pool)
        checkpointer = build_checkpointer(pool)
        app.state.graph = build_graph(checkpointer)

        # Purge conversations that have been idle longer than the configured
        # retention window. Runs once per startup rather than on a timer —
        # simple and sufficient for this app's scale, and avoids needing a
        # background scheduler/task queue just for housekeeping. The sidebar
        # row and the underlying checkpoint data are two separate stores
        # (see agent/conversations_db.py), so both have to be cleaned up
        # here — purge_stale_conversations() only knows about the former.
        purged_thread_ids = purge_stale_conversations(pool, CONVERSATION_RETENTION_DAYS)
        for stale_thread_id in purged_thread_ids:
            checkpointer.delete_thread(stale_thread_id)
        if purged_thread_ids:
            logger.info(
                "Purged %d conversation(s) older than %d days.",
                len(purged_thread_ids),
                CONVERSATION_RETENTION_DAYS,
            )

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


class ConversationSummary(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class ConversationHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ConversationMessage]


def _to_conversation_messages(raw_messages: list) -> list[ConversationMessage]:
    """Filter a thread's full message list down to what the chat UI actually shows.

    Keeps HumanMessages (user turns) and AIMessages that carry a final
    answer (no pending tool_calls, non-empty content). Drops ToolMessages,
    intermediate tool-calling AIMessages, and the summarizer's SystemMessage
    placeholder — none of those are ever rendered in app.py's chat window,
    so resuming a conversation shouldn't surface them either.
    """
    result: list[ConversationMessage] = []
    for message in raw_messages:
        role = getattr(message, "type", None)
        if role == "human":
            result.append(ConversationMessage(role="user", content=_extract_text(message.content)))
        elif role == "ai" and not getattr(message, "tool_calls", None) and message.content:
            result.append(
                ConversationMessage(role="assistant", content=_extract_text(message.content))
            )
    return result


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
    guessable/replayable across accounts. Also records this thread in the
    sidebar's conversation list (agent/conversations_db.py) — a new entry
    on the first message of a thread, or just a recency bump on later ones.
    """
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim must not be empty.")

    if request.thread_id and not _is_valid_uuid(request.thread_id):
        raise HTTPException(status_code=400, detail="thread_id must be a valid UUID.")

    if request.thread_id:
        owner_id = _get_thread_owner(app.state.graph, request.thread_id)
        if owner_id is not None and owner_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This conversation belongs to a different account.",
            )
        thread_id = request.thread_id
        # A client-provided thread_id doesn't mean the thread has actually
        # been used before — app.py generates one upfront for every new
        # conversation, before the first message is ever sent. owner_id is
        # only set once a thread has a real checkpoint, so "never had an
        # owner" is the real signal for "this is new," not "was an id sent."
        is_new_thread = owner_id is None
    else:
        thread_id = str(uuid.uuid4())
        is_new_thread = True

    try:
        result = run_claim(app.state.graph, request.claim, thread_id, user_id=current_user["id"])
    except Exception:
        logger.exception("Unexpected error running claim for thread %s.", thread_id)
        raise HTTPException(status_code=500, detail="Something went wrong processing this claim.")

    # Record this thread in the sidebar's conversation list on its first
    # message, or just bump its recency on every later one — kept after a
    # successful run_claim so a failed claim never creates a phantom
    # sidebar entry for a conversation that has no actual content yet.
    #
    # Deliberately caught and logged rather than left to propagate: by
    # this point the claim has already been fully processed and answered
    # (a real Gemini/Tavily/Pinecone pass), and that real conversation
    # data is already safe in the checkpoint regardless of what happens
    # here. thread_id is already guaranteed to be a well-formed UUID by
    # now (see the _is_valid_uuid check above), but the write itself can
    # still fail for other reasons — a dropped connection, Neon
    # auto-suspending mid-request, any transient Postgres hiccup. Failing
    # this bookkeeping step is a much smaller problem than discarding a
    # real answer and returning a 500 for it, which would also make a
    # client naively retrying re-run the entire expensive pipeline for
    # nothing. Worst case here is just a conversation that doesn't show up
    # (or doesn't bump recency) in the sidebar list — not a lost answer.
    try:
        if is_new_thread:
            title = derive_title(request.claim)
            create_conversation(app.state.pool, thread_id, current_user["id"], title)
        else:
            touch_conversation(app.state.pool, thread_id)
    except Exception:
        logger.exception("Failed to record conversation %s in the sidebar list.", thread_id)

    messages = result["messages"]
    answer = _extract_text(messages[-1].content) if messages else ""

    return ChatResponse(
        answer=answer,
        thread_id=thread_id,
        recursion_limit_hit=result["recursion_limit_hit"],
    )


@app.get("/conversations", response_model=list[ConversationSummary])
def get_conversations(current_user: dict = Depends(get_current_user)) -> list[ConversationSummary]:
    """List the authenticated user's conversations, most recently active first."""
    rows = list_conversations(app.state.pool, current_user["id"])
    return [ConversationSummary(**row) for row in rows]


@app.get("/conversations/{thread_id}/messages", response_model=ConversationHistoryResponse)
def get_conversation_messages(
    thread_id: str, current_user: dict = Depends(get_current_user)
) -> ConversationHistoryResponse:
    """Return a thread's message history, so the UI can resume it.

    Same ownership check as /chat — a thread belonging to a different
    user is rejected with 403 rather than leaking its contents. An
    unknown thread_id (owner_id is None) is also rejected rather than
    treated as an empty-but-valid conversation, since it was never
    created through this user's own /chat calls.
    """
    owner_id = _get_thread_owner(app.state.graph, thread_id)
    if owner_id is None or owner_id != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This conversation belongs to a different account.",
        )
    state = app.state.graph.get_state({"configurable": {"thread_id": thread_id}})
    raw_messages = state.values.get("messages", []) if state else []
    return ConversationHistoryResponse(
        thread_id=thread_id, messages=_to_conversation_messages(raw_messages)
    )


@app.delete("/conversations/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation_endpoint(
    thread_id: str, current_user: dict = Depends(get_current_user)
) -> None:
    """Delete a conversation entirely — both its sidebar entry and its checkpoint data.

    Same ownership check as the other /conversations endpoints: an unknown
    thread_id or one belonging to a different user is rejected with 403
    rather than treated as "already deleted," for the same reasons as
    get_conversation_messages above. Deletes checkpoint data first and the
    sidebar row second — if the checkpoint delete fails, the sidebar entry
    is left in place so the conversation isn't silently hidden while its
    underlying data still exists; if the sidebar delete were to fail after
    a successful checkpoint delete, the next GET /conversations simply
    won't show a stale entry for long since a following retry or the
    retention purge would still clean it up, whereas the reverse ordering
    could leak a resumable-looking conversation whose data is actually gone.
    """
    owner_id = _get_thread_owner(app.state.graph, thread_id)
    if owner_id is None or owner_id != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This conversation belongs to a different account.",
        )
    try:
        app.state.graph.checkpointer.delete_thread(thread_id)
        delete_conversation(app.state.pool, thread_id)
    except Exception:
        logger.exception("Unexpected error deleting thread %s.", thread_id)
        raise HTTPException(
            status_code=500, detail="Something went wrong deleting this conversation."
        )
