"""Password hashing and JWT session tokens for multi-user authentication.

Two independent concerns kept in one small module: turning a plaintext
password into something safe to store (argon2id, OWASP's current
recommendation over bcrypt/scrypt for new systems), and turning a verified
user identity into a signed, time-limited token the client can hold and
replay on later requests instead of resending a password every time.
"""

import logging
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError, VerifyMismatchError

from agent.config import JWT_ALGORITHM, JWT_EXPIRY_MINUTES, JWT_SECRET_KEY

logger = logging.getLogger(__name__)

# A stateless config object with no I/O or credentials involved in building
# it (unlike the Gemini/Pinecone/Langfuse clients elsewhere in this
# codebase) — safe to construct at import time rather than lazily.
_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a plaintext password with argon2id for storage.

    This function's output is the only form of a password that should ever
    reach the database or a log line — never store or log the plaintext.
    """
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a plaintext password against a stored argon2id hash.

    Returns False for a wrong password or a corrupted/foreign hash rather
    than letting argon2's exceptions escape — callers (the login endpoint)
    should only ever have to handle a plain bool, not argon2's internal
    exception hierarchy.
    """
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHash):
        return False


def create_access_token(user_id: str, email: str) -> str:
    """Build a signed JWT identifying this user, valid for JWT_EXPIRY_MINUTES.

    user_id goes in the "sub" claim (JWT convention for "subject"). email is
    included only for convenience/debugging — never put anything sensitive
    in a JWT payload, since it's base64-encoded, not encrypted, and readable
    by anyone holding the token.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """Validate and decode a JWT, or return None if it's invalid/expired.

    Returns None instead of raising so callers can turn every failure mode
    (bad signature, expired, malformed) into the same 401 response, instead
    of a long except chain producing inconsistent errors per failure type.
    """
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        logger.info("Rejected invalid or expired JWT.")
        return None
