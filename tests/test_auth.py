"""Tests for agent/auth.py — password hashing and JWT session tokens."""

from datetime import datetime, timedelta, timezone

import jwt

from agent.auth import create_access_token, decode_access_token, hash_password, verify_password
from agent.config import JWT_ALGORITHM, JWT_SECRET_KEY


def test_hash_and_verify_password_roundtrip() -> None:
    password_hash = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", password_hash) is True


def test_verify_password_rejects_wrong_password() -> None:
    password_hash = hash_password("correct horse battery staple")
    assert verify_password("wrong password", password_hash) is False


def test_verify_password_rejects_garbage_hash() -> None:
    """A corrupted/foreign hash should return False, not raise."""
    assert verify_password("anything", "not-a-real-argon2-hash") is False


def test_create_and_decode_access_token_roundtrip() -> None:
    token = create_access_token(user_id="user-123", email="person@example.com")
    payload = decode_access_token(token)

    assert payload is not None
    assert payload["sub"] == "user-123"
    assert payload["email"] == "person@example.com"


def test_decode_access_token_rejects_garbage_token() -> None:
    assert decode_access_token("not-a-real-jwt") is None


def test_decode_access_token_rejects_expired_token() -> None:
    """A token whose exp claim is already in the past should be rejected, not raise."""
    expired_payload = {
        "sub": "user-123",
        "email": "person@example.com",
        "iat": datetime.now(timezone.utc) - timedelta(minutes=10),
        "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
    }
    expired_token = jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    assert decode_access_token(expired_token) is None


def test_decode_access_token_rejects_wrong_signature() -> None:
    """A token signed with a different secret than this app's should be rejected."""
    payload = {
        "sub": "user-123",
        "email": "person@example.com",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    # A 32+ byte dummy secret — long enough that PyJWT doesn't emit an
    # InsecureKeyLengthWarning for this deliberately-wrong-signature test.
    wrongly_signed_secret = "a-different-secret-thats-at-least-32-bytes-long"
    wrongly_signed_token = jwt.encode(payload, wrongly_signed_secret, algorithm=JWT_ALGORITHM)

    assert decode_access_token(wrongly_signed_token) is None
