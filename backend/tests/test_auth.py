import pytest
from datetime import datetime, timezone, timedelta
from jose import jwt
from fastapi import HTTPException
from core.auth import decode_token, SECRET_KEY, ALGORITHM


def test_decode_token_valid():
    payload = {
        "sub": "user_abc_123",
        "email": "student@example.com",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    user = decode_token(token)

    assert user["id"] == "user_abc_123"
    assert user["email"] == "student@example.com"


def test_decode_token_missing_sub():
    payload = {
        "email": "student@example.com",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        decode_token(token)
    assert exc_info.value.status_code == 401


def test_decode_token_expired():
    payload = {
        "sub": "user_abc_123",
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        decode_token(token)
    assert exc_info.value.status_code == 401
