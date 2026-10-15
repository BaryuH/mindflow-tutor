import os
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.auth import validate_jwt_secret, SECRET_KEY, ALGORITHM
from routers.auth import router, _oauth_success_redirect

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_validate_jwt_secret_production_fail_fast():
    # Insecure default secret in production must raise RuntimeError
    with patch.dict(os.environ, {"ENVIRONMENT": "production", "JWT_SECRET_KEY": "change-me-in-production"}):
        with pytest.raises(RuntimeError, match="FATAL: Insecure or missing JWT_SECRET_KEY"):
            validate_jwt_secret()

    # Empty secret in production must raise RuntimeError
    with patch.dict(os.environ, {"ENVIRONMENT": "production", "JWT_SECRET_KEY": ""}):
        with pytest.raises(RuntimeError, match="FATAL: Insecure or missing JWT_SECRET_KEY"):
            validate_jwt_secret()

    # Short secret in production must raise RuntimeError
    with patch.dict(os.environ, {"ENVIRONMENT": "production", "JWT_SECRET_KEY": "short-secret"}):
        with pytest.raises(RuntimeError, match="FATAL: Insecure or missing JWT_SECRET_KEY"):
            validate_jwt_secret()

    # Secure 32+ char secret in production must succeed
    strong_secret = "a" * 32
    with patch.dict(os.environ, {"ENVIRONMENT": "production", "JWT_SECRET_KEY": strong_secret}):
        validate_jwt_secret()  # Should not raise


def test_validate_jwt_secret_fail_on_flag():
    # FAIL_ON_DEFAULT_SECRET=true should fail fast even in dev
    with patch.dict(os.environ, {"FAIL_ON_DEFAULT_SECRET": "true", "JWT_SECRET_KEY": "default"}):
        with pytest.raises(RuntimeError, match="FATAL: Insecure or missing JWT_SECRET_KEY"):
            validate_jwt_secret()


def test_oauth_success_redirect_uses_code_not_raw_token():
    redirect_url = _oauth_success_redirect(
        "http://localhost:5173",
        user_id="user_123",
        email="test@example.com",
        full_name="Test User",
    )
    # Must NOT expose access token in URL query
    assert "token=" not in redirect_url
    assert "user=" not in redirect_url

    # Must contain short-lived exchange code
    assert "?code=" in redirect_url


def test_oauth_code_exchange_and_replay_protection():
    # 1. Generate an exchange redirect URL
    redirect_url = _oauth_success_redirect(
        "http://localhost:5173",
        user_id="oauth_user_99",
        email="oauth@example.com",
        full_name="OAuth Student",
    )
    code = redirect_url.split("code=")[1]

    # 2. Exchange code via POST /auth/oauth/exchange
    res = client.post("/auth/oauth/exchange", json={"code": code})
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["id"] == "oauth_user_99"
    assert data["user"]["email"] == "oauth@example.com"
    assert data["user"]["full_name"] == "OAuth Student"

    # 3. Replay attack: attempt to use the same code again -> 400
    res_replay = client.post("/auth/oauth/exchange", json={"code": code})
    assert res_replay.status_code == 400
    assert "already been used" in res_replay.json()["detail"]

def test_oauth_code_exchange_invalid_code():
    res = client.post("/auth/oauth/exchange", json={"code": "completely-invalid-code"})
    assert "Invalid or expired exchange code" in res.json()["detail"]
