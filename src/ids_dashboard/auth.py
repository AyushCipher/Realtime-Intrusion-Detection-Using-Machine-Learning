"""HTTP Basic auth for the REST API, plus a short-lived token flow for the
WebSocket endpoint (browsers cannot set an Authorization header on a
WebSocket handshake, so Basic auth doesn't reach it directly).

This is the module's "basic auth on the API" requirement, implemented
rather than left undone -- but it is still a minimal, single shared
credential pair with no per-user accounts, no rate limiting/lockout, and no
built-in TLS. See the README's known-limitations section for what that
means and what a real deployment needs on top of this.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from typing import Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

DEFAULT_WS_TOKEN_TTL_SECONDS = 300.0


@dataclass
class AuthSettings:
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "AuthSettings":
        username = os.environ.get("IDS_DASHBOARD_USERNAME")
        password = os.environ.get("IDS_DASHBOARD_PASSWORD")
        if not username or not password:
            raise RuntimeError(
                "IDS_DASHBOARD_USERNAME and IDS_DASHBOARD_PASSWORD environment variables "
                "must both be set -- see the README's authentication section."
            )
        return cls(username=username, password=password)


class TokenStore:
    """In-memory, single-process store of short-lived WebSocket tokens.

    Tokens are lost on restart and are not shared across multiple API
    replicas -- see the README's known-limitations section.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_WS_TOKEN_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._tokens: Dict[str, float] = {}

    def issue(self) -> str:
        token = secrets.token_urlsafe(32)
        self._tokens[token] = time.time() + self.ttl_seconds
        return token

    def validate(self, token: str) -> bool:
        expiry = self._tokens.get(token)
        if expiry is None:
            return False
        if expiry < time.time():
            del self._tokens[token]
            return False
        return True


_security = HTTPBasic()


def make_basic_auth_dependency(settings: AuthSettings):
    """Returns a FastAPI dependency that enforces `settings`' credentials."""

    def _check(credentials: HTTPBasicCredentials = Depends(_security)) -> str:
        # secrets.compare_digest for constant-time comparison; still a
        # single shared secret pair, not per-user authentication.
        correct_username = secrets.compare_digest(credentials.username, settings.username)
        correct_password = secrets.compare_digest(credentials.password, settings.password)
        if not (correct_username and correct_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username

    return _check
