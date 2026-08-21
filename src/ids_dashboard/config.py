"""Environment-driven configuration for the dashboard API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from .auth import DEFAULT_WS_TOKEN_TTL_SECONDS
from .schema import ALERT_TOPIC


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    alert_topic: str = ALERT_TOPIC
    db_path: str = "alerts.db"
    use_stub_source: bool = False
    ws_token_ttl_seconds: float = DEFAULT_WS_TOKEN_TTL_SECONDS
    cors_allow_origins: List[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "Settings":
        servers = os.environ.get("IDS_DASHBOARD_BOOTSTRAP_SERVERS", "localhost:9092")
        origins = os.environ.get("IDS_DASHBOARD_CORS_ORIGINS", "*")
        return cls(
            bootstrap_servers=[s.strip() for s in servers.split(",") if s.strip()],
            alert_topic=os.environ.get("IDS_DASHBOARD_ALERT_TOPIC", ALERT_TOPIC),
            db_path=os.environ.get("IDS_DASHBOARD_DB_PATH", "alerts.db"),
            use_stub_source=_env_bool("IDS_DASHBOARD_USE_STUB"),
            ws_token_ttl_seconds=float(os.environ.get("IDS_DASHBOARD_WS_TOKEN_TTL", str(DEFAULT_WS_TOKEN_TTL_SECONDS))),
            cors_allow_origins=[s.strip() for s in origins.split(",") if s.strip()],
        )
