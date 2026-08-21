"""FastAPI application factory: wiring, CORS, and startup/shutdown."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .alert_consumer import AlertEventSource, KafkaAlertEventSource, StubAlertEventSource
from .auth import AuthSettings, TokenStore, make_basic_auth_dependency
from .broadcaster import AlertBroadcaster
from .config import Settings
from .ingest_service import IngestService
from .routes_alerts import build_router as build_alerts_router
from .routes_ws import build_router as build_ws_router
from .store import AlertStore

logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    auth_settings: Optional[AuthSettings] = None,
    source: Optional[AlertEventSource] = None,
    store: Optional[AlertStore] = None,
) -> FastAPI:
    """Builds the app. `source`/`store` are injectable so tests (and
    `--use-stub` local runs) don't need a real Kafka broker or DB file."""
    settings = settings or Settings.from_env()
    auth_settings = auth_settings or AuthSettings.from_env()

    store = store or AlertStore(settings.db_path)
    broadcaster = AlertBroadcaster()
    token_store = TokenStore(ttl_seconds=settings.ws_token_ttl_seconds)

    if source is None:
        if settings.use_stub_source:
            source = StubAlertEventSource([])
        else:
            source = KafkaAlertEventSource(settings.bootstrap_servers, topic=settings.alert_topic)

    ingest_service = IngestService(source, store, broadcaster)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        loop = asyncio.get_running_loop()
        ingest_service.start(loop)
        logger.info(
            "Dashboard API started; ingesting from %s",
            "stub source" if settings.use_stub_source else settings.bootstrap_servers,
        )
        try:
            yield
        finally:
            ingest_service.stop()
            store.close()

    app = FastAPI(title="IDS Alert Dashboard API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.store = store
    app.state.broadcaster = broadcaster
    app.state.token_store = token_store
    app.state.ingest_service = ingest_service

    get_auth_user = make_basic_auth_dependency(auth_settings)
    app.include_router(build_alerts_router(get_auth_user))
    app.include_router(build_ws_router(token_store.validate))

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok",
            "alerts_processed": ingest_service.processed,
            "ws_clients": broadcaster.client_count,
        }

    return app
