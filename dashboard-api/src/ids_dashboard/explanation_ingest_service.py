"""Ties the explanation source, the store, and the WebSocket broadcaster
together -- the explanation-side counterpart to ingest_service.py.

A second, independent background consumer rather than a branch inside
IngestService, since explanations come from a different topic, a
different producer (tier2_reasoner, not ml), and only ever arrive for the
escalated minority of alerts -- conflating the two loops would make both
harder to reason about for no real benefit.

Broadcasts reuse the same `AlertBroadcaster` alerts already go through
(one WebSocket connection, one client-side listener, not two) -- wrapped
with a `__type: "explanation"` marker so `routes_ws.py`'s single endpoint
stays unchanged and the frontend can tell the two message shapes apart
without a breaking change to the existing (unmarked) alert broadcast
format.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Optional

from .broadcaster import AlertBroadcaster
from .explanation_consumer import ExplanationEventSource
from .schema import validate_explanation_event
from .store import AlertStore

logger = logging.getLogger(__name__)


class ExplanationIngestService:
    def __init__(self, source: ExplanationEventSource, store: AlertStore, broadcaster: AlertBroadcaster) -> None:
        self.source = source
        self.store = store
        self.broadcaster = broadcaster
        self.processed = 0
        self.broadcast_scheduled = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def process_explanation(self, explanation: Dict[str, Any]) -> bool:
        """Validates and persists one explanation. Returns True if newly
        inserted (vs. a duplicate explanation_id), same redelivery
        tolerance as IngestService.process_alert."""
        validate_explanation_event(explanation)
        self.processed += 1
        return self.store.insert_explanation(explanation)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._thread = threading.Thread(target=self._run, name="ExplanationIngestService", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for explanation in self.source:
                if self._stop_event.is_set():
                    break
                self._handle_explanation(explanation)
        except Exception:  # noqa: BLE001 - keep the thread from dying silently
            logger.exception("ExplanationIngestService consumer loop terminated unexpectedly")
        finally:
            self.source.close()

    def _handle_explanation(self, explanation: Dict[str, Any]) -> None:
        try:
            inserted = self.process_explanation(explanation)
        except ValueError:
            logger.warning("Dropping explanation that failed schema validation", exc_info=True)
            return
        if inserted and self._loop is not None:
            message = {"__type": "explanation", **explanation}
            asyncio.run_coroutine_threadsafe(self.broadcaster.broadcast(message), self._loop)
            self.broadcast_scheduled += 1

    def stop(self) -> None:
        self._stop_event.set()
        self.source.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
