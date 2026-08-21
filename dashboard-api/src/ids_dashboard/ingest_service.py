"""Ties the alert source, the store, and the WebSocket broadcaster together.

`process_alert` is synchronous and side-effect-testable on its own (store
insert only); `start`/`stop` wrap it for the live path, where the Kafka
client (synchronous, kafka-python) runs on a background thread and hands
each newly-stored alert to the asyncio event loop via
`run_coroutine_threadsafe` so it can be broadcast without blocking the
consumer thread on slow or many WebSocket clients.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Optional

from .alert_consumer import AlertEventSource
from .broadcaster import AlertBroadcaster
from .schema import validate_alert_event
from .store import AlertStore

logger = logging.getLogger(__name__)


class IngestService:
    def __init__(self, source: AlertEventSource, store: AlertStore, broadcaster: AlertBroadcaster) -> None:
        self.source = source
        self.store = store
        self.broadcaster = broadcaster
        self.processed = 0
        self.broadcast_scheduled = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def process_alert(self, alert: Dict[str, Any]) -> bool:
        """Validates and persists one alert. Returns True if it was newly
        inserted (vs. a duplicate alert_id, e.g. from consumer redelivery),
        which is what callers use to decide whether to broadcast it."""
        validate_alert_event(alert)
        self.processed += 1
        return self.store.insert_alert(alert)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Starts the background consumer thread. `loop` is the asyncio
        event loop WebSocket broadcasts must run on (the FastAPI app's)."""
        self._loop = loop
        self._thread = threading.Thread(target=self._run, name="IngestService", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for alert in self.source:
                if self._stop_event.is_set():
                    break
                self._handle_alert(alert)
        except Exception:  # noqa: BLE001 - keep the thread from dying silently
            logger.exception("IngestService consumer loop terminated unexpectedly")
        finally:
            self.source.close()

    def _handle_alert(self, alert: Dict[str, Any]) -> None:
        try:
            inserted = self.process_alert(alert)
        except ValueError:
            logger.warning("Dropping alert that failed schema validation", exc_info=True)
            return
        if inserted and self._loop is not None:
            asyncio.run_coroutine_threadsafe(self.broadcaster.broadcast(alert), self._loop)
            self.broadcast_scheduled += 1

    def stop(self) -> None:
        """Signals the background thread to stop and closes the source.

        Known limitation: if the underlying Kafka consumer is blocked
        waiting on the next record (no `consumer_timeout_ms` set), the
        `_stop_event` check inside `_run` isn't re-evaluated until the next
        record arrives. Calling `source.close()` here is expected to
        unblock a real KafkaConsumer, but kafka-python does not document
        `close()` from another thread while `__iter__` is blocked as
        officially thread-safe, so shutdown timing on the live path isn't
        guaranteed to be immediate -- see the README's known-limitations
        section.
        """
        self._stop_event.set()
        self.source.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
