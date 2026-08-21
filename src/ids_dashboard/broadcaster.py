"""In-process fan-out of live alerts to connected WebSocket clients.

Deliberately simple: an in-memory set of connections on a single process.
Multiple dashboard API replicas would each have their own independent set
and would not see each other's clients -- see the README's known-limitations
section.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)


class AlertBroadcaster:
    def __init__(self) -> None:
        self._clients: Set[Any] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)

    async def disconnect(self, websocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, alert: Dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)

        stale = []
        for ws in clients:
            try:
                await ws.send_json(alert)
            except Exception:  # noqa: BLE001 - any send failure means a dead/closing client
                stale.append(ws)

        if stale:
            async with self._lock:
                for ws in stale:
                    self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)
