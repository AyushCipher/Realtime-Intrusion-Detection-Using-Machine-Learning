"""The live alert WebSocket endpoint.

Browsers cannot set an `Authorization` header on a WebSocket handshake, so
this endpoint is gated by a short-lived token issued via the Basic-auth-
protected `POST /api/ws-token` REST endpoint instead (see auth.py and
routes_alerts.py) rather than left unauthenticated.
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect


def build_router(validate_token: Callable[[str], bool]) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/alerts")
    async def alerts_ws(websocket: WebSocket, token: str = Query(...)) -> None:
        if not validate_token(token):
            await websocket.close(code=1008)  # policy violation; closed before accept()
            return

        broadcaster = websocket.app.state.broadcaster
        await broadcaster.connect(websocket)
        try:
            while True:
                # No client->server protocol is defined; we only read to
                # notice a disconnect promptly rather than leak the socket.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await broadcaster.disconnect(websocket)

    return router
