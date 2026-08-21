"""REST endpoints: historical alert query/filter, triage, and summary stats.

All routes are gated by the Basic-auth dependency the caller passes into
`build_router` (see auth.py and app.py) -- built as a function rather than a
module-level router so tests and `create_app` can supply the concrete,
settings-bound auth dependency instead of relying on global state.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .store import TRIAGE_STATUSES, AlertStore


class TriageUpdate(BaseModel):
    status: str = Field(..., description=f"One of {TRIAGE_STATUSES}")
    note: Optional[str] = None


class WsTokenResponse(BaseModel):
    token: str
    expires_in: float


def _store(request: Request) -> AlertStore:
    return request.app.state.store


def build_router(get_auth_user) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(get_auth_user)])

    @router.get("/alerts")
    def list_alerts(
        request: Request,
        severity: Optional[str] = Query(None),
        attack_type: Optional[str] = Query(None, description="stage2_predicted_class, e.g. 'DoS/DDoS'"),
        start_time: Optional[float] = Query(None, description="Unix epoch seconds, inclusive"),
        end_time: Optional[float] = Query(None, description="Unix epoch seconds, inclusive"),
        triage_status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        store = _store(request)
        alerts = store.list_alerts(
            severity=severity,
            attack_type=attack_type,
            start_time=start_time,
            end_time=end_time,
            triage_status=triage_status,
            limit=limit,
            offset=offset,
        )
        total = store.count_alerts(
            severity=severity,
            attack_type=attack_type,
            start_time=start_time,
            end_time=end_time,
            triage_status=triage_status,
        )
        return {"alerts": alerts, "total": total, "limit": limit, "offset": offset}

    @router.get("/alerts/summary")
    def summary(
        request: Request,
        start_time: Optional[float] = Query(None),
        end_time: Optional[float] = Query(None),
    ):
        return _store(request).summary(start_time=start_time, end_time=end_time)

    @router.get("/alerts/{alert_id}")
    def get_alert(alert_id: str, request: Request):
        alert = _store(request).get_alert(alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="alert not found")
        return alert

    @router.patch("/alerts/{alert_id}/triage")
    def set_triage(alert_id: str, body: TriageUpdate, request: Request):
        if body.status not in TRIAGE_STATUSES:
            raise HTTPException(status_code=422, detail=f"status must be one of {TRIAGE_STATUSES}")
        updated = _store(request).set_triage(alert_id, body.status, body.note)
        if not updated:
            raise HTTPException(status_code=404, detail="alert not found")
        return _store(request).get_alert(alert_id)

    @router.post("/ws-token", response_model=WsTokenResponse)
    def issue_ws_token(request: Request):
        token_store = request.app.state.token_store
        token = token_store.issue()
        return WsTokenResponse(token=token, expires_in=token_store.ttl_seconds)

    return router
