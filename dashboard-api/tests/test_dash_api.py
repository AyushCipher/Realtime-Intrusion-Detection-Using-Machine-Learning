"""Tests for the FastAPI REST + WebSocket surface, using TestClient with an
injected stub alert source/store -- no live Kafka broker needed, matching
the pattern used throughout this project for testing Kafka-facing modules.
"""

import queue as queue_module
import time

import pytest
from fastapi.testclient import TestClient

from ids_dashboard.alert_consumer import AlertEventSource, StubAlertEventSource
from ids_dashboard.app import create_app
from ids_dashboard.auth import AuthSettings
from ids_dashboard.config import Settings
from ids_dashboard.explanation_consumer import ExplanationEventSource, StubExplanationEventSource
from ids_dashboard.store import AlertStore


def _valid_alert(
    alert_id="a1",
    severity="high",
    predicted_class="DoS/DDoS",
    scored_at=1_700_000_000.0,
    escalated=False,
    escalation_trigger="",
    unknown_mass=0.0,
):
    return {
        "alert_id": alert_id,
        "flow_id": f"flow-{alert_id}",
        "src_ip": "10.0.0.1",
        "src_port": 1234,
        "dst_ip": "10.0.0.2",
        "dst_port": 443,
        "protocol": 6,
        "flow_start_time": scored_at - 1.0,
        "scored_at": scored_at,
        "stage1_anomaly_score": 0.8,
        "stage1_flagged": True,
        "stage2_predicted_class": predicted_class,
        "stage2_confidence": 0.9,
        "stage2_class_probabilities": {predicted_class: 0.9},
        "severity": severity,
        "explanation": [{"feature": "flow_duration", "value": 1.0, "shap_value": 0.4}],
        "unknown_mass": unknown_mass,
        "escalated": escalated,
        "escalation_trigger": escalation_trigger,
        "model_version": "two-stage-v1",
        "schema_version": 1,
    }


def _valid_explanation(explanation_id="e1", alert_id="a1"):
    return {
        "explanation_id": explanation_id,
        "alert_id": alert_id,
        "flow_id": f"flow-{alert_id}",
        "generated_at": 1_700_000_010.0,
        "suspected_technique_id": "T1110",
        "suspected_technique_name": "Brute Force",
        "risk_explanation": "Traffic pattern matches brute-force indicators.",
        "recommended_action": "Block the source IP.",
        "retrieved_technique_ids": ["T1110"],
        "rag_enabled": True,
        "llm_latency_ms": 5750.0,
        "model_version": "gemini-3.6-flash",
        "schema_version": 1,
    }


AUTH = ("tester", "secret123")


def _app_with_alerts(alerts, explanations=()):
    store = AlertStore(":memory:")
    app = create_app(
        settings=Settings(use_stub_source=True, db_path=":memory:"),
        auth_settings=AuthSettings(username=AUTH[0], password=AUTH[1]),
        source=StubAlertEventSource(alerts),
        store=store,
        explanation_source=StubExplanationEventSource(explanations),
    )
    return app, store


def _wait_for(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_healthz_does_not_require_auth():
    app, _ = _app_with_alerts([])
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_alerts_endpoint_requires_auth():
    app, _ = _app_with_alerts([])
    with TestClient(app) as client:
        resp = client.get("/api/alerts")
        assert resp.status_code == 401


def test_alerts_endpoint_rejects_wrong_credentials():
    app, _ = _app_with_alerts([])
    with TestClient(app) as client:
        resp = client.get("/api/alerts", auth=("tester", "wrong-password"))
        assert resp.status_code == 401


def test_list_alerts_returns_ingested_events():
    alerts = [_valid_alert("a1", severity="high"), _valid_alert("a2", severity="low")]
    app, store = _app_with_alerts(alerts)
    with TestClient(app) as client:
        assert _wait_for(lambda: store.count_alerts() == 2)

        resp = client.get("/api/alerts", auth=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert {a["alert_id"] for a in body["alerts"]} == {"a1", "a2"}


def test_list_alerts_filters_by_severity():
    alerts = [_valid_alert("a1", severity="high"), _valid_alert("a2", severity="low")]
    app, store = _app_with_alerts(alerts)
    with TestClient(app) as client:
        assert _wait_for(lambda: store.count_alerts() == 2)

        resp = client.get("/api/alerts", params={"severity": "high"}, auth=AUTH)
        body = resp.json()
        assert [a["alert_id"] for a in body["alerts"]] == ["a1"]


def test_list_alerts_filters_by_escalated():
    alerts = [
        _valid_alert("a1", escalated=True, escalation_trigger="openset", unknown_mass=0.63),
        _valid_alert("a2", escalated=False),
    ]
    app, store = _app_with_alerts(alerts)
    with TestClient(app) as client:
        assert _wait_for(lambda: store.count_alerts() == 2)

        resp = client.get("/api/alerts", params={"escalated": "true"}, auth=AUTH)
        body = resp.json()
        assert [a["alert_id"] for a in body["alerts"]] == ["a1"]
        assert body["alerts"][0]["unknown_mass"] == pytest.approx(0.63)
        assert body["alerts"][0]["escalation_trigger"] == "openset"


def test_get_explanation_404_when_alert_does_not_exist():
    app, _ = _app_with_alerts([])
    with TestClient(app) as client:
        resp = client.get("/api/alerts/nope/explanation", auth=AUTH)
        assert resp.status_code == 404


def test_get_explanation_404_when_alert_exists_but_not_yet_explained():
    app, store = _app_with_alerts([_valid_alert("a1", escalated=True)])
    with TestClient(app) as client:
        assert _wait_for(lambda: store.count_alerts() == 1)
        resp = client.get("/api/alerts/a1/explanation", auth=AUTH)
        assert resp.status_code == 404


def test_get_explanation_returns_it_once_ingested():
    app, store = _app_with_alerts(
        [_valid_alert("a1", escalated=True, escalation_trigger="openset")],
        explanations=[_valid_explanation("e1", "a1")],
    )
    with TestClient(app) as client:
        assert _wait_for(lambda: store.get_explanation_for_alert("a1") is not None)

        resp = client.get("/api/alerts/a1/explanation", auth=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["suspected_technique_id"] == "T1110"
        assert body["recommended_action"] == "Block the source IP."


def test_get_explanation_requires_auth():
    app, _ = _app_with_alerts([])
    with TestClient(app) as client:
        resp = client.get("/api/alerts/nope/explanation")
        assert resp.status_code == 401


def test_get_alert_by_id_and_404():
    app, store = _app_with_alerts([_valid_alert("a1")])
    with TestClient(app) as client:
        assert _wait_for(lambda: store.count_alerts() == 1)

        found = client.get("/api/alerts/a1", auth=AUTH)
        assert found.status_code == 200
        assert found.json()["alert_id"] == "a1"

        missing = client.get("/api/alerts/does-not-exist", auth=AUTH)
        assert missing.status_code == 404


def test_triage_update_success_and_validation():
    app, store = _app_with_alerts([_valid_alert("a1")])
    with TestClient(app) as client:
        assert _wait_for(lambda: store.count_alerts() == 1)

        ok = client.patch("/api/alerts/a1/triage", json={"status": "confirmed", "note": "looked legit"}, auth=AUTH)
        assert ok.status_code == 200
        assert ok.json()["triage_status"] == "confirmed"
        assert ok.json()["triage_note"] == "looked legit"

        bad_status = client.patch("/api/alerts/a1/triage", json={"status": "not_a_status"}, auth=AUTH)
        assert bad_status.status_code == 422

        missing = client.patch("/api/alerts/nope/triage", json={"status": "confirmed"}, auth=AUTH)
        assert missing.status_code == 404


def test_summary_endpoint():
    alerts = [_valid_alert("a1", predicted_class="DoS/DDoS"), _valid_alert("a2", predicted_class="BENIGN", severity="info")]
    app, store = _app_with_alerts(alerts)
    with TestClient(app) as client:
        assert _wait_for(lambda: store.count_alerts() == 2)

        resp = client.get("/api/alerts/summary", auth=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_alerts"] == 2
        assert body["stage1_proxy_false_positive"]["available"] is True


def test_ws_token_endpoint_requires_auth_and_issues_token():
    app, _ = _app_with_alerts([])
    with TestClient(app) as client:
        unauthed = client.post("/api/ws-token")
        assert unauthed.status_code == 401

        resp = client.post("/api/ws-token", auth=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert "token" in body and len(body["token"]) > 10
        assert body["expires_in"] > 0


def test_websocket_rejects_invalid_token():
    app, _ = _app_with_alerts([])
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/ws/alerts?token=not-a-real-token"):
                raise AssertionError("expected the connection to be rejected")
        except Exception:
            pass  # rejection surfaces as a WebSocket/connection error, not a clean connect


class _QueueAlertEventSource(AlertEventSource):
    """Test-only source: blocks for alerts pushed after the app has started,
    so a test can simulate a live alert arriving while a WebSocket client is
    already connected -- exercising the real background-thread -> store ->
    broadcast -> WebSocket path end to end."""

    def __init__(self) -> None:
        self._queue: "queue_module.Queue" = queue_module.Queue()

    def push(self, alert) -> None:
        self._queue.put(alert)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            yield item

    def close(self) -> None:
        self._queue.put(None)


def test_websocket_receives_live_broadcast_alert():
    source = _QueueAlertEventSource()
    store = AlertStore(":memory:")
    app = create_app(
        settings=Settings(use_stub_source=True, db_path=":memory:"),
        auth_settings=AuthSettings(username=AUTH[0], password=AUTH[1]),
        source=source,
        store=store,
    )
    with TestClient(app) as client:
        token = client.post("/api/ws-token", auth=AUTH).json()["token"]

        with client.websocket_connect(f"/ws/alerts?token={token}") as ws:
            source.push(_valid_alert("live-1"))
            received = ws.receive_json()
            assert received["alert_id"] == "live-1"

        assert _wait_for(lambda: store.count_alerts() == 1)


class _QueueExplanationEventSource(ExplanationEventSource):
    """Explanation-side counterpart to _QueueAlertEventSource above."""

    def __init__(self) -> None:
        self._queue: "queue_module.Queue" = queue_module.Queue()

    def push(self, explanation) -> None:
        self._queue.put(explanation)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            yield item

    def close(self) -> None:
        self._queue.put(None)


def test_websocket_receives_live_broadcast_explanation_with_type_marker():
    alert_source = _QueueAlertEventSource()
    explanation_source = _QueueExplanationEventSource()
    store = AlertStore(":memory:")
    app = create_app(
        settings=Settings(use_stub_source=True, db_path=":memory:"),
        auth_settings=AuthSettings(username=AUTH[0], password=AUTH[1]),
        source=alert_source,
        store=store,
        explanation_source=explanation_source,
    )
    with TestClient(app) as client:
        token = client.post("/api/ws-token", auth=AUTH).json()["token"]

        with client.websocket_connect(f"/ws/alerts?token={token}") as ws:
            alert_source.push(_valid_alert("live-1", escalated=True, escalation_trigger="openset"))
            alert_msg = ws.receive_json()
            assert "__type" not in alert_msg  # unmarked -- the pre-existing, unchanged alert broadcast shape

            explanation_source.push(_valid_explanation("live-e1", "live-1"))
            explanation_msg = ws.receive_json()
            assert explanation_msg["__type"] == "explanation"
            assert explanation_msg["explanation_id"] == "live-e1"
            assert explanation_msg["alert_id"] == "live-1"

        assert _wait_for(lambda: store.get_explanation_for_alert("live-1") is not None)
