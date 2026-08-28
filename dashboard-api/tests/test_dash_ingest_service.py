import asyncio

import pytest

from ids_dashboard.alert_consumer import StubAlertEventSource
from ids_dashboard.broadcaster import AlertBroadcaster
from ids_dashboard.ingest_service import IngestService
from ids_dashboard.store import AlertStore


def _valid_alert(alert_id="a1"):
    return {
        "alert_id": alert_id,
        "flow_id": "flow-1",
        "src_ip": "10.0.0.1",
        "src_port": 1234,
        "dst_ip": "10.0.0.2",
        "dst_port": 443,
        "protocol": 6,
        "flow_start_time": 0.0,
        "scored_at": 1.0,
        "stage1_anomaly_score": 0.5,
        "stage1_flagged": True,
        "stage2_predicted_class": "DoS/DDoS",
        "stage2_confidence": 0.9,
        "stage2_class_probabilities": {"DoS/DDoS": 0.9, "BENIGN": 0.1},
        "severity": "high",
        "explanation": [],
        "unknown_mass": 0.0,
        "escalated": False,
        "escalation_trigger": "",
        "model_version": "two-stage-v1",
        "schema_version": 1,
    }


def test_process_alert_persists_valid_events():
    service = IngestService(StubAlertEventSource([]), AlertStore(":memory:"), AlertBroadcaster())
    assert service.process_alert(_valid_alert("a1")) is True
    assert service.processed == 1
    assert service.store.get_alert("a1") is not None


def test_process_alert_rejects_invalid_events():
    service = IngestService(StubAlertEventSource([]), AlertStore(":memory:"), AlertBroadcaster())
    bad = _valid_alert("a1")
    del bad["severity"]
    with pytest.raises(ValueError):
        service.process_alert(bad)


def test_process_alert_returns_false_for_duplicate():
    service = IngestService(StubAlertEventSource([]), AlertStore(":memory:"), AlertBroadcaster())
    alert = _valid_alert("a1")
    assert service.process_alert(alert) is True
    assert service.process_alert(alert) is False
    assert service.processed == 2  # both attempts count as "processed"


def test_start_stop_consumes_stub_source_and_broadcasts():
    async def run():
        events = [_valid_alert("a1"), _valid_alert("a2")]
        store = AlertStore(":memory:")
        broadcaster = AlertBroadcaster()
        service = IngestService(StubAlertEventSource(events), store, broadcaster)

        loop = asyncio.get_running_loop()
        service.start(loop)

        for _ in range(50):  # poll for the background thread to finish, up to ~1s
            if store.count_alerts() == 2:
                break
            await asyncio.sleep(0.02)

        service.stop()

        assert store.count_alerts() == 2
        assert service.processed == 2
        assert service.broadcast_scheduled == 2

    asyncio.run(run())
