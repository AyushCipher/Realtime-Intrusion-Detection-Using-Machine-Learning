import pytest

from ids_dashboard.alert_consumer import StubAlertEventSource


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


def test_stub_source_yields_validated_events():
    events = [_valid_alert("a1"), _valid_alert("a2")]
    source = StubAlertEventSource(events)
    assert list(source) == events


def test_stub_source_raises_on_invalid_event():
    bad = _valid_alert("a1")
    del bad["severity"]
    source = StubAlertEventSource([bad])
    with pytest.raises(ValueError):
        list(source)


def test_stub_source_close_is_a_no_op():
    source = StubAlertEventSource([])
    source.close()  # must not raise
