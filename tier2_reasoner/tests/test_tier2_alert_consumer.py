import pytest

from ids_tier2.alert_consumer import StubEscalatedAlertSource

_ESCALATED = {
    "alert_id": "a-1", "flow_id": "f-1", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
    "dst_port": 443, "protocol": 6, "stage2_predicted_class": "Brute Force",
    "stage2_confidence": 0.7, "severity": "medium", "explanation": [],
    "unknown_mass": 0.6, "escalated": True, "escalation_trigger": "openset",
}
_NOT_ESCALATED = {**_ESCALATED, "alert_id": "a-2", "escalated": False}


def test_stub_source_yields_only_escalated_alerts():
    source = StubEscalatedAlertSource([_ESCALATED, _NOT_ESCALATED])
    results = list(source)
    assert len(results) == 1
    assert results[0]["alert_id"] == "a-1"


def test_stub_source_yields_nothing_when_none_escalated():
    source = StubEscalatedAlertSource([_NOT_ESCALATED])
    assert list(source) == []


def test_stub_source_validates_each_alert():
    bad = dict(_ESCALATED)
    del bad["escalated"]
    source = StubEscalatedAlertSource([bad])
    with pytest.raises(ValueError):
        list(source)


def test_stub_source_close_is_a_noop():
    source = StubEscalatedAlertSource([_ESCALATED])
    source.close()  # should not raise
