import pytest

from ids_dashboard.store import AlertStore


def _make_alert(alert_id, severity="high", predicted_class="DoS/DDoS", scored_at=1_700_000_000.0, stage1_flagged=True):
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
        "stage1_anomaly_score": 0.9,
        "stage1_flagged": stage1_flagged,
        "stage2_predicted_class": predicted_class,
        "stage2_confidence": 0.95,
        "stage2_class_probabilities": {predicted_class: 0.95, "BENIGN": 0.05},
        "severity": severity,
        "explanation": [{"feature": "flow_duration", "value": 0.1, "shap_value": 0.5}],
        "model_version": "two-stage-v1",
        "schema_version": 1,
    }


def _store():
    return AlertStore(":memory:")


def test_insert_and_get_round_trip():
    store = _store()
    alert = _make_alert("a1")
    assert store.insert_alert(alert) is True

    fetched = store.get_alert("a1")
    assert fetched["alert_id"] == "a1"
    assert fetched["severity"] == "high"
    assert fetched["stage1_flagged"] is True
    assert fetched["stage2_class_probabilities"] == {"DoS/DDoS": 0.95, "BENIGN": 0.05}
    assert fetched["explanation"] == [{"feature": "flow_duration", "value": 0.1, "shap_value": 0.5}]
    assert "received_at" in fetched
    assert fetched["triage_status"] == "new"


def test_get_missing_alert_returns_none():
    store = _store()
    assert store.get_alert("does-not-exist") is None


def test_duplicate_insert_is_ignored():
    store = _store()
    alert = _make_alert("dup")
    assert store.insert_alert(alert) is True
    assert store.insert_alert(alert) is False
    assert store.count_alerts() == 1


def test_list_alerts_filters_by_severity_and_attack_type():
    store = _store()
    store.insert_alert(_make_alert("a1", severity="high", predicted_class="DoS/DDoS"))
    store.insert_alert(_make_alert("a2", severity="low", predicted_class="PortScan"))
    store.insert_alert(_make_alert("a3", severity="high", predicted_class="PortScan"))

    high = store.list_alerts(severity="high")
    assert {a["alert_id"] for a in high} == {"a1", "a3"}

    portscan = store.list_alerts(attack_type="PortScan")
    assert {a["alert_id"] for a in portscan} == {"a2", "a3"}

    high_portscan = store.list_alerts(severity="high", attack_type="PortScan")
    assert {a["alert_id"] for a in high_portscan} == {"a3"}


def test_list_alerts_filters_by_time_range_and_orders_newest_first():
    store = _store()
    store.insert_alert(_make_alert("old", scored_at=1000.0))
    store.insert_alert(_make_alert("mid", scored_at=2000.0))
    store.insert_alert(_make_alert("new", scored_at=3000.0))

    windowed = store.list_alerts(start_time=1500.0, end_time=2500.0)
    assert [a["alert_id"] for a in windowed] == ["mid"]

    all_alerts = store.list_alerts()
    assert [a["alert_id"] for a in all_alerts] == ["new", "mid", "old"]


def test_list_alerts_pagination():
    store = _store()
    for i in range(5):
        store.insert_alert(_make_alert(f"a{i}", scored_at=1000.0 + i))

    page1 = store.list_alerts(limit=2, offset=0)
    page2 = store.list_alerts(limit=2, offset=2)
    assert [a["alert_id"] for a in page1] == ["a4", "a3"]
    assert [a["alert_id"] for a in page2] == ["a2", "a1"]
    assert store.count_alerts() == 5


def test_set_triage_updates_status_and_note():
    store = _store()
    store.insert_alert(_make_alert("a1"))

    assert store.set_triage("a1", "confirmed", note="verified via siem") is True
    fetched = store.get_alert("a1")
    assert fetched["triage_status"] == "confirmed"
    assert fetched["triage_note"] == "verified via siem"
    assert fetched["triage_updated_at"] is not None


def test_set_triage_missing_alert_returns_false():
    store = _store()
    assert store.set_triage("nope", "confirmed") is False


def test_set_triage_invalid_status_raises():
    store = _store()
    store.insert_alert(_make_alert("a1"))
    with pytest.raises(ValueError):
        store.set_triage("a1", "not_a_real_status")


def test_summary_basic_counts():
    store = _store()
    store.insert_alert(_make_alert("a1", severity="high", predicted_class="DoS/DDoS", scored_at=1_700_000_000.0))
    store.insert_alert(_make_alert("a2", severity="critical", predicted_class="Heartbleed", scored_at=1_700_003_600.0))
    store.insert_alert(_make_alert("a3", severity="high", predicted_class="DoS/DDoS", scored_at=1_700_086_400.0))

    summary = store.summary()
    assert summary["total_alerts"] == 3
    assert summary["by_severity"] == {"high": 2, "critical": 1}
    assert summary["by_attack_type"] == {"DoS/DDoS": 2, "Heartbleed": 1}
    assert len(summary["volume_by_day"]) == 2  # two distinct calendar days


def test_summary_stage1_proxy_false_positive_requires_benign_alerts():
    store = _store()
    store.insert_alert(_make_alert("a1", predicted_class="DoS/DDoS"))

    summary = store.summary()
    assert summary["stage1_proxy_false_positive"]["available"] is False
    assert summary["stage1_proxy_false_positive"]["rate"] is None

    store.insert_alert(_make_alert("a2", severity="info", predicted_class="BENIGN"))
    summary = store.summary()
    assert summary["stage1_proxy_false_positive"]["available"] is True
    assert summary["stage1_proxy_false_positive"]["benign_count"] == 1
    assert summary["stage1_proxy_false_positive"]["rate"] == pytest.approx(1 / 2)


def test_summary_analyst_reviewed_false_positive_rate():
    store = _store()
    for i in range(4):
        store.insert_alert(_make_alert(f"a{i}"))

    summary = store.summary()
    assert summary["analyst_reviewed_false_positive"]["reviewed_count"] == 0
    assert summary["analyst_reviewed_false_positive"]["rate"] is None

    store.set_triage("a0", "false_positive")
    store.set_triage("a1", "confirmed")
    # a2, a3 left as 'new' -- unreviewed, must not count toward the rate

    summary = store.summary()
    reviewed = summary["analyst_reviewed_false_positive"]
    assert reviewed["reviewed_count"] == 2
    assert reviewed["false_positive_count"] == 1
    assert reviewed["rate"] == pytest.approx(0.5)
    assert reviewed["total_count"] == 4
