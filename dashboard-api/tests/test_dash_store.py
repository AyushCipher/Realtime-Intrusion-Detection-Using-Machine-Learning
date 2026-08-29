import sqlite3

import pytest

from ids_dashboard.store import AlertStore


def _make_alert(
    alert_id,
    severity="high",
    predicted_class="DoS/DDoS",
    scored_at=1_700_000_000.0,
    stage1_flagged=True,
    unknown_mass=0.0,
    escalated=False,
    escalation_trigger="",
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
        "stage1_anomaly_score": 0.9,
        "stage1_flagged": stage1_flagged,
        "stage2_predicted_class": predicted_class,
        "stage2_confidence": 0.95,
        "stage2_class_probabilities": {predicted_class: 0.95, "BENIGN": 0.05},
        "severity": severity,
        "explanation": [{"feature": "flow_duration", "value": 0.1, "shap_value": 0.5}],
        "unknown_mass": unknown_mass,
        "escalated": escalated,
        "escalation_trigger": escalation_trigger,
        "model_version": "two-stage-v1",
        "schema_version": 1,
    }


def _make_explanation(
    explanation_id,
    alert_id,
    generated_at=1_700_000_010.0,
    suspected_technique_id="T1110",
    rag_enabled=True,
):
    return {
        "explanation_id": explanation_id,
        "alert_id": alert_id,
        "flow_id": f"flow-{alert_id}",
        "generated_at": generated_at,
        "suspected_technique_id": suspected_technique_id,
        "suspected_technique_name": "Brute Force",
        "risk_explanation": "Traffic pattern matches brute-force indicators.",
        "recommended_action": "Block the source IP.",
        "retrieved_technique_ids": [suspected_technique_id],
        "rag_enabled": rag_enabled,
        "llm_latency_ms": 5750.0,
        "model_version": "gemini-3.6-flash",
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


# --- Open-set fields (unknown_mass/escalated/escalation_trigger) ----------


def test_insert_and_get_round_trips_open_set_fields():
    store = _store()
    store.insert_alert(_make_alert("a1", unknown_mass=0.63, escalated=True, escalation_trigger="openset"))

    fetched = store.get_alert("a1")
    assert fetched["unknown_mass"] == pytest.approx(0.63)
    assert fetched["escalated"] is True
    assert fetched["escalation_trigger"] == "openset"


def test_insert_defaults_open_set_fields_when_not_escalated():
    store = _store()
    store.insert_alert(_make_alert("a1"))
    fetched = store.get_alert("a1")
    assert fetched["unknown_mass"] == 0.0
    assert fetched["escalated"] is False
    assert fetched["escalation_trigger"] == ""


def test_list_and_count_alerts_filter_by_escalated():
    store = _store()
    store.insert_alert(_make_alert("a1", escalated=True, escalation_trigger="openset"))
    store.insert_alert(_make_alert("a2", escalated=False))
    store.insert_alert(_make_alert("a3", escalated=True, escalation_trigger="softmax"))

    escalated_only = store.list_alerts(escalated=True)
    assert {a["alert_id"] for a in escalated_only} == {"a1", "a3"}
    assert store.count_alerts(escalated=True) == 2

    not_escalated = store.list_alerts(escalated=False)
    assert {a["alert_id"] for a in not_escalated} == {"a2"}
    assert store.count_alerts(escalated=False) == 1

    assert store.count_alerts() == 3  # escalated=None (default): no filter


def test_opening_a_pre_tier2_database_migrates_the_alerts_table(tmp_path):
    # Reproduces a real bug found integrating against a stale docker-compose
    # volume: a database file created before Tier 2 (unknown_mass/escalated/
    # escalation_trigger, and the whole explanations table) existed used to
    # crash AlertStore.__init__ with "no such column: escalated", because
    # CREATE TABLE IF NOT EXISTS is a no-op against an existing table and the
    # idx_alerts_escalated index creation then failed against it.
    db_path = tmp_path / "legacy_alerts.db"
    legacy_conn = sqlite3.connect(str(db_path))
    legacy_conn.executescript(
        """
        CREATE TABLE alerts (
            alert_id TEXT PRIMARY KEY,
            flow_id TEXT NOT NULL,
            src_ip TEXT NOT NULL,
            src_port INTEGER NOT NULL,
            dst_ip TEXT NOT NULL,
            dst_port INTEGER NOT NULL,
            protocol INTEGER NOT NULL,
            flow_start_time REAL NOT NULL,
            scored_at REAL NOT NULL,
            stage1_anomaly_score REAL NOT NULL,
            stage1_flagged INTEGER NOT NULL,
            stage2_predicted_class TEXT NOT NULL,
            stage2_confidence REAL NOT NULL,
            stage2_class_probabilities TEXT NOT NULL,
            severity TEXT NOT NULL,
            explanation TEXT NOT NULL,
            model_version TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            received_at REAL NOT NULL,
            triage_status TEXT NOT NULL DEFAULT 'new',
            triage_note TEXT,
            triage_updated_at REAL
        );
        """
    )
    legacy_conn.execute(
        """
        INSERT INTO alerts (
            alert_id, flow_id, src_ip, src_port, dst_ip, dst_port, protocol,
            flow_start_time, scored_at, stage1_anomaly_score, stage1_flagged,
            stage2_predicted_class, stage2_confidence, stage2_class_probabilities,
            severity, explanation, model_version, schema_version, received_at
        ) VALUES ('pre-existing', 'flow-1', '10.0.0.1', 1234, '10.0.0.2', 443, 6,
            1.0, 2.0, 0.9, 1, 'DoS/DDoS', 0.95, '{}', 'high', '[]', 'v1', 1, 3.0)
        """
    )
    legacy_conn.commit()
    legacy_conn.close()

    store = AlertStore(str(db_path))  # must not raise

    pre_existing = store.get_alert("pre-existing")
    assert pre_existing["unknown_mass"] == 0.0
    assert pre_existing["escalated"] is False
    assert pre_existing["escalation_trigger"] == ""

    store.insert_alert(_make_alert("new-alert", unknown_mass=0.7, escalated=True, escalation_trigger="openset"))
    assert store.count_alerts(escalated=True) == 1


# --- Tier 2 explanations ---------------------------------------------------


def test_insert_and_get_explanation_round_trip():
    store = _store()
    store.insert_alert(_make_alert("a1", escalated=True, escalation_trigger="openset"))
    assert store.insert_explanation(_make_explanation("e1", "a1")) is True

    fetched = store.get_explanation_for_alert("a1")
    assert fetched["explanation_id"] == "e1"
    assert fetched["suspected_technique_id"] == "T1110"
    assert fetched["retrieved_technique_ids"] == ["T1110"]
    assert fetched["rag_enabled"] is True
    assert fetched["llm_latency_ms"] == pytest.approx(5750.0)


def test_get_explanation_for_alert_returns_none_when_absent():
    store = _store()
    store.insert_alert(_make_alert("a1", escalated=True))
    assert store.get_explanation_for_alert("a1") is None


def test_get_explanation_for_alert_returns_the_most_recent():
    store = _store()
    store.insert_alert(_make_alert("a1", escalated=True))
    store.insert_explanation(_make_explanation("e1", "a1", generated_at=1000.0))
    store.insert_explanation(_make_explanation("e2", "a1", generated_at=2000.0, suspected_technique_id="T1046"))

    fetched = store.get_explanation_for_alert("a1")
    assert fetched["explanation_id"] == "e2"
    assert fetched["suspected_technique_id"] == "T1046"


def test_duplicate_explanation_insert_is_ignored():
    store = _store()
    store.insert_alert(_make_alert("a1", escalated=True))
    explanation = _make_explanation("e1", "a1")
    assert store.insert_explanation(explanation) is True
    assert store.insert_explanation(explanation) is False


def test_insert_explanation_does_not_require_the_alert_to_exist_yet():
    # Kafka delivery order between network.ids.alerts and
    # network.ids.explanations isn't guaranteed -- see insert_explanation's
    # docstring.
    store = _store()
    assert store.insert_explanation(_make_explanation("e1", "not-yet-inserted")) is True
    assert store.get_explanation_for_alert("not-yet-inserted")["explanation_id"] == "e1"
