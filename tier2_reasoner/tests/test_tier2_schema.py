import pytest

from ids_tier2.schema import (
    EXPLANATION_EVENT_FIELDS,
    event_from_json,
    event_to_json,
    validate_alert_event,
    validate_explanation_event,
)

_VALID_ALERT = {
    "alert_id": "a-1",
    "flow_id": "f-1",
    "src_ip": "10.0.0.1",
    "dst_ip": "10.0.0.2",
    "dst_port": 443,
    "protocol": 6,
    "stage2_predicted_class": "Brute Force",
    "stage2_confidence": 0.7,
    "severity": "medium",
    "explanation": [],
    "unknown_mass": 0.4,
    "escalated": True,
    "escalation_trigger": "openset",
}


def _valid_explanation():
    return {
        "explanation_id": "e-1",
        "alert_id": "a-1",
        "flow_id": "f-1",
        "generated_at": 1_700_000_000.0,
        "suspected_technique_id": "T1110",
        "suspected_technique_name": "Brute Force",
        "risk_explanation": "text",
        "recommended_action": "text",
        "retrieved_technique_ids": ["T1110"],
        "rag_enabled": True,
        "llm_latency_ms": 12.3,
        "model_version": "stub-v1",
        "schema_version": 1,
    }


def test_validate_alert_event_accepts_valid_alert():
    validate_alert_event(_VALID_ALERT)  # should not raise


def test_validate_alert_event_rejects_missing_fields():
    bad = dict(_VALID_ALERT)
    del bad["escalated"]
    with pytest.raises(ValueError):
        validate_alert_event(bad)


def test_validate_explanation_event_accepts_valid_event():
    validate_explanation_event(_valid_explanation())  # should not raise


def test_validate_explanation_event_rejects_missing_field():
    bad = _valid_explanation()
    del bad["risk_explanation"]
    with pytest.raises(ValueError):
        validate_explanation_event(bad)


def test_validate_explanation_event_rejects_wrong_type():
    bad = _valid_explanation()
    bad["rag_enabled"] = "yes"  # should be bool
    with pytest.raises(ValueError):
        validate_explanation_event(bad)


def test_explanation_event_fields_are_all_type_checked():
    # every declared field must actually appear in a valid example, so a
    # future field addition to EXPLANATION_EVENT_FIELDS can't silently
    # skip validation coverage
    example = _valid_explanation()
    assert set(EXPLANATION_EVENT_FIELDS) == set(example)


def test_event_json_round_trip():
    payload = event_to_json(_valid_explanation())
    assert event_from_json(payload) == _valid_explanation()
