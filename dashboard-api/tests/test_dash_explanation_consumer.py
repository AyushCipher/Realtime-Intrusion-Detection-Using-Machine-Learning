import pytest

from ids_dashboard.explanation_consumer import StubExplanationEventSource


def _valid_explanation(explanation_id="e1", alert_id="a1"):
    return {
        "explanation_id": explanation_id,
        "alert_id": alert_id,
        "flow_id": "flow-1",
        "generated_at": 1_700_000_010.0,
        "suspected_technique_id": "T1110",
        "suspected_technique_name": "Brute Force",
        "risk_explanation": "text",
        "recommended_action": "text",
        "retrieved_technique_ids": ["T1110"],
        "rag_enabled": True,
        "llm_latency_ms": 5750.0,
        "model_version": "gemini-3.6-flash",
        "schema_version": 1,
    }


def test_stub_source_yields_validated_events():
    events = [_valid_explanation("e1"), _valid_explanation("e2")]
    source = StubExplanationEventSource(events)
    assert list(source) == events


def test_stub_source_raises_on_invalid_event():
    bad = _valid_explanation("e1")
    del bad["risk_explanation"]
    source = StubExplanationEventSource([bad])
    with pytest.raises(ValueError):
        list(source)


def test_stub_source_close_is_a_no_op():
    source = StubExplanationEventSource([])
    source.close()  # must not raise
