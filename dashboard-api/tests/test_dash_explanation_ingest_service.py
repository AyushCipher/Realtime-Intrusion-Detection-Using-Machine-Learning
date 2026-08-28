import asyncio

import pytest

from ids_dashboard.broadcaster import AlertBroadcaster
from ids_dashboard.explanation_consumer import StubExplanationEventSource
from ids_dashboard.explanation_ingest_service import ExplanationIngestService
from ids_dashboard.store import AlertStore


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


def test_process_explanation_persists_valid_events():
    service = ExplanationIngestService(StubExplanationEventSource([]), AlertStore(":memory:"), AlertBroadcaster())
    assert service.process_explanation(_valid_explanation("e1")) is True
    assert service.processed == 1
    assert service.store.get_explanation_for_alert("a1") is not None


def test_process_explanation_rejects_invalid_events():
    service = ExplanationIngestService(StubExplanationEventSource([]), AlertStore(":memory:"), AlertBroadcaster())
    bad = _valid_explanation("e1")
    del bad["risk_explanation"]
    with pytest.raises(ValueError):
        service.process_explanation(bad)


def test_process_explanation_returns_false_for_duplicate():
    service = ExplanationIngestService(StubExplanationEventSource([]), AlertStore(":memory:"), AlertBroadcaster())
    explanation = _valid_explanation("e1")
    assert service.process_explanation(explanation) is True
    assert service.process_explanation(explanation) is False
    assert service.processed == 2  # both attempts count as "processed"


def test_start_stop_consumes_stub_source_and_broadcasts_with_type_marker():
    async def run():
        events = [_valid_explanation("e1", "a1"), _valid_explanation("e2", "a2")]
        store = AlertStore(":memory:")
        broadcaster = AlertBroadcaster()

        broadcasts = []
        orig_broadcast = broadcaster.broadcast

        async def spy_broadcast(message):
            broadcasts.append(message)
            await orig_broadcast(message)

        broadcaster.broadcast = spy_broadcast  # type: ignore[method-assign]

        service = ExplanationIngestService(StubExplanationEventSource(events), store, broadcaster)
        loop = asyncio.get_running_loop()
        service.start(loop)

        for _ in range(50):
            if len(broadcasts) == 2:
                break
            await asyncio.sleep(0.02)

        service.stop()

        assert service.processed == 2
        assert service.broadcast_scheduled == 2
        assert all(b["__type"] == "explanation" for b in broadcasts)
        assert {b["explanation_id"] for b in broadcasts} == {"e1", "e2"}

    asyncio.run(run())
