from ids_tier2.alert_consumer import StubEscalatedAlertSource
from ids_tier2.explanation_producer import StubExplanationProducer
from ids_tier2.llm_client import StubLLMClient
from ids_tier2.reasoner import Tier2Reasoner
from ids_tier2.schema import validate_explanation_event
from ids_tier2.service import Tier2Service

_ESCALATED_ALERT = {
    "alert_id": "a-1", "flow_id": "f-1", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
    "dst_port": 22, "protocol": 6, "stage2_predicted_class": "Brute Force",
    "stage2_confidence": 0.7, "severity": "medium", "explanation": [],
    "unknown_mass": 0.6, "escalated": True, "escalation_trigger": "openset",
}


def _build_service(alerts):
    source = StubEscalatedAlertSource(alerts)
    reasoner = Tier2Reasoner(StubLLMClient(), use_rag=True)
    producer = StubExplanationProducer()
    return Tier2Service(source, reasoner, producer), producer


def test_run_processes_every_escalated_alert_and_publishes_explanations():
    service, producer = _build_service([_ESCALATED_ALERT, {**_ESCALATED_ALERT, "alert_id": "a-2"}])
    service.run()

    assert service.processed == 2
    assert service.explanations_published == 2
    assert len(producer.published) == 2
    for event in producer.published:
        validate_explanation_event(event)


def test_run_skips_non_escalated_alerts_entirely():
    not_escalated = {**_ESCALATED_ALERT, "alert_id": "a-3", "escalated": False}
    service, producer = _build_service([not_escalated])
    service.run()

    assert service.processed == 0  # never reached process_alert -- filtered by the source
    assert producer.published == []


def test_process_alert_joins_back_to_originating_alert_id():
    service, producer = _build_service([])
    event = service.process_alert(_ESCALATED_ALERT)
    assert event["alert_id"] == _ESCALATED_ALERT["alert_id"]
    assert event["flow_id"] == _ESCALATED_ALERT["flow_id"]
    validate_explanation_event(event)


def test_process_alert_returns_none_and_does_not_publish_on_reasoner_failure():
    class _BrokenReasoner:
        def explain(self, alert):
            raise RuntimeError("boom")

    source = StubEscalatedAlertSource([])
    producer = StubExplanationProducer()
    service = Tier2Service(source, _BrokenReasoner(), producer)

    result = service.process_alert(_ESCALATED_ALERT)
    assert result is None
    assert producer.published == []
    assert service.processed == 1
    assert service.explanations_published == 0


def test_source_and_producer_are_closed_after_run():
    service, producer = _build_service([_ESCALATED_ALERT])
    service.run()
    assert producer.closed is True
