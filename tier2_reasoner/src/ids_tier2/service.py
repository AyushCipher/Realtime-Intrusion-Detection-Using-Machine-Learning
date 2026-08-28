"""Ties the escalated-alert source, reasoner, and explanation producer
together into the live Tier 2 service -- this module's Kafka-facing
entrypoint: consume escalated alerts, reason over each one, publish
`schema.EXPLANATION_TOPIC` events.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from .alert_consumer import EscalatedAlertSource
from .explanation_producer import ExplanationProducer
from .reasoner import Tier2Reasoner
from .schema import EXPLANATION_SCHEMA_VERSION

logger = logging.getLogger(__name__)


def build_explanation_event(alert: Dict[str, Any], explanation) -> Dict[str, Any]:
    return {
        "explanation_id": str(uuid.uuid4()),
        "alert_id": alert.get("alert_id", ""),
        "flow_id": alert.get("flow_id", ""),
        "generated_at": time.time(),
        "suspected_technique_id": explanation.suspected_technique_id,
        "suspected_technique_name": explanation.suspected_technique_name,
        "risk_explanation": explanation.risk_explanation,
        "recommended_action": explanation.recommended_action,
        "retrieved_technique_ids": explanation.retrieved_technique_ids,
        "rag_enabled": explanation.rag_enabled,
        "llm_latency_ms": explanation.llm_latency_ms,
        "model_version": explanation.model_version,
        "schema_version": EXPLANATION_SCHEMA_VERSION,
    }


class Tier2Service:
    def __init__(self, source: EscalatedAlertSource, reasoner: Tier2Reasoner, producer: ExplanationProducer) -> None:
        self.source = source
        self.reasoner = reasoner
        self.producer = producer
        self.processed = 0
        self.explanations_published = 0

    def run(self) -> None:
        try:
            for alert in self.source:
                self.process_alert(alert)
        finally:
            self.source.close()
            self.producer.close()

    def process_alert(self, alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reasons over one escalated alert; publishes and returns the
        resulting explanation. Exposed separately from run() so tests and
        the stub-source path can drive it one alert at a time."""
        self.processed += 1
        try:
            explanation = self.reasoner.explain(alert)
        except Exception:
            logger.exception("Tier 2 reasoning failed for alert %s; skipping", alert.get("alert_id"))
            return None

        event = build_explanation_event(alert, explanation)
        self.producer.publish(event)
        self.explanations_published += 1
        return event
