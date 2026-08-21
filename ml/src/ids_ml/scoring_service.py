"""Ties the flow-event source, two-stage detector, SHAP explanations, and
alert producer together into the live scoring service.

This is the module's Kafka-facing entrypoint: consume `schema.FLOW_TOPIC`,
score each event, and publish `schema.ALERT_TOPIC` events for anything
worth a human's attention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .alert_producer import AlertEventProducer
from .explainability import ShapExplainer, explanation_to_alert_field
from .features import event_to_feature_vector
from .flow_consumer import FlowEventSource
from .pipeline import ScoredFlow, TwoStageDetector, build_alert

logger = logging.getLogger(__name__)


@dataclass
class ScoringServiceConfig:
    explain_top_k: int = 5
    # Publishing an alert for every benign flow would flood the alerts
    # topic with nothing for the dashboard to act on, so by default only
    # non-BENIGN stage-2 predictions are published. Set True to also
    # publish "stage 1 flagged it, but stage 2 says BENIGN" events -- useful
    # for measuring the pre-filter's false-positive rate from the dashboard
    # side, at the cost of a noisier topic.
    alert_on_stage1_flag_only: bool = False


class ScoringService:
    def __init__(
        self,
        source: FlowEventSource,
        detector: TwoStageDetector,
        producer: AlertEventProducer,
        explainer: Optional[ShapExplainer] = None,
        config: Optional[ScoringServiceConfig] = None,
    ) -> None:
        self.source = source
        self.detector = detector
        self.producer = producer
        self.explainer = explainer
        self.config = config or ScoringServiceConfig()
        self.processed = 0
        self.alerts_published = 0

    def _should_alert(self, scored: ScoredFlow) -> bool:
        if scored.stage2_predicted_class != "BENIGN":
            return True
        return self.config.alert_on_stage1_flag_only and scored.stage1_flagged

    def run(self) -> None:
        try:
            for event in self.source:
                self.process_event(event)
        finally:
            self.source.close()
            self.producer.close()

    def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scores one flow event; publishes and returns an alert if warranted,
        else returns None. Exposed separately from run() so tests and the
        stub-source path can drive it one event at a time."""
        self.processed += 1
        X = event_to_feature_vector(event).reshape(1, -1)
        scored = self.detector.score(X)[0]

        if not self._should_alert(scored):
            return None

        explanation: List[Dict[str, float]] = []
        if self.explainer is not None and scored.stage2_ran:
            contributions = self.explainer.explain(X[0], scored.stage2_predicted_class, top_k=self.config.explain_top_k)
            explanation = explanation_to_alert_field(contributions)

        alert = build_alert(event, scored, explanation=explanation)
        self.producer.publish(alert)
        self.alerts_published += 1
        return alert
