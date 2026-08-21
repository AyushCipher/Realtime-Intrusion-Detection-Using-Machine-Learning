"""Combines stage 1 and stage 2 into the deployed detection cascade.

`TwoStageDetector.score` is the production path: stage 2 only runs on rows
stage 1 flags. Rows stage 1 doesn't flag are reported as BENIGN without ever
touching the (more expensive) stage-2 model -- so the pipeline's overall
recall on attacks is bounded by stage 1's recall. `evaluation.py` separately
runs stage 2 on *all* rows to characterize it as a standalone classifier;
that comparison is what makes stage 1's miss rate visible instead of hidden
inside a single blended number.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .stage1_iforest import AnomalyPreFilter
from .stage2_xgboost import AttackClassifier

MODEL_VERSION = "two-stage-v1"

_SEVERITY_ORDER = ["low", "medium", "high", "critical"]

# Base severity by predicted attack family. Deliberately mirrors
# data.ATTACK_CATEGORY_MAP's families. "Other" (unmapped future labels)
# defaults to "medium" -- unknown is treated as worth a human look, not
# ignored and not auto-escalated to critical.
_BASE_SEVERITY = {
    "Heartbleed": "critical",
    "Infiltration": "critical",
    "DoS/DDoS": "high",
    "Botnet": "high",
    "Brute Force": "medium",
    "Web Attack": "medium",
    "PortScan": "low",
    "Other": "medium",
}


def severity_for(predicted_class: str, confidence: float) -> str:
    """Maps a stage-2 prediction to a dashboard severity level.

    BENIGN is always "info". Otherwise severity starts from the predicted
    family's base level and is downgraded one notch when the classifier
    itself isn't confident (< 0.5), so a low-confidence "critical" guess
    doesn't page someone as loudly as a confident one.
    """
    if predicted_class == "BENIGN":
        return "info"
    base = _BASE_SEVERITY.get(predicted_class, "medium")
    if confidence < 0.5 and base in _SEVERITY_ORDER:
        idx = max(_SEVERITY_ORDER.index(base) - 1, 0)
        return _SEVERITY_ORDER[idx]
    return base


@dataclass
class ScoredFlow:
    stage1_anomaly_score: float
    stage1_flagged: bool
    stage2_ran: bool
    stage2_predicted_class: str
    stage2_confidence: float
    stage2_class_probabilities: Dict[str, float]


class TwoStageDetector:
    """The deployed stage1 -> stage2 cascade."""

    def __init__(self, stage1: AnomalyPreFilter, stage2: AttackClassifier) -> None:
        self.stage1 = stage1
        self.stage2 = stage2

    def score(self, X: np.ndarray) -> List[ScoredFlow]:
        n = X.shape[0]
        stage1_scores = self.stage1.anomaly_score(X)
        stage1_flags = self.stage1.flag(X)

        results: List[Optional[ScoredFlow]] = [None] * n
        flagged_idx = np.where(stage1_flags)[0]

        if len(flagged_idx) > 0:
            proba = self.stage2.predict_proba(X[flagged_idx])
            classes = self.stage2.classes_
            best = proba.argmax(axis=1)
            for pos, row in enumerate(flagged_idx):
                class_probs = {classes[c]: float(proba[pos, c]) for c in range(len(classes))}
                results[row] = ScoredFlow(
                    stage1_anomaly_score=float(stage1_scores[row]),
                    stage1_flagged=True,
                    stage2_ran=True,
                    stage2_predicted_class=str(classes[best[pos]]),
                    stage2_confidence=float(proba[pos, best[pos]]),
                    stage2_class_probabilities=class_probs,
                )

        for row in range(n):
            if results[row] is None:
                results[row] = ScoredFlow(
                    stage1_anomaly_score=float(stage1_scores[row]),
                    stage1_flagged=False,
                    stage2_ran=False,
                    stage2_predicted_class="BENIGN",
                    stage2_confidence=1.0,
                    stage2_class_probabilities={},
                )

        return results  # type: ignore[return-value]


def build_alert(
    event: Dict[str, Any],
    scored: ScoredFlow,
    explanation: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Builds a Kafka alert event (schema.ALERT_EVENT_FIELDS) from a source
    flow event and its ScoredFlow result. `explanation` is the optional
    SHAP/TreeSHAP feature-contribution list from explainability.py,
    JSON-serializable via explainability.explanation_to_alert_field."""
    from .schema import ALERT_SCHEMA_VERSION

    return {
        "alert_id": str(uuid.uuid4()),
        "flow_id": event.get("flow_id", ""),
        "src_ip": event.get("src_ip", ""),
        "src_port": int(event.get("src_port", 0)),
        "dst_ip": event.get("dst_ip", ""),
        "dst_port": int(event.get("dst_port", 0)),
        "protocol": int(event.get("protocol", 0)),
        "flow_start_time": float(event.get("flow_start_time", 0.0)),
        "scored_at": time.time(),
        "stage1_anomaly_score": scored.stage1_anomaly_score,
        "stage1_flagged": scored.stage1_flagged,
        "stage2_predicted_class": scored.stage2_predicted_class,
        "stage2_confidence": scored.stage2_confidence,
        "stage2_class_probabilities": scored.stage2_class_probabilities,
        "severity": severity_for(scored.stage2_predicted_class, scored.stage2_confidence),
        "explanation": explanation or [],
        "model_version": MODEL_VERSION,
        "schema_version": ALERT_SCHEMA_VERSION,
    }
