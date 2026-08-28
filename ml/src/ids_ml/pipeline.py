"""Combines stage 1 and stage 2 into the deployed detection cascade.

`TwoStageDetector.score` is the production path: stage 2 only runs on rows
stage 1 flags. Rows stage 1 doesn't flag are reported as BENIGN without ever
touching the (more expensive) stage-2 model -- so the pipeline's overall
recall on attacks is bounded by stage 1's recall. `evaluation.py` separately
runs stage 2 on *all* rows to characterize it as a standalone classifier;
that comparison is what makes stage 1's miss rate visible instead of hidden
inside a single blended number.

Optionally, a `gate` (`softmax_gate.SoftmaxGate` or `openset_head.
OpenMaxHead`) recalibrates stage 2's output on flagged rows into a
three-way decision -- `known-benign` / `known-attack` / `escalated` --
instead of the plain "flagged or not" binary, per the open-set upgrade
(see `ml/README.md`'s open-set section). Without a `gate`, `score()`
behaves exactly as before: every flagged row is `known-benign` or
`known-attack` from stage 2's raw prediction, `unknown_mass` is always
`0.0`, and nothing is ever escalated.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .conformal_gate import ConformalGate
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
    # Open-set fields (see module docstring). Defaults match the pre-open-set
    # behavior exactly when no `gate` is passed to TwoStageDetector.
    decision: str = "known-benign"  # "known-benign" | "known-attack" | "escalated"
    unknown_mass: float = 0.0
    escalated: bool = False
    escalation_trigger: str = ""  # "openset" | "softmax" | "" (no gate configured)


class TwoStageDetector:
    """The deployed stage1 -> stage2 cascade.

    `gate` and `escalation_gate` are both optional and independent:
    - No `gate`: original behavior, `decision` is `known-benign`/
      `known-attack` from stage 2's raw prediction, nothing escalates.
    - `gate` without `escalation_gate`: stage 2's flagged-row predictions
      are recalibrated (`unknown_mass` is real) but nothing escalates,
      since there's no calibrated threshold to escalate against yet --
      useful for collecting `unknown_mass` scores to calibrate one
      (`conformal_gate.calibrate_threshold`).
    - Both: the full three-way router -- `escalated` when `unknown_mass`
      exceeds `escalation_gate.threshold`, else `known-benign`/
      `known-attack` from the gate's recalibrated prediction.
    """

    def __init__(
        self,
        stage1: AnomalyPreFilter,
        stage2: AttackClassifier,
        gate: Optional[Any] = None,
        escalation_gate: Optional[ConformalGate] = None,
        escalation_trigger_name: str = "",
    ) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.gate = gate
        self.escalation_gate = escalation_gate
        self.escalation_trigger_name = escalation_trigger_name

    def score(self, X: np.ndarray) -> List[ScoredFlow]:
        n = X.shape[0]
        stage1_scores = self.stage1.anomaly_score(X)
        stage1_flags = self.stage1.flag(X)

        results: List[Optional[ScoredFlow]] = [None] * n
        flagged_idx = np.where(stage1_flags)[0]

        if len(flagged_idx) > 0:
            X_flagged = X[flagged_idx]
            proba = self.stage2.predict_proba(X_flagged)
            classes = self.stage2.classes_
            best = proba.argmax(axis=1)

            gate_results = self.gate.recalibrate_batch(X_flagged) if self.gate is not None else None

            for pos, row in enumerate(flagged_idx):
                class_probs = {classes[c]: float(proba[pos, c]) for c in range(len(classes))}
                predicted_class = str(classes[best[pos]])
                confidence = float(proba[pos, best[pos]])
                unknown_mass = 0.0
                escalated = False
                trigger = ""

                if gate_results is not None:
                    gr = gate_results[pos]
                    predicted_class = gr.predicted_class
                    confidence = gr.known_class_probabilities.get(predicted_class, confidence)
                    class_probs = gr.known_class_probabilities
                    unknown_mass = gr.unknown_mass
                    trigger = self.escalation_trigger_name
                    if self.escalation_gate is not None and self.escalation_gate.should_escalate(unknown_mass):
                        escalated = True

                decision = "escalated" if escalated else ("known-benign" if predicted_class == "BENIGN" else "known-attack")

                results[row] = ScoredFlow(
                    stage1_anomaly_score=float(stage1_scores[row]),
                    stage1_flagged=True,
                    stage2_ran=True,
                    stage2_predicted_class=predicted_class,
                    stage2_confidence=confidence,
                    stage2_class_probabilities=class_probs,
                    decision=decision,
                    unknown_mass=unknown_mass,
                    escalated=escalated,
                    escalation_trigger=trigger,
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
                    decision="known-benign",
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
        "unknown_mass": scored.unknown_mass,
        "escalated": scored.escalated,
        "escalation_trigger": scored.escalation_trigger,
        "model_version": MODEL_VERSION,
        "schema_version": ALERT_SCHEMA_VERSION,
    }
