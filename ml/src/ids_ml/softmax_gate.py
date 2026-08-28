"""Softmax-confidence escalation gate: the closed-set baseline that
`openset_head.OpenMaxHead` is benchmarked against.

This is the pipeline's original behavior (stage2_confidence as the
uncertainty signal), pulled out into its own module rather than deleted.
The open-set upgrade's central empirical claim is that an open-set trigger
beats a softmax-confidence trigger at an equal escalation budget -- proving
that requires both to exist as literal, swappable alternatives behind the
same interface, not one replacing the other. See `pipeline.TwoStageDetector`
for how a `gate` (this class or `OpenMaxHead`) plugs into the router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class GateResult:
    predicted_class: str
    known_class_probabilities: Dict[str, float]
    unknown_mass: float  # 1 - stage2_confidence for this gate


class SoftmaxGate:
    """Wraps an already-fit `stage2_xgboost.AttackClassifier`. No separate
    fit() step: the baseline's uncertainty signal is just stage 2's own
    softmax output, unchanged."""

    def __init__(self, classifier) -> None:
        self.classifier = classifier

    def recalibrate_batch(self, X: np.ndarray) -> List[GateResult]:
        proba = self.classifier.predict_proba(X)
        classes = self.classifier.classes_
        results = []
        for row in range(X.shape[0]):
            best = int(proba[row].argmax())
            class_probs = {classes[c]: float(proba[row, c]) for c in range(len(classes))}
            results.append(
                GateResult(
                    predicted_class=str(classes[best]),
                    known_class_probabilities=class_probs,
                    unknown_mass=1.0 - float(proba[row, best]),
                )
            )
        return results
