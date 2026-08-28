"""Conformal calibration of the escalation threshold against an operator-
set escalation budget, instead of an arbitrary fixed number on
`unknown_mass`.

Run a gate (`openset_head.OpenMaxHead` or `softmax_gate.SoftmaxGate`) over
a held-out calibration set of *known* (in-distribution) traffic, then set
the escalation threshold to the smallest value such that, by the standard
split-conformal quantile guarantee (Vovk et al.; see Angelopoulos & Bates,
2021 for a modern treatment), at most `budget` fraction of held-out known
traffic exceeds it -- with the usual finite-sample correction, so the
guarantee holds even for small calibration sets. This is what makes the
threshold defensible instead of "unknown_mass > 0.5": it is set from
measured data against an explicit, human-legible operating point
("escalate at most 5% of known flows"), which is also the quantity
`evaluation.py`'s escalation-rate-vs-unknown-recall tradeoff curve sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import joblib
import numpy as np


@dataclass
class ConformalGate:
    threshold: float
    budget: float
    n_calibration: int

    def should_escalate(self, unknown_mass: float) -> bool:
        return unknown_mass > self.threshold

    def save(self, path) -> None:
        joblib.dump({"threshold": self.threshold, "budget": self.budget, "n_calibration": self.n_calibration}, path)

    @classmethod
    def load(cls, path) -> "ConformalGate":
        payload = joblib.load(path)
        return cls(**payload)


def calibrate_threshold(unknown_mass_scores: Sequence[float], budget: float) -> ConformalGate:
    """`unknown_mass_scores` must come from a calibration set of KNOWN
    (in-distribution) traffic only -- never mix in held-out/unknown-family
    flows here, or the threshold calibrates against the wrong population
    and the escalation-rate guarantee no longer means what it says.
    """
    scores = np.asarray(list(unknown_mass_scores), dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("calibration set must be non-empty")
    if not (0.0 < budget < 1.0):
        raise ValueError("budget must be in (0, 1)")

    # Smallest score such that, by the split-conformal guarantee, at most
    # `budget` fraction of a fresh draw from the same distribution is
    # expected to exceed it.
    level = min(np.ceil((n + 1) * (1 - budget)) / n, 1.0)
    threshold = float(np.quantile(scores, level, method="higher"))
    return ConformalGate(threshold=threshold, budget=budget, n_calibration=n)
