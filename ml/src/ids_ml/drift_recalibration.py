"""Detect-then-recalibrate baseline for the conformal thresholds.

This project adapts its thresholds continuously (ACI:
`adaptive_conformal_gate.AdaptiveConformalGate` for the escalation budget,
`adaptive_stage1.AdaptiveStage1Threshold` for the Stage 1 miss budget). The
main alternative in the intrusion-detection literature is to keep a static
conformal threshold and recalibrate it only when a drift detector fires --
the approach of FIRCE / FADES (Barrett et al., 2026), which monitor
conformal-evaluation signals and adapt when they indicate drift. A paper
claiming continuous adaptation is better has to beat that, so this module
implements it as a baseline.

What this is, and is not: a FIRCE-STYLE baseline, not a reimplementation of
FIRCE. FIRCE's drift signal is built from conformal credibility/confidence
p-values over a supervised classifier. Here the detector is a two-sample
Kolmogorov-Smirnov test between the recent window of scores and the scores
the current threshold was calibrated on. That keeps it label-free on the
escalation side (like the adaptive gate) and comparable on the Stage 1 side.
Report it as "detect-then-recalibrate (KS)".

Mechanics, shared by both sides:
  * a static conformal threshold is computed from a reference set of scores;
  * every score is folded into a sliding window;
  * every `check_every` updates, once the window is full, a KS test compares
    the window with the reference set; if p < `drift_alpha`, the threshold
    is recalibrated on the window, and the window becomes the new reference.

`tail="upper"` is the escalation gate: scores are unknown_mass on (assumed)
known traffic, a flow escalates if its score exceeds the threshold, and
recalibration uses `conformal_gate.calibrate_threshold`. `tail="lower"` is
Stage 1: scores are anomaly scores of confirmed attacks, an attack is missed
if its score is below the threshold, and recalibration uses
`conformal_stage1.calibrate_stage1_threshold`.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List

import numpy as np
from scipy.stats import ks_2samp

from .conformal_gate import calibrate_threshold
from .conformal_stage1 import calibrate_stage1_threshold


class DetectThenRecalibrate:
    def __init__(
        self,
        budget: float,
        tail: str = "upper",
        window_size: int = 500,
        check_every: int = 100,
        drift_alpha: float = 0.01,
    ) -> None:
        if tail not in ("upper", "lower"):
            raise ValueError("tail must be 'upper' (escalation gate) or 'lower' (Stage 1)")
        if not (0.0 < budget < 1.0):
            raise ValueError("budget must be in (0, 1)")
        self.budget = budget
        self.tail = tail
        self.window_size = window_size
        self.check_every = check_every
        self.drift_alpha = drift_alpha
        self._window: Deque[float] = deque(maxlen=window_size)
        self._reference = np.array([])
        self.threshold = float("nan")
        self.n_updates = 0
        self.recalibrated_at: List[int] = []

    def _calibrate(self, scores: np.ndarray) -> float:
        if self.tail == "upper":
            return calibrate_threshold(scores, self.budget).threshold
        return calibrate_stage1_threshold(scores, self.budget).threshold

    def seed(self, scores: Iterable[float]) -> "DetectThenRecalibrate":
        """Calibrate the initial threshold on `scores`; they also become the
        reference set the first drift test compares against."""
        self._reference = np.asarray(list(scores), dtype=float)
        self.threshold = self._calibrate(self._reference)
        return self

    def decide(self, score: float) -> bool:
        """Upper tail: escalate? Lower tail: flag (i.e. not missed)?"""
        if self.tail == "upper":
            return (not np.isnan(self.threshold)) and score > self.threshold
        return score >= self.threshold

    def update(self, score: float) -> None:
        """Fold one score in (known-traffic score for the upper tail, a
        confirmed attack's score for the lower tail) and run the periodic
        drift check."""
        self._window.append(float(score))
        self.n_updates += 1
        if len(self._window) == self.window_size and self.n_updates % self.check_every == 0:
            recent = np.asarray(self._window)
            if ks_2samp(recent, self._reference).pvalue < self.drift_alpha:
                self.threshold = self._calibrate(recent)
                self._reference = recent
                self.recalibrated_at.append(self.n_updates)

    @property
    def n_recalibrations(self) -> int:
        return len(self.recalibrated_at)
