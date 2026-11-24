"""Online-adaptive recall control for the Stage 1 pre-filter.

`conformal_stage1.calibrate_stage1_threshold` sets Stage 1's flag threshold
once, from held-out known attacks, so that a fresh attack is passed as
benign with probability at most `miss_budget` -- if calibration and live
attacks are exchangeable. Real traffic breaks that assumption: on the
chronological split the realised miss rate overshoots the budget (see
results/stage1_recall_guarantee.csv). This module is the Stage 1 analogue of
`adaptive_conformal_gate.AdaptiveConformalGate`: it applies Gibbs & Candès's
adaptive conformal inference (ACI) update to the miss side instead of the
escalation side.

Update rule, once per attack whose label becomes known:

    err_t       = 1 if that attack was passed as benign (missed), else 0
    alpha_{t+1} = alpha_t + gamma * (miss_budget - err_t)
    threshold   = the alpha_{t+1} lower quantile of a sliding window of
                  recent attack scores

A run of misses lowers alpha_t and so the threshold, flagging more; a run of
caught attacks raises it, flagging less. Gibbs & Candès's Proposition 4.1
then bounds the long-run average miss rate over the T attacks fed back:

    |mean(err_t) - miss_budget| <= (max(alpha_1, 1 - alpha_1) + gamma) / (T * gamma)

with no exchangeability assumption. As with AdaptiveConformalGate, alpha_t
is clipped to [1e-6, 1 - 1e-6] and the threshold comes from a finite window,
so the bound holds only approximately; experiments measure the realised
rate rather than assume it.

The assumption this needs, stated up front because it is the price of
dropping exchangeability: labels. The escalation gate can adapt label-free
(it treats the stream as known traffic), but a miss can only be counted once
someone confirms the flow was an attack -- in practice analyst triage,
an incident report, or a delayed signature match. Benign flows carry no
information about misses and never call `update`. `evaluation.
stage1_stream_report` measures how much a feedback delay costs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List

import numpy as np


@dataclass
class AdaptiveStage1Threshold:
    miss_budget: float
    gamma: float = 0.01
    window_size: int = 500

    def __post_init__(self) -> None:
        if not (0.0 < self.miss_budget < 1.0):
            raise ValueError("miss_budget must be in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.alpha_t: float = self.miss_budget
        self._window: Deque[float] = deque(maxlen=self.window_size)
        self.alpha_history: List[float] = []
        self.n_feedback = 0
        self.n_missed = 0

    def seed(self, attack_scores: Iterable[float]) -> "AdaptiveStage1Threshold":
        """Warm-start the window from the calibration attacks the static
        threshold would use. Does not touch alpha_t."""
        for s in attack_scores:
            self._window.append(float(s))
        return self

    @property
    def threshold(self) -> float:
        """Current flag threshold. -inf (flag everything) while the window is
        empty -- the safe default for a recall-side control."""
        if not self._window:
            return float("-inf")
        level = min(max(self.alpha_t, 0.0), 1.0)
        return float(np.quantile(np.array(self._window), level, method="lower"))

    def flag(self, score: float) -> bool:
        return score >= self.threshold

    def update(self, attack_score: float, threshold_at_decision: float) -> bool:
        """Feed back one CONFIRMED attack's Stage 1 score.

        `threshold_at_decision` is the threshold that was in force when the
        flow was scored. With delayed feedback the threshold may have moved
        since, and the miss that matters is the one that actually happened.
        Returns whether that attack was missed, then performs the ACI step
        and folds the score into the window.
        """
        missed = attack_score < threshold_at_decision
        err_t = 1.0 if missed else 0.0
        self.alpha_t = min(max(self.alpha_t + self.gamma * (self.miss_budget - err_t), 1e-6), 1.0 - 1e-6)
        self.alpha_history.append(self.alpha_t)
        self._window.append(float(attack_score))
        self.n_feedback += 1
        self.n_missed += int(missed)
        return missed

    @property
    def realised_miss_rate(self) -> float:
        """Miss rate over every attack fed back so far. NaN before the first."""
        return self.n_missed / self.n_feedback if self.n_feedback else float("nan")
