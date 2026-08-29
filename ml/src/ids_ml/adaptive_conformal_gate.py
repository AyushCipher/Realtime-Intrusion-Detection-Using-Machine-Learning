"""Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) for the
escalation-budget threshold, extending `conformal_gate.py`'s static
split-conformal calibration to keep its budget guarantee under concept
drift.

`conformal_gate.ConformalGate` calibrates a single threshold once, from
one held-out batch of known traffic, and its guarantee implicitly assumes
the deployment distribution stays exchangeable with that batch. This
project's own `evaluation.concept_drift_report` already demonstrates that
assumption is false here (day-2 traffic's feature distributions shift from
day-1's, by construction in the synthetic fixture and by nature in real
network traffic). Under that shift, a statically-calibrated threshold's
true escalation rate silently drifts away from the requested budget, with
no signal that it's happening -- the threshold just sits where it was set.

`AdaptiveConformalGate` instead updates its significance level online,
after every scored known-traffic flow, following Gibbs & Candès' ACI
update rule. That rule was originally formulated for prediction-interval
miscoverage; it's adapted here to escalation-rate control by treating
"this known flow got escalated" as the failure event a fixed-rate
miscoverage rule normally tracks. At step t:

    threshold_t = quantile_{1 - alpha_t}(recent known-traffic unknown_mass scores)
    escalated_t = 1{score_t > threshold_t}
    alpha_{t+1} = alpha_t + gamma * (budget - escalated_t)

If the realized escalation rate over recent flows runs above the budget,
alpha_t falls (each escalation subtracts gamma*(1-budget)), which raises
the quantile level 1-alpha_t and so raises the threshold -- harder to
escalate at the next step. Symmetrically, a run of non-escalations pushes
alpha_t up, lowering the threshold and making escalation easier again. So
the escalation rate is pulled back toward the target budget continuously,
rather than assumed to hold from a single calibration pass. This is the
same mechanism Gibbs & Candès prove tracks a target rate under arbitrary
(even adversarial) distribution shift, with no i.i.d./exchangeability
assumption -- exactly the assumption a static calibration silently makes.

Checked against the literature before building this (not assumed novel):
CALIBURN (arXiv 2605.24696) uses only static Conformal Risk Control and
names this exact adaptation as unimplemented future work ("under
significant concept drift... the empirical test FPR can exceed the
nominal alpha"). FIRCE/FADES (arXiv 2605.01962; MDPI Electronics
15(10):2114) use periodic recalibration triggered by a drift-detection
signal, not continuous online updating of a rate-control target -- a
related but distinct mechanism. None of the three combine this with
open-set/zero-day detection, which is the context this gate is deployed
in via `pipeline.TwoStageDetector`.

## Production ground-truth-proxy assumption -- flagged for operator sign-off

`update()`'s docstring says "only ever call this with known/in-distribution
traffic". Offline (this module's own tests, `evaluation.
static_vs_adaptive_conformal_drift_report`), that's enforceable directly --
the test harness knows which rows are the held-out family. **Live, it
isn't**: whether an incoming flow is genuinely known/benign or a genuine
novel attack is exactly the question open-set detection exists to answer,
so gating `update()` on that answer is circular, and no ground truth is
available at scoring time to break the circularity.

The proxy this project adopts (`pipeline.TwoStageDetector.score`, when
`escalation_gate` is this class): call `update()` on **every** stage-1-
flagged, stage-2-scored flow, unconditionally -- i.e. treat the entire
live stream as if it were known traffic for adaptation purposes. This
rests on one explicit, unvalidated assumption:

    Genuine novel/zero-day traffic is a small enough fraction of live
    scored flows that folding it into the "known" update stream biases
    alpha_t negligibly rather than corrupting it.

This is not a guess pulled from nowhere -- real network traffic's extreme
class imbalance (attack prevalence <<1% is this project's own stated
real-world expectation; see the top-level README's known-limitations
section) supports it as a reasonable default, and it's the same
"eventually-labeled-or-approximately-labeled stream" compromise online
conformal methods generally make when true per-sample labels aren't
available at prediction time (Gibbs & Candès's own follow-up work
discusses delayed-label variants of ACI). But it has never been checked
against this project's own real traffic, and a deployment with unusually
high novel-attack prevalence (e.g. mid-campaign, or a network already
compromised) would silently violate it -- with no self-correction, since
the algorithm has no way to tell a true escalation-worthy flow apart from
a false one without exactly the ground truth it doesn't have.

**The mitigation is monitoring, not a fix**: `realized_escalation_rate`
tracks what the assumption actually produced; `evaluation.
audit_gate_against_triage` compares it against analyst-confirmed triage
outcomes (`dashboard-api`'s existing `triage_status` workflow) once
enough alerts have been reviewed, surfacing a persistent gap as a signal
for a human operator to investigate -- not something this gate corrects
on its own. Whether that residual risk is acceptable for a given
deployment is exactly the sign-off this module's assumption needs before
`--adaptive-escalation` (see `train.py`) is used outside a demo/prototype
setting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List

import joblib
import numpy as np


@dataclass
class AdaptiveConformalGate:
    budget: float
    gamma: float = 0.01
    window_size: int = 200

    def __post_init__(self) -> None:
        if not (0.0 < self.budget < 1.0):
            raise ValueError("budget must be in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.alpha_t: float = self.budget
        self._window: Deque[float] = deque(maxlen=self.window_size)
        self.alpha_history: List[float] = []
        self._escalation_history: Deque[bool] = deque(maxlen=self.window_size)

    def seed(self, unknown_mass_scores: Iterable[float]) -> "AdaptiveConformalGate":
        """Warm-start the sliding window from an initial calibration batch
        (e.g. the same batch `conformal_gate.calibrate_threshold` would
        use), so the first live threshold isn't computed from an empty
        buffer. Does not touch alpha_t -- only static calibration sets a
        threshold outright; this gate always starts at alpha_t = budget
        and adapts from there.
        """
        for s in unknown_mass_scores:
            self._window.append(float(s))
        return self

    @property
    def threshold(self) -> float:
        """Current escalation threshold, from the current alpha_t and the
        sliding window of recent known-traffic scores. NaN (never escalate)
        if the window is still empty."""
        if not self._window:
            return float("nan")
        level = min(max(1.0 - self.alpha_t, 0.0), 1.0)
        return float(np.quantile(np.array(self._window), level, method="higher"))

    def should_escalate(self, unknown_mass: float) -> bool:
        t = self.threshold
        return (not np.isnan(t)) and unknown_mass > t

    def update(self, unknown_mass: float) -> bool:
        """Feed one KNOWN-traffic flow's score, in stream order.

        Returns whether it was escalated, evaluated against the threshold
        as it stood *before* this call (i.e. the decision a live system
        would actually have made), then performs the ACI step and folds
        the score into the sliding window. Only ever call this with
        known/in-distribution traffic -- same requirement as
        `conformal_gate.calibrate_threshold`'s calibration set.
        """
        escalated = self.should_escalate(unknown_mass)
        err_t = 1.0 if escalated else 0.0
        self.alpha_t = min(max(self.alpha_t + self.gamma * (self.budget - err_t), 1e-6), 1.0 - 1e-6)
        self.alpha_history.append(self.alpha_t)
        self._escalation_history.append(escalated)
        self._window.append(float(unknown_mass))
        return escalated

    @property
    def realized_escalation_rate(self) -> float:
        """Fraction of the last `window_size` `update()` calls that
        escalated -- a monitoring signal, not an input to the ACI update
        itself (that's `alpha_t`/`threshold`, driven by `budget`). Compare
        this against `budget` to see whether the online adaptation is
        actually tracking its target on recent traffic; compare it against
        an analyst-triage-confirmed rate (`audit_against_triage` in
        `evaluation.py`) to check whether the production ground-truth-proxy
        assumption this gate is fed under (see `pipeline.TwoStageDetector`'s
        docstring) is holding up. NaN before the first `update()` call.
        """
        if not self._escalation_history:
            return float("nan")
        return float(np.mean(self._escalation_history))

    def save(self, path) -> None:
        """Saves full online state (alpha_t, the sliding window, budget/
        gamma/window_size, and both history buffers) so a restarted serving
        process resumes adaptation instead of silently resetting to
        alpha_t=budget and an empty window -- a silent reset would discard
        exactly the drift-tracking state this gate exists to maintain.
        """
        joblib.dump(
            {
                "budget": self.budget,
                "gamma": self.gamma,
                "window_size": self.window_size,
                "alpha_t": self.alpha_t,
                "window": list(self._window),
                "alpha_history": self.alpha_history,
                "escalation_history": list(self._escalation_history),
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "AdaptiveConformalGate":
        payload = joblib.load(path)
        instance = cls(budget=payload["budget"], gamma=payload["gamma"], window_size=payload["window_size"])
        instance.alpha_t = payload["alpha_t"]
        instance._window = deque(payload["window"], maxlen=instance.window_size)
        instance.alpha_history = payload["alpha_history"]
        instance._escalation_history = deque(payload["escalation_history"], maxlen=instance.window_size)
        return instance
