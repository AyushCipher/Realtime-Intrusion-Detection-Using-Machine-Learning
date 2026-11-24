"""End-to-end miss-rate control for the two-stage cascade.

The cascade raises an alert for a flow only if Stage 1 flags it AND Stage 2
does not let it through silently (Stage 2 predicts an attack class, or the
open-set gate escalates it). Each stage is calibrated on its own today:
`conformal_stage1` bounds Stage 1's miss rate at budget alpha, and Stage 2's
BENIGN decision is left as argmax. Nothing bounds the END-TO-END miss rate,

    P(no alert | attack) = P(Stage 1 misses) + P(Stage 1 flags, Stage 2 silent),

which can exceed alpha because the second term is extra.

This module calibrates the two stages jointly with conformal risk control
(Angelopoulos, Bates, Fisch, Lei and Schuster, "Conformal Risk Control",
ICLR 2024). One parameter lambda in [0, 1] moves both thresholds together:

    t1(lambda) = Stage 1 flag threshold: the lambda-quantile of Stage 1 scores
                 of attack flows in a SHAPING split (-inf at lambda = 0);
    u2(lambda) = Stage 2 is silent on a flagged flow when P(BENIGN) >= u2:
                 the (1 - lambda)-quantile of P(BENIGN) on shaping attacks
                 (+inf at lambda = 0, i.e. never silent).

For a known-family attack i, the loss L_i(lambda) = 1[no alert] is
non-decreasing in lambda and L_i(0) = 0 (lambda = 0 alerts on everything).
With n exchangeable calibration attacks, CRC picks

    lambda_hat = max{ lambda in grid : (n * R_n(lambda) + 1) / (n + 1) <= alpha },

where R_n is the empirical miss rate on calibration attacks, and guarantees
E[L_{n+1}(lambda_hat)] <= alpha for a new exchangeable attack (CRC Theorem 1
with bounded loss B = 1, written for a non-decreasing loss). The shaping split
only fixes the threshold maps; it never sees the calibration set, so the maps
are fixed functions when CRC is applied.

Scope, stated plainly: the guarantee covers attacks exchangeable with the
calibration attacks, i.e. KNOWN families under no drift. Zero-day families
and chronological drift are outside it; experiments measure them separately.

Two baselines share the same scores:
  separate   -- Stage 1 conformal at alpha, Stage 2 argmax (current practice);
  bonferroni -- Stage 1 conformal at alpha/2, Stage 2's silence threshold
                calibrated conformally at alpha/2 on calibration attacks that
                Stage 1 flags. Valid by a union bound, but conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .conformal_stage1 import calibrate_stage1_threshold


@dataclass
class CascadeScores:
    """Per-flow quantities the cascade's alert decision depends on."""

    s1: np.ndarray  # Stage 1 anomaly score (higher = more anomalous)
    p_benign: np.ndarray  # Stage 2 probability of BENIGN
    argmax_benign: np.ndarray  # Stage 2's argmax class is BENIGN
    escalated: np.ndarray  # the open-set gate escalates (fixed threshold)

    def subset(self, mask: np.ndarray) -> "CascadeScores":
        return CascadeScores(self.s1[mask], self.p_benign[mask], self.argmax_benign[mask], self.escalated[mask])


@dataclass
class CascadeDecision:
    """t1: Stage 1 flags when s1 >= t1. u2: Stage 2 is silent on a flagged
    flow when p_benign >= u2 and the gate does not escalate. u2 = None means
    the argmax rule (silent when argmax is BENIGN)."""

    method: str
    t1: float
    u2: float = None
    lam: float = float("nan")

    def alert(self, sc: CascadeScores) -> np.ndarray:
        flagged = sc.s1 >= self.t1
        silent = sc.argmax_benign if self.u2 is None else sc.p_benign >= self.u2
        return flagged & ~(silent & ~sc.escalated)


def calibrate_separate(calib_attacks: CascadeScores, alpha: float) -> CascadeDecision:
    return CascadeDecision("separate", calibrate_stage1_threshold(calib_attacks.s1, alpha).threshold)


def _upper_conformal_threshold(scores: np.ndarray, budget: float) -> float:
    """Smallest u such that a new exchangeable score is >= u with probability
    at most `budget` (the (n+1)-corrected upper quantile); +inf if the
    calibration set is too small to certify the budget."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.floor(budget * (n + 1)))
    if k == 0:
        return float("inf")
    # Rank argument: a new exchangeable score is strictly above the j-th
    # smallest calibration score, j = n + 1 - k, with probability at most
    # (n + 1 - j) / (n + 1) = k / (n + 1) <= budget. "Strictly above" makes
    # ties safe, hence u = the next float above s_(j).
    return float(np.nextafter(s[n - k], np.inf))


def calibrate_bonferroni(calib_attacks: CascadeScores, alpha: float) -> CascadeDecision:
    t1 = calibrate_stage1_threshold(calib_attacks.s1, alpha / 2).threshold
    flagged = calib_attacks.subset(calib_attacks.s1 >= t1)
    # Silence needs p_benign >= u2 AND no escalation; only non-escalated
    # flagged attacks can be silenced, but calibrating on all flagged
    # attacks (escalated ones scored as -inf) keeps the subpopulation intact.
    silenceable = np.where(flagged.escalated, -np.inf, flagged.p_benign)
    u2 = _upper_conformal_threshold(silenceable, alpha / 2)
    return CascadeDecision("bonferroni", t1, u2)


@dataclass
class JointCRC:
    """Threshold maps fixed on a shaping split; lambda picked on calibration."""

    shaping_attacks: CascadeScores
    grid_size: int = 1001

    def thresholds(self, lam: float):
        if lam <= 0.0:
            return float("-inf"), float("inf")
        t1 = float(np.quantile(self.shaping_attacks.s1, lam, method="lower"))
        u2 = float(np.quantile(self.shaping_attacks.p_benign, 1.0 - lam, method="higher"))
        return t1, u2

    def decision(self, lam: float) -> CascadeDecision:
        t1, u2 = self.thresholds(lam)
        return CascadeDecision("joint_crc", t1, u2, lam)

    def calibrate(self, calib_attacks: CascadeScores, alpha: float) -> CascadeDecision:
        n = len(calib_attacks.s1)
        best = 0.0
        for lam in np.linspace(0.0, 1.0, self.grid_size):
            risk = float(np.mean(~self.decision(lam).alert(calib_attacks))) if n else 1.0
            if (n * risk + 1.0) / (n + 1.0) <= alpha:
                best = lam  # loss is non-decreasing in lambda, so keep the largest valid
        return self.decision(best)


@dataclass
class CostAwareJointCRC:
    """Joint CRC along a cost-aware path instead of lockstep quantiles.

    On the shaping split (attacks AND benign flows), for every target miss
    level m in a grid, pick the (t1, u2) pair from a grid of candidates that
    raises the fewest benign alerts while missing at most m of the shaping
    attacks. The path is then made monotone (t1 non-decreasing, u2
    non-increasing in m) by running max / min, so every attack's loss is
    still non-decreasing in the path index and CRC's guarantee holds
    unchanged. Only the shaping split shapes the path; CRC then picks the
    index on the calibration split exactly as in JointCRC.
    """

    shaping_attacks: CascadeScores
    shaping_benign: CascadeScores
    n_candidates: int = 41
    grid_size: int = 201

    def __post_init__(self) -> None:
        qs = np.linspace(0.0, 1.0, self.n_candidates)
        t1s = np.r_[-np.inf, np.quantile(self.shaping_attacks.s1, qs[1:], method="lower")]
        u2s = np.r_[np.inf, np.quantile(self.shaping_attacks.p_benign, 1.0 - qs[1:], method="higher")]
        miss = np.empty((len(t1s), len(u2s)))
        cost = np.empty_like(miss)
        for i, t1 in enumerate(t1s):
            for j, u2 in enumerate(u2s):
                d = CascadeDecision("joint_crc_cost", t1, u2)
                miss[i, j] = np.mean(~d.alert(self.shaping_attacks))
                cost[i, j] = np.mean(d.alert(self.shaping_benign))
        path_t1, path_u2 = [], []
        for m in np.linspace(0.0, 1.0, self.grid_size):
            feasible = miss <= m + 1e-12
            if not feasible.any():
                i, j = 0, 0
            else:
                i, j = np.unravel_index(np.where(feasible, cost, np.inf).argmin(), cost.shape)
            path_t1.append(t1s[i])
            path_u2.append(u2s[j])
        self._t1 = np.maximum.accumulate(np.array(path_t1))
        self._u2 = np.minimum.accumulate(np.array(path_u2))
        self._t1[0], self._u2[0] = -np.inf, np.inf  # index 0 alerts on everything: L_i(0) = 0

    def decision(self, k: int) -> CascadeDecision:
        return CascadeDecision("joint_crc_cost", float(self._t1[k]), float(self._u2[k]), k / (self.grid_size - 1))

    def calibrate(self, calib_attacks: CascadeScores, alpha: float) -> CascadeDecision:
        n = len(calib_attacks.s1)
        best = 0
        for k in range(self.grid_size):
            risk = float(np.mean(~self.decision(k).alert(calib_attacks))) if n else 1.0
            if (n * risk + 1.0) / (n + 1.0) <= alpha:
                best = k
        return self.decision(best)


def evaluate(decision: CascadeDecision, test: CascadeScores, is_attack: np.ndarray) -> Dict[str, float]:
    alerted = decision.alert(test)
    flagged = test.s1 >= decision.t1
    att, ben = is_attack, ~is_attack
    return {
        "method": decision.method,
        "lambda": decision.lam,
        "end_to_end_miss": float((~alerted[att]).mean()) if att.any() else float("nan"),
        "stage1_miss": float((~flagged[att]).mean()) if att.any() else float("nan"),
        "stage2_silent_given_flagged": float((~alerted[att & flagged]).mean()) if (att & flagged).any() else float("nan"),
        "benign_alert_rate": float(alerted[ben].mean()) if ben.any() else float("nan"),
        "benign_flag_rate": float(flagged[ben].mean()) if ben.any() else float("nan"),
    }
