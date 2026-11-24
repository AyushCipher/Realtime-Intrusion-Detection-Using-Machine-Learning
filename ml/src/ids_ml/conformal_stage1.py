"""Conformal recall control for the Stage 1 pre-filter.

The cascade's weakest point is Stage 1: an attack it scores as benign never
reaches Stage 2, so the open-set gate and the classifier cannot recover it.
`conformal_gate.py` already bounds the *escalation* rate of known flows
(a false-positive-side budget). This module adds the missing other side: it
sets Stage 1's flag threshold from held-out known *attack* flows so that, by
the same split-conformal argument, a fresh known attack is passed as benign
with probability at most `miss_budget`.

Guarantee (finite-sample, distribution-free): if the calibration attack
scores and a new attack's score are exchangeable, then with n calibration
attacks and k = floor(miss_budget * (n + 1)),

    P(new attack score < threshold) <= k / (n + 1) <= miss_budget,

where `threshold` is the k-th smallest calibration attack score (and -inf,
i.e. flag everything, when k = 0). Stage 1 flags a flow when its anomaly
score is >= threshold (see `AnomalyPreFilter.flag`).

What this does NOT guarantee:
  * Only attack families represented in the calibration set are covered. A
    zero-day family's miss rate is not bounded; `evaluation.py`'s
    leave-one-family-out experiments measure that separately.
  * Exchangeability is assumed. Under drift the realised miss rate can
    exceed the budget, the same caveat as the escalation gate.
  * Tighter recall costs load: a lower threshold also flags more benign
    flows, which Stage 2 must then classify. `miss_budget` trades recall
    against Stage 2 throughput, and is reported alongside the flag rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class Stage1RecallGuarantee:
    threshold: float
    miss_budget: float
    n_calibration: int
    guaranteed_miss_rate: float

    def apply(self, stage1) -> None:
        """Pin `stage1` (an `AnomalyPreFilter`) to this flag threshold."""
        stage1.config.flag_threshold = self.threshold


def calibrate_stage1_threshold(attack_scores: Sequence[float], miss_budget: float) -> Stage1RecallGuarantee:
    """`attack_scores` are Stage 1 anomaly scores (higher = more anomalous) of
    KNOWN attack flows from a held-out calibration set. Benign flows must not
    be included: the guarantee is about attacks.
    """
    scores = np.sort(np.asarray(list(attack_scores), dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("calibration set must contain at least one attack flow")
    if not (0.0 < miss_budget < 1.0):
        raise ValueError("miss_budget must be in (0, 1)")

    k = int(np.floor(miss_budget * (n + 1)))
    if k == 0:
        # Budget tighter than 1/(n+1): the calibration set cannot certify any
        # miss rate that small, so flag everything rather than overclaim.
        return Stage1RecallGuarantee(float("-inf"), miss_budget, n, 0.0)
    return Stage1RecallGuarantee(float(scores[k - 1]), miss_budget, n, k / (n + 1))


@dataclass
class PerFamilyRecallGuarantee:
    """Result of `calibrate_stage1_per_family`.

    `threshold` is the flag threshold actually used. `guaranteed_families` are
    the families whose miss rate is bounded by `miss_budget` at that threshold
    (each with its own finite-sample bound); `waived_families` are families
    whose budget could only be met by flagging more benign traffic than
    `max_benign_flag_rate` allows, so their guarantee was given up -- reported,
    never silent. `benign_flag_rate` is measured on the calibration benign flows.
    """

    threshold: float
    miss_budget: float
    family_thresholds: dict
    guaranteed_families: list
    waived_families: list
    uncertifiable_families: list
    benign_flag_rate: float

    def apply(self, stage1) -> None:
        stage1.config.flag_threshold = self.threshold


def calibrate_stage1_per_family(
    attack_scores_by_family: dict,
    miss_budget: float,
    benign_scores: Optional[Sequence[float]] = None,
    max_benign_flag_rate: Optional[float] = None,
) -> PerFamilyRecallGuarantee:
    """Class-conditional (Mondrian) version of `calibrate_stage1_threshold`.

    The pooled calibration bounds the miss rate averaged over the calibration
    attack mix, so a family that is rare in calibration can be missed far
    more often than the budget, and a family that looks like benign traffic
    drags the single threshold down until almost all benign flows are flagged
    (on CSE-CIC-IDS2018's chronological split, Infiltration alone does this).

    Here each family gets its own conformal threshold, the deployed threshold
    is the lowest of them (so every kept family is covered), and, if a benign
    cost cap is given, families whose threshold would breach the cap are
    waived and named. A family with too few calibration flows to certify the
    budget (fewer than 1/miss_budget - 1) is reported as uncertifiable and not
    allowed to set the threshold, since its only certifiable threshold is -inf.
    """
    if not attack_scores_by_family:
        raise ValueError("need at least one attack family")
    fam_thr = {f: calibrate_stage1_threshold(s, miss_budget).threshold for f, s in attack_scores_by_family.items() if len(s)}
    uncertifiable = sorted(f for f, t in fam_thr.items() if t == float("-inf"))
    certifiable = {f: t for f, t in fam_thr.items() if t != float("-inf")}

    benign = np.asarray(list(benign_scores), dtype=float) if benign_scores is not None else None
    cap_thr = float("-inf")
    if benign is not None and max_benign_flag_rate is not None and len(benign):
        cap_thr = float(np.quantile(benign, 1.0 - max_benign_flag_rate, method="higher"))

    kept = {f: t for f, t in certifiable.items() if t >= cap_thr}
    waived = sorted(f for f in certifiable if f not in kept)
    if kept:
        threshold = min(kept.values())
    elif certifiable:
        threshold = cap_thr  # every family would breach the cap: hold the cost cap, guarantee none
    else:
        threshold = float("-inf")
    rate = float((benign >= threshold).mean()) if benign is not None and len(benign) else float("nan")
    return PerFamilyRecallGuarantee(threshold, miss_budget, fam_thr, sorted(kept), waived, uncertifiable, rate)


@dataclass
class DualBudgetThreshold:
    """Result of `calibrate_stage1_dual`: the deployed threshold and which budget set it."""

    threshold: float
    miss_budget: float
    compute_budget: float
    recall_threshold: float
    compute_threshold: float
    binding: str  # "compute" when the compute budget lowered the threshold, else "recall"

    def apply(self, stage1) -> None:
        stage1.config.flag_threshold = self.threshold


def calibrate_stage1_dual(
    attack_scores: Sequence[float],
    benign_scores: Sequence[float],
    miss_budget: float,
    compute_budget: float,
) -> DualBudgetThreshold:
    """Stage 1 threshold under a miss budget AND a compute budget.

    The recall threshold (`calibrate_stage1_threshold`) is set from KNOWN attacks only.
    When the pre-filter scores known attacks near the top of its range, that threshold is
    high and Stage 1 flags very little benign traffic -- and drops most flows from attack
    families it was never trained on (on LYCOS-IDS2017 with a 5% miss budget: 0.1% of
    benign traffic flagged, 0-35% of each withheld family passed on).

    `compute_budget` is the share of (benign) traffic the operator can afford to send to
    Stage 2. The deployed threshold is the LOWER of the recall threshold and the
    (1 - compute_budget) quantile of calibration benign scores, so the known-attack miss
    guarantee still holds (lowering the threshold only flags more) and any spare compute
    goes to the most unusual flows, which is where unseen families sit. If the recall
    threshold already flags more than `compute_budget` of benign traffic, it is kept and
    the compute budget is exceeded: recall takes priority, and `binding` says "recall".
    """
    if not (0.0 <= compute_budget < 1.0):
        raise ValueError("compute_budget must be in [0, 1)")
    rec = calibrate_stage1_threshold(attack_scores, miss_budget).threshold
    benign = np.asarray(list(benign_scores), dtype=float)
    if compute_budget == 0.0 or len(benign) == 0:
        comp = float("inf")
    else:
        comp = float(np.quantile(benign, 1.0 - compute_budget, method="higher"))
    thr = min(rec, comp)
    return DualBudgetThreshold(thr, miss_budget, compute_budget, rec, comp, "compute" if comp < rec else "recall")
