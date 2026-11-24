"""Evaluation: per-category metrics, split-leakage comparison, concept
drift, and a low-and-slow adversarial robustness probe.

Every report function here returns real numbers computed from whatever data
is passed in -- none of it is canned. When run against
tests/fixtures/synthetic_cicids_sample.csv (see tests/test_ml_evaluation.py)
the numbers describe that synthetic fixture, not real-world CICIDS2017
performance; see the README for how to point this module at the real
dataset and what to expect to be different.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score
from scipy.stats import wilcoxon

from .adaptive_conformal_gate import AdaptiveConformalGate
from .adaptive_stage1 import AdaptiveStage1Threshold
from .conformal_gate import calibrate_threshold
from .conformal_stage1 import calibrate_stage1_threshold
from .drift_recalibration import DetectThenRecalibrate
from .ood_gates import OOD_GATES
from .openset_head import OpenMaxConfig, OpenMaxHead
from .softmax_gate import SoftmaxGate
from .split import leave_one_family_out, random_holdout, random_split, time_based_split
from .stage1_hybrid import HybridConfig, HybridPreFilter
from .stage1_iforest import AnomalyPreFilter, Stage1Config
from .stage2_xgboost import AttackClassifier, Stage2Config


def per_category_report(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Precision/recall/F1/support per attack category, sorted by support
    (descending) so common categories (DoS, easy) and rare ones (Infiltration,
    Web Attack -- hard) are both visible rather than averaged away.
    """
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = pd.DataFrame(
        {"category": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    return report.sort_values("support", ascending=False).reset_index(drop=True)


def leakage_comparison_report(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "attack_category",
    stage2_config: Optional[Stage2Config] = None,
    random_state: int = 0,
) -> Dict[str, object]:
    """Trains the same classifier under a time-based split and a random
    split of the same data, and compares macro-F1. A random split is
    expected to score at or above the time-based split -- if it scores
    noticeably higher, that gap is leakage inflation, not real skill; it
    must never be used to select or report a deployed model's metrics
    (see split.py's module docstring for why).
    """
    cfg = stage2_config or Stage2Config(random_state=random_state)

    t_train, _t_val, t_test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    time_clf = AttackClassifier(cfg).fit(t_train[feature_cols].to_numpy(), t_train[label_col].tolist())
    time_report = per_category_report(
        t_test[label_col].tolist(), list(time_clf.predict(t_test[feature_cols].to_numpy()))
    )

    r_train, _r_val, r_test = random_split(df, train_frac=0.7, val_frac=0.15, random_state=random_state)
    rand_clf = AttackClassifier(cfg).fit(r_train[feature_cols].to_numpy(), r_train[label_col].tolist())
    rand_report = per_category_report(
        r_test[label_col].tolist(), list(rand_clf.predict(r_test[feature_cols].to_numpy()))
    )

    time_macro_f1 = float(time_report["f1"].mean())
    rand_macro_f1 = float(rand_report["f1"].mean())

    return {
        "time_based_macro_f1": time_macro_f1,
        "random_split_macro_f1": rand_macro_f1,
        "inflation": rand_macro_f1 - time_macro_f1,
        "time_based_report": time_report,
        "random_split_report": rand_report,
    }


def concept_drift_report(
    window_a_df: pd.DataFrame,
    window_b_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    stage2_config: Optional[Stage2Config] = None,
) -> pd.DataFrame:
    """Trains on the chronological first 80% of window_a, then compares
    per-category F1 on (1) window_a's own held-out chronological tail
    ("in-window", same distribution the model was trained on) versus (2) all
    of window_b ("later window"). The gap is concept drift: how much worse
    the model gets on traffic from a later period it never saw, even before
    any retraining.
    """
    ordered = window_a_df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    split_at = int(len(ordered) * 0.8)
    train_part = ordered.iloc[:split_at]
    in_window_test = ordered.iloc[split_at:]

    clf = AttackClassifier(stage2_config or Stage2Config(random_state=0)).fit(
        train_part[feature_cols].to_numpy(), train_part[label_col].tolist()
    )

    in_window_report = per_category_report(
        in_window_test[label_col].tolist(), list(clf.predict(in_window_test[feature_cols].to_numpy()))
    )
    later_report = per_category_report(
        window_b_df[label_col].tolist(), list(clf.predict(window_b_df[feature_cols].to_numpy()))
    )

    merged = in_window_report.merge(
        later_report, on="category", suffixes=("_in_window", "_later_window"), how="outer"
    ).fillna(0.0)
    merged["f1_degradation"] = merged["f1_in_window"] - merged["f1_later_window"]
    return merged.sort_values("f1_degradation", ascending=False).reset_index(drop=True)


# --- Adversarial robustness: low-and-slow probe ----------------------------

_RATE_DEPENDENT_COLUMNS = [
    "flow_duration",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_min",
    "flow_iat_max",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_min",
    "fwd_iat_max",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_min",
    "bwd_iat_max",
]


def simulate_low_and_slow(df: pd.DataFrame, slowdown_factor: float = 8.0) -> pd.DataFrame:
    """Returns a copy of df approximating a "low-and-slow" variant of the
    same flows: identical packet/byte totals, spread over
    `slowdown_factor`x longer duration -- i.e. the same action, paced to
    look less bursty. Byte/packet-per-flow columns (flow_bytes_per_sec,
    flow_packets_per_sec) are divided by the factor accordingly; inter-
    arrival times are multiplied by it; total packet/byte counts are
    unchanged, since a low-and-slow attack still delivers the same payload.
    """
    out = df.copy()
    out["flow_duration"] = out["flow_duration"] * slowdown_factor
    out["flow_bytes_per_sec"] = out["flow_bytes_per_sec"] / slowdown_factor
    out["flow_packets_per_sec"] = out["flow_packets_per_sec"] / slowdown_factor
    for col in _RATE_DEPENDENT_COLUMNS:
        if col in out.columns and col.endswith(("mean", "std", "min", "max")):
            out[col] = out[col] * slowdown_factor
    return out


def adversarial_robustness_report(
    detector,
    attack_df: pd.DataFrame,
    feature_cols: List[str],
    slowdown_factor: float = 8.0,
) -> Dict[str, float]:
    """Compares stage-1's flag rate on real attack flows against a
    low-and-slow variant of the same flows. A large drop means spreading an
    attack out in time meaningfully evades the anomaly pre-filter -- a real
    risk for a detector that leans on rate/timing features the way this
    one does. This is a heuristic probe, not a formal adversarial-ML
    evaluation (no gradient-based or query-based attack search is
    performed); see the README's adversarial robustness section.
    """
    X_original = attack_df[feature_cols].to_numpy()
    slowed_df = simulate_low_and_slow(attack_df, slowdown_factor)
    X_slowed = slowed_df[feature_cols].to_numpy()

    original_flag_rate = float(detector.stage1.flag(X_original).mean()) if len(attack_df) else 0.0
    slowed_flag_rate = float(detector.stage1.flag(X_slowed).mean()) if len(attack_df) else 0.0

    return {
        "n_flows": len(attack_df),
        "slowdown_factor": slowdown_factor,
        "original_stage1_flag_rate": original_flag_rate,
        "low_and_slow_stage1_flag_rate": slowed_flag_rate,
        "evasion_delta": original_flag_rate - slowed_flag_rate,
    }


# --- Open-set: OpenMax vs softmax head-to-head on leave-one-family-out ----
#
# H1 (this project's central open-set claim): an open-set escalation
# trigger (openset_head.OpenMaxHead) beats a closed-set softmax-confidence
# trigger (softmax_gate.SoftmaxGate) on unknown-family recall at an equal
# escalation budget. The functions below are the metrics half of testing
# that; split.leave_one_family_out / rotate_holdout are the data half.
#
# Every number here still inherits this repo's standing caveat: run
# against tests/fixtures/synthetic_cicids_sample.csv, these describe that
# tiny, deliberately caricatured fixture, not real CICIDS2017/2018
# performance -- and with single-digit support for several families
# (Heartbleed, Infiltration, Web Attack), the per-family numbers are noisy
# almost by construction. See ml/README.md's open-set section.

_GATE_NAMES = ("softmax", "openset")
# Every gate `_fit_gate` can build. H1 as originally run used only
# _GATE_NAMES; the stronger OOD baselines (ood_gates.py) are opt-in through
# `gate_names` so earlier results and tests stay reproducible unchanged.
ALL_GATE_NAMES = _GATE_NAMES + tuple(OOD_GATES)
# Gates whose unknown_mass is a probability in [0, 1]; ECE and Brier are
# only meaningful for these.
_BOUNDED_GATES = {"softmax", "openset"}


@dataclass
class OpenSetTrialResult:
    family: str
    seed: int
    gate_name: str  # one of ALL_GATE_NAMES
    n_known_calib: int
    n_known_test: int
    n_unknown_test: int
    escalation_rate_on_known: float  # should track `budget` on the calibration set by construction; this is its out-of-sample analogue on the test split's known rows
    unknown_recall: float  # fraction of held-out-family test rows escalated
    unknown_auroc: float  # unknown_mass as a score distinguishing known vs held-out-family test rows
    open_auc: float  # stricter joint correct-classification + confidence-ranking metric -- see open_auc()
    ece: float  # NaN for gates whose score is not a probability (see _BOUNDED_GATES)
    brier: float  # same
    open_auc_score: float = float("nan")  # OpenAUC ranked by the gate's own novelty score -- see open_auc_from_scores()
    # Unknown recall at the threshold where exactly `budget` of the TEST
    # known rows escalate. Conformal thresholds overshoot by different
    # amounts per gate under drift (e.g. kNN escalates far more known
    # traffic than its budget), so `unknown_recall` compares gates at
    # unequal false-alarm rates; this compares them at the same one. It is
    # an oracle operating point (uses test labels), for comparison only.
    recall_at_matched_fpr: float = float("nan")


def _fit_gate(gate_name: str, stage2: AttackClassifier, X_fit, y_fit, seed: int = 0):
    if gate_name == "softmax":
        return SoftmaxGate(stage2)
    if gate_name == "openset":
        return OpenMaxHead(stage2, OpenMaxConfig()).fit(X_fit, y_fit)
    if gate_name in ("knn", "efc"):
        return OOD_GATES[gate_name](stage2, random_state=seed).fit(X_fit, y_fit)
    if gate_name in OOD_GATES:
        return OOD_GATES[gate_name](stage2).fit(X_fit, y_fit)
    raise ValueError(f"unknown gate_name: {gate_name!r}")


def _gate_scores(gate, X: np.ndarray) -> np.ndarray:
    """unknown_mass for every row of X. Uses the gate's vectorised
    `scores()` when it has one: building a GateResult (with a dict of class
    probabilities) per row costs about a kilobyte each, which at real-data
    sizes (hundreds of thousands of rows x several gates) exhausts memory.
    OpenMax has no vectorised form, so it falls back to the batch path."""
    if len(X) == 0:
        return np.array([])
    if hasattr(gate, "scores"):
        return np.asarray(gate.scores(X), dtype=float)
    return np.array([r.unknown_mass for r in gate.recalibrate_batch(X)])


def open_auc_from_scores(
    known_true_labels: Sequence[str],
    known_predicted_labels: Sequence[str],
    known_scores: Sequence[float],
    unknown_scores: Sequence[float],
) -> float:
    """OpenAUC (Wang et al., NeurIPS 2022) with each gate's own novelty
    score as the open-set ranking, so gates are compared on the score they
    actually escalate on:

    OpenAUC = mean over (correctly classified known i, unknown j) pairs of
              1[ score_i < score_j ]

    A misclassified known row contributes no wins, as in `open_auc`. That
    function ranks by the gate's known-class probabilities instead, which
    makes every OOD gate in ood_gates.py score identically to softmax (they
    share Stage 2's probabilities) -- hence this score-based variant for
    cross-gate comparisons. For the softmax gate the two coincide.
    """
    known_true = np.asarray(list(known_true_labels))
    known_pred = np.asarray(list(known_predicted_labels))
    known_scores = np.asarray(known_scores, dtype=float)
    unknown_scores = np.sort(np.asarray(unknown_scores, dtype=float))
    correct = known_scores[known_true == known_pred]
    if len(correct) == 0 or len(unknown_scores) == 0:
        return float("nan")
    # For each correct known score, count unknown scores strictly above it;
    # sorting turns the pairwise comparison into one searchsorted call.
    wins = len(unknown_scores) - np.searchsorted(unknown_scores, correct, side="right")
    return float(wins.sum() / (len(correct) * len(unknown_scores)))


def expected_calibration_error(y_true: Sequence[int], y_score: Sequence[float], n_bins: int = 10) -> float:
    """Standard equal-width-binned ECE: the weighted-by-bin-size average gap
    between mean predicted score and actual positive rate within each bin.
    Here the binary label is "does this row belong to the held-out family"
    (1) vs "a known family" (0), and the score is `unknown_mass` -- so this
    asks whether `unknown_mass` is calibrated as a probability of novelty,
    not whether stage 2's known-class softmax is calibrated (a different,
    already-answered question elsewhere in this module).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_score)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (y_score >= lo) & (y_score <= hi if hi == 1.0 else y_score < hi)
        if not in_bin.any():
            continue
        bin_confidence = y_score[in_bin].mean()
        bin_accuracy = y_true[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_confidence - bin_accuracy)
    return float(ece)


def open_auc(
    known_true_labels: Sequence[str],
    known_class_probabilities: Sequence[Dict[str, float]],
    unknown_class_probabilities: Sequence[Dict[str, float]],
) -> float:
    """OpenAUC (Wang et al., NeurIPS 2022): a stricter open-set metric than
    `unknown_auroc` (plain AUROC of `unknown_mass` separating known vs.
    unknown rows). Plain AUROC only asks whether the novelty score ranks
    unknown rows above known rows -- it says nothing about whether stage 2
    actually got the known rows' *class* right. OpenAUC jointly requires
    both: a known test row only counts as a positive comparison if the
    gate's predicted class matches its true label, and even then it's only
    "beaten" an unknown row if its true-class probability exceeds that
    unknown row's highest known-class probability.

    OpenAUC = (1 / (|correct known| * |unknown|)) * sum over every
    (correct known i, unknown j) pair of 1[ p_i(y_i) > max_k p_j(k) ]

    where p_i(y_i) is known row i's probability on its own true class, and
    max_k p_j(k) is unknown row j's highest probability over known classes.
    A misclassified known row contributes zero comparisons -- it can never
    help this score, mirroring the paper's intent that a model shouldn't
    get open-set credit for a closed-set mistake. Returns nan if there are
    no correctly-classified known rows or no unknown rows to compare
    against (open-set recognition is undefined without both).
    """
    correct_known_scores = [
        probs[true_label]
        for true_label, probs in zip(known_true_labels, known_class_probabilities)
        if max(probs, key=probs.get) == true_label
    ]
    unknown_max_scores = [max(probs.values()) if probs else 0.0 for probs in unknown_class_probabilities]

    if not correct_known_scores or not unknown_max_scores:
        return float("nan")

    correct_known_scores = np.asarray(correct_known_scores)
    unknown_max_scores = np.sort(np.asarray(unknown_max_scores))
    # For each correct known score, count unknown scores strictly below it.
    # A pairwise broadcast needs n_known x n_unknown memory (about 2 GB at
    # real-data sizes); sorting gives the same count in O(n log n).
    wins = np.searchsorted(unknown_max_scores, correct_known_scores, side="left").sum()
    return float(wins / (len(correct_known_scores) * len(unknown_max_scores)))


def run_openset_trial(
    df: pd.DataFrame,
    family: str,
    feature_cols: List[str],
    budget: float = 0.1,
    seed: int = 0,
    fit_frac: float = 0.75,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    train_frac: float = 0.7,
    stage2_n_estimators: int = 300,
    gate_names: Sequence[str] = _GATE_NAMES,
) -> List[OpenSetTrialResult]:
    """One leave-one-family-out fold at one seed, every gate in `gate_names`
    head-to-head on the same fitted Stage 2.

    `train` (from `split.leave_one_family_out`) is further split into a
    stage-2 *fit* portion and a *calibration* portion (`split.
    random_holdout` -- a random split is fine here, since this holdout's
    purpose is estimating each gate's score distribution on in-distribution
    data, not simulating "future" traffic the way `time_based_split` is
    for). Calibrating the escalation threshold on the same rows stage 2 was
    fit on would read the model's training-set overconfidence as evidence
    the threshold can be strict -- calibrating on a genuine holdout avoids
    that.
    """
    train, test = leave_one_family_out(
        df, family, label_col=label_col, timestamp_col=timestamp_col, train_frac=train_frac
    )
    fit_df, calib_df = random_holdout(train, holdout_frac=1.0 - fit_frac, random_state=seed)

    X_fit = fit_df[feature_cols].to_numpy()
    y_fit = fit_df[label_col].tolist()
    stage2 = AttackClassifier(Stage2Config(n_estimators=stage2_n_estimators, random_state=seed)).fit(X_fit, y_fit)

    X_calib = calib_df[feature_cols].to_numpy()
    known_test = test[test[label_col] != family]
    unknown_test = test[test[label_col] == family]
    X_known_test = known_test[feature_cols].to_numpy()
    X_unknown_test = unknown_test[feature_cols].to_numpy()

    known_true = known_test[label_col].tolist()
    # Every gate except OpenMax shares Stage 2's closed-set prediction.
    stage2_known_pred = list(stage2.predict(X_known_test)) if len(X_known_test) else []

    results: List[OpenSetTrialResult] = []
    for gate_name in gate_names:
        gate = _fit_gate(gate_name, stage2, X_fit, y_fit, seed=seed)

        calib_scores = list(_gate_scores(gate, X_calib))
        conformal = calibrate_threshold(calib_scores, budget=budget) if calib_scores else None

        if gate_name in _BOUNDED_GATES:
            # The legacy probability-based open_auc needs per-row class
            # probabilities; only these two gates have distinct ones.
            known_gate_results = gate.recalibrate_batch(X_known_test) if len(X_known_test) else []
            unknown_gate_results = gate.recalibrate_batch(X_unknown_test) if len(X_unknown_test) else []
            known_um = np.array([r.unknown_mass for r in known_gate_results])
            unknown_um = np.array([r.unknown_mass for r in unknown_gate_results])
            known_pred = [r.predicted_class for r in known_gate_results]
            oauc = open_auc(
                known_true,
                [r.known_class_probabilities for r in known_gate_results],
                [r.known_class_probabilities for r in unknown_gate_results],
            )
            del known_gate_results, unknown_gate_results
        else:
            known_um = _gate_scores(gate, X_known_test)
            unknown_um = _gate_scores(gate, X_unknown_test)
            known_pred = stage2_known_pred
            # Ranked by Stage 2's probabilities, this would equal softmax's
            # value by construction; open_auc_score is the meaningful one.
            oauc = float("nan")

        if conformal is not None and len(known_um):
            escalation_rate_on_known = float(np.mean(known_um > conformal.threshold))
        else:
            escalation_rate_on_known = float("nan")
        if conformal is not None and len(unknown_um):
            unknown_recall = float(np.mean(unknown_um > conformal.threshold))
        else:
            unknown_recall = float("nan")

        oauc_score = open_auc_from_scores(known_true, known_pred, known_um, unknown_um)
        if len(known_um) and len(unknown_um):
            matched_threshold = float(np.quantile(known_um, 1.0 - budget, method="higher"))
            recall_at_matched_fpr = float(np.mean(unknown_um > matched_threshold))
        else:
            recall_at_matched_fpr = float("nan")

        y_true = np.concatenate([np.zeros(len(known_um)), np.ones(len(unknown_um))])
        y_score = np.concatenate([known_um, unknown_um])
        if len(known_um) and len(unknown_um):
            auroc = float(roc_auc_score(y_true, y_score))
            if gate_name in _BOUNDED_GATES:
                brier = float(brier_score_loss(y_true, y_score))
                ece = expected_calibration_error(y_true, y_score)
            else:
                brier = ece = float("nan")
        else:
            auroc = brier = ece = float("nan")

        results.append(
            OpenSetTrialResult(
                family=family,
                seed=seed,
                gate_name=gate_name,
                n_known_calib=len(calib_scores),
                n_known_test=len(known_um),
                n_unknown_test=len(unknown_um),
                escalation_rate_on_known=escalation_rate_on_known,
                unknown_recall=unknown_recall,
                unknown_auroc=auroc,
                open_auc=oauc,
                ece=ece,
                brier=brier,
                open_auc_score=oauc_score,
                recall_at_matched_fpr=recall_at_matched_fpr,
            )
        )
    return results


def openset_vs_softmax_report(
    df: pd.DataFrame,
    feature_cols: List[str],
    families: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    budget: float = 0.1,
    fit_frac: float = 0.75,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    train_frac: float = 0.7,
    stage2_n_estimators: int = 300,
    gate_names: Sequence[str] = _GATE_NAMES,
) -> pd.DataFrame:
    """Runs `run_openset_trial` for every (family, seed) pair, every gate in
    `gate_names` -- the full leave-one-family-out rotation, repeated across
    seeds so a single lucky/unlucky train/calibration split doesn't decide
    the result. One row per (family, seed, gate). See
    `openset_vs_softmax_significance` and `paired_gate_significance` for
    the paired comparisons on top of this raw table.
    """
    if families is None:
        families = sorted(f for f in df[label_col].unique() if f != "BENIGN")
    rows = []
    for family in families:
        for seed in seeds:
            for result in run_openset_trial(
                df,
                family,
                feature_cols,
                budget=budget,
                seed=seed,
                fit_frac=fit_frac,
                label_col=label_col,
                timestamp_col=timestamp_col,
                train_frac=train_frac,
                stage2_n_estimators=stage2_n_estimators,
                gate_names=gate_names,
            ):
                rows.append(asdict(result))
    return pd.DataFrame(rows)


def paired_gate_significance(
    report_df: pd.DataFrame,
    gate_a: str,
    gate_b: str,
    metric: str = "unknown_recall",
    cluster_col: str = "family",
) -> Dict[str, object]:
    """Paired comparison of two gates that respects how the trials cluster.

    Seeds within one held-out family share the same unknown attack, so
    they are not independent: 5 families x 5 seeds is closer to 5
    observations than 25. `openset_vs_softmax_significance` pairs by
    (family, seed) and so overstates the evidence. This function first
    averages each gate's `metric` over seeds within each `cluster_col`
    value, then runs a Wilcoxon signed-rank test over the clusters, and
    also reports a sign count (clusters where a beats b). The number of
    held-out families is the honest sample size; adding families (more
    datasets, or finer raw-label LOFO) is what raises it.
    """
    sub = report_df[report_df["gate_name"].isin([gate_a, gate_b])]
    per_cluster = sub.groupby([cluster_col, "gate_name"])[metric].mean().unstack("gate_name").dropna()
    n = len(per_cluster)
    out: Dict[str, object] = {
        "metric": metric,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "n_clusters": n,
        "a_mean": float(per_cluster[gate_a].mean()) if n else float("nan"),
        "b_mean": float(per_cluster[gate_b].mean()) if n else float("nan"),
    }
    if n == 0:
        out.update(statistic=float("nan"), p_value=float("nan"), a_wins=0, b_wins=0, note="no complete clusters")
        return out
    diffs = per_cluster[gate_a] - per_cluster[gate_b]
    out["a_wins"] = int((diffs > 0).sum())
    out["b_wins"] = int((diffs < 0).sum())
    if n < 2 or (diffs == 0).all():
        out.update(statistic=float("nan"), p_value=float("nan"), note="too few clusters or no differences")
    else:
        statistic, p_value = wilcoxon(per_cluster[gate_a], per_cluster[gate_b])
        out.update(statistic=float(statistic), p_value=float(p_value), note="")
    return out


def openset_vs_softmax_significance(report_df: pd.DataFrame, metric: str = "unknown_recall") -> Dict[str, object]:
    """Paired Wilcoxon signed-rank test comparing the two gates on `metric`,
    paired by (family, seed) -- i.e. every pair is "same fold, same seed,
    different gate", which is what makes the pairing valid.

    With few LOFO folds and few seeds (as on this project's tiny synthetic
    fixture -- currently 7 non-benign families), this test has very little
    statistical power; a non-significant p-value here means "not enough
    data to tell", not "no difference". Always report the raw per-family
    numbers in `report_df` alongside this, never this p-value alone -- see
    H5 in the project's research plan (explicit null-result fallback).
    """
    pivot = report_df.pivot_table(index=["family", "seed"], columns="gate_name", values=metric)
    pivot = pivot.dropna()

    if len(pivot) < 2:
        return {
            "metric": metric,
            "n_pairs": len(pivot),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "openset_mean": float(pivot["openset"].mean()) if "openset" in pivot and len(pivot) else float("nan"),
            "softmax_mean": float(pivot["softmax"].mean()) if "softmax" in pivot and len(pivot) else float("nan"),
            "note": "fewer than 2 paired observations -- cannot run a Wilcoxon test",
        }

    diffs = pivot["openset"] - pivot["softmax"]
    if (diffs == 0).all():
        statistic, p_value = float("nan"), 1.0
        note = "openset and softmax were identical on every pair"
    else:
        statistic, p_value = wilcoxon(pivot["openset"], pivot["softmax"])
        note = ""

    return {
        "metric": metric,
        "n_pairs": len(pivot),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "openset_mean": float(pivot["openset"].mean()),
        "softmax_mean": float(pivot["softmax"].mean()),
        "openset_std": float(pivot["openset"].std()),
        "softmax_std": float(pivot["softmax"].std()),
        "note": note,
    }


# --- Escalation-budget sweep: recall/AUROC vs. budget, not just budget=0.1 -
#
# H1 above (and this project's demo model) reports both gates at a single
# fixed budget (0.1). That one point can't show whether OpenMax's advantage
# holds across the operating range an analyst would actually choose from,
# or is an artifact of that particular budget. `escalation_budget_sweep_trial`
# reuses one LOFO fold's already-fit stage 2 and already-fit gate across
# every budget in the sweep -- only `conformal_gate.calibrate_threshold`
# (a quantile over already-computed calibration scores) depends on budget,
# so sweeping it costs nothing beyond the single-budget trial's fit cost.


def escalation_budget_sweep_trial(
    df: pd.DataFrame,
    family: str,
    feature_cols: List[str],
    budgets: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
    seed: int = 0,
    fit_frac: float = 0.75,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    train_frac: float = 0.7,
    stage2_n_estimators: int = 300,
) -> List[Dict[str, object]]:
    """One leave-one-family-out fold at one seed, both gates, swept across
    every budget in `budgets`. One dict per (gate, budget); see the module-
    level comment above for why stage 2 and the gate are each fit only
    once, not once per budget.
    """
    train, test = leave_one_family_out(
        df, family, label_col=label_col, timestamp_col=timestamp_col, train_frac=train_frac
    )
    fit_df, calib_df = random_holdout(train, holdout_frac=1.0 - fit_frac, random_state=seed)

    X_fit = fit_df[feature_cols].to_numpy()
    y_fit = fit_df[label_col].tolist()
    stage2 = AttackClassifier(Stage2Config(n_estimators=stage2_n_estimators, random_state=seed)).fit(X_fit, y_fit)

    X_calib = calib_df[feature_cols].to_numpy()
    known_test = test[test[label_col] != family]
    unknown_test = test[test[label_col] == family]
    X_known_test = known_test[feature_cols].to_numpy()
    X_unknown_test = unknown_test[feature_cols].to_numpy()

    rows: List[Dict[str, object]] = []
    for gate_name in _GATE_NAMES:
        gate = _fit_gate(gate_name, stage2, X_fit, y_fit)

        calib_scores = list(_gate_scores(gate, X_calib))
        known_um = _gate_scores(gate, X_known_test)
        unknown_um = _gate_scores(gate, X_unknown_test)

        for budget in budgets:
            conformal = calibrate_threshold(calib_scores, budget=budget) if calib_scores else None
            if conformal is not None and len(known_um):
                escalation_rate_on_known = float(np.mean([conformal.should_escalate(s) for s in known_um]))
            else:
                escalation_rate_on_known = float("nan")
            if conformal is not None and len(unknown_um):
                unknown_recall = float(np.mean([conformal.should_escalate(s) for s in unknown_um]))
            else:
                unknown_recall = float("nan")

            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "gate_name": gate_name,
                    "budget": budget,
                    "n_known_calib": len(calib_scores),
                    "n_known_test": len(known_um),
                    "n_unknown_test": len(unknown_um),
                    "escalation_rate_on_known": escalation_rate_on_known,
                    "unknown_recall": unknown_recall,
                }
            )
    return rows


def escalation_budget_sweep_report(
    df: pd.DataFrame,
    feature_cols: List[str],
    families: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    budgets: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
    fit_frac: float = 0.75,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    train_frac: float = 0.7,
    stage2_n_estimators: int = 300,
) -> pd.DataFrame:
    """Runs `escalation_budget_sweep_trial` for every (family, seed) pair,
    both gates, every budget. One row per (family, seed, gate, budget);
    `.groupby(["gate_name", "budget"])["unknown_recall"].mean()` on the
    result is the recall-vs-budget curve -- the tradeoff `ml/README.md`'s
    open-set section has flagged as missing (only budget=0.1 was reported).
    """
    if families is None:
        families = sorted(f for f in df[label_col].unique() if f != "BENIGN")
    rows: List[Dict[str, object]] = []
    for family in families:
        for seed in seeds:
            rows.extend(
                escalation_budget_sweep_trial(
                    df,
                    family,
                    feature_cols,
                    budgets=budgets,
                    seed=seed,
                    fit_frac=fit_frac,
                    label_col=label_col,
                    timestamp_col=timestamp_col,
                    train_frac=train_frac,
                    stage2_n_estimators=stage2_n_estimators,
                )
            )
    return pd.DataFrame(rows)


# --- Static vs. adaptive conformal calibration under concept drift --------
#
# H2 (this project's second research claim): a statically-calibrated
# escalation threshold's true escalation rate drifts away from its target
# budget under concept drift, while an online-adaptive one (Gibbs &
# Candès-style ACI, adaptive_conformal_gate.AdaptiveConformalGate) tracks
# the budget back down. This reuses the exact day-1/day-2 drift scenario
# `concept_drift_report` already uses -- the same fixture property that
# motivates this comparison in the first place.


@dataclass
class DriftCalibrationSegmentResult:
    segment: str  # "pre_drift" | "post_drift"
    n: int
    budget: float
    static_escalation_rate: float
    adaptive_escalation_rate: float
    static_error: float  # |static_escalation_rate - budget|
    adaptive_error: float  # |adaptive_escalation_rate - budget|
    # Detect-then-recalibrate baseline (drift_recalibration.py), streamed
    # over the same scores; NaN when the report is run without it.
    recalibrated_escalation_rate: float = float("nan")
    recalibrated_error: float = float("nan")
    n_recalibrations: int = 0  # cumulative, up to the end of this segment


def static_vs_adaptive_conformal_drift_report(
    window_a_df: pd.DataFrame,
    window_b_df: pd.DataFrame,
    feature_cols: List[str],
    budget: float = 0.1,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    fit_frac: float = 0.5,
    calib_frac: float = 0.3,
    gamma: float = 0.05,
    window_size: int = 50,
    stage2_n_estimators: int = 300,
    random_state: int = 0,
    gate_name: str = "softmax",
    recalib_window_size: int = 500,
    recalib_check_every: int = 100,
    recalib_drift_alpha: float = 0.01,
) -> List[DriftCalibrationSegmentResult]:
    """Compares a statically-calibrated escalation threshold against an
    online-adaptive one, streamed over known traffic across a real
    distribution shift.

    `window_a_df` (e.g. day 1) is split three ways in stream/time order:
    a `fit_frac` portion trains stage 2 (and the softmax gate riding on
    it), the next `calib_frac` of the remainder calibrates the static
    threshold and seeds the adaptive gate's window, and everything after
    that is the "pre_drift" evaluation segment -- still window_a's
    distribution, just temporally later than calibration. `window_b_df`
    (e.g. day 2, already shifted by `tests/generate_ml_fixtures.py`'s
    `DRIFT_FACTOR`) is the "post_drift" segment.

    Both gates are streamed over both segments in chronological order.
    The static gate's threshold never changes after calibration; the
    adaptive gate's `update()` is called on every flow, in order, so its
    threshold evolves exactly as it would in a live deployment. Only
    known traffic is used throughout -- this measures whether the
    escalation-rate *budget* holds under drift, which is orthogonal to
    the unknown-family recall question `openset_vs_softmax_report`
    answers.

    A third method, detect-then-recalibrate (`drift_recalibration.
    DetectThenRecalibrate`, the FIRCE-style baseline), is streamed over the
    same scores: static until a KS drift test on the recent window fires,
    then recalibrated on that window. `gate_name` picks the gate whose
    unknown_mass is thresholded (H2 was originally run on "softmax").
    """
    fit_df, remainder = random_holdout(window_a_df, holdout_frac=1.0 - fit_frac, random_state=random_state)
    remainder_sorted = remainder.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    calib_end = int(len(remainder_sorted) * calib_frac)
    calib_df = remainder_sorted.iloc[:calib_end]
    pre_drift_df = remainder_sorted.iloc[calib_end:]
    post_drift_df = window_b_df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

    X_fit = fit_df[feature_cols].to_numpy()
    y_fit = fit_df[label_col].tolist()
    stage2 = AttackClassifier(Stage2Config(n_estimators=stage2_n_estimators, random_state=random_state)).fit(
        X_fit, y_fit
    )
    gate = _fit_gate(gate_name, stage2, X_fit, y_fit, seed=random_state)

    calib_scores = list(_gate_scores(gate, calib_df[feature_cols].to_numpy()))
    static_gate = calibrate_threshold(calib_scores, budget=budget)
    adaptive_gate = AdaptiveConformalGate(budget=budget, gamma=gamma, window_size=window_size).seed(calib_scores)
    recalibrated_gate = DetectThenRecalibrate(
        budget,
        tail="upper",
        window_size=recalib_window_size,
        check_every=recalib_check_every,
        drift_alpha=recalib_drift_alpha,
    ).seed(calib_scores)

    results: List[DriftCalibrationSegmentResult] = []
    for segment_name, segment_df in (("pre_drift", pre_drift_df), ("post_drift", post_drift_df)):
        if len(segment_df) == 0:
            continue
        scores = _gate_scores(gate, segment_df[feature_cols].to_numpy())

        static_escalations = [static_gate.should_escalate(s) for s in scores]
        adaptive_escalations = [adaptive_gate.update(s) for s in scores]
        recalibrated_escalations = []
        for s in scores:
            recalibrated_escalations.append(recalibrated_gate.decide(s))
            recalibrated_gate.update(s)

        static_rate = float(np.mean(static_escalations))
        adaptive_rate = float(np.mean(adaptive_escalations))
        recalibrated_rate = float(np.mean(recalibrated_escalations))

        results.append(
            DriftCalibrationSegmentResult(
                segment=segment_name,
                n=len(scores),
                budget=budget,
                static_escalation_rate=static_rate,
                adaptive_escalation_rate=adaptive_rate,
                static_error=abs(static_rate - budget),
                adaptive_error=abs(adaptive_rate - budget),
                recalibrated_escalation_rate=recalibrated_rate,
                recalibrated_error=abs(recalibrated_rate - budget),
                n_recalibrations=recalibrated_gate.n_recalibrations,
            )
        )
    return results


def static_vs_adaptive_conformal_report(
    window_a_df: pd.DataFrame,
    window_b_df: pd.DataFrame,
    feature_cols: List[str],
    budget: float = 0.1,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    **kwargs,
) -> pd.DataFrame:
    """Runs `static_vs_adaptive_conformal_drift_report` across multiple
    seeds (each seed re-draws the fit/calibration split), returning one row
    per (seed, segment). `**kwargs` forwards to the per-seed call (e.g.
    `gamma`, `window_size`, `calib_frac`)."""
    rows = []
    for seed in seeds:
        for result in static_vs_adaptive_conformal_drift_report(
            window_a_df, window_b_df, feature_cols, budget=budget, random_state=seed, **kwargs
        ):
            row = asdict(result)
            row["seed"] = seed
            rows.append(row)
    return pd.DataFrame(rows)


def static_vs_adaptive_conformal_significance(report_df: pd.DataFrame, segment: Optional[str] = None) -> Dict[str, object]:
    """Paired Wilcoxon signed-rank test comparing `static_error` against
    `adaptive_error` (smaller is better -- distance from the target
    budget), paired by seed. Pass `segment` to restrict to "pre_drift" or
    "post_drift"; omit to pool both (each (seed, segment) pair is still one
    matched observation either way).

    Same statistical-power caveat as `openset_vs_softmax_significance`:
    few seeds means little power. Report the raw per-segment error means
    alongside this, never the p-value alone.
    """
    df = report_df if segment is None else report_df[report_df["segment"] == segment]
    static_errs = df["static_error"].to_numpy()
    adaptive_errs = df["adaptive_error"].to_numpy()
    n = len(df)

    if n < 2:
        return {
            "segment": segment or "pooled",
            "n_pairs": n,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "static_error_mean": float(static_errs.mean()) if n else float("nan"),
            "adaptive_error_mean": float(adaptive_errs.mean()) if n else float("nan"),
            "note": "fewer than 2 paired observations -- cannot run a Wilcoxon test",
        }

    diffs = static_errs - adaptive_errs
    if (diffs == 0).all():
        statistic, p_value = float("nan"), 1.0
        note = "static and adaptive errors were identical on every pair"
    else:
        statistic, p_value = wilcoxon(static_errs, adaptive_errs)
        note = ""

    return {
        "segment": segment or "pooled",
        "n_pairs": n,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "static_error_mean": float(static_errs.mean()),
        "adaptive_error_mean": float(adaptive_errs.mean()),
        "static_error_std": float(static_errs.std()),
        "adaptive_error_std": float(adaptive_errs.std()),
        "note": note,
    }


def drift_method_significance(
    report_df: pd.DataFrame, method_a: str, method_b: str, segment: Optional[str] = None
) -> Dict[str, object]:
    """Paired Wilcoxon on `{method}_error` (distance from the budget, smaller
    is better) for any two of "static", "adaptive", "recalibrated" in a
    `static_vs_adaptive_conformal_report` table, paired by (seed, segment).
    Same power caveat as `static_vs_adaptive_conformal_significance`."""
    df = report_df if segment is None else report_df[report_df["segment"] == segment]
    a = df[f"{method_a}_error"].to_numpy(dtype=float)
    b = df[f"{method_b}_error"].to_numpy(dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    out: Dict[str, object] = {
        "segment": segment or "pooled",
        "method_a": method_a,
        "method_b": method_b,
        "n_pairs": int(len(a)),
        "a_error_mean": float(a.mean()) if len(a) else float("nan"),
        "b_error_mean": float(b.mean()) if len(b) else float("nan"),
    }
    if len(a) < 2 or np.all(a == b):
        out.update(statistic=float("nan"), p_value=float("nan"), note="too few pairs or identical errors")
    else:
        statistic, p_value = wilcoxon(a, b)
        out.update(statistic=float(statistic), p_value=float(p_value), note="")
    return out


# --- Stage 1 recall control under drift: static vs adaptive vs recalibrate -
#
# The two-sided budget claim: the escalation gate holds a false-positive-
# side budget (H2), and Stage 1 should hold a miss-side budget. The static
# conformal guarantee (conformal_stage1.py) assumes calibration and live
# attacks are exchangeable; on a chronological stream they are not. This
# streams real traffic in time order and compares, at the same miss budget:
#   static        -- conformal threshold calibrated once;
#   adaptive      -- ACI on confirmed-attack feedback (adaptive_stage1.py);
#   recalibrated  -- static, recalibrated when a KS drift test fires
#                    (drift_recalibration.py, FIRCE-style baseline).
# Feedback for the two online methods arrives `feedback_delay` flows after
# the attack was scored, modelling triage latency.


_STAGE1_METHODS = ("static", "adaptive", "recalibrated")


def make_stage1(kind: str, fit_df: pd.DataFrame, feature_cols: List[str], seed: int = 0,
                label_col: str = "attack_category", contamination="auto", n_estimators: int = 200):
    """Fit the Stage 1 pre-filter used by an experiment.

    kind="iforest" is the original unsupervised Isolation Forest (kept so older
    results stay reproducible); kind="hybrid" is `stage1_hybrid.HybridPreFilter`,
    the current default of the deployed system, which needs labels and so is
    fitted on the same window with its attack labels.
    """
    X = fit_df[feature_cols].to_numpy()
    if kind == "iforest":
        return AnomalyPreFilter(Stage1Config(n_estimators=n_estimators, contamination=contamination, random_state=seed)).fit(X)
    if kind == "hybrid":
        return HybridPreFilter(HybridConfig(random_state=seed)).fit(X, (fit_df[label_col] != "BENIGN").to_numpy())
    raise ValueError(f"unknown stage1 kind: {kind!r}")


def stage1_stream_trial(
    df: pd.DataFrame,
    feature_cols: List[str],
    miss_budget: float = 0.1,
    seed: int = 0,
    fit_frac: float = 0.3,
    calib_frac: float = 0.15,
    gamma: float = 0.01,
    window_size: int = 500,
    feedback_delay: int = 0,
    contamination="auto",
    stage1_n_estimators: int = 200,
    recalib_window_size: int = 500,
    recalib_check_every: int = 100,
    recalib_drift_alpha: float = 0.01,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    stage1_kind: str = "iforest",
) -> List[Dict[str, object]]:
    """One seed, one budget, one feedback delay. The stream is the data in
    time order: the first `fit_frac` fits Stage 1 (unsupervised, as in
    training), the attacks in the next `calib_frac` calibrate the static
    threshold and seed both online methods, and everything after is the
    evaluation stream.

    Returns one dict per (method, segment). Segments: "overall", one per
    calendar day of the evaluation stream, and "seen_families" /
    "new_families" -- attacks of a family that does / does not appear in the
    calibration set. The static guarantee only covers seen families, so the
    split shows how much of any overshoot comes from never-calibrated
    families versus drift within known ones.
    """
    ordered = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    fit_end = int(n * fit_frac)
    calib_end = fit_end + int(n * calib_frac)
    fit_df = ordered.iloc[:fit_end]
    calib_df = ordered.iloc[fit_end:calib_end]
    eval_df = ordered.iloc[calib_end:].reset_index(drop=True)

    stage1 = make_stage1(stage1_kind, fit_df, feature_cols, seed=seed, label_col=label_col,
                         contamination=contamination, n_estimators=stage1_n_estimators)

    calib_attacks = calib_df[calib_df[label_col] != "BENIGN"]
    if len(calib_attacks) == 0:
        raise ValueError("calibration window contains no attacks; widen calib_frac")
    calib_scores = stage1.anomaly_score(calib_attacks[feature_cols].to_numpy())
    seen_families = set(calib_attacks[label_col])

    static_threshold = calibrate_stage1_threshold(calib_scores, miss_budget).threshold
    adaptive = AdaptiveStage1Threshold(miss_budget, gamma=gamma, window_size=window_size).seed(calib_scores)
    recalibrated = DetectThenRecalibrate(
        miss_budget,
        tail="lower",
        window_size=recalib_window_size,
        check_every=recalib_check_every,
        drift_alpha=recalib_drift_alpha,
    ).seed(calib_scores)

    scores = stage1.anomaly_score(eval_df[feature_cols].to_numpy())
    is_attack = (eval_df[label_col] != "BENIGN").to_numpy()
    flagged = {m: np.zeros(len(eval_df), dtype=bool) for m in _STAGE1_METHODS}
    flagged["static"] = scores >= static_threshold

    # Pending feedback: (release_index, score, adaptive threshold at decision).
    pending: List[tuple] = []
    head = 0
    adaptive_threshold = adaptive.threshold
    for i in range(len(eval_df)):
        while head < len(pending) and pending[head][0] <= i:
            _, s, thr = pending[head]
            adaptive.update(s, thr)
            recalibrated.update(s)
            head += 1
            adaptive_threshold = adaptive.threshold
        s = scores[i]
        flagged["adaptive"][i] = s >= adaptive_threshold
        flagged["recalibrated"][i] = recalibrated.decide(s)
        if is_attack[i]:
            pending.append((i + 1 + feedback_delay, s, adaptive_threshold))

    families = eval_df[label_col].to_numpy()
    days = pd.to_datetime(eval_df[timestamp_col]).dt.date.astype(str).to_numpy()
    seen_mask = np.array([f in seen_families for f in families])
    segments = {"overall": np.ones(len(eval_df), dtype=bool)}
    segments["seen_families"] = ~is_attack | seen_mask
    segments["new_families"] = ~is_attack | ~seen_mask
    for day in sorted(set(days)):
        segments[f"day_{day}"] = days == day

    rows: List[Dict[str, object]] = []
    for method in _STAGE1_METHODS:
        for seg_name, seg in segments.items():
            att = seg & is_attack
            ben = seg & ~is_attack
            n_att = int(att.sum())
            miss_rate = float((~flagged[method][att]).mean()) if n_att else float("nan")
            rows.append(
                {
                    "seed": seed,
                    "miss_budget": miss_budget,
                    "feedback_delay": feedback_delay,
                    "method": method,
                    "segment": seg_name,
                    "n_attacks": n_att,
                    "n_benign": int(ben.sum()),
                    "miss_rate": miss_rate,
                    "miss_error": abs(miss_rate - miss_budget) if n_att else float("nan"),
                    "miss_excess": max(miss_rate - miss_budget, 0.0) if n_att else float("nan"),
                    "benign_flag_rate": float(flagged[method][ben].mean()) if ben.any() else float("nan"),
                    "n_calibration_attacks": int(len(calib_scores)),
                    "n_recalibrations": recalibrated.n_recalibrations if method == "recalibrated" else 0,
                    "stage1_kind": stage1_kind,
                }
            )
    return rows


def stage1_stream_report(
    df: pd.DataFrame,
    feature_cols: List[str],
    miss_budgets: Sequence[float] = (0.05, 0.1, 0.2),
    seeds: Sequence[int] = (0, 1, 2),
    feedback_delays: Sequence[int] = (0,),
    **kwargs,
) -> pd.DataFrame:
    """`stage1_stream_trial` over every (budget, seed, delay). `**kwargs`
    forwards to the trial (fit_frac, gamma, window_size, ...)."""
    rows: List[Dict[str, object]] = []
    for budget in miss_budgets:
        for seed in seeds:
            for delay in feedback_delays:
                rows.extend(
                    stage1_stream_trial(
                        df, feature_cols, miss_budget=budget, seed=seed, feedback_delay=delay, **kwargs
                    )
                )
    return pd.DataFrame(rows)


# --- Cross-dataset transfer -----------------------------------------------
#
# Train and calibrate on one dataset (source), deploy on another (target)
# that shares the canonical CICFlowMeter features. This is the most severe
# drift available: different networks, hosts and attack tools. It asks
# whether each budget survives deployment somewhere new, and whether online
# adaptation recovers it.
#
# The adaptive gate is fed EVERY target flow, attacks included, exactly as
# pipeline.TwoStageDetector feeds it in production (it cannot know which
# flows are attacks). That is the honest setting, and it has a cost the
# report exposes: a gate that adapts to a stream containing attacks raises
# its threshold and escalates fewer of them.


def cross_dataset_trial(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: List[str],
    family_map: Optional[Dict[str, str]] = None,
    budget: float = 0.1,
    miss_budget: float = 0.1,
    seed: int = 0,
    gate_names: Sequence[str] = ALL_GATE_NAMES,
    stage2_n_estimators: int = 300,
    gamma: float = 0.05,
    window_size: int = 50,
    stage1_gamma: float = 0.01,
    stage1_window_size: int = 500,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    stage1_kind: str = "iforest",
) -> List[Dict[str, object]]:
    """One seed of source -> target transfer. `family_map` maps target
    attack names onto source families they correspond to (e.g.
    {"ddos": "DoS/DDoS"}); mapped target attacks are "shared", the rest are
    "novel" -- genuinely unseen families. Returns one row per gate (plus a
    "stage1" row) with escalation and miss metrics under the static,
    adaptive and recalibrated thresholds.
    """
    family_map = family_map or {}
    fit_df, calib_df = random_holdout(source_df, holdout_frac=0.25, random_state=seed)
    X_fit = fit_df[feature_cols].to_numpy()
    y_fit = fit_df[label_col].tolist()
    stage2 = AttackClassifier(Stage2Config(n_estimators=stage2_n_estimators, random_state=seed)).fit(X_fit, y_fit)

    target = (
        target_df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
        if timestamp_col in target_df.columns
        else target_df.reset_index(drop=True)
    )
    X_target = target[feature_cols].to_numpy()
    t_labels = target[label_col].to_numpy()
    is_benign = t_labels == "BENIGN"
    mapped = np.array([family_map.get(l) for l in t_labels], dtype=object)
    is_shared = ~is_benign & np.array([m is not None for m in mapped])
    is_novel = ~is_benign & ~is_shared

    rows: List[Dict[str, object]] = []
    for gate_name in gate_names:
        gate = _fit_gate(gate_name, stage2, X_fit, y_fit, seed=seed)
        calib_scores = list(_gate_scores(gate, calib_df[feature_cols].to_numpy()))
        if gate_name == "openset":
            results = gate.recalibrate_batch(X_target)
            scores = np.array([r.unknown_mass for r in results])
            predicted = np.array([r.predicted_class for r in results])
            del results
        else:
            scores = _gate_scores(gate, X_target)
            predicted = np.asarray(stage2.predict(X_target))

        static = calibrate_threshold(calib_scores, budget=budget)
        adaptive = AdaptiveConformalGate(budget=budget, gamma=gamma, window_size=window_size).seed(calib_scores)
        recalibrated = DetectThenRecalibrate(budget, tail="upper").seed(calib_scores)
        escalated = {"static": scores > static.threshold, "adaptive": np.zeros(len(scores), bool), "recalibrated": np.zeros(len(scores), bool)}
        for i, s in enumerate(scores):
            escalated["adaptive"][i] = adaptive.update(s)
            escalated["recalibrated"][i] = recalibrated.decide(s)
            recalibrated.update(s)

        row: Dict[str, object] = {
            "seed": seed,
            "gate_name": gate_name,
            "n_target_benign": int(is_benign.sum()),
            "n_target_shared": int(is_shared.sum()),
            "n_target_novel": int(is_novel.sum()),
            "shared_family_accuracy": float((predicted[is_shared] == mapped[is_shared]).mean()) if is_shared.any() else float("nan"),
            "novel_auroc": float(roc_auc_score(np.r_[np.zeros(is_benign.sum()), np.ones(is_novel.sum())], np.r_[scores[is_benign], scores[is_novel]]))
            if is_benign.any() and is_novel.any()
            else float("nan"),
            "n_recalibrations": recalibrated.n_recalibrations,
        }
        for method, esc in escalated.items():
            rate = float(esc[is_benign].mean()) if is_benign.any() else float("nan")
            row[f"{method}_benign_escalation_rate"] = rate
            row[f"{method}_budget_error"] = abs(rate - budget) if is_benign.any() else float("nan")
            row[f"{method}_novel_recall"] = float(esc[is_novel].mean()) if is_novel.any() else float("nan")
        rows.append(row)

    # Stage 1: IsolationForest fit on source, miss budget calibrated on source
    # attacks; target attacks fed back to the adaptive threshold in order.
    stage1 = make_stage1(stage1_kind, fit_df, feature_cols, seed=seed, label_col=label_col)
    src_attack_scores = stage1.anomaly_score(calib_df[calib_df[label_col] != "BENIGN"][feature_cols].to_numpy())
    t_scores = stage1.anomaly_score(X_target)
    static_thr = calibrate_stage1_threshold(src_attack_scores, miss_budget).threshold
    adaptive1 = AdaptiveStage1Threshold(miss_budget, gamma=stage1_gamma, window_size=stage1_window_size).seed(src_attack_scores)
    flagged_adaptive = np.zeros(len(t_scores), bool)
    thr = adaptive1.threshold
    for i, s in enumerate(t_scores):
        flagged_adaptive[i] = s >= thr
        if not is_benign[i]:
            adaptive1.update(s, thr)
            thr = adaptive1.threshold
    attacks = ~is_benign
    rows.append(
        {
            "seed": seed,
            "gate_name": "stage1",
            "n_target_benign": int(is_benign.sum()),
            "n_target_shared": int(is_shared.sum()),
            "n_target_novel": int(is_novel.sum()),
            "static_miss_rate": float((t_scores[attacks] < static_thr).mean()) if attacks.any() else float("nan"),
            "adaptive_miss_rate": float((~flagged_adaptive[attacks]).mean()) if attacks.any() else float("nan"),
            "static_benign_flag_rate": float((t_scores[is_benign] >= static_thr).mean()) if is_benign.any() else float("nan"),
            "adaptive_benign_flag_rate": float(flagged_adaptive[is_benign].mean()) if is_benign.any() else float("nan"),
            "miss_budget": miss_budget,
        }
    )
    return rows


def cross_dataset_report(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: List[str],
    seeds: Sequence[int] = (0, 1, 2),
    **kwargs,
) -> pd.DataFrame:
    """`cross_dataset_trial` over seeds; `**kwargs` forwards to the trial."""
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        rows.extend(cross_dataset_trial(source_df, target_df, feature_cols, seed=seed, **kwargs))
    return pd.DataFrame(rows)


# --- Auditing AdaptiveConformalGate's production ground-truth-proxy -------
#
# adaptive_conformal_gate.py's module docstring ("Production ground-truth-
# proxy assumption") documents the assumption `pipeline.TwoStageDetector`
# adopts in production: every scored flow is fed to `update()` as if it
# were known traffic, on the assumption that genuine novel/attack traffic
# is a small enough share of the stream not to matter. This can't be
# validated in real time (that's exactly the ground truth the assumption
# exists to work around), but it can be *audited* after the fact against
# analyst triage -- dashboard-api's existing `triage_status` workflow,
# which already reviews escalated alerts for other reasons.


def audit_gate_against_triage(triage_records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Estimates how often the production ground-truth-proxy assumption was
    actually violated, from analyst-triaged escalated alerts.

    `triage_records` -- one dict per alert, at minimum `escalated: bool` and
    `triage_status: str` (dashboard-api's `TRIAGE_STATUSES`: "new"/
    "acknowledged"/"confirmed"/"false_positive") -- e.g. rows pulled from
    dashboard-api's `alerts` table.

    Only escalated alerts are ever triaged in this project's workflow, so
    this can only audit "of what we escalated, how much genuinely wasn't
    known traffic" -- it has no visibility into the symmetric failure (a
    genuine novel attack that never got escalated at all, so no analyst
    ever saw it to triage). That blind spot is inherent to this proxy, not
    a gap in this function; see the module docstring above.

    A `false_positive` triage means the analyst determined the escalated
    flow genuinely was known/benign -- exactly what the proxy assumed when
    it fed that flow into `update()`, so this is the *expected*, assumption-
    consistent outcome. A `confirmed` triage means the flow genuinely
    wasn't known traffic, so the proxy's "treat as known" assumption was
    wrong for that flow -- `confirmed_fraction` is the empirical rate of
    that violation among triaged escalations, and is the number to watch:
    a value the operator judges too high (this function doesn't set that
    threshold -- it depends on the deployment's own risk tolerance) is the
    signal the assumption may no longer hold, not something this function
    corrects on its own.
    """
    escalated = [r for r in triage_records if r.get("escalated")]
    triaged = [r for r in escalated if r.get("triage_status") in ("confirmed", "false_positive")]
    n_confirmed = sum(1 for r in triaged if r["triage_status"] == "confirmed")
    n_false_positive = sum(1 for r in triaged if r["triage_status"] == "false_positive")

    return {
        "n_escalated": len(escalated),
        "n_escalated_triaged": len(triaged),
        "n_confirmed": n_confirmed,
        "n_false_positive": n_false_positive,
        "triage_coverage": float(len(triaged) / len(escalated)) if escalated else float("nan"),
        "confirmed_fraction": float(n_confirmed / len(triaged)) if triaged else float("nan"),
    }


# --- Latency / throughput ---------------------------------------------
#
# Real-time feasibility evidence: every open-set/LLM-escalation paper this
# project checked against (DQN-IDS most directly -- arXiv, NDSS SDIoTSec
# 2026) reports per-flow inference latency as the first thing a reviewer
# looks for, not an afterthought. Measured here the same way DQN-IDS
# measured its own two-stage CNN+DQN pipeline: warm-up iterations
# discarded, then median/p95/p99 over single-flow calls (how a live
# Kafka consumer actually calls `TwoStageDetector.score`, one flow at a
# time) plus a batched-throughput number.


@dataclass
class LatencyReport:
    n_warmup: int
    n_trials: int
    single_median_ms: float
    single_p95_ms: float
    single_p99_ms: float
    single_throughput_per_sec: float
    batch_size: int
    batch_median_ms: float  # per batched score() call, not per row
    batch_throughput_per_sec: float  # rows/sec when batched


def latency_report(detector, X: np.ndarray, warmup: int = 50, n_trials: int = 200, batch_size: int = 64) -> LatencyReport:
    """Measures `pipeline.TwoStageDetector.score`'s inference latency on
    real (already-fit) stage1/stage2 models, both single-flow (one row per
    call -- how `scoring_service.ScoringService.process_event` actually
    calls it) and batched. `warmup` calls are timed but discarded first --
    XGBoost/sklearn's first few calls include cache/allocation overhead
    that isn't representative of steady-state serving latency (the same
    warm-up pattern DQN-IDS's own runtime table uses).
    """
    n = X.shape[0]
    if n < 1:
        raise ValueError("X must have at least one row")

    for i in range(warmup):
        detector.score(X[i % n : i % n + 1])

    single_times_ms = np.empty(n_trials)
    for i in range(n_trials):
        idx = (warmup + i) % n
        t0 = time.perf_counter()
        detector.score(X[idx : idx + 1])
        t1 = time.perf_counter()
        single_times_ms[i] = (t1 - t0) * 1000.0

    actual_batch_size = min(batch_size, n)
    # Non-overlapping windows of actual_batch_size rows, up to 20 of them
    # -- bounded by n // actual_batch_size so every window stays in range.
    n_batches = max(1, min(20, n // actual_batch_size))
    batch_times_ms = np.empty(n_batches)
    for b in range(n_batches):
        start = b * actual_batch_size
        batch = X[start : start + actual_batch_size]
        t0 = time.perf_counter()
        detector.score(batch)
        t1 = time.perf_counter()
        batch_times_ms[b] = (t1 - t0) * 1000.0

    batch_median = float(np.median(batch_times_ms))

    return LatencyReport(
        n_warmup=warmup,
        n_trials=n_trials,
        single_median_ms=float(np.median(single_times_ms)),
        single_p95_ms=float(np.percentile(single_times_ms, 95)),
        single_p99_ms=float(np.percentile(single_times_ms, 99)),
        single_throughput_per_sec=float(1000.0 / np.median(single_times_ms)),
        batch_size=actual_batch_size,
        batch_median_ms=batch_median,
        batch_throughput_per_sec=float(actual_batch_size * 1000.0 / batch_median) if batch_median else float("nan"),
    )


def amortized_latency_ms(tier1_median_ms: float, escalation_rate: float, tier2_latency_ms: float) -> float:
    """Expected end-to-end per-flow latency: every flow pays Tier 1's
    (stage1+stage2) cost; only the `escalation_rate` fraction additionally
    pays Tier 2's cost. This -- not Tier 2's raw latency alone -- is the
    number that determines real-time feasibility, which is the entire
    point of routing only the escalated minority to a heavier downstream
    tier instead of running it on every flow (see `tier2_reasoner`'s
    README for the measured Tier 2 latency this gets combined with, and
    `ml/README.md`'s real-data escalation rates for realistic
    `escalation_rate` values instead of a guessed one).
    """
    return tier1_median_ms + escalation_rate * tier2_latency_ms
