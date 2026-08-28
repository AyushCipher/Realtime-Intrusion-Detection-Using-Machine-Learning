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
from .conformal_gate import calibrate_threshold
from .openset_head import OpenMaxConfig, OpenMaxHead
from .softmax_gate import SoftmaxGate
from .split import leave_one_family_out, random_holdout, random_split, time_based_split
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


@dataclass
class OpenSetTrialResult:
    family: str
    seed: int
    gate_name: str  # "softmax" | "openset"
    n_known_calib: int
    n_known_test: int
    n_unknown_test: int
    escalation_rate_on_known: float  # should track `budget` on the calibration set by construction; this is its out-of-sample analogue on the test split's known rows
    unknown_recall: float  # fraction of held-out-family test rows escalated
    unknown_auroc: float  # unknown_mass as a score distinguishing known vs held-out-family test rows
    ece: float
    brier: float


def _fit_gate(gate_name: str, stage2: AttackClassifier, X_fit, y_fit):
    if gate_name == "softmax":
        return SoftmaxGate(stage2)
    if gate_name == "openset":
        return OpenMaxHead(stage2, OpenMaxConfig()).fit(X_fit, y_fit)
    raise ValueError(f"unknown gate_name: {gate_name!r}")


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
) -> List[OpenSetTrialResult]:
    """One leave-one-family-out fold at one seed, both gates head-to-head.

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

    results: List[OpenSetTrialResult] = []
    for gate_name in _GATE_NAMES:
        gate = _fit_gate(gate_name, stage2, X_fit, y_fit)

        calib_scores = [r.unknown_mass for r in gate.recalibrate_batch(X_calib)] if len(X_calib) else []
        conformal = calibrate_threshold(calib_scores, budget=budget) if calib_scores else None

        known_um = np.array(
            [r.unknown_mass for r in gate.recalibrate_batch(X_known_test)] if len(X_known_test) else []
        )
        unknown_um = np.array(
            [r.unknown_mass for r in gate.recalibrate_batch(X_unknown_test)] if len(X_unknown_test) else []
        )

        if conformal is not None and len(known_um):
            escalation_rate_on_known = float(np.mean([conformal.should_escalate(s) for s in known_um]))
        else:
            escalation_rate_on_known = float("nan")
        if conformal is not None and len(unknown_um):
            unknown_recall = float(np.mean([conformal.should_escalate(s) for s in unknown_um]))
        else:
            unknown_recall = float("nan")

        y_true = np.concatenate([np.zeros(len(known_um)), np.ones(len(unknown_um))])
        y_score = np.concatenate([known_um, unknown_um])
        if len(known_um) and len(unknown_um):
            auroc = float(roc_auc_score(y_true, y_score))
            brier = float(brier_score_loss(y_true, y_score))
            ece = expected_calibration_error(y_true, y_score)
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
                ece=ece,
                brier=brier,
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
) -> pd.DataFrame:
    """Runs `run_openset_trial` for every (family, seed) pair, both gates --
    the full leave-one-family-out rotation, repeated across seeds so a
    single lucky/unlucky train/calibration split doesn't decide the result.
    One row per (family, seed, gate). See `openset_vs_softmax_significance`
    for the paired comparison on top of this raw table.
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
            ):
                rows.append(asdict(result))
    return pd.DataFrame(rows)


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
    gate = SoftmaxGate(stage2)

    calib_scores = [r.unknown_mass for r in gate.recalibrate_batch(calib_df[feature_cols].to_numpy())]
    static_gate = calibrate_threshold(calib_scores, budget=budget)
    adaptive_gate = AdaptiveConformalGate(budget=budget, gamma=gamma, window_size=window_size).seed(calib_scores)

    results: List[DriftCalibrationSegmentResult] = []
    for segment_name, segment_df in (("pre_drift", pre_drift_df), ("post_drift", post_drift_df)):
        if len(segment_df) == 0:
            continue
        scores = [r.unknown_mass for r in gate.recalibrate_batch(segment_df[feature_cols].to_numpy())]

        static_escalations = [static_gate.should_escalate(s) for s in scores]
        adaptive_escalations = [adaptive_gate.update(s) for s in scores]

        static_rate = float(np.mean(static_escalations))
        adaptive_rate = float(np.mean(adaptive_escalations))

        results.append(
            DriftCalibrationSegmentResult(
                segment=segment_name,
                n=len(scores),
                budget=budget,
                static_escalation_rate=static_rate,
                adaptive_escalation_rate=adaptive_rate,
                static_error=abs(static_rate - budget),
                adaptive_error=abs(adaptive_rate - budget),
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
