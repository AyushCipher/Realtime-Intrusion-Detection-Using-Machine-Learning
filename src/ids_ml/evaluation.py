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

from typing import Dict, List, Optional, Sequence

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .split import random_split, time_based_split
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
