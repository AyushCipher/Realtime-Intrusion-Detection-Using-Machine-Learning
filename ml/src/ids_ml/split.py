"""Time-based train/val/test splitting.

CRITICAL: this module's default (and the only split used for reported
metrics) is chronological, not random. A random split lets flows from the
same near-simultaneous attack burst -- which share near-identical feature
values -- land on both sides of the split, so the model effectively
memorizes attack instances it will "predict" in the test set. That inflates
scores in a way that does not hold up on genuinely future traffic.
`random_split` exists only so `evaluation.leakage_comparison_report` can
demonstrate the size of that inflation; it must not be used to select or
report a deployed model's metrics.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by timestamp and cut chronologically into train/val/test.

    Every row in val is strictly later than every row in train, and every
    row in test is strictly later than every row in val (ties at the cut
    boundary aside), so val/test genuinely simulate "future" traffic.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"'{timestamp_col}' column required for a time-based split")
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum to < 1")

    ordered = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = ordered.iloc[:train_end]
    val_df = ordered.iloc[train_end:val_end]
    test_df = ordered.iloc[val_end:]
    return train_df, val_df, test_df


def random_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A conventional shuffled split. Only for quantifying leakage inflation
    (see module docstring) -- never for selecting or reporting a model."""
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum to < 1")

    rng = np.random.default_rng(random_state)
    shuffled = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return shuffled.iloc[:train_end], shuffled.iloc[train_end:val_end], shuffled.iloc[val_end:]


def time_window_split(
    df: pd.DataFrame,
    cutoff,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (before cutoff, at-or-after cutoff) for concept-drift testing:
    train on one window, evaluate on a later one."""
    if timestamp_col not in df.columns:
        raise ValueError(f"'{timestamp_col}' column required for a time-window split")
    before = df[df[timestamp_col] < cutoff]
    after = df[df[timestamp_col] >= cutoff]
    return before, after


def random_holdout(
    df: pd.DataFrame,
    holdout_frac: float = 0.25,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A random 2-way split with no chronological meaning, for holdout sets
    where order genuinely doesn't matter -- e.g. calibrating an escalation
    threshold (`conformal_gate.calibrate_threshold`) on a slice of known
    traffic held out from the classifier's own training data, which is a
    different concern from same-burst leakage across time (that's what
    `time_based_split` guards against). Not for train/test *model*
    evaluation splits -- use `time_based_split` for those.

    Returns (kept, holdout).
    """
    if not (0 < holdout_frac < 1):
        raise ValueError("holdout_frac must be in (0, 1)")
    rng = np.random.default_rng(random_state)
    shuffled = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    holdout_end = int(len(shuffled) * holdout_frac)
    return shuffled.iloc[holdout_end:].reset_index(drop=True), shuffled.iloc[:holdout_end].reset_index(drop=True)


def leave_one_family_out(
    df: pd.DataFrame,
    held_out_family: str,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    train_frac: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Zero-day / open-set holdout split for one attack family.

    `held_out_family`'s rows never appear in train, regardless of
    timestamp -- the model must not see even an early instance of the
    family it's being tested on for unknown-attack detection. The
    remaining ("known") rows still use this module's chronological
    same-burst-leakage rule (see module docstring): they're time-sorted and
    cut at `train_frac`, so known-class evaluation isn't accidentally
    inflated by the leakage `time_based_split` exists to avoid. Test is the
    union of the known rows' chronological tail and every held-out-family
    row -- i.e. known classes are evaluated on genuinely future traffic,
    and the held-out family (whose true label is only used for scoring,
    never for training) represents "an attack family never seen before".

    Holding out `BENIGN` is refused: stage 1 (`stage1_iforest.py`) is fit
    unsupervised on training data assumed to be overwhelmingly benign, so a
    training set with no benign rows at all isn't a meaningful "known"
    baseline for anything downstream.
    """
    if held_out_family == "BENIGN":
        raise ValueError("cannot hold out BENIGN -- stage 1 needs benign rows to train on")
    if held_out_family not in set(df[label_col]):
        raise ValueError(f"'{held_out_family}' does not appear in '{label_col}'")

    known = df[df[label_col] != held_out_family]
    unknown = df[df[label_col] == held_out_family]

    known_sorted = known.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    train_end = int(len(known_sorted) * train_frac)
    train_df = known_sorted.iloc[:train_end]
    known_test_df = known_sorted.iloc[train_end:]

    test_df = pd.concat([known_test_df, unknown], ignore_index=True)
    return train_df, test_df


def rotate_holdout(
    df: pd.DataFrame,
    families: Optional[Sequence[str]] = None,
    label_col: str = "attack_category",
    timestamp_col: str = "timestamp",
    train_frac: float = 0.7,
) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yields (family, train_df, test_df) for one `leave_one_family_out`
    fold per attack family (BENIGN excluded -- see that function), so
    `evaluation.py` can run the full leave-one-family-out rotation and
    report per-family known-class and unknown-detection metrics
    separately, never merged into one blended accuracy number.
    """
    if families is None:
        families = sorted(f for f in df[label_col].unique() if f != "BENIGN")
    for family in families:
        train_df, test_df = leave_one_family_out(
            df, family, label_col=label_col, timestamp_col=timestamp_col, train_frac=train_frac
        )
        yield family, train_df, test_df
