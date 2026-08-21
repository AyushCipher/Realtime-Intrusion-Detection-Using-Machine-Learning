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

from typing import Tuple

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
