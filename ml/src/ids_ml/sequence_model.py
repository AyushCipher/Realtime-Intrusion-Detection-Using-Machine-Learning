"""Sequence-aware model over per-source-IP flow history (optional).

Motivation: a single flow's features can't distinguish "one odd packet" from
step 3 of a recon -> brute-force -> exfiltration campaign -- that needs
looking at what the same source IP did recently. The natural model for that
is a recurrent one (LSTM/GRU) over each source IP's flow sequence, as the
project scope calls for.

This environment's PyTorch install is blocked by the host's Application
Control (WDAC) security policy -- `import torch` fails at DLL load time
with `OSError: [WinError 4551] An Application Control policy has blocked
this file... shm.dll`. That's a machine-level security policy, not
something this module works around. Rather than silently drop the
"optional" sequence-model component, this implements the same
sequence-aware objective -- classify a flow using its source IP's recent
flow history, in a sliding window -- with an MLPClassifier over flattened
windows: a lightweight, dependency-light stand-in available in this
environment.

`build_sequences` produces the same (windowed samples, next-flow label)
contract a recurrent model would consume, so swapping in a real LSTM/GRU
later (where PyTorch is actually loadable) means swapping
SequenceAttackModel's internals only, not the data pipeline. See the
README's known-limitations section for what a real recurrent model would
likely add over this: an MLP over a flattened window treats each position
independently and has no notion of order beyond concatenation order, where
an LSTM/GRU would learn temporal dynamics directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .features import CANONICAL_FEATURE_COLUMNS


def build_sequences(
    df: pd.DataFrame,
    window: int = 3,
    group_col: str = "src_ip",
    timestamp_col: str = "timestamp",
    feature_cols: Optional[List[str]] = None,
    label_col: str = "attack_category",
) -> Tuple[np.ndarray, np.ndarray]:
    """Slides a `window`-flow window per source IP; each window becomes one
    sample: X = that window's features flattened in chronological order,
    y = the label of the window's most recent (last) flow.

    A source IP with fewer than `window` flows contributes no samples --
    there's no history yet to look back on for it.
    """
    feature_cols = feature_cols or CANONICAL_FEATURE_COLUMNS
    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' column required to build per-source-IP sequences")

    X_rows: List[np.ndarray] = []
    y_rows: List[object] = []
    for _, group in df.groupby(group_col):
        ordered = group.sort_values(timestamp_col, kind="mergesort") if timestamp_col in group.columns else group
        feats = ordered[feature_cols].to_numpy()
        labels = ordered[label_col].to_numpy()
        for end in range(window, len(ordered) + 1):
            X_rows.append(feats[end - window : end].flatten())
            y_rows.append(labels[end - 1])

    if not X_rows:
        return np.empty((0, window * len(feature_cols))), np.empty((0,), dtype=object)
    return np.vstack(X_rows), np.array(y_rows)


@dataclass
class SequenceModelConfig:
    window: int = 3
    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    max_iter: int = 500
    random_state: int = 0


class SequenceAttackModel:
    """MLP over flattened per-source-IP flow windows.

    See the module docstring for why this stands in for an LSTM/GRU in this
    environment, and what a real recurrent model would add on top of it.
    """

    def __init__(self, config: Optional[SequenceModelConfig] = None) -> None:
        self.config = config or SequenceModelConfig()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y_labels) -> "SequenceAttackModel":
        X_scaled = self.scaler.fit_transform(X)
        y = self.label_encoder.fit_transform(list(y_labels))
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        y = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y.astype(int))

    @property
    def classes_(self) -> List[str]:
        return list(self.label_encoder.classes_)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SequenceAttackModel must be fit() before use")
