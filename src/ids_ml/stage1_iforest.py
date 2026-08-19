"""Stage 1: Isolation Forest as a cheap, unsupervised anomaly pre-filter.

Isolation Forest is fit without labels (in practice, on training data that
is overwhelmingly benign, as real traffic is) and used to flag flows that
look statistically unusual. Its job is cheap recall on "this looks weird",
not attack-type precision -- that's stage 2 (stage2_xgboost.py). Keeping
stage 1 unsupervised and inexpensive is what lets the pipeline run stage 2's
heavier classifier on only a small flagged subset of live traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class Stage1Config:
    n_estimators: int = 200
    contamination: str = "auto"
    random_state: int = 0
    # None -> use IsolationForest's own inlier/outlier boundary (predict() == -1).
    # A float pins the flag to a specific point on anomaly_score()'s scale
    # instead, for callers who want to tune the pre-filter's flag rate directly.
    flag_threshold: Optional[float] = None


class AnomalyPreFilter:
    """Fit/score/flag wrapper around IsolationForest + feature scaling."""

    def __init__(self, config: Optional[Stage1Config] = None) -> None:
        self.config = config or Stage1Config()
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> "AnomalyPreFilter":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous.

        Note this is the negation of sklearn's `score_samples` (where more
        negative means more anomalous) so callers get an intuitive
        "bigger number = weirder" scale.
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return -self.model.score_samples(X_scaled)

    def flag(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.config.flag_threshold is None:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled) == -1
        return self.anomaly_score(X) >= self.config.flag_threshold

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("AnomalyPreFilter must be fit() before use")

    def save(self, path) -> None:
        self._check_fitted()
        joblib.dump({"scaler": self.scaler, "model": self.model, "config": self.config}, path)

    @classmethod
    def load(cls, path) -> "AnomalyPreFilter":
        payload = joblib.load(path)
        instance = cls(payload["config"])
        instance.scaler = payload["scaler"]
        instance.model = payload["model"]
        instance._fitted = True
        return instance
