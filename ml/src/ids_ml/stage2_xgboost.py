"""Stage 2: XGBoost multi-class classifier for attack-type classification.

Trained on the full labeled training set (BENIGN plus every attack
category), not just the subset stage 1 would flag -- that keeps its
per-category metrics (see evaluation.py) meaningful as a standalone
classifier. In the live pipeline (pipeline.py), it is only *run* on flows
stage 1 flags, so the cascade's actual cost profile matches the "cheap
pre-filter, heavier classifier on the minority" design in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


@dataclass
class Stage2Config:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 0
    n_jobs: int = -1


class AttackClassifier:
    """fit/predict/predict_proba wrapper around XGBClassifier with label encoding."""

    def __init__(self, config: Optional[Stage2Config] = None) -> None:
        self.config = config or Stage2Config()
        self.label_encoder = LabelEncoder()
        self.model: Optional[xgb.XGBClassifier] = None
        self._fitted = False

    def fit(self, X: np.ndarray, y_labels: List[str]) -> "AttackClassifier":
        y = self.label_encoder.fit_transform(list(y_labels))
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            eval_metric="mlogloss",
        )
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y = self.model.predict(X)
        return self.label_encoder.inverse_transform(y.astype(int))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns an (n_samples, n_classes) array; column order matches `classes_`."""
        self._check_fitted()
        return self.model.predict_proba(X)

    @property
    def classes_(self) -> List[str]:
        return list(self.label_encoder.classes_)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("AttackClassifier must be fit() before use")

    def save(self, path) -> None:
        self._check_fitted()
        joblib.dump({"model": self.model, "label_encoder": self.label_encoder, "config": self.config}, path)

    @classmethod
    def load(cls, path) -> "AttackClassifier":
        payload = joblib.load(path)
        instance = cls(payload["config"])
        instance.model = payload["model"]
        instance.label_encoder = payload["label_encoder"]
        instance._fitted = True
        return instance
