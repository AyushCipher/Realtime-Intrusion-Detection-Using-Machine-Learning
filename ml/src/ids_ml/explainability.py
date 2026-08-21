"""Per-prediction SHAP (TreeSHAP) explanations for the dashboard.

Uses XGBoost's own `pred_contribs=True` prediction mode rather than the
`shap` package's `TreeExplainer`. Both compute the same TreeSHAP algorithm
(Lundberg & Lee, 2017) for tree ensembles -- but at the versions available
in this environment (xgboost>=2.x, which serializes a per-class
`base_score` for multiclass models) the `shap` package's external model
parser fails to load that base_score format:

    ValueError: could not convert string to float:
    '[2.9169626E0,-4.0982914E-1,...]'

This is a known version-compatibility gap between `shap`'s XGBoost model
loader and newer XGBoost releases, not a bug in this module. XGBoost's
native `pred_contribs` path computes identical Shapley values without
depending on `shap` being able to parse the model file, so it's used here
directly. See the README's known-limitations section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import xgboost as xgb

from .features import CANONICAL_FEATURE_COLUMNS
from .stage2_xgboost import AttackClassifier


@dataclass
class FeatureContribution:
    feature: str
    value: float
    shap_value: float

    def to_dict(self) -> Dict[str, float]:
        return {"feature": self.feature, "value": self.value, "shap_value": self.shap_value}


class ShapExplainer:
    """TreeSHAP explanations for a fitted stage-2 `AttackClassifier`."""

    def __init__(self, classifier: AttackClassifier, feature_names: Optional[List[str]] = None) -> None:
        if classifier.model is None:
            raise RuntimeError("AttackClassifier must be fit() before building an explainer")
        self.classifier = classifier
        self.feature_names = feature_names or CANONICAL_FEATURE_COLUMNS

    def explain_batch(
        self, X: np.ndarray, predicted_classes: Sequence[str], top_k: int = 5
    ) -> List[List[FeatureContribution]]:
        X = np.asarray(X)
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        # shape: (n_samples, n_classes, n_features + 1); last column is the bias term.
        contribs = self.classifier.model.get_booster().predict(dmatrix, pred_contribs=True)
        class_index = {c: i for i, c in enumerate(self.classifier.classes_)}

        results: List[List[FeatureContribution]] = []
        for row in range(X.shape[0]):
            class_idx = class_index[predicted_classes[row]]
            row_contribs = contribs[row, class_idx, : len(self.feature_names)]
            order = np.argsort(-np.abs(row_contribs))[:top_k]
            results.append(
                [
                    FeatureContribution(
                        feature=self.feature_names[i],
                        value=float(X[row, i]),
                        shap_value=float(row_contribs[i]),
                    )
                    for i in order
                ]
            )
        return results

    def explain(self, x_row: np.ndarray, predicted_class: str, top_k: int = 5) -> List[FeatureContribution]:
        return self.explain_batch(np.asarray(x_row).reshape(1, -1), [predicted_class], top_k=top_k)[0]


def explanation_to_alert_field(contributions: List[FeatureContribution]) -> List[Dict[str, float]]:
    """Converts a FeatureContribution list into the JSON-serializable form
    expected by schema.ALERT_EVENT_FIELDS["explanation"]."""
    return [c.to_dict() for c in contributions]
