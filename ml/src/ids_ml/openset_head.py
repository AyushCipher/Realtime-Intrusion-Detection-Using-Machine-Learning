"""OpenMax open-set recalibration (Bendale & Boult, CVPR 2016) over stage
2's raw per-class activations.

OpenMax was proposed for a deep network's pre-softmax logit vector. This
module adapts it to XGBoost's per-class raw margins
(`Booster.predict(output_margin=True)`), which play the same structural
role here: an unnormalized per-class score vector, computed before the
softmax that turns it into `stage2_xgboost`'s `predict_proba`. That's a
deliberate substitution, not the original formulation -- XGBoost's
multiclass margins are additive log-odds-like scores per class, so the
same "mean activation vector + Weibull tail" recalibration applies to them
structurally, but this does not inherit whatever regularities a trained
deep network's logit geometry might otherwise have.

Algorithm, per known class c (Algorithm 2 in the paper, adapted):
1. Fit time: collect the margin vectors of every *correctly classified*
   training example of class c; compute their mean ("mean activation
   vector", MAV_c) and fit a Weibull distribution to the `tail_size`
   largest distances from MAV_c -- how far a genuine example of class c
   can plausibly land, in the tail.
2. Inference time: for a query's margin vector v, rank classes by raw
   margin descending. For each of the top `alpha_ranks` classes (rank i,
   1-indexed), shrink that class's margin by a weight
   `w_i = 1 - ((alpha - i) / alpha) * weibull_cdf(||v - MAV_c||)` -- the
   further v sits from that class's MAV (relative to how spread out
   genuine examples of that class were), the more of its score gets
   redistributed away.
3. The total amount shrunk off every class becomes an extra "unknown"
   logit. Softmax over (shrunk known-class logits, unknown logit) gives
   final probabilities, including `unknown_mass` = P(unknown).

This is a heuristic recalibration, not a guarantee: it only ever *shrinks*
a class's score toward unknown based on distance-to-mean in margin space,
so a genuinely novel attack whose margins happen to land close to a known
class's MAV will not be caught by this signal alone. See `ml/README.md`'s
open-set section for how this is evaluated (`split.rotate_holdout` +
`evaluation.py`'s unknown-detection metrics) rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import xgboost as xgb
from scipy.stats import weibull_min

from .softmax_gate import GateResult
from .stage2_xgboost import AttackClassifier


@dataclass
class OpenMaxConfig:
    tail_size: int = 20
    alpha_ranks: int = 3


class OpenMaxHead:
    """Open-set recalibration over an already-fit `AttackClassifier`.
    Produces the same `softmax_gate.GateResult` shape as `SoftmaxGate`, so
    the two are swappable behind `pipeline.TwoStageDetector`'s `gate`.
    """

    _MIN_EXAMPLES_FOR_TAIL = 4

    def __init__(self, classifier: AttackClassifier, config: Optional[OpenMaxConfig] = None) -> None:
        self.classifier = classifier
        self.config = config or OpenMaxConfig()
        self._mavs: Dict[str, np.ndarray] = {}
        self._tails: Dict[str, Optional[tuple]] = {}
        self._fitted = False

    def _margins(self, X: np.ndarray) -> np.ndarray:
        """Raw per-class margins (pre-softmax), shape (n_samples, n_classes)."""
        booster = self.classifier.model.get_booster()
        dmatrix = xgb.DMatrix(X)
        return np.asarray(booster.predict(dmatrix, output_margin=True))

    def fit(self, X_train: np.ndarray, y_train_labels: Sequence[str]) -> "OpenMaxHead":
        margins = self._margins(X_train)
        y_train = np.asarray(list(y_train_labels))
        preds = np.asarray(self.classifier.predict(X_train))

        for class_idx, class_name in enumerate(self.classifier.classes_):
            correct_mask = (y_train == class_name) & (preds == class_name)
            n_correct = int(correct_mask.sum())
            if n_correct < self._MIN_EXAMPLES_FOR_TAIL:
                # Too few correctly-classified training examples to fit a
                # stable Weibull tail (2-3 near-identical distances produce a
                # degenerate fit -- observed in practice on this project's
                # tiny synthetic fixture's rarest categories, e.g. Heartbleed
                # at support=2). This class never contributes a
                # distance-based reduction (treated as "no evidence either
                # way", not "safe") until it has enough correctly-classified
                # training examples to characterize a genuine tail.
                self._mavs[class_name] = margins[correct_mask].mean(axis=0) if n_correct else margins.mean(axis=0) * 0.0
                self._tails[class_name] = None
                continue

            class_margins = margins[correct_mask]
            mav = class_margins.mean(axis=0)
            distances = np.linalg.norm(class_margins - mav, axis=1)
            tail_size = min(self.config.tail_size, len(distances))
            tail_distances = np.sort(distances)[-tail_size:]

            self._mavs[class_name] = mav
            if np.ptp(tail_distances) < 1e-9:
                # Every correctly-classified training example of this class
                # landed at (numerically) the same distance from its own
                # mean -- there's no real spread to fit a Weibull shape to.
                # Happens on this project's synthetic fixture, whose
                # deliberately caricatured categories are tight enough that
                # a handful of examples can be near-identical in margin
                # space (see ml/README.md's known-limitations pattern for
                # this fixture). Rather than let scipy fit a degenerate
                # distribution to near-zero variance, treat this class the
                # same as "too few examples": no distance-based reduction.
                self._tails[class_name] = None
                continue
            shape, loc, scale = weibull_min.fit(tail_distances, floc=0)
            self._tails[class_name] = (shape, loc, scale)

        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("OpenMaxHead must be fit() before use")

    def recalibrate_batch(self, X: np.ndarray) -> List[GateResult]:
        self._check_fitted()
        margins = self._margins(X)
        classes = self.classifier.classes_
        n, k = margins.shape
        alpha = min(self.config.alpha_ranks, k)

        results: List[GateResult] = []
        for row in range(n):
            v = margins[row].copy()
            order = np.argsort(-v)  # class indices, descending raw margin
            unknown_logit = 0.0

            for rank, class_idx in enumerate(order[:alpha], start=1):
                class_name = classes[class_idx]
                tail = self._tails.get(class_name)
                if tail is None:
                    continue
                shape, loc, scale = tail
                mav = self._mavs[class_name]
                distance = float(np.linalg.norm(v - mav))
                cdf = float(weibull_min.cdf(distance, shape, loc=loc, scale=scale))
                weight = 1.0 - ((alpha - rank) / alpha) * cdf
                weight = min(max(weight, 0.0), 1.0)
                original = v[class_idx]
                v[class_idx] = original * weight
                unknown_logit += original * (1.0 - weight)

            full = np.append(v, unknown_logit)
            full = full - full.max()  # numerically stable softmax
            exp = np.exp(full)
            probs = exp / exp.sum()

            known_probs = {classes[i]: float(probs[i]) for i in range(k)}
            unknown_mass = float(probs[-1])
            predicted_class = max(known_probs, key=known_probs.get)

            results.append(
                GateResult(
                    predicted_class=predicted_class,
                    known_class_probabilities=known_probs,
                    unknown_mass=unknown_mass,
                )
            )
        return results
