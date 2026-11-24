"""Stronger unknown-attack (out-of-distribution) baselines for the
escalation gate, behind the same interface as `softmax_gate.SoftmaxGate`
and `openset_head.OpenMaxHead`.

H1 compares OpenMax only against the softmax-confidence trigger, which is
the weakest baseline in the out-of-distribution (OOD) detection literature.
A reviewer will ask how OpenMax does against the standard stronger scores,
so this module adds four of them:

  * MaxLogitGate    -- negative largest raw class margin (Hendrycks et al.,
                       "Scaling Out-of-Distribution Detection for Real-World
                       Settings", ICML 2022).
  * EnergyGate      -- energy score, -logsumexp(margins) (Liu et al.,
                       "Energy-based Out-of-distribution Detection",
                       NeurIPS 2020).
  * MahalanobisGate -- smallest class-conditional Mahalanobis distance with a
                       shared covariance (Lee et al., NeurIPS 2018).
  * KNNGate         -- distance to the k-th nearest training flow in
                       L2-normalised space (Sun et al., "Out-of-Distribution
                       Detection with Deep Nearest Neighbors", ICML 2022).

Adaptation note, stated plainly because it changes what the comparison
means: Lee et al. and Sun et al. compute their distances on a deep network's
penultimate-layer embedding. XGBoost has no such embedding, so
MahalanobisGate and KNNGate work on the input flow features instead, after a
signed log1p transform and standardisation (flow features are heavy-tailed:
byte and packet rates span many orders of magnitude). MaxLogitGate and
EnergyGate use XGBoost's raw per-class margins, the same signal OpenMax uses.

Every gate's `unknown_mass` here is a novelty score (higher = more likely an
unknown attack), not a probability. `conformal_gate.calibrate_threshold`
only needs the scores to be ordered, so the escalation budget works the same
way for every gate. Probability-only metrics (ECE, Brier) are not
meaningful for these scores and `evaluation.run_openset_trial` reports them
as NaN.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import joblib
import numpy as np
import xgboost as xgb
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .softmax_gate import GateResult


def _raw_margins(classifier, X: np.ndarray) -> np.ndarray:
    """Raw per-class margins (pre-softmax), shape (n_samples, n_classes).

    XGBoost returns a single margin column for a two-class problem; that is
    the log-odds of class 1, so it is expanded to [0, m] to keep one column
    per class.
    """
    booster = classifier.model.get_booster()
    margins = np.asarray(booster.predict(xgb.DMatrix(X), output_margin=True))
    if margins.ndim == 1:
        margins = np.column_stack([np.zeros_like(margins), margins])
    return margins


def _results(classifier, X: np.ndarray, scores: np.ndarray) -> List[GateResult]:
    """Known-class prediction and probabilities from Stage 2's own softmax,
    novelty score from the gate. Only the score differs between these
    gates; the closed-set prediction is the same classifier's."""
    proba = classifier.predict_proba(X)
    classes = classifier.classes_
    results = []
    for row in range(X.shape[0]):
        best = int(proba[row].argmax())
        results.append(
            GateResult(
                predicted_class=str(classes[best]),
                known_class_probabilities={classes[c]: float(proba[row, c]) for c in range(len(classes))},
                unknown_mass=float(scores[row]),
            )
        )
    return results


def _signed_log1p(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


class MaxLogitGate:
    """Novelty score = -max_k margin_k. No fitting beyond Stage 2's own."""

    bounded_score = False

    def __init__(self, classifier) -> None:
        self.classifier = classifier

    def fit(self, X_fit: np.ndarray, y_fit: Sequence[str]) -> "MaxLogitGate":
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        return -_raw_margins(self.classifier, X).max(axis=1) if len(X) else np.array([])

    def recalibrate_batch(self, X: np.ndarray) -> List[GateResult]:
        if len(X) == 0:
            return []
        return _results(self.classifier, X, self.scores(X))


class EnergyGate:
    """Novelty score = energy = -T * logsumexp(margins / T), T = 1 by default
    (the setting Liu et al. recommend when not tuning on OOD data, which we
    do not have by construction: the held-out family is unseen)."""

    bounded_score = False

    def __init__(self, classifier, temperature: float = 1.0) -> None:
        self.classifier = classifier
        self.temperature = temperature

    def fit(self, X_fit: np.ndarray, y_fit: Sequence[str]) -> "EnergyGate":
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])
        t = self.temperature
        return -t * logsumexp(_raw_margins(self.classifier, X) / t, axis=1)

    def recalibrate_batch(self, X: np.ndarray) -> List[GateResult]:
        if len(X) == 0:
            return []
        return _results(self.classifier, X, self.scores(X))


class MahalanobisGate:
    """Novelty score = min over known classes of the squared Mahalanobis
    distance to that class's mean, with one covariance shared by all classes
    (Lee et al.'s tied-covariance form), estimated with Ledoit-Wolf shrinkage
    because some flow features are near-constant within a class."""

    bounded_score = False

    def __init__(self, classifier, chunk_size: int = 20000) -> None:
        self.classifier = classifier
        self.chunk_size = chunk_size
        self.scaler = StandardScaler()
        self._means: Optional[np.ndarray] = None
        self._precision: Optional[np.ndarray] = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(_signed_log1p(X))

    def fit(self, X_fit: np.ndarray, y_fit: Sequence[str]) -> "MahalanobisGate":
        Z = self.scaler.fit_transform(_signed_log1p(X_fit))
        y = np.asarray(list(y_fit))
        classes = sorted(set(y))
        means = np.vstack([Z[y == c].mean(axis=0) for c in classes])
        centred = Z - means[np.searchsorted(classes, y)]
        self._means = means
        self._precision = LedoitWolf().fit(centred).precision_
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        if self._means is None:
            raise RuntimeError("MahalanobisGate must be fit() before use")
        out = np.empty(len(X))
        for start in range(0, len(X), self.chunk_size):
            Z = self._transform(X[start : start + self.chunk_size])
            # (n, k, d) differences would not fit in memory for large n, so
            # expand d^T P d = z^T P z - 2 z^T P mu + mu^T P mu per class.
            zp = Z @ self._precision
            zpz = np.einsum("ij,ij->i", zp, Z)
            mpm = np.einsum("kj,kj->k", self._means @ self._precision, self._means)
            d2 = zpz[:, None] - 2.0 * zp @ self._means.T + mpm[None, :]
            out[start : start + len(Z)] = d2.min(axis=1)
        return out

    def recalibrate_batch(self, X: np.ndarray) -> List[GateResult]:
        if len(X) == 0:
            return []
        return _results(self.classifier, X, self.scores(X))


class KNNGate:
    """Novelty score = Euclidean distance to the k-th nearest training flow,
    after the same transform as MahalanobisGate plus L2 normalisation (Sun
    et al. normalise features before the kNN search).

    The reference set is a seeded random subsample of at most
    `max_reference` training flows, to keep queries tractable on hundreds of
    thousands of test flows. Sun et al. show the score is robust to
    subsampling the reference set; `k` is scaled down with it (their k = 50
    is for the full CIFAR training set).
    """

    bounded_score = False

    def __init__(self, classifier, k: int = 10, max_reference: int = 20000, random_state: int = 0) -> None:
        self.classifier = classifier
        self.k = k
        self.max_reference = max_reference
        self.random_state = random_state
        self.scaler = StandardScaler()
        self._nn: Optional[NearestNeighbors] = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(_signed_log1p(X))
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.maximum(norms, 1e-12)

    def fit(self, X_fit: np.ndarray, y_fit: Sequence[str]) -> "KNNGate":
        self.scaler.fit(_signed_log1p(X_fit))
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(X_fit))[: self.max_reference]
        reference = self._transform(np.asarray(X_fit)[idx])
        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(reference)), algorithm="brute").fit(reference)
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("KNNGate must be fit() before use")
        out = np.empty(len(X))
        chunk = 5000
        for start in range(0, len(X), chunk):
            dist, _ = self._nn.kneighbors(self._transform(X[start : start + chunk]))
            out[start : start + len(dist)] = dist[:, -1]
        return out

    def recalibrate_batch(self, X: np.ndarray) -> List[GateResult]:
        if len(X) == 0:
            return []
        return _results(self.classifier, X, self.scores(X))

    def save(self, path) -> None:
        """Saves this gate's own fitted state (scaler and reference set), not
        the Stage 2 classifier, which is saved separately -- same convention as
        `OpenMaxHead.save`. `load()` takes the loaded classifier back."""
        if self._nn is None:
            raise RuntimeError("KNNGate must be fit() before use")
        joblib.dump({"k": self.k, "max_reference": self.max_reference, "random_state": self.random_state,
                     "scaler": self.scaler, "nn": self._nn}, path)

    @classmethod
    def load(cls, path, classifier) -> "KNNGate":
        payload = joblib.load(path)
        gate = cls(classifier, k=payload["k"], max_reference=payload["max_reference"], random_state=payload["random_state"])
        gate.scaler = payload["scaler"]
        gate._nn = payload["nn"]
        return gate


class EFCGate:
    """Novelty score from a reimplemented multi-class Energy-based Flow Classifier
    (Souza et al., arXiv:2109.11224; see efc.py): min over known classes of the
    class energy minus that class's 95th-percentile threshold, scaled by the class's
    energy spread. > 0 is what EFC itself would call "suspicious". Known-class
    predictions still come from Stage 2, as for every other gate, so gates differ
    only in their novelty score."""

    bounded_score = False

    def __init__(self, classifier, random_state: int = 0) -> None:
        from .efc import EFCConfig, EnergyFlowClassifier

        self.classifier = classifier
        # The paper's pseudocount (0.5); smaller values misroute known flows between class models (efc.py).
        self.efc = EnergyFlowClassifier(EFCConfig(n_bins=10, pseudocount=0.5, random_state=random_state))

    def fit(self, X_fit: np.ndarray, y_fit: Sequence[str]) -> "EFCGate":
        self.efc.fit(np.asarray(X_fit), list(y_fit))
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        return self.efc.novelty_score(np.asarray(X))

    def recalibrate_batch(self, X: np.ndarray) -> List[GateResult]:
        if len(X) == 0:
            return []
        return _results(self.classifier, X, self.scores(X))


# Every class above takes the fitted Stage 2 classifier and exposes
# fit(X_fit, y_fit) -> self, so evaluation.py builds them all the same way.
OOD_GATES = {
    "maxlogit": MaxLogitGate,
    "energy": EnergyGate,
    "mahalanobis": MahalanobisGate,
    "knn": KNNGate,
    "efc": EFCGate,
}
