"""Hybrid Stage 1 pre-filter: a supervised score for known attacks plus a
benign-density score for novel ones.

Why this exists. On real CSE-CIC-IDS2018 data the original unsupervised
Isolation Forest pre-filter is at chance (attack-vs-benign AUROC about 0.5; see
results/paper/*/stage1_separability.json), so it cannot drop benign traffic
cheaply. Leave-one-family-out validation (scripts/stage1_selection.py) showed
the two natural cheap alternatives fail in opposite ways:

  * a small supervised benign-vs-attack XGBoost is excellent on known families
    (AUROC 0.97) but weak on families it never saw (0.74);
  * an Isolation Forest fitted on benign flows only is the reverse (0.76 on
    known, 0.73 on unseen).

Averaging their percentile-calibrated scores was best on unseen families
(about 0.80) and still good on known ones (about 0.92). Replacing the Isolation
Forest by a nearest-benign-neighbour distance (`density="knn"`, now the default)
improved unseen-family AUROC to about 0.83 with 0.95 on known families, and a
depth-6 supervised model (the current default: 100 trees, depth 6, equal weights)
reaches about 0.84 on unseen families, 0.96 on known families and 0.98 zero-day
AUROC on LYCOS-IDS2017 (results/paper/*/stage1_selection.csv and closr_protocol.csv,
candidate hyb_s6k_w50). Raising `supervised_weight` to 0.7 or 0.85 gives 0.80 / 0.76
on unseen families but 0.97 / 0.98 on known ones and lower zero-day FPR on LYCOS.

That weighted average forces a choice of threat model. Taking the MAX of the two
percentiles instead (`fusion="max"`, the default) removes most of the trade-off: 0.835
on unseen families and 0.975 on known ones (CSE-CIC-IDS2018), and 0.999 known / 0.992
zero-day AUROC with FPR-at-95%-TPR 0.017 on LYCOS-IDS2017, within about 0.01 of the best
single detector on each protocol (candidate max_s6k). `supervised_weight`
moves along that trade-off: with conformal recall control at the same miss
budget, w = 1 flags the fewest benign flows on known families, while w = 0.5
misses 2-4x fewer unseen-family attacks (see results/paper/*/stage1_hybrid_eval.csv). This class implements
that average as a per-flow scorer, so it works one flow at a time:

    score(x) = w * ECDF_sup(p_attack(x)) + (1 - w) * mean over density views of ECDF(view(x))
               (w = supervised_weight; default density view: distance to the k-th nearest benign flow)

where each ECDF is the empirical CDF over a held-out reference sample of
BENIGN flows. A score near 1 therefore means "more attack-like than almost all
benign reference flows" on both views. The reference sample is held out from
both models' training so the percentiles are not distorted by overfitting.

The interface matches `AnomalyPreFilter` (`anomaly_score`, `flag`,
`config.flag_threshold`, `save`/`load`) so `TwoStageDetector`,
`conformal_stage1` and `adaptive_stage1` work with it unchanged. Unlike the
Isolation Forest it needs labels to fit (`fit(X, y_is_attack)`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _slog(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


@dataclass
class HybridConfig:
    xgb_trees: int = 100
    xgb_depth: int = 6
    if_trees: int = 100
    # Which benign-density view(s) complement the supervised score: "knn" (distance to the k-th
    # nearest benign reference flow), "iforest" (Isolation Forest fitted on benign flows), or "both".
    density: str = "knn"
    knn_k: int = 10
    knn_reference: int = 20000
    reference_fraction: float = 0.2  # share of benign flows held out as the percentile reference
    default_benign_flag_rate: float = 0.05  # default flag threshold: this share of reference benign flows flagged
    random_state: int = 0
    # How the supervised and density percentiles are combined: "max" flags a flow if EITHER view
    # finds it unusual (default: near-best on both known-like zero-days and entirely new families);
    # "weighted" uses supervised_weight * sup + (1 - supervised_weight) * density.
    fusion: str = "max"
    supervised_weight: float = 0.5  # 1.0 = supervised only (best on known families); 0.5 = hybrid; 0.0 = density only
    flag_threshold: Optional[float] = None


class HybridPreFilter:
    def __init__(self, config: Optional[HybridConfig] = None) -> None:
        self.config = config or HybridConfig()
        self._fitted = False

    def fit(self, X: np.ndarray, y_is_attack: np.ndarray) -> "HybridPreFilter":
        c = self.config
        rng = np.random.default_rng(c.random_state)
        y = np.asarray(y_is_attack).astype(int)
        benign = np.where(y == 0)[0]
        attack = np.where(y == 1)[0]
        if len(benign) < 50 or len(attack) < 1:
            raise ValueError("HybridPreFilter needs benign and attack training flows")
        benign = rng.permutation(benign)
        n_ref = max(int(len(benign) * c.reference_fraction), 20)
        ref_idx, fit_benign = benign[:n_ref], benign[n_ref:]

        fit_idx = np.concatenate([fit_benign, attack])
        self.xgb = XGBClassifier(
            n_estimators=c.xgb_trees, max_depth=c.xgb_depth, learning_rate=0.2, n_jobs=4, verbosity=0, random_state=c.random_state
        ).fit(X[fit_idx], y[fit_idx])

        if c.fusion not in ("max", "weighted"):
            raise ValueError("fusion must be 'max' or 'weighted'")
        if c.density not in ("knn", "iforest", "both"):
            raise ValueError("density must be 'knn', 'iforest' or 'both'")
        self.scaler = StandardScaler().fit(_slog(X[fit_benign]))
        Z = self.scaler.transform(_slog(X[fit_benign]))
        self.iforest = self.knn = None
        if c.density in ("iforest", "both"):
            self.iforest = IsolationForest(n_estimators=c.if_trees, random_state=c.random_state, n_jobs=4).fit(Z)
        if c.density in ("knn", "both"):
            ref = Z[rng.permutation(len(Z))[: c.knn_reference]]
            self.knn = NearestNeighbors(n_neighbors=c.knn_k, n_jobs=4).fit(ref)

        Xr = X[ref_idx]
        self._ref_sup = np.sort(self.xgb.predict_proba(Xr)[:, 1])
        self._ref_dens = {name: np.sort(f(Xr)) for name, f in self._density_scorers().items()}
        self._fitted = True
        ref_scores = self.anomaly_score(Xr)
        self.default_threshold = float(np.quantile(ref_scores, 1.0 - c.default_benign_flag_rate))
        return self

    def _density_scorers(self) -> dict:
        scorers = {}
        if self.iforest is not None:
            scorers["iforest"] = lambda X: -self.iforest.score_samples(self.scaler.transform(_slog(X)))
        if self.knn is not None:
            scorers["knn"] = lambda X: self.knn.kneighbors(self.scaler.transform(_slog(X)))[0][:, -1]
        return scorers

    @staticmethod
    def _ecdf(ref_sorted: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.searchsorted(ref_sorted, v, side="right") / len(ref_sorted)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more attack-like. Range [0, 1]."""
        self._check_fitted()
        sup = self._ecdf(self._ref_sup, self.xgb.predict_proba(X)[:, 1])
        dens = np.mean([self._ecdf(self._ref_dens[n], f(X)) for n, f in self._density_scorers().items()], axis=0)
        if self.config.fusion == "max":
            return np.maximum(sup, dens)
        w = self.config.supervised_weight
        return w * sup + (1.0 - w) * dens

    def flag(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        thr = self.config.flag_threshold if self.config.flag_threshold is not None else self.default_threshold
        return self.anomaly_score(X) >= thr

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("HybridPreFilter must be fit() before use")

    def save(self, path) -> None:
        self._check_fitted()
        joblib.dump({"kind": "hybrid", "obj": self}, path)

    @classmethod
    def load(cls, path) -> "HybridPreFilter":
        payload = joblib.load(path)
        return payload["obj"]


def load_stage1(path):
    """Load either stage-1 kind: the Isolation Forest `AnomalyPreFilter` or `HybridPreFilter`."""
    from .stage1_iforest import AnomalyPreFilter

    payload = joblib.load(path)
    if isinstance(payload, dict) and payload.get("kind") == "hybrid":
        return payload["obj"]
    return AnomalyPreFilter.load(path)
