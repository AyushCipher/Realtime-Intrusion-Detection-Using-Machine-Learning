"""Energy-based Flow Classifier (EFC): a reimplementation for head-to-head comparison.

Reimplements the single-class EFC of Pontes et al. and its multi-class open-set
extension by Souza et al., "A Novel Open Set Energy-based Flow Classifier for
Network Intrusion Detection" (arXiv:2109.11224), from the papers' description,
so it can be run on exactly the features and splits our cascade uses.

Model (one per class): every feature is discretised into Q bins (quantile edges
fitted on the training flows). With single-site and pair frequencies
f_i(a), f_ij(a, b), mixed with a pseudocount lambda (uniform 1/Q and 1/Q^2), the
mean-field Potts inference gives couplings and fields with the last bin as gauge:

    C_ij(a, b) = f_ij(a, b) - f_i(a) f_j(b)            a, b < Q
    e_ij(a, b) = -(C^-1)_ij(a, b)                       (i != j; 0 on the diagonal blocks)
    h_i(a)     = ln(f_i(a) / f_i(Q)) - sum_{j != i, b < Q} e_ij(a, b) f_j(b)

and a flow's energy is H(x) = -sum_i h_i(x_i) - sum_{i<j} e_ij(x_i, x_j)
(terms involving a gauge bin are zero). Lower energy = more typical of the class.
Each class's threshold is the 95th percentile of its own training energies.

Multi-class decision (Souza et al.): the class with the lowest energy, or
"suspicious" (unknown) if that energy is above the class's threshold; the novelty
score is that energy minus that threshold (> 0 means suspicious). Comparing raw
energies across classes favours narrow classes, whose models can assign low energy
to flows they never saw; with a small pseudocount this misroutes known flows, which
is why the paper's lambda = 0.5 is kept for the multi-class gate (EFCGate).

Deviations, stated because they affect comparability: our 35 canonical features
rather than the paper's feature set; quantile discretisation; the single-class
detector's Q = 20 and lambda = 0.2 were chosen on a VALIDATION split of CSE-CIC-IDS2018 (grid Q in {10, 20, 30}, lambda in
{0.05, 0.2, 0.5}; single-class benign-vs-attack AUROC 0.855 against 0.766 for Q = 30,
lambda = 0.5); large classes are subsampled to `max_per_class` flows for fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse


@dataclass
class EFCConfig:
    n_bins: int = 20
    pseudocount: float = 0.2
    percentile: float = 95.0
    max_per_class: int = 30000
    random_state: int = 0


class _PottsModel:
    def __init__(self, n_features: int, q: int, lam: float) -> None:
        self.n, self.q, self.lam = n_features, q, lam

    def _onehot(self, B: np.ndarray) -> sparse.csr_matrix:
        """One-hot of bins 0..Q-2 (bin Q-1 is the gauge and has no column)."""
        n, q = self.n, self.q
        rows, cols = np.nonzero(B < q - 1)
        data = np.ones(len(rows), dtype=np.float64)
        col_idx = cols * (q - 1) + B[rows, cols]
        return sparse.csr_matrix((data, (rows, col_idx)), shape=(B.shape[0], n * (q - 1)))

    def fit(self, B: np.ndarray) -> "_PottsModel":
        n, q, lam = self.n, self.q, self.lam
        m = B.shape[0]
        O = self._onehot(B)
        fi = np.asarray(O.mean(axis=0)).ravel()
        fij = (O.T @ O).toarray() / m
        fi = (1 - lam) * fi + lam / q
        fij = (1 - lam) * fij + lam / (q * q)
        # Diagonal blocks: f_ii(a, b) = f_i(a) delta_ab (a feature takes one value).
        for i in range(n):
            s = slice(i * (q - 1), (i + 1) * (q - 1))
            fij[s, s] = np.diag(fi[s])
        C = fij - np.outer(fi, fi)
        E = -np.linalg.inv(C)
        for i in range(n):
            s = slice(i * (q - 1), (i + 1) * (q - 1))
            E[s, s] = 0.0
        # gauge-bin frequency per feature: 1 - sum of the other bins
        fi_mat = fi.reshape(n, q - 1)
        fq = np.clip(1.0 - fi_mat.sum(axis=1), 1e-12, None)
        h = np.log(fi_mat / fq[:, None]) - (E @ fi).reshape(n, q - 1)
        self.E, self.h = E, h.ravel()
        return self

    def energy(self, B: np.ndarray, chunk: int = 20000) -> np.ndarray:
        out = np.empty(B.shape[0])
        for s in range(0, B.shape[0], chunk):
            O = self._onehot(B[s : s + chunk])
            field = O @ self.h
            pair = np.asarray(O.multiply(O @ self.E).sum(axis=1)).ravel() / 2.0
            out[s : s + chunk] = -field - pair
        return out


class EnergyFlowClassifier:
    """Single-class (fit with y=None: one model of all given flows) or multi-class EFC."""

    def __init__(self, config: Optional[EFCConfig] = None) -> None:
        self.config = config or EFCConfig()
        self.models: Dict[str, _PottsModel] = {}

    def _bin(self, X: np.ndarray) -> np.ndarray:
        q = self.config.n_bins
        B = np.empty(X.shape, dtype=np.int64)
        for j in range(X.shape[1]):
            B[:, j] = np.clip(np.searchsorted(self.edges[j], X[:, j], side="right"), 0, q - 1)
        return B

    def fit(self, X: np.ndarray, y: Optional[Sequence[str]] = None) -> "EnergyFlowClassifier":
        c = self.config
        X = np.asarray(X, dtype=float)
        qs = np.linspace(0, 1, c.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        labels = np.array(["__all__"] * len(X)) if y is None else np.asarray(y)
        rng = np.random.default_rng(c.random_state)
        self.thresholds, self.scales = {}, {}
        for cls in np.unique(labels):
            idx = np.where(labels == cls)[0]
            if len(idx) > c.max_per_class:
                idx = rng.choice(idx, c.max_per_class, replace=False)
            B = self._bin(X[idx])
            m = _PottsModel(X.shape[1], c.n_bins, c.pseudocount).fit(B)
            e = m.energy(B)
            self.models[str(cls)] = m
            self.thresholds[str(cls)] = float(np.percentile(e, c.percentile))
            q75, q25 = np.percentile(e, [75, 25])
            self.scales[str(cls)] = float(max(q75 - q25, 1e-9))
        self.classes_ = sorted(self.models)
        return self

    def energies(self, X: np.ndarray) -> np.ndarray:
        B = self._bin(np.asarray(X, dtype=float))
        return np.column_stack([self.models[c].energy(B) for c in self.classes_])

    def novelty_score(self, X: np.ndarray) -> np.ndarray:
        """The paper's decision quantity: H_{c*}(x) - theta_{c*}, where c* is the class with the
        lowest energy. > 0 is exactly when EFC labels the flow 'suspicious'. For a single-class
        model this is the energy minus the 95th-percentile training energy."""
        H = self.energies(X)
        th = np.array([self.thresholds[c] for c in self.classes_])
        best = H.argmin(axis=1)
        return H[np.arange(len(H)), best] - th[best]

    def predict(self, X: np.ndarray) -> List[str]:
        H = self.energies(X)
        th = np.array([self.thresholds[c] for c in self.classes_])
        best = H.argmin(axis=1)
        return [self.classes_[b] if H[i, b] <= th[b] else "suspicious" for i, b in enumerate(best)]
