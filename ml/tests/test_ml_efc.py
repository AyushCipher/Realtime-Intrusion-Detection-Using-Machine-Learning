import numpy as np
from sklearn.metrics import roc_auc_score

from ids_ml.efc import EFCConfig, EnergyFlowClassifier


def _data(seed=0, n=1500, d=6):
    """Class A: strongly positively correlated features. Class B: alternating signs,
    i.e. a combination A never shows. EFC scores combinations through its couplings."""
    rng = np.random.default_rng(seed)
    cov = np.full((d, d), 0.9) + 0.1 * np.eye(d)
    # method="cholesky": the default SVD factorisation's singular-vector signs
    # depend on the LAPACK/BLAS build, so the same seed drew different samples on
    # different CI machines (AUROC 0.972 locally, 0.888 on one runner).
    a = rng.multivariate_normal(np.zeros(d), cov, n, method="cholesky")
    signs = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
    b = rng.normal(0, 0.3, (n, d)) + 1.5 * signs
    return a, b


def test_single_class_energy_separates_outliers():
    a, b = _data()
    efc = EnergyFlowClassifier(EFCConfig(n_bins=5, pseudocount=0.1)).fit(a)
    a2, b2 = _data(seed=1)
    e = np.r_[efc.novelty_score(a2), efc.novelty_score(b2)]
    y = np.r_[np.zeros(len(a2)), np.ones(len(b2))]
    assert roc_auc_score(y, e) > 0.9


def test_multiclass_flags_unknown_combinations_with_paper_pseudocount():
    a, b = _data()
    X = np.vstack([a, b])
    y = ["A"] * len(a) + ["B"] * len(b)
    efc = EnergyFlowClassifier(EFCConfig(n_bins=10, pseudocount=0.5)).fit(X, y)
    _, b2 = _data(seed=2)
    rng = np.random.default_rng(3)
    unknown = np.column_stack([rng.normal(-1.5, 0.3, 300) * (1 if j < 3 else -1) for j in range(6)])
    assert np.mean(efc.novelty_score(unknown) > 0) > 0.9
    assert np.mean(efc.novelty_score(b2) > 0) < 0.2
    assert set(efc.predict(unknown)) == {"suspicious"} or np.mean(np.array(efc.predict(unknown)) == "suspicious") > 0.9


def test_energy_matches_explicit_sum():
    a, _ = _data(n=400)
    efc = EnergyFlowClassifier(EFCConfig(n_bins=5)).fit(a)
    m = efc.models["__all__"]
    B = efc._bin(a[:5])
    q, n = m.q, m.n
    for r in range(5):
        h = 0.0
        pair = 0.0
        for i in range(n):
            if B[r, i] < q - 1:
                h += m.h[i * (q - 1) + B[r, i]]
                for j in range(i + 1, n):
                    if B[r, j] < q - 1:
                        pair += m.E[i * (q - 1) + B[r, i], j * (q - 1) + B[r, j]]
        assert np.isclose(m.energy(B[r : r + 1])[0], -h - pair)
