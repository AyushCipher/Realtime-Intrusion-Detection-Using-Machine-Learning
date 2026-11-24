import numpy as np
import pytest

from ids_ml.conformal_stage1 import calibrate_stage1_threshold
from ids_ml.stage1_hybrid import HybridConfig, HybridPreFilter, load_stage1


def _data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    benign = rng.normal(0, 1, size=(n, 6))
    attack = rng.normal(2.5, 1, size=(n // 3, 6))
    X = np.vstack([benign, attack])
    y = np.r_[np.zeros(n), np.ones(n // 3)]
    return X, y


def test_scores_in_unit_range_and_attacks_score_higher():
    X, y = _data()
    f = HybridPreFilter(HybridConfig(xgb_trees=20, xgb_depth=3, if_trees=30)).fit(X, y)
    s = f.anomaly_score(X)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s[y == 1].mean() > s[y == 0].mean() + 0.2


def test_default_flag_rate_on_benign_is_small():
    X, y = _data()
    f = HybridPreFilter(HybridConfig(xgb_trees=20, xgb_depth=3, if_trees=30)).fit(X, y)
    Xb, _ = _data(seed=1)
    flagged = f.flag(Xb[:1500])
    assert flagged.mean() < 0.15


def test_works_with_conformal_recall_control():
    X, y = _data()
    f = HybridPreFilter(HybridConfig(xgb_trees=20, xgb_depth=3, if_trees=30)).fit(X, y)
    Xv, yv = _data(seed=2)
    g = calibrate_stage1_threshold(f.anomaly_score(Xv[yv == 1]), 0.1)
    g.apply(f)
    Xt, yt = _data(seed=3)
    miss = (~f.flag(Xt[yt == 1])).mean()
    assert miss <= 0.1 + 0.05


def test_save_and_load_round_trip(tmp_path):
    X, y = _data()
    f = HybridPreFilter(HybridConfig(xgb_trees=20, xgb_depth=3, if_trees=30)).fit(X, y)
    p = tmp_path / "stage1.joblib"
    f.save(p)
    g = load_stage1(p)
    assert np.allclose(f.anomaly_score(X[:50]), g.anomaly_score(X[:50]))


def test_requires_both_classes():
    X, y = _data()
    with pytest.raises(ValueError):
        HybridPreFilter().fit(X[y == 0], y[y == 0])


def test_supervised_weight_one_matches_supervised_only_ordering():
    X, y = _data()
    f = HybridPreFilter(HybridConfig(xgb_trees=20, xgb_depth=3, if_trees=30, fusion="weighted", supervised_weight=1.0)).fit(X, y)
    s = f.anomaly_score(X)
    assert s[y == 1].mean() > s[y == 0].mean()


@pytest.mark.parametrize("density", ["knn", "iforest", "both"])
def test_every_density_view_separates_attacks(density):
    X, y = _data()
    f = HybridPreFilter(HybridConfig(xgb_trees=20, xgb_depth=3, if_trees=30, density=density, knn_reference=800)).fit(X, y)
    s = f.anomaly_score(X)
    assert s[y == 1].mean() > s[y == 0].mean() + 0.2


def test_rejects_unknown_density():
    X, y = _data()
    with pytest.raises(ValueError):
        HybridPreFilter(HybridConfig(density="nope")).fit(X, y)


def test_max_fusion_is_at_least_each_view():
    X, y = _data()
    f = HybridPreFilter(HybridConfig(xgb_trees=20, xgb_depth=3, knn_reference=800, fusion="max")).fit(X, y)
    sup = f._ecdf(f._ref_sup, f.xgb.predict_proba(X)[:, 1])
    assert np.all(f.anomaly_score(X) >= sup - 1e-12)


def test_rejects_unknown_fusion():
    X, y = _data()
    with pytest.raises(ValueError):
        HybridPreFilter(HybridConfig(fusion="mean")).fit(X, y)


def test_stage1_stream_trial_runs_with_hybrid_stage1():
    import pandas as pd

    from ids_ml.evaluation import stage1_stream_trial

    rng = np.random.default_rng(0)
    n = 1200
    X = np.vstack([rng.normal(0, 1, (n, 3)), rng.normal(3, 1, (n // 3, 3))])
    lab = ["BENIGN"] * n + ["DoS/DDoS"] * (n // 3)
    order = rng.permutation(len(lab))
    df = pd.DataFrame(X[order], columns=["a", "b", "c"])
    df["attack_category"] = np.array(lab)[order]
    df["timestamp"] = pd.date_range("2018-02-14", periods=len(df), freq="s")
    rows = stage1_stream_trial(df, ["a", "b", "c"], miss_budget=0.1, stage1_kind="hybrid", window_size=50)
    assert rows and all(r["stage1_kind"] == "hybrid" for r in rows)
    overall = [r for r in rows if r["method"] == "static" and r["segment"] == "overall"][0]
    assert overall["benign_flag_rate"] < 0.6
