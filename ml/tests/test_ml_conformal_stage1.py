import numpy as np
import pytest

from ids_ml.conformal_stage1 import calibrate_stage1_threshold
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config


def test_threshold_is_kth_smallest_score():
    scores = np.arange(1.0, 100.0)  # n = 99
    g = calibrate_stage1_threshold(scores, miss_budget=0.10)
    k = int(np.floor(0.10 * 100))
    assert g.threshold == scores[k - 1]
    assert g.guaranteed_miss_rate == pytest.approx(k / 100)
    assert g.guaranteed_miss_rate <= 0.10


def test_budget_tighter_than_certifiable_flags_everything():
    g = calibrate_stage1_threshold([0.4, 0.5, 0.6], miss_budget=0.05)  # 0.05 * 4 < 1
    assert g.threshold == float("-inf")
    assert g.guaranteed_miss_rate == 0.0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        calibrate_stage1_threshold([], 0.1)
    with pytest.raises(ValueError):
        calibrate_stage1_threshold([0.5], 1.0)


def test_apply_pins_prefilter_threshold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    stage1 = AnomalyPreFilter(Stage1Config(n_estimators=20, random_state=0)).fit(X)
    g = calibrate_stage1_threshold(stage1.anomaly_score(X[:50]), 0.2)
    g.apply(stage1)
    assert stage1.config.flag_threshold == g.threshold
    # every calibration attack at or above the threshold is flagged
    flagged = stage1.flag(X[:50])
    assert flagged.sum() >= 50 - int(np.floor(0.2 * 51))


@pytest.mark.parametrize("budget", [0.05, 0.10, 0.20])
def test_empirical_miss_rate_respects_budget(budget):
    """Monte Carlo check of the guarantee under exchangeability: averaged over
    many fresh calibration/test draws, the miss rate must not exceed the budget
    (beyond sampling noise)."""
    rng = np.random.default_rng(1)
    n_cal, n_trials = 99, 3000
    misses = 0
    for _ in range(n_trials):
        cal = rng.normal(size=n_cal)
        new = rng.normal()
        g = calibrate_stage1_threshold(cal, budget)
        misses += new < g.threshold
    assert misses / n_trials <= budget + 0.02


def test_per_family_threshold_is_lowest_family_threshold():
    from ids_ml.conformal_stage1 import calibrate_stage1_per_family

    rng = np.random.default_rng(0)
    fams = {"easy": rng.normal(3, 1, 200), "hard": rng.normal(1, 1, 200)}
    g = calibrate_stage1_per_family(fams, 0.1)
    hard_t = calibrate_stage1_threshold(fams["hard"], 0.1).threshold
    assert g.threshold == hard_t
    assert g.guaranteed_families == ["easy", "hard"] and not g.waived_families


def test_per_family_waives_family_that_breaks_benign_cap():
    from ids_ml.conformal_stage1 import calibrate_stage1_per_family

    rng = np.random.default_rng(1)
    benign = rng.normal(0, 1, 2000)
    fams = {"separable": rng.normal(4, 0.5, 300), "benign_like": rng.normal(0, 1, 300)}
    g = calibrate_stage1_per_family(fams, 0.05, benign_scores=benign, max_benign_flag_rate=0.1)
    assert g.waived_families == ["benign_like"]
    assert g.guaranteed_families == ["separable"]
    assert g.benign_flag_rate <= 0.1 + 1e-9


def test_per_family_flags_small_family_as_uncertifiable():
    from ids_ml.conformal_stage1 import calibrate_stage1_per_family

    g = calibrate_stage1_per_family({"tiny": [0.5, 0.6], "big": np.linspace(0, 1, 200)}, 0.05)
    assert g.uncertifiable_families == ["tiny"]
    assert g.threshold > float("-inf")


def test_dual_budget_lowers_threshold_only_when_compute_allows():
    from ids_ml.conformal_stage1 import calibrate_stage1_dual

    rng = np.random.default_rng(0)
    benign = rng.normal(0, 1, 5000)
    attacks = rng.normal(6, 0.5, 500)  # known attacks score far above benign
    g = calibrate_stage1_dual(attacks, benign, miss_budget=0.05, compute_budget=0.10)
    assert g.binding == "compute" and g.threshold < g.recall_threshold
    assert abs((benign >= g.threshold).mean() - 0.10) < 0.01
    assert (attacks < g.threshold).mean() <= 0.05  # known-attack guarantee still holds

    hard = rng.normal(0.5, 1, 500)  # known attacks that look benign: recall must win
    h = calibrate_stage1_dual(hard, benign, miss_budget=0.05, compute_budget=0.10)
    assert h.binding == "recall" and h.threshold == h.recall_threshold
