"""End-to-end cascade calibration (cascade_crc.py): the guarantees must hold
in Monte Carlo on exchangeable synthetic data, and the separate calibration
the cascade used before must be shown to exceed the end-to-end budget."""

import numpy as np
import pytest

from ids_ml.cascade_crc import (
    CascadeScores,
    CostAwareJointCRC,
    JointCRC,
    _upper_conformal_threshold,
    calibrate_bonferroni,
    calibrate_separate,
    evaluate,
)


def _attacks(rng, n):
    s1 = rng.normal(2.0, 1.0, n)
    # Stage 2 sometimes calls an attack BENIGN, more often when Stage 1 is unsure.
    p_benign = 1 / (1 + np.exp(2.0 * s1 - 1.0 + rng.normal(0, 1.5, n)))
    escalated = rng.random(n) < 0.05
    return CascadeScores(s1, p_benign, p_benign > 0.5, escalated)


def _benign(rng, n):
    s1 = rng.normal(0.0, 1.0, n)
    p_benign = 1 / (1 + np.exp(-(3.0 - s1) + rng.normal(0, 1.0, n)))
    return CascadeScores(s1, p_benign, p_benign > 0.5, rng.random(n) < 0.02)


def _concat(a, b):
    return CascadeScores(*(np.concatenate([getattr(a, f), getattr(b, f)]) for f in ("s1", "p_benign", "argmax_benign", "escalated")))


def test_upper_conformal_threshold_is_valid_in_monte_carlo():
    rng = np.random.default_rng(0)
    rates = []
    for _ in range(2000):
        cal, new = rng.random(50), rng.random()
        rates.append(new >= _upper_conformal_threshold(cal, 0.1))
    assert np.mean(rates) <= 0.1 + 0.015


def test_upper_conformal_threshold_is_inf_when_budget_too_small_and_handles_ties():
    assert _upper_conformal_threshold(np.arange(5.0), 0.1) == float("inf")
    u = _upper_conformal_threshold(np.ones(100), 0.1)
    assert not (1.0 >= u)  # all-tied scores: nothing is certified as silent


def test_joint_loss_is_monotone_in_lambda_and_zero_at_zero():
    rng = np.random.default_rng(1)
    shaping, cal = _attacks(rng, 500), _attacks(rng, 300)
    crc = JointCRC(shaping)
    misses = [float(np.mean(~crc.decision(l).alert(cal))) for l in np.linspace(0, 1, 51)]
    assert misses[0] == 0.0
    assert all(b >= a - 1e-12 for a, b in zip(misses, misses[1:]))


def test_separate_violates_while_joint_and_bonferroni_hold_end_to_end():
    rng = np.random.default_rng(2)
    alpha = 0.1
    res = {"separate": [], "bonferroni": [], "joint_crc": [], "joint_crc_cost": []}
    for _ in range(60):
        shaping, shaping_ben, cal = _attacks(rng, 400), _benign(rng, 400), _attacks(rng, 400)
        test_att, test_ben = _attacks(rng, 2000), _benign(rng, 2000)
        test = _concat(test_att, test_ben)
        is_attack = np.r_[np.ones(2000, bool), np.zeros(2000, bool)]
        for d in (
            calibrate_separate(cal, alpha),
            calibrate_bonferroni(cal, alpha),
            JointCRC(shaping).calibrate(cal, alpha),
            CostAwareJointCRC(shaping, shaping_ben, n_candidates=21, grid_size=101).calibrate(cal, alpha),
        ):
            res[d.method].append(evaluate(d, test, is_attack)["end_to_end_miss"])
    # Validity only. Which valid method raises fewer benign alerts is an
    # empirical question for the experiments, not a property to assert here.
    assert np.mean(res["separate"]) > alpha + 0.02  # the extra Stage 2 misses break the budget
    for m in ("bonferroni", "joint_crc", "joint_crc_cost"):
        assert np.mean(res[m]) <= alpha + 0.01, m


def test_evaluate_reports_stagewise_decomposition():
    rng = np.random.default_rng(3)
    att, ben = _attacks(rng, 300), _benign(rng, 300)
    test = _concat(att, ben)
    is_attack = np.r_[np.ones(300, bool), np.zeros(300, bool)]
    r = evaluate(calibrate_separate(att, 0.1), test, is_attack)
    # miss = Stage 1 miss + (1 - Stage 1 miss) * P(silent | flagged)
    assert r["end_to_end_miss"] == pytest.approx(r["stage1_miss"] + (1 - r["stage1_miss"]) * r["stage2_silent_given_flagged"])


def test_cost_aware_path_is_monotone_and_alerts_everything_at_zero():
    rng = np.random.default_rng(4)
    crc = CostAwareJointCRC(_attacks(rng, 400), _benign(rng, 400), n_candidates=21, grid_size=51)
    cal = _attacks(rng, 300)
    misses = [float(np.mean(~crc.decision(k).alert(cal))) for k in range(51)]
    assert misses[0] == 0.0
    assert all(b >= a - 1e-12 for a, b in zip(misses, misses[1:]))
