"""Tests for the paper-extension modules: stronger OOD baselines
(ood_gates.py), adaptive Stage 1 recall control (adaptive_stage1.py), the
detect-then-recalibrate baseline (drift_recalibration.py), and the
evaluation functions that use them."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_ml.adaptive_stage1 import AdaptiveStage1Threshold
from ids_ml.conformal_stage1 import calibrate_stage1_threshold
from ids_ml.data import load_and_map
from ids_ml.drift_recalibration import DetectThenRecalibrate
from ids_ml.evaluation import (
    ALL_GATE_NAMES,
    drift_method_significance,
    open_auc_from_scores,
    openset_vs_softmax_report,
    paired_gate_significance,
    run_openset_trial,
    stage1_stream_report,
    stage1_stream_trial,
    static_vs_adaptive_conformal_drift_report,
    static_vs_adaptive_conformal_report,
)
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.ood_gates import OOD_GATES
from ids_ml.split import time_window_split
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"
_FAST_N_ESTIMATORS = 30


# --- ood_gates -------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_stage2():
    df = load_and_map(FIXTURE_PATH)
    X = df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y = df["attack_category"].tolist()
    return AttackClassifier(Stage2Config(n_estimators=_FAST_N_ESTIMATORS)).fit(X, y), X, y


@pytest.mark.parametrize("name", sorted(OOD_GATES))
def test_ood_gate_scores_are_finite_and_shaped_like_other_gates(name, fitted_stage2):
    clf, X, y = fitted_stage2
    gate = OOD_GATES[name](clf).fit(X, y)
    results = gate.recalibrate_batch(X[:50])
    assert len(results) == 50
    for r in results:
        assert np.isfinite(r.unknown_mass)
        assert r.predicted_class in clf.classes_
        assert sum(r.known_class_probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert gate.recalibrate_batch(X[:0]) == []


@pytest.mark.parametrize("name", ["mahalanobis", "knn"])
def test_distance_gates_score_far_away_flows_as_more_novel(name, fitted_stage2):
    clf, X, y = fitted_stage2
    gate = OOD_GATES[name](clf).fit(X, y)
    near = np.array([r.unknown_mass for r in gate.recalibrate_batch(X[:100])])
    far = np.array([r.unknown_mass for r in gate.recalibrate_batch(X[:100] * 1e4 + 1e6)])
    assert far.mean() > near.mean()


def test_energy_is_lower_for_confident_rows_than_maxlogit_tie_free_ordering(fitted_stage2):
    # Energy and max-logit both decrease as the top margin grows; they should
    # rank the same rows broadly the same way.
    clf, X, y = fitted_stage2
    e = np.array([r.unknown_mass for r in OOD_GATES["energy"](clf).recalibrate_batch(X[:200])])
    m = np.array([r.unknown_mass for r in OOD_GATES["maxlogit"](clf).recalibrate_batch(X[:200])])
    assert np.corrcoef(e, m)[0, 1] > 0.8


# --- open_auc_from_scores --------------------------------------------------


def test_open_auc_from_scores_perfect_and_excludes_misclassified():
    assert open_auc_from_scores(["a", "b"], ["a", "b"], [0.1, 0.2], [0.9, 0.8]) == pytest.approx(1.0)
    # The misclassified known row (score 0.95) is dropped, not counted as a loss.
    assert open_auc_from_scores(["a", "b"], ["a", "a"], [0.1, 0.95], [0.9]) == pytest.approx(1.0)
    # Ties are not wins.
    assert open_auc_from_scores(["a"], ["a"], [0.5], [0.5]) == pytest.approx(0.0)
    assert np.isnan(open_auc_from_scores(["a"], ["b"], [0.1], [0.9]))
    assert np.isnan(open_auc_from_scores(["a"], ["a"], [0.1], []))


# --- adaptive_stage1 -------------------------------------------------------


def test_adaptive_stage1_tracks_budget_when_attack_scores_drift_down():
    rng = np.random.default_rng(0)
    calib = rng.normal(5.0, 1.0, 400)
    stream = rng.normal(3.5, 1.0, 6000)  # attacks get stealthier: lower anomaly scores
    budget = 0.1

    static_thr = calibrate_stage1_threshold(calib, budget).threshold
    static_miss = float((stream < static_thr).mean())

    adaptive = AdaptiveStage1Threshold(budget, gamma=0.02, window_size=300).seed(calib)
    for s in stream:
        adaptive.update(s, adaptive.threshold)

    assert static_miss > 0.4
    assert abs(adaptive.realised_miss_rate - budget) < 0.05


def test_adaptive_stage1_threshold_is_minus_inf_without_data():
    gate = AdaptiveStage1Threshold(0.1)
    assert gate.threshold == float("-inf")
    assert gate.flag(-1e9)
    assert np.isnan(gate.realised_miss_rate)


def test_adaptive_stage1_counts_miss_against_threshold_at_decision():
    gate = AdaptiveStage1Threshold(0.1).seed([1.0, 2.0, 3.0])
    assert gate.update(0.5, threshold_at_decision=1.0) is True
    assert gate.update(5.0, threshold_at_decision=1.0) is False
    assert gate.n_feedback == 2 and gate.n_missed == 1


@pytest.mark.parametrize("bad", [{"miss_budget": 0.0}, {"miss_budget": 0.1, "gamma": 0.0}, {"miss_budget": 0.1, "window_size": 1}])
def test_adaptive_stage1_validates_arguments(bad):
    with pytest.raises(ValueError):
        AdaptiveStage1Threshold(**bad)


# --- drift_recalibration ---------------------------------------------------


def test_detect_then_recalibrate_fires_on_shift_and_restores_upper_tail_rate():
    rng = np.random.default_rng(1)
    budget = 0.1
    gate = DetectThenRecalibrate(budget, tail="upper", window_size=500, check_every=100).seed(rng.normal(0, 1, 2000))
    shifted = rng.normal(2.0, 1.0, 5000)
    decisions = []
    for s in shifted:
        decisions.append(gate.decide(s))
        gate.update(s)
    assert gate.n_recalibrations >= 1
    # After the first recalibration the escalation rate should be near budget.
    first = gate.recalibrated_at[0]
    assert abs(np.mean(decisions[first + 500 :]) - budget) < 0.05


def test_detect_then_recalibrate_does_not_fire_without_shift():
    rng = np.random.default_rng(2)
    gate = DetectThenRecalibrate(0.1, tail="upper", window_size=500, check_every=100, drift_alpha=0.001)
    gate.seed(rng.normal(0, 1, 3000))
    for s in rng.normal(0, 1, 3000):
        gate.update(s)
    assert gate.n_recalibrations <= 1


def test_detect_then_recalibrate_lower_tail_flags_by_threshold():
    gate = DetectThenRecalibrate(0.1, tail="lower").seed(np.arange(100, dtype=float))
    assert gate.decide(gate.threshold)
    assert not gate.decide(gate.threshold - 1.0)


def test_detect_then_recalibrate_rejects_bad_tail():
    with pytest.raises(ValueError):
        DetectThenRecalibrate(0.1, tail="middle")


# --- evaluation integration on the synthetic fixture -----------------------


def test_run_openset_trial_with_all_gates():
    df = load_and_map(FIXTURE_PATH)
    results = run_openset_trial(
        df, "Botnet", CANONICAL_FEATURE_COLUMNS, budget=0.2, seed=0,
        stage2_n_estimators=_FAST_N_ESTIMATORS, gate_names=ALL_GATE_NAMES,
    )
    by_gate = {r.gate_name: r for r in results}
    assert set(by_gate) == set(ALL_GATE_NAMES)
    for name, r in by_gate.items():
        for value in (r.escalation_rate_on_known, r.unknown_recall, r.unknown_auroc, r.open_auc_score, r.recall_at_matched_fpr):
            assert np.isnan(value) or 0.0 <= value <= 1.0
        if name in ("softmax", "openset"):
            assert np.isnan(r.ece) or 0.0 <= r.ece <= 1.0
        else:
            assert np.isnan(r.ece) and np.isnan(r.brier)


def test_paired_gate_significance_counts_families_not_seeds():
    df = load_and_map(FIXTURE_PATH)
    report = openset_vs_softmax_report(
        df, CANONICAL_FEATURE_COLUMNS, families=["Botnet", "PortScan", "Brute Force"], seeds=[0, 1],
        budget=0.2, stage2_n_estimators=_FAST_N_ESTIMATORS, gate_names=["softmax", "energy"],
    )
    result = paired_gate_significance(report, "energy", "softmax")
    assert result["n_clusters"] == 3
    assert result["a_wins"] + result["b_wins"] <= 3
    assert np.isnan(result["p_value"]) or 0.0 <= result["p_value"] <= 1.0


def test_drift_report_includes_recalibration_baseline():
    df = load_and_map(FIXTURE_PATH)
    window_a, window_b = time_window_split(df, pd.Timestamp("2017-07-04"))
    results = static_vs_adaptive_conformal_drift_report(
        window_a, window_b, CANONICAL_FEATURE_COLUMNS, budget=0.1,
        stage2_n_estimators=_FAST_N_ESTIMATORS, recalib_window_size=50, recalib_check_every=10,
    )
    for r in results:
        assert 0.0 <= r.recalibrated_escalation_rate <= 1.0
        assert r.recalibrated_error == pytest.approx(abs(r.recalibrated_escalation_rate - 0.1))
        assert r.n_recalibrations >= 0

    report = static_vs_adaptive_conformal_report(
        window_a, window_b, CANONICAL_FEATURE_COLUMNS, budget=0.1, seeds=[0, 1],
        stage2_n_estimators=_FAST_N_ESTIMATORS,
    )
    sig = drift_method_significance(report, "recalibrated", "adaptive")
    assert sig["n_pairs"] == len(report)


def test_stage1_stream_trial_segments_and_ranges():
    df = load_and_map(FIXTURE_PATH)
    rows = stage1_stream_trial(df, CANONICAL_FEATURE_COLUMNS, miss_budget=0.1, seed=0, stage1_n_estimators=50)
    out = pd.DataFrame(rows)
    assert set(out["method"]) == {"static", "adaptive", "recalibrated"}
    assert {"overall", "seen_families", "new_families"} <= set(out["segment"])
    assert any(s.startswith("day_") for s in out["segment"])
    overall = out[out["segment"] == "overall"]
    assert (overall["n_attacks"] > 0).all()
    for col in ("miss_rate", "benign_flag_rate"):
        vals = overall[col].dropna()
        assert ((vals >= 0) & (vals <= 1)).all()


def test_stage1_stream_report_grid_and_feedback_delay():
    df = load_and_map(FIXTURE_PATH)
    report = stage1_stream_report(
        df, CANONICAL_FEATURE_COLUMNS, miss_budgets=[0.1], seeds=[0], feedback_delays=[0, 50],
        stage1_n_estimators=50,
    )
    assert set(report["feedback_delay"]) == {0, 50}
    # The static method ignores feedback, so its result cannot depend on the delay.
    static = report[(report["method"] == "static") & (report["segment"] == "overall")]
    assert static["miss_rate"].nunique() == 1


# --- cross-dataset and generic loader --------------------------------------


def test_cross_dataset_trial_rows_and_ranges():
    from ids_ml.evaluation import cross_dataset_trial

    df = load_and_map(FIXTURE_PATH)
    # Stand-in for two datasets: first and second half of the fixture in
    # time, with the target's "PortScan" renamed so it looks foreign.
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    source = ordered.iloc[: len(ordered) // 2]
    target = ordered.iloc[len(ordered) // 2 :].copy()
    target.loc[target["attack_category"] == "PortScan", "attack_category"] = "scanning"
    target.loc[target["attack_category"] == "Botnet", "attack_category"] = "never-seen"
    rows = cross_dataset_trial(
        source, target, CANONICAL_FEATURE_COLUMNS, family_map={"scanning": "PortScan"},
        stage2_n_estimators=_FAST_N_ESTIMATORS, gate_names=["softmax", "energy"],
    )
    by_gate = {r["gate_name"]: r for r in rows}
    assert set(by_gate) == {"softmax", "energy", "stage1"}
    assert by_gate["softmax"]["n_target_novel"] > 0 and by_gate["softmax"]["n_target_shared"] > 0
    for g in ("softmax", "energy"):
        for m in ("static", "adaptive", "recalibrated"):
            assert 0.0 <= by_gate[g][f"{m}_benign_escalation_rate"] <= 1.0
    assert 0.0 <= by_gate["stage1"]["static_miss_rate"] <= 1.0
    assert 0.0 <= by_gate["stage1"]["adaptive_miss_rate"] <= 1.0


def test_load_and_map_cicflowmeter_uses_attack_column_and_benign_aliases(tmp_path):
    from ids_ml.data import CICIDS_COLUMN_MAP, load_and_map_cicflowmeter

    n = 4
    frame = {col: [1.0] * n for col in CICIDS_COLUMN_MAP}
    frame["Label"] = [0, 1, 1, 0]
    frame["Attack"] = ["Benign", "ddos", "xss", "normal"]
    frame["Timestamp"] = ["2019-01-01 00:00:0%d" % i for i in range(n)]
    path = tmp_path / "foreign.csv"
    pd.DataFrame(frame).to_csv(path, index=False)

    out = load_and_map_cicflowmeter(path)
    assert out["attack_category"].tolist() == ["BENIGN", "ddos", "xss", "BENIGN"]
    assert out["is_attack"].tolist() == [False, True, True, False]
    assert list(out.columns[: len(CANONICAL_FEATURE_COLUMNS)]) == CANONICAL_FEATURE_COLUMNS


def test_load_and_map_cicflowmeter_fails_loudly_without_label(tmp_path):
    from ids_ml.data import CICIDS_COLUMN_MAP, load_and_map_cicflowmeter

    path = tmp_path / "nolabel.csv"
    pd.DataFrame({col: [1.0] for col in CICIDS_COLUMN_MAP}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no label column"):
        load_and_map_cicflowmeter(path)
