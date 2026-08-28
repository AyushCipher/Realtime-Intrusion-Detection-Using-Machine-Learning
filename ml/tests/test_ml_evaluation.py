from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_ml.data import load_and_map
from ids_ml.evaluation import (
    adversarial_robustness_report,
    concept_drift_report,
    expected_calibration_error,
    leakage_comparison_report,
    openset_vs_softmax_report,
    openset_vs_softmax_significance,
    per_category_report,
    run_openset_trial,
    amortized_latency_ms,
    latency_report,
    simulate_low_and_slow,
    static_vs_adaptive_conformal_drift_report,
    static_vs_adaptive_conformal_report,
    static_vs_adaptive_conformal_significance,
)
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.pipeline import TwoStageDetector
from ids_ml.split import time_based_split, time_window_split
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"
_FAST_N_ESTIMATORS = 30  # keep open-set trial tests fast; not a metric-quality claim


def test_per_category_report_basic_correctness():
    y_true = ["BENIGN", "BENIGN", "DoS/DDoS", "DoS/DDoS", "PortScan"]
    y_pred = ["BENIGN", "DoS/DDoS", "DoS/DDoS", "DoS/DDoS", "BENIGN"]

    report = per_category_report(y_true, y_pred)
    by_cat = report.set_index("category")

    # DoS/DDoS: true positives=2 (both actual DoS predicted DoS), false
    # positive=1 (a BENIGN predicted as DoS) -> precision=2/3, recall=2/2=1.0
    assert by_cat.loc["DoS/DDoS", "recall"] == 1.0
    assert abs(by_cat.loc["DoS/DDoS", "precision"] - (2 / 3)) < 1e-9
    assert by_cat.loc["PortScan", "support"] == 1
    # sorted by support descending: BENIGN(2) and DoS/DDoS(2) before PortScan(1)
    assert report.iloc[-1]["category"] == "PortScan"


def test_leakage_comparison_report_returns_valid_ranges():
    df = load_and_map(FIXTURE_PATH)
    result = leakage_comparison_report(
        df, CANONICAL_FEATURE_COLUMNS, stage2_config=Stage2Config(n_estimators=50, random_state=0)
    )

    assert 0.0 <= result["time_based_macro_f1"] <= 1.0
    assert 0.0 <= result["random_split_macro_f1"] <= 1.0
    assert isinstance(result["time_based_report"], pd.DataFrame)
    assert isinstance(result["random_split_report"], pd.DataFrame)
    assert set(result["time_based_report"]["category"]) == set(result["random_split_report"]["category"])


def test_concept_drift_report_structure():
    df = load_and_map(FIXTURE_PATH)
    window_a, window_b = time_window_split(df, pd.Timestamp("2017-07-04"))
    assert len(window_a) > 0 and len(window_b) > 0

    report = concept_drift_report(
        window_a, window_b, CANONICAL_FEATURE_COLUMNS, stage2_config=Stage2Config(n_estimators=50, random_state=0)
    )

    assert {"category", "f1_in_window", "f1_later_window", "f1_degradation"}.issubset(report.columns)
    assert report["f1_in_window"].between(0.0, 1.0).all()
    assert report["f1_later_window"].between(0.0, 1.0).all()


def test_simulate_low_and_slow_rescales_rate_features():
    df = load_and_map(FIXTURE_PATH).head(5).copy()
    slowed = simulate_low_and_slow(df, slowdown_factor=4.0)

    assert (slowed["flow_duration"] == df["flow_duration"] * 4.0).all()
    assert (slowed["flow_bytes_per_sec"] == df["flow_bytes_per_sec"] / 4.0).all()
    assert (slowed["flow_packets_per_sec"] == df["flow_packets_per_sec"] / 4.0).all()
    # total packet/byte counts are unchanged -- same payload, just paced differently
    assert (slowed["total_fwd_packets"] == df["total_fwd_packets"]).all()
    assert (slowed["total_fwd_bytes"] == df["total_fwd_bytes"]).all()


def test_adversarial_robustness_report_runs_and_returns_valid_rates():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()

    stage1 = AnomalyPreFilter(Stage1Config(n_estimators=100, contamination=0.25, random_state=0)).fit(X_train)
    stage2 = AttackClassifier(Stage2Config(n_estimators=50, random_state=0)).fit(
        X_train, train["attack_category"].tolist()
    )
    detector = TwoStageDetector(stage1, stage2)

    attack_rows = test[test["is_attack"]]
    report = adversarial_robustness_report(detector, attack_rows, CANONICAL_FEATURE_COLUMNS, slowdown_factor=8.0)

    assert 0.0 <= report["original_stage1_flag_rate"] <= 1.0
    assert 0.0 <= report["low_and_slow_stage1_flag_rate"] <= 1.0
    assert report["n_flows"] == len(attack_rows)


# --- Open-set: OpenMax vs softmax head-to-head ------------------------------


def test_expected_calibration_error_is_zero_for_a_perfectly_calibrated_score():
    # Half the mass at score 0.0 with true label 0, half at score 1.0 with
    # true label 1 -- every bin's mean score exactly matches its accuracy.
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert expected_calibration_error(y_true, y_score) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_is_positive_for_overconfident_scores():
    # Score claims near-certainty (0.95) but the true positive rate in that
    # bin is only 50% -- a large calibration gap.
    y_true = [0, 1, 0, 1]
    y_score = [0.95, 0.95, 0.95, 0.95]
    ece = expected_calibration_error(y_true, y_score)
    assert ece == pytest.approx(0.45, abs=1e-9)


def test_expected_calibration_error_returns_nan_for_empty_input():
    assert np.isnan(expected_calibration_error([], []))


def test_run_openset_trial_returns_both_gates_with_valid_ranges():
    df = load_and_map(FIXTURE_PATH)
    results = run_openset_trial(
        df, "Botnet", CANONICAL_FEATURE_COLUMNS, budget=0.2, seed=0, stage2_n_estimators=_FAST_N_ESTIMATORS
    )

    assert {r.gate_name for r in results} == {"softmax", "openset"}
    for r in results:
        assert r.family == "Botnet"
        assert r.n_unknown_test > 0
        for value in (r.escalation_rate_on_known, r.unknown_recall, r.unknown_auroc, r.ece, r.brier):
            assert np.isnan(value) or 0.0 <= value <= 1.0


def test_run_openset_trial_refuses_benign_holdout():
    df = load_and_map(FIXTURE_PATH)
    with pytest.raises(ValueError):
        run_openset_trial(df, "BENIGN", CANONICAL_FEATURE_COLUMNS, stage2_n_estimators=_FAST_N_ESTIMATORS)


def test_openset_vs_softmax_report_has_one_row_per_family_seed_gate():
    df = load_and_map(FIXTURE_PATH)
    families = ["Botnet", "PortScan"]
    seeds = [0, 1]
    report = openset_vs_softmax_report(
        df, CANONICAL_FEATURE_COLUMNS, families=families, seeds=seeds, budget=0.2, stage2_n_estimators=_FAST_N_ESTIMATORS
    )

    assert len(report) == len(families) * len(seeds) * 2
    assert set(report["family"]) == set(families)
    assert set(report["seed"]) == set(seeds)
    assert set(report["gate_name"]) == {"softmax", "openset"}


def test_openset_vs_softmax_significance_structure_and_p_value_range():
    df = load_and_map(FIXTURE_PATH)
    report = openset_vs_softmax_report(
        df,
        CANONICAL_FEATURE_COLUMNS,
        families=["Botnet", "PortScan", "Brute Force"],
        seeds=[0, 1, 2],
        budget=0.2,
        stage2_n_estimators=_FAST_N_ESTIMATORS,
    )
    result = openset_vs_softmax_significance(report, metric="unknown_recall")

    assert {"metric", "n_pairs", "statistic", "p_value", "openset_mean", "softmax_mean"}.issubset(result.keys())
    if result["n_pairs"] >= 2:
        assert np.isnan(result["p_value"]) or 0.0 <= result["p_value"] <= 1.0


def test_openset_vs_softmax_significance_handles_too_few_pairs():
    tiny_report = pd.DataFrame(
        [{"family": "Botnet", "seed": 0, "gate_name": "softmax", "unknown_recall": 0.5}]
    )
    result = openset_vs_softmax_significance(tiny_report, metric="unknown_recall")
    assert result["n_pairs"] < 2
    assert np.isnan(result["p_value"])


# --- Static vs. adaptive conformal calibration under drift -----------------


def test_static_vs_adaptive_conformal_drift_report_structure():
    df = load_and_map(FIXTURE_PATH)
    window_a, window_b = time_window_split(df, pd.Timestamp("2017-07-04"))
    assert len(window_a) > 0 and len(window_b) > 0

    results = static_vs_adaptive_conformal_drift_report(
        window_a,
        window_b,
        CANONICAL_FEATURE_COLUMNS,
        budget=0.1,
        stage2_n_estimators=_FAST_N_ESTIMATORS,
    )

    assert {r.segment for r in results} == {"pre_drift", "post_drift"}
    for r in results:
        assert r.n > 0
        assert 0.0 <= r.static_escalation_rate <= 1.0
        assert 0.0 <= r.adaptive_escalation_rate <= 1.0
        assert r.static_error == pytest.approx(abs(r.static_escalation_rate - 0.1))
        assert r.adaptive_error == pytest.approx(abs(r.adaptive_escalation_rate - 0.1))


def test_static_vs_adaptive_conformal_report_has_one_row_per_seed_segment():
    df = load_and_map(FIXTURE_PATH)
    window_a, window_b = time_window_split(df, pd.Timestamp("2017-07-04"))
    seeds = [0, 1, 2]

    report = static_vs_adaptive_conformal_report(
        window_a, window_b, CANONICAL_FEATURE_COLUMNS, budget=0.1, seeds=seeds, stage2_n_estimators=_FAST_N_ESTIMATORS
    )

    assert set(report["seed"]) == set(seeds)
    assert set(report["segment"]) == {"pre_drift", "post_drift"}
    assert len(report) == len(seeds) * 2


def test_static_vs_adaptive_conformal_significance_structure():
    df = load_and_map(FIXTURE_PATH)
    window_a, window_b = time_window_split(df, pd.Timestamp("2017-07-04"))
    report = static_vs_adaptive_conformal_report(
        window_a,
        window_b,
        CANONICAL_FEATURE_COLUMNS,
        budget=0.1,
        seeds=[0, 1, 2, 3, 4],
        stage2_n_estimators=_FAST_N_ESTIMATORS,
    )
    result = static_vs_adaptive_conformal_significance(report)

    assert {"n_pairs", "statistic", "p_value", "static_error_mean", "adaptive_error_mean"}.issubset(result.keys())
    if result["n_pairs"] >= 2:
        assert np.isnan(result["p_value"]) or 0.0 <= result["p_value"] <= 1.0


def test_static_vs_adaptive_conformal_significance_handles_too_few_pairs():
    tiny_report = pd.DataFrame([{"segment": "pre_drift", "static_error": 0.05, "adaptive_error": 0.01}])
    result = static_vs_adaptive_conformal_significance(tiny_report)
    assert result["n_pairs"] < 2
    assert np.isnan(result["p_value"])


# --- Latency / throughput ---------------------------------------------


def _fit_detector_for_latency():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()
    stage1 = AnomalyPreFilter(Stage1Config(n_estimators=50, contamination=0.25, random_state=0)).fit(X_train)
    stage2 = AttackClassifier(Stage2Config(n_estimators=30, random_state=0)).fit(X_train, train["attack_category"].tolist())
    detector = TwoStageDetector(stage1, stage2)
    X_test = test[CANONICAL_FEATURE_COLUMNS].to_numpy()
    return detector, X_test


def test_latency_report_returns_positive_finite_numbers():
    detector, X_test = _fit_detector_for_latency()
    report = latency_report(detector, X_test, warmup=5, n_trials=20, batch_size=16)

    for value in (
        report.single_median_ms,
        report.single_p95_ms,
        report.single_p99_ms,
        report.single_throughput_per_sec,
        report.batch_median_ms,
        report.batch_throughput_per_sec,
    ):
        assert value > 0
        assert np.isfinite(value)
    # p99 should be at or above the median (same distribution, higher percentile)
    assert report.single_p99_ms >= report.single_median_ms
    assert report.single_p95_ms >= report.single_median_ms


def test_latency_report_batch_size_is_capped_by_available_rows():
    detector, X_test = _fit_detector_for_latency()
    report = latency_report(detector, X_test, warmup=2, n_trials=5, batch_size=10_000)
    assert report.batch_size == len(X_test)


def test_latency_report_rejects_empty_input():
    detector, _X_test = _fit_detector_for_latency()
    with pytest.raises(ValueError):
        latency_report(detector, np.empty((0, len(CANONICAL_FEATURE_COLUMNS))))


def test_amortized_latency_is_tier1_when_escalation_rate_is_zero():
    assert amortized_latency_ms(tier1_median_ms=5.0, escalation_rate=0.0, tier2_latency_ms=500.0) == pytest.approx(5.0)


def test_amortized_latency_adds_full_tier2_cost_when_escalation_rate_is_one():
    assert amortized_latency_ms(tier1_median_ms=5.0, escalation_rate=1.0, tier2_latency_ms=500.0) == pytest.approx(505.0)


def test_amortized_latency_scales_linearly_with_escalation_rate():
    assert amortized_latency_ms(tier1_median_ms=2.0, escalation_rate=0.1, tier2_latency_ms=1000.0) == pytest.approx(102.0)
