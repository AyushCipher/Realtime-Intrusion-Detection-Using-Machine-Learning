from pathlib import Path

import pandas as pd

from ids_ml.data import load_and_map
from ids_ml.evaluation import (
    adversarial_robustness_report,
    concept_drift_report,
    leakage_comparison_report,
    per_category_report,
    simulate_low_and_slow,
)
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.pipeline import TwoStageDetector
from ids_ml.split import time_based_split, time_window_split
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


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
