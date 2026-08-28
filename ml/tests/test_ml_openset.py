from pathlib import Path

import pytest

from ids_ml.conformal_gate import calibrate_threshold
from ids_ml.data import load_and_map
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.openset_head import OpenMaxConfig, OpenMaxHead
from ids_ml.pipeline import TwoStageDetector
from ids_ml.softmax_gate import SoftmaxGate
from ids_ml.split import leave_one_family_out, time_based_split
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


def _fit_stage2(train):
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_train = train["attack_category"].tolist()
    return AttackClassifier(Stage2Config(n_estimators=100, random_state=0)).fit(X_train, y_train)


def test_softmax_gate_unknown_mass_is_one_minus_confidence():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    stage2 = _fit_stage2(train)
    gate = SoftmaxGate(stage2)

    X_test = test[CANONICAL_FEATURE_COLUMNS].to_numpy()
    proba = stage2.predict_proba(X_test)
    results = gate.recalibrate_batch(X_test)

    for i, r in enumerate(results):
        assert r.unknown_mass == pytest.approx(1.0 - proba[i].max(), abs=1e-6)
        assert 0.0 <= r.unknown_mass <= 1.0
        assert sum(r.known_class_probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_openmax_head_known_class_probabilities_plus_unknown_mass_sum_to_one():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    stage2 = _fit_stage2(train)

    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_train = train["attack_category"].tolist()
    head = OpenMaxHead(stage2, OpenMaxConfig(tail_size=10, alpha_ranks=3)).fit(X_train, y_train)

    X_test = test[CANONICAL_FEATURE_COLUMNS].to_numpy()
    results = head.recalibrate_batch(X_test)

    assert len(results) == len(test)
    for r in results:
        total = sum(r.known_class_probabilities.values()) + r.unknown_mass
        assert total == pytest.approx(1.0, abs=1e-4)
        assert 0.0 <= r.unknown_mass <= 1.0
        assert r.predicted_class in set(stage2.classes_)


def test_openmax_head_must_be_fit_before_use():
    df = load_and_map(FIXTURE_PATH)
    train, _val, _test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    stage2 = _fit_stage2(train)
    head = OpenMaxHead(stage2)

    with pytest.raises(RuntimeError):
        head.recalibrate_batch(train[CANONICAL_FEATURE_COLUMNS].to_numpy())


def test_openmax_flags_higher_unknown_mass_on_a_held_out_family_than_on_known_traffic():
    # The core open-set sanity check: a family the model never trained on
    # should look "more unknown" on average than traffic from families it
    # did train on. This is a coarse average-level check (the synthetic
    # fixture's tiny per-family counts make per-sample guarantees noisy,
    # see ml/README.md's known-limitations pattern for this fixture), not a
    # claim about a specific unknown-recall number -- that's evaluation.py's
    # job once it's wired up to the LOFO rotation.
    df = load_and_map(FIXTURE_PATH)
    train, test = leave_one_family_out(df, "Botnet", train_frac=0.7)
    stage2 = _fit_stage2(train)

    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_train = train["attack_category"].tolist()
    head = OpenMaxHead(stage2, OpenMaxConfig(tail_size=10, alpha_ranks=3)).fit(X_train, y_train)

    known_test = test[test["attack_category"] != "Botnet"]
    unknown_test = test[test["attack_category"] == "Botnet"]
    assert len(unknown_test) > 0

    known_results = head.recalibrate_batch(known_test[CANONICAL_FEATURE_COLUMNS].to_numpy())
    unknown_results = head.recalibrate_batch(unknown_test[CANONICAL_FEATURE_COLUMNS].to_numpy())

    known_mean = sum(r.unknown_mass for r in known_results) / len(known_results)
    unknown_mean = sum(r.unknown_mass for r in unknown_results) / len(unknown_results)
    assert unknown_mean > known_mean


def test_calibrate_threshold_controls_escalation_rate_on_held_out_known_traffic():
    df = load_and_map(FIXTURE_PATH)
    train, calib, test = time_based_split(df, train_frac=0.5, val_frac=0.25)
    stage2 = _fit_stage2(train)
    gate = SoftmaxGate(stage2)

    calib_scores = [r.unknown_mass for r in gate.recalibrate_batch(calib[CANONICAL_FEATURE_COLUMNS].to_numpy())]
    conformal = calibrate_threshold(calib_scores, budget=0.2)

    escalated = sum(1 for s in calib_scores if conformal.should_escalate(s))
    # Split-conformal's finite-sample guarantee is about a *fresh* draw from
    # the same distribution, not the calibration set itself -- but on the
    # calibration set the escalation rate at the calibrated threshold should
    # still land close to the requested budget, not wildly off it.
    assert escalated / len(calib_scores) <= 0.2 + 0.05


def test_calibrate_threshold_rejects_bad_budget():
    with pytest.raises(ValueError):
        calibrate_threshold([0.1, 0.2, 0.3], budget=1.5)


def test_calibrate_threshold_rejects_empty_scores():
    with pytest.raises(ValueError):
        calibrate_threshold([], budget=0.2)


def test_two_stage_detector_without_gate_never_escalates():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()

    stage1 = AnomalyPreFilter(Stage1Config(n_estimators=100, contamination=0.25, random_state=0)).fit(X_train)
    stage2 = _fit_stage2(train)
    detector = TwoStageDetector(stage1, stage2)

    results = detector.score(test[CANONICAL_FEATURE_COLUMNS].to_numpy())
    assert all(not r.escalated for r in results)
    assert all(r.unknown_mass == 0.0 for r in results)
    assert all(r.escalation_trigger == "" for r in results)
    assert all(r.decision in ("known-benign", "known-attack") for r in results)


def test_two_stage_detector_with_gate_and_escalation_gate_routes_three_ways():
    df = load_and_map(FIXTURE_PATH)
    train, calib, test = time_based_split(df, train_frac=0.5, val_frac=0.25)
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()

    stage1 = AnomalyPreFilter(Stage1Config(n_estimators=100, contamination=0.35, random_state=0)).fit(X_train)
    stage2 = _fit_stage2(train)
    gate = SoftmaxGate(stage2)

    X_calib_flagged = calib[CANONICAL_FEATURE_COLUMNS].to_numpy()[stage1.flag(calib[CANONICAL_FEATURE_COLUMNS].to_numpy())]
    calib_scores = [r.unknown_mass for r in gate.recalibrate_batch(X_calib_flagged)] if len(X_calib_flagged) else [0.5]
    escalation_gate = calibrate_threshold(calib_scores, budget=0.3)

    detector = TwoStageDetector(stage1, stage2, gate=gate, escalation_gate=escalation_gate, escalation_trigger_name="softmax")
    results = detector.score(test[CANONICAL_FEATURE_COLUMNS].to_numpy())

    decisions = {r.decision for r in results}
    assert decisions <= {"known-benign", "known-attack", "escalated"}
    for r in results:
        if r.stage1_flagged:
            assert r.escalation_trigger == "softmax"
        if r.escalated:
            assert r.decision == "escalated"
            assert r.unknown_mass > escalation_gate.threshold
