"""Tests for the --gate wiring between ids_ml.train (fits + calibrates +
saves a gate) and ids_ml.serve (loads it back) -- the artifact contract
(gate_type.txt / openset_gate.joblib / escalation_gate.joblib) each side
depends on the other producing correctly.
"""

from pathlib import Path

import pytest

from ids_ml import serve, train
from ids_ml.pipeline import TwoStageDetector

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic_cicids_sample.csv"


def _train_with_gate(tmp_path, gate: str, escalation_budget: float = 0.2, adaptive: bool = False) -> Path:
    model_dir = tmp_path / "models"
    args = [
        "--data",
        str(FIXTURE_PATH),
        "--model-dir",
        str(model_dir),
        "--stage1-n-estimators",
        "50",
        "--stage2-n-estimators",
        "30",
        "--skip-leakage-check",
        "--skip-robustness-probe",
        "--gate",
        gate,
        "--escalation-budget",
        str(escalation_budget),
    ]
    if adaptive:
        args.append("--adaptive-escalation")
    train.main(args)
    return model_dir


def test_train_without_gate_writes_no_gate_artifacts(tmp_path):
    model_dir = tmp_path / "models"
    train.main(
        [
            "--data",
            str(FIXTURE_PATH),
            "--model-dir",
            str(model_dir),
            "--stage1-n-estimators",
            "50",
            "--stage2-n-estimators",
            "30",
            "--skip-leakage-check",
            "--skip-robustness-probe",
        ]
    )
    assert (model_dir / "stage1_iforest.joblib").exists()
    assert (model_dir / "stage2_xgboost.joblib").exists()
    assert not (model_dir / train.GATE_TYPE_FILENAME).exists()


@pytest.mark.parametrize("gate", ["softmax", "openset", "knn", "fused"])
def test_train_with_gate_writes_expected_artifacts(tmp_path, gate):
    model_dir = _train_with_gate(tmp_path, gate)
    assert (model_dir / train.GATE_TYPE_FILENAME).read_text().strip() == gate
    assert (model_dir / train.ESCALATION_GATE_FILENAME).exists()
    assert (model_dir / train.ESCALATION_KIND_FILENAME).read_text().strip() == "static"
    assert (model_dir / train.OPENSET_GATE_FILENAME).exists() == (gate == "openset")
    assert (model_dir / train.KNN_GATE_FILENAME).exists() == (gate == "knn")
    assert (model_dir / train.FUSED_GATE_FILENAME).exists() == (gate == "fused")


def test_train_with_adaptive_escalation_writes_adaptive_kind(tmp_path):
    model_dir = _train_with_gate(tmp_path, "openset", adaptive=True)
    assert (model_dir / train.ESCALATION_KIND_FILENAME).read_text().strip() == "adaptive"


def test_train_adaptive_escalation_requires_a_gate(tmp_path):
    model_dir = tmp_path / "models"
    with pytest.raises(ValueError):
        train.main(
            [
                "--data",
                str(FIXTURE_PATH),
                "--model-dir",
                str(model_dir),
                "--stage1-n-estimators",
                "50",
                "--stage2-n-estimators",
                "30",
                "--skip-leakage-check",
                "--skip-robustness-probe",
                "--adaptive-escalation",
            ]
        )


def test_serve_load_gate_reconstructs_a_working_adaptive_detector(tmp_path):
    from ids_ml.adaptive_conformal_gate import AdaptiveConformalGate

    model_dir = _train_with_gate(tmp_path, "openset", adaptive=True)

    stage1 = serve.AnomalyPreFilter.load(model_dir / "stage1_iforest.joblib")
    stage2 = serve.AttackClassifier.load(model_dir / "stage2_xgboost.joblib")
    loaded_gate, escalation_gate, trigger_name = serve.load_gate(model_dir, stage2)

    assert isinstance(escalation_gate, AdaptiveConformalGate)
    alpha_before = escalation_gate.alpha_t

    detector = TwoStageDetector(stage1, stage2, gate=loaded_gate, escalation_gate=escalation_gate, escalation_trigger_name=trigger_name)

    from ids_ml.data import load_and_map
    from ids_ml.features import CANONICAL_FEATURE_COLUMNS

    df = load_and_map(FIXTURE_PATH)
    X = df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    results = detector.score(X)

    n_flagged = sum(1 for r in results if r.stage1_flagged)
    assert n_flagged > 0
    # Proves score() drove update(), not just should_escalate() -- a
    # frozen adaptive gate would leave alpha_t exactly at its seeded value.
    assert escalation_gate.alpha_t != alpha_before


@pytest.mark.parametrize("gate", ["softmax", "openset", "knn", "fused"])
def test_serve_load_gate_reconstructs_a_working_detector(tmp_path, gate):
    model_dir = _train_with_gate(tmp_path, gate)

    stage1 = serve.AnomalyPreFilter.load(model_dir / "stage1_iforest.joblib")
    stage2 = serve.AttackClassifier.load(model_dir / "stage2_xgboost.joblib")
    loaded_gate, escalation_gate, trigger_name = serve.load_gate(model_dir, stage2)

    assert loaded_gate is not None
    assert escalation_gate is not None
    assert trigger_name == gate

    detector = TwoStageDetector(stage1, stage2, gate=loaded_gate, escalation_gate=escalation_gate, escalation_trigger_name=trigger_name)

    from ids_ml.data import load_and_map
    from ids_ml.features import CANONICAL_FEATURE_COLUMNS

    df = load_and_map(FIXTURE_PATH)
    X = df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    results = detector.score(X)

    assert any(r.stage1_flagged for r in results)  # sanity: the fixture actually exercises stage 2
    for r in results:
        if r.stage1_flagged:
            assert r.escalation_trigger == gate
        if r.escalated:
            assert r.decision == "escalated"


def test_serve_load_gate_returns_none_when_no_artifacts_present(tmp_path):
    model_dir = tmp_path / "models"
    train.main(
        [
            "--data",
            str(FIXTURE_PATH),
            "--model-dir",
            str(model_dir),
            "--stage1-n-estimators",
            "50",
            "--stage2-n-estimators",
            "30",
            "--skip-leakage-check",
            "--skip-robustness-probe",
        ]
    )
    stage2 = serve.AttackClassifier.load(model_dir / "stage2_xgboost.joblib")
    gate, escalation_gate, trigger_name = serve.load_gate(model_dir, stage2)
    assert gate is None
    assert escalation_gate is None
    assert trigger_name == ""


def test_train_gate_requires_nonempty_validation_split(tmp_path):
    model_dir = tmp_path / "models"
    with pytest.raises(ValueError):
        train.main(
            [
                "--data",
                str(FIXTURE_PATH),
                "--model-dir",
                str(model_dir),
                "--stage1-n-estimators",
                "50",
                "--stage2-n-estimators",
                "30",
                "--skip-leakage-check",
                "--skip-robustness-probe",
                "--val-frac",
                "0.0001",  # rounds down to an empty validation split
                "--gate",
                "softmax",
            ]
        )


def test_train_random_split_hybrid_fused_adaptive_end_to_end(tmp_path):
    """The deployment configuration: hybrid Stage 1 with recall control, fused
    gate, adaptive escalation, random split. Trains and reloads into a working
    detector."""
    model_dir = tmp_path / "m"
    train.main([
        "--data", str(FIXTURE_PATH), "--model-dir", str(model_dir), "--split", "random",
        "--stage1-kind", "hybrid", "--stage1-miss-budget", "0.05", "--gate", "fused", "--adaptive-escalation",
        "--stage2-n-estimators", "20", "--skip-leakage-check", "--skip-robustness-probe",
    ])
    from ids_ml.serve import load_gate
    from ids_ml.stage1_hybrid import load_stage1
    from ids_ml.stage2_xgboost import AttackClassifier

    stage2 = AttackClassifier.load(model_dir / "stage2_xgboost.joblib")
    gate, escalation_gate, name = load_gate(model_dir, stage2)
    assert name == "fused" and gate is not None and escalation_gate is not None
    stage1 = load_stage1(model_dir / "stage1_iforest.joblib")
    assert type(stage1).__name__ == "HybridPreFilter"


def test_joint_miss_budget_writes_calibration_and_serve_applies_it(tmp_path):
    """--joint-miss-budget calibrates both stages together; serve.py must load
    the silence threshold into the detector, and a flagged flow that Stage 2
    calls BENIGN with P(BENIGN) below it must be raised as an attack."""
    import json

    import numpy as np

    model_dir = tmp_path / "m"
    train.main([
        "--data", str(FIXTURE_PATH), "--model-dir", str(model_dir), "--split", "random",
        "--stage1-kind", "hybrid", "--gate", "fused", "--adaptive-escalation", "--joint-miss-budget", "0.1",
        "--stage2-n-estimators", "20", "--skip-leakage-check", "--skip-robustness-probe",
    ])
    cal = json.loads((model_dir / train.CASCADE_CALIBRATION_FILENAME).read_text())
    assert cal["joint_miss_budget"] == 0.1 and "benign_silence_threshold" in cal

    from ids_ml.stage2_xgboost import AttackClassifier

    stage2 = AttackClassifier.load(model_dir / "stage2_xgboost.joblib")
    gate, esc, name = serve.load_gate(model_dir, stage2)

    class AlwaysFlag:
        def anomaly_score(self, X):
            return np.ones(len(X))

        def flag(self, X):
            return np.ones(len(X), dtype=bool)

    from ids_ml.data import load_and_map
    from ids_ml.features import CANONICAL_FEATURE_COLUMNS

    X_all = load_and_map(FIXTURE_PATH)[CANONICAL_FEATURE_COLUMNS].to_numpy()
    benign_idx = stage2.classes_.index("BENIGN")
    proba = stage2.predict_proba(X_all)
    X = X_all[proba.argmax(axis=1) == benign_idx][:1]
    assert len(X) == 1  # the fixture has flows Stage 2 calls BENIGN
    p_benign = float(stage2.predict_proba(X)[0, benign_idx])
    strict = TwoStageDetector(AlwaysFlag(), stage2, benign_silence_threshold=p_benign + 1e-9).score(X)[0]
    lenient = TwoStageDetector(AlwaysFlag(), stage2, benign_silence_threshold=0.0).score(X)[0]
    default = TwoStageDetector(AlwaysFlag(), stage2).score(X)[0]
    assert strict.decision == "known-attack" and strict.stage2_predicted_class != "BENIGN"
    assert lenient.decision == "known-benign" and default.decision == "known-benign"


def test_joint_miss_budget_requires_a_gate(tmp_path):
    with pytest.raises(ValueError):
        train.main(["--data", str(FIXTURE_PATH), "--model-dir", str(tmp_path / "m"), "--joint-miss-budget", "0.1",
                    "--skip-leakage-check", "--skip-robustness-probe"])
