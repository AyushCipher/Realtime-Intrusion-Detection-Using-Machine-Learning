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


@pytest.mark.parametrize("gate", ["softmax", "openset"])
def test_train_with_gate_writes_expected_artifacts(tmp_path, gate):
    model_dir = _train_with_gate(tmp_path, gate)
    assert (model_dir / train.GATE_TYPE_FILENAME).read_text().strip() == gate
    assert (model_dir / train.ESCALATION_GATE_FILENAME).exists()
    assert (model_dir / train.ESCALATION_KIND_FILENAME).read_text().strip() == "static"
    if gate == "openset":
        assert (model_dir / train.OPENSET_GATE_FILENAME).exists()
    else:
        assert not (model_dir / train.OPENSET_GATE_FILENAME).exists()


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


@pytest.mark.parametrize("gate", ["softmax", "openset"])
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
