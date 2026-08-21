from pathlib import Path

from ids_ml.data import load_and_map
from ids_ml.features import CANONICAL_FEATURE_COLUMNS, event_to_feature_vector
from ids_ml.pipeline import TwoStageDetector, build_alert, severity_for
from ids_ml.schema import validate_alert_event
from ids_ml.split import time_based_split
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


def _fit_detector():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()

    stage1 = AnomalyPreFilter(Stage1Config(n_estimators=100, contamination=0.25, random_state=0)).fit(X_train)
    stage2 = AttackClassifier(Stage2Config(n_estimators=100, random_state=0)).fit(
        X_train, train["attack_category"].tolist()
    )
    detector = TwoStageDetector(stage1, stage2)
    return detector, test


def test_score_only_runs_stage2_on_flagged_rows():
    detector, test_df = _fit_detector()
    X_test = test_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    results = detector.score(X_test)

    assert len(results) == len(test_df)
    for r in results:
        if not r.stage1_flagged:
            assert r.stage2_ran is False
            assert r.stage2_predicted_class == "BENIGN"
            assert r.stage2_class_probabilities == {}
        else:
            assert r.stage2_ran is True
            assert r.stage2_predicted_class in set(detector.stage2.classes_)
            assert 0.0 <= r.stage2_confidence <= 1.0


def test_severity_mapping():
    assert severity_for("BENIGN", 0.99) == "info"
    assert severity_for("Heartbleed", 0.9) == "critical"
    assert severity_for("Heartbleed", 0.3) == "high"  # low confidence downgrades one level
    assert severity_for("PortScan", 0.9) == "low"


def test_build_alert_produces_a_schema_valid_event():
    detector, test_df = _fit_detector()
    X_test = test_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    results = detector.score(X_test)

    row = test_df.iloc[0]
    event = {
        "flow_id": "flow-123",
        "src_ip": row.get("src_ip", "10.0.0.1"),
        "src_port": 1234,
        "dst_ip": "10.0.0.2",
        "dst_port": 443,
        "protocol": 6,
        "flow_start_time": 1_700_000_000.0,
    }
    alert = build_alert(event, results[0])
    validate_alert_event(alert)  # raises on any schema mismatch
    assert alert["flow_id"] == "flow-123"


def test_event_to_feature_vector_matches_canonical_column_order():
    event = {name: float(i) for i, name in enumerate(CANONICAL_FEATURE_COLUMNS)}
    vec = event_to_feature_vector(event)
    assert list(vec) == [float(i) for i in range(len(CANONICAL_FEATURE_COLUMNS))]


def test_event_to_feature_vector_defaults_missing_fields_to_zero():
    vec = event_to_feature_vector({})
    assert vec.shape == (len(CANONICAL_FEATURE_COLUMNS),)
    assert (vec == 0.0).all()
