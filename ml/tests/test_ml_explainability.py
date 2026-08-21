from pathlib import Path

from ids_ml.data import load_and_map
from ids_ml.explainability import ShapExplainer, explanation_to_alert_field
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.pipeline import build_alert
from ids_ml.schema import validate_alert_event
from ids_ml.split import time_based_split
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


def _fit_classifier():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    clf = AttackClassifier(Stage2Config(n_estimators=50, random_state=0)).fit(
        train[CANONICAL_FEATURE_COLUMNS].to_numpy(), train["attack_category"].tolist()
    )
    return clf, test


def test_explain_returns_top_k_contributions_with_valid_feature_names():
    clf, test_df = _fit_classifier()
    explainer = ShapExplainer(clf)
    X_test = test_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    preds = clf.predict(X_test)

    contributions = explainer.explain(X_test[0], preds[0], top_k=5)

    assert len(contributions) == 5
    for c in contributions:
        assert c.feature in CANONICAL_FEATURE_COLUMNS
        assert isinstance(c.shap_value, float)


def test_explain_batch_matches_explain_row_by_row():
    clf, test_df = _fit_classifier()
    explainer = ShapExplainer(clf)
    X_test = test_df[CANONICAL_FEATURE_COLUMNS].to_numpy()[:4]
    preds = list(clf.predict(X_test))

    batch = explainer.explain_batch(X_test, preds, top_k=3)
    assert len(batch) == 4
    for i in range(4):
        single = explainer.explain(X_test[i], preds[i], top_k=3)
        assert [c.feature for c in single] == [c.feature for c in batch[i]]


def test_contributions_are_sorted_by_absolute_shap_value():
    clf, test_df = _fit_classifier()
    explainer = ShapExplainer(clf)
    X_test = test_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    preds = clf.predict(X_test)

    contributions = explainer.explain(X_test[0], preds[0], top_k=6)
    magnitudes = [abs(c.shap_value) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explanation_embeds_into_a_schema_valid_alert():
    clf, test_df = _fit_classifier()
    explainer = ShapExplainer(clf)
    X_test = test_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    preds = clf.predict(X_test)

    contributions = explainer.explain(X_test[0], preds[0], top_k=5)
    explanation_field = explanation_to_alert_field(contributions)
    assert isinstance(explanation_field, list)
    assert all(set(d.keys()) == {"feature", "value", "shap_value"} for d in explanation_field)

    from ids_ml.pipeline import ScoredFlow

    scored = ScoredFlow(
        stage1_anomaly_score=0.5,
        stage1_flagged=True,
        stage2_ran=True,
        stage2_predicted_class=str(preds[0]),
        stage2_confidence=0.9,
        stage2_class_probabilities={},
    )
    event = {"flow_id": "f1", "src_ip": "1.1.1.1", "src_port": 1, "dst_ip": "2.2.2.2",
              "dst_port": 2, "protocol": 6, "flow_start_time": 0.0}
    alert = build_alert(event, scored, explanation=explanation_field)
    validate_alert_event(alert)
    assert alert["explanation"] == explanation_field


def test_unfit_classifier_raises():
    import pytest
    from ids_ml.stage2_xgboost import AttackClassifier

    with pytest.raises(RuntimeError):
        ShapExplainer(AttackClassifier())
