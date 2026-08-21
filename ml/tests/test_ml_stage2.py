from pathlib import Path

import numpy as np

from ids_ml.data import load_and_map
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.split import time_based_split
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


def _train_test():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_train = train["attack_category"].tolist()
    X_test = test[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_test = test["attack_category"].tolist()
    return X_train, y_train, X_test, y_test


def test_fit_predict_and_predict_proba_shapes():
    X_train, y_train, X_test, y_test = _train_test()
    clf = AttackClassifier(Stage2Config(n_estimators=50, random_state=0)).fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    assert preds.shape == (X_test.shape[0],)
    assert proba.shape == (X_test.shape[0], len(clf.classes_))
    assert set(preds).issubset(set(clf.classes_))
    # predict_proba rows should sum to ~1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_classifier_separates_the_caricatured_synthetic_categories_well():
    # The synthetic fixture's categories are deliberately far apart (see
    # tests/generate_ml_fixtures.py), so a reasonably well-fit classifier
    # should do well on it. This is a sanity check on the training code
    # path, not a claim about real-world CICIDS2017 accuracy.
    X_train, y_train, X_test, y_test = _train_test()
    clf = AttackClassifier(Stage2Config(n_estimators=100, random_state=0)).fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracy = (preds == np.array(y_test)).mean()
    assert accuracy > 0.8


def test_unfitted_model_raises():
    clf = AttackClassifier()
    try:
        clf.predict(np.zeros((1, len(CANONICAL_FEATURE_COLUMNS))))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError before fit()")


def test_save_and_load_round_trip(tmp_path):
    X_train, y_train, X_test, _ = _train_test()
    clf = AttackClassifier(Stage2Config(n_estimators=30, random_state=0)).fit(X_train, y_train)
    path = tmp_path / "stage2.joblib"
    clf.save(path)

    loaded = AttackClassifier.load(path)
    assert list(loaded.predict(X_test)) == list(clf.predict(X_test))
