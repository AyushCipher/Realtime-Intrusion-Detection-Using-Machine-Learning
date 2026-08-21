from pathlib import Path

from ids_ml.data import load_and_map
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.split import time_based_split
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


def _train_test_matrices():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    return train[CANONICAL_FEATURE_COLUMNS].to_numpy(), test[CANONICAL_FEATURE_COLUMNS].to_numpy(), test


def test_fit_and_score_shapes():
    X_train, X_test, _ = _train_test_matrices()
    model = AnomalyPreFilter(Stage1Config(n_estimators=50, random_state=0)).fit(X_train)

    scores = model.anomaly_score(X_test)
    flags = model.flag(X_test)

    assert scores.shape == (X_test.shape[0],)
    assert flags.shape == (X_test.shape[0],)
    assert flags.dtype == bool


def test_unfitted_model_raises():
    model = AnomalyPreFilter()
    try:
        model.anomaly_score(_train_test_matrices()[0][:5])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError before fit()")


def test_flags_a_higher_share_of_attacks_than_benign_traffic():
    # Isolation Forest is unsupervised, so this is a sanity check on
    # separability, not a precision/recall guarantee -- the caricatured
    # synthetic attack categories should still look more "unusual" to it
    # than the (also synthetic, but centrally-clustered) benign traffic.
    X_train, X_test, test_df = _train_test_matrices()
    model = AnomalyPreFilter(Stage1Config(n_estimators=100, contamination=0.2, random_state=0)).fit(X_train)
    flags = model.flag(X_test)

    is_attack = test_df["is_attack"].to_numpy()
    attack_flag_rate = flags[is_attack].mean() if is_attack.any() else 0.0
    benign_flag_rate = flags[~is_attack].mean() if (~is_attack).any() else 0.0

    assert attack_flag_rate > benign_flag_rate


def test_save_and_load_round_trip(tmp_path):
    X_train, X_test, _ = _train_test_matrices()
    model = AnomalyPreFilter(Stage1Config(n_estimators=30, random_state=0)).fit(X_train)
    path = tmp_path / "stage1.joblib"
    model.save(path)

    loaded = AnomalyPreFilter.load(path)
    import numpy as np
    assert np.allclose(loaded.anomaly_score(X_test), model.anomaly_score(X_test))
