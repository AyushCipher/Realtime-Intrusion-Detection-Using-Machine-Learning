"""Tests for the per-source-IP sequence model (see sequence_model.py's
module docstring for why this is an MLP-over-windows stand-in for an
LSTM/GRU in this environment).

Uses tests/fixtures/synthetic_sequence_sample.csv, which -- unlike the main
synthetic_cicids_sample.csv fixture -- gives each "attacker" source IP a
deliberate recon (PortScan) -> brute-force -> DoS/exfiltration campaign, so
there's real per-IP history for build_sequences() to window over.
"""

from pathlib import Path

import numpy as np
import pytest

from ids_ml.data import load_and_map
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.sequence_model import SequenceAttackModel, SequenceModelConfig, build_sequences

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_sequence_sample.csv"


def _load():
    return load_and_map(FIXTURE_PATH)


def test_build_sequences_shape_matches_window_and_feature_count():
    df = _load()
    window = 3
    X, y = build_sequences(df, window=window)

    assert X.shape[1] == window * len(CANONICAL_FEATURE_COLUMNS)
    assert X.shape[0] == len(y)

    expected_samples = sum(max(0, n - window + 1) for n in df.groupby("src_ip").size())
    assert X.shape[0] == expected_samples


def test_build_sequences_requires_group_column():
    df = _load().drop(columns=["src_ip"])
    with pytest.raises(ValueError):
        build_sequences(df, window=3)


def test_campaign_windows_include_dos_ddos_as_a_terminal_label():
    # Each attacker IP's campaign ends with DoS Hulk flows -- a window whose
    # last flow is one of those should be labeled "DoS/DDoS".
    df = _load()
    X, y = build_sequences(df, window=3)
    assert "DoS/DDoS" in set(y)


def test_sequence_model_fit_and_predict():
    df = _load()
    X, y = build_sequences(df, window=3)
    model = SequenceAttackModel(SequenceModelConfig(window=3, hidden_layer_sizes=(16,), max_iter=300)).fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert set(preds).issubset(set(model.classes_))


def test_unfitted_model_raises():
    model = SequenceAttackModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 3 * len(CANONICAL_FEATURE_COLUMNS))))
