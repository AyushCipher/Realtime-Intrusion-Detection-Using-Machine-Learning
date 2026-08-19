"""Tests for CICIDS CSV loading/mapping against the synthetic fixture.

tests/fixtures/synthetic_cicids_sample.csv is synthetic (see
tests/generate_ml_fixtures.py) but shaped exactly like a real CICIDS2017
CSV, including its inconsistent leading-whitespace column-naming quirk, so
these tests exercise the real parsing/mapping logic.
"""

from pathlib import Path

from ids_ml.data import attack_category, load_and_map, load_cicids_csv
from ids_ml.features import CANONICAL_FEATURE_COLUMNS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


def test_load_cicids_csv_strips_column_whitespace():
    df = load_cicids_csv(FIXTURE_PATH)
    assert "Flow Duration" in df.columns
    assert " Flow Duration" not in df.columns


def test_map_to_canonical_produces_expected_columns_and_row_count():
    df = load_and_map(FIXTURE_PATH)
    assert len(df) == 1224
    for col in CANONICAL_FEATURE_COLUMNS:
        assert col in df.columns
    assert set(["label", "attack_category", "is_attack", "timestamp", "src_ip"]).issubset(df.columns)


def test_microsecond_columns_converted_to_seconds():
    df = load_and_map(FIXTURE_PATH)
    # BASE_PARAMS["BENIGN"]["dur"] = 2_000_000 microseconds = 2.0 seconds;
    # jittered per-row but should be on the order of single-digit seconds,
    # not millions.
    benign = df[df["label"] == "BENIGN"]
    assert benign["flow_duration"].median() < 100.0
    assert benign["flow_duration"].median() > 0.01


def test_attack_category_mapping():
    assert attack_category("BENIGN") == "BENIGN"
    assert attack_category("DoS Hulk") == "DoS/DDoS"
    assert attack_category("PortScan") == "PortScan"
    assert attack_category("Web Attack - Brute Force") == "Web Attack"
    assert attack_category("Some Unknown Future Attack") == "Other"


def test_no_missing_or_infinite_values_in_canonical_columns():
    df = load_and_map(FIXTURE_PATH)
    feature_df = df[CANONICAL_FEATURE_COLUMNS]
    assert not feature_df.isna().any().any()
    assert not (feature_df.abs() == float("inf")).any().any()


def test_missing_required_column_raises():
    import pandas as pd
    from ids_ml.data import map_to_canonical

    bad_df = pd.DataFrame({"Label": ["BENIGN"]})
    try:
        map_to_canonical(bad_df)
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected ValueError for missing columns")
