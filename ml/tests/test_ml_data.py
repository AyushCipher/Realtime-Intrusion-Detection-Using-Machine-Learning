"""Tests for CICIDS CSV loading/mapping against the synthetic fixture.

tests/fixtures/synthetic_cicids_sample.csv is synthetic (see
tests/generate_ml_fixtures.py) but shaped exactly like a real CICIDS2017
CSV, including its inconsistent leading-whitespace column-naming quirk, so
these tests exercise the real parsing/mapping logic.
"""

from pathlib import Path

import pandas as pd

from ids_ml.data import attack_category, load_and_map, load_and_map_2018, load_cicids_csv
from ids_ml.features import CANONICAL_FEATURE_COLUMNS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"

# A handful of CSE-CIC-IDS2018-shaped rows (CICFlowMeter-V3's abbreviated
# column names, e.g. "Tot Fwd Pkts" not "Total Fwd Packets"), including the
# two verified real-world data-quality issues load_cicids2018_csv exists to
# handle: a leaked-header row (every column literally holds its own header
# name) and a truncated row (Label == "Be").
_CICIDS2018_HEADER = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "SYN Flag Cnt", "ACK Flag Cnt", "FIN Flag Cnt", "RST Flag Cnt",
    "PSH Flag Cnt", "URG Flag Cnt", "ECE Flag Cnt", "CWE Flag Count", "Label",
]


def _cicids2018_row(label: str, dur: float = 2_000_000.0, timestamp: str = "01/03/2018 08:00:00") -> list:
    # Numeric filler values are arbitrary but consistent (dur used for
    # Flow Duration so the microsecond->second conversion is checkable).
    # Default timestamp "01/03/2018" is deliberately day/month-ambiguous
    # (both <=12) -- day-first is March 1st, month-first is January 3rd --
    # to actually exercise load_and_map_2018's dayfirst=True, matching the
    # real Thursday-01-03-2018 file this is modeled on.
    row = {c: 1.0 for c in _CICIDS2018_HEADER}
    row["Flow Duration"] = dur
    row["Timestamp"] = timestamp
    row["Label"] = label
    return [row[c] for c in _CICIDS2018_HEADER]


def _write_cicids2018_fixture(tmp_path: Path) -> Path:
    rows = [
        _cicids2018_row("Benign"),
        _cicids2018_row("Bot"),
        _cicids2018_row("FTP-BruteForce"),
        _CICIDS2018_HEADER,  # leaked header row: every column == its own header name
        _cicids2018_row("Be"),  # truncated/corrupted label
    ]
    path = tmp_path / "cicids2018_sample.csv"
    pd.DataFrame(rows, columns=_CICIDS2018_HEADER).to_csv(path, index=False)
    return path


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


# --- CIC-IDS2018 (CICFlowMeter-V3 / AWS release) loader ---------------


def test_load_and_map_2018_drops_corrupted_rows(tmp_path):
    path = _write_cicids2018_fixture(tmp_path)
    df = load_and_map_2018(path)

    # 5 rows written, 2 corrupted (leaked header + truncated "Be") -> 3 kept
    assert len(df) == 3
    assert set(df["label"]) == {"BENIGN", "Bot", "FTP-BruteForce"}


def test_load_and_map_2018_normalizes_benign_case_and_maps_categories(tmp_path):
    path = _write_cicids2018_fixture(tmp_path)
    df = load_and_map_2018(path)

    by_label = df.set_index("label")
    assert by_label.loc["BENIGN", "attack_category"] == "BENIGN"
    assert by_label.loc["BENIGN", "is_attack"] == False  # noqa: E712
    assert by_label.loc["Bot", "attack_category"] == "Botnet"
    assert by_label.loc["FTP-BruteForce", "attack_category"] == "Brute Force"
    assert by_label.loc["Bot", "is_attack"] == True  # noqa: E712


def test_load_and_map_2018_produces_canonical_columns(tmp_path):
    path = _write_cicids2018_fixture(tmp_path)
    df = load_and_map_2018(path)
    for col in CANONICAL_FEATURE_COLUMNS:
        assert col in df.columns
    assert not df[CANONICAL_FEATURE_COLUMNS].isna().any().any()


def test_load_and_map_2018_converts_microseconds_to_seconds(tmp_path):
    path = _write_cicids2018_fixture(tmp_path)
    df = load_and_map_2018(path)
    # _cicids2018_row uses dur=2_000_000 microseconds = 2.0 seconds for every row
    assert (df["flow_duration"] == 2.0).all()


def test_load_and_map_2018_parses_ambiguous_dates_day_first(tmp_path):
    # "01/03/2018" is ambiguous (day and month both <=12): month-first
    # would silently misread it as January 3rd instead of March 1st.
    # Regression test for a real bug caught while building this loader --
    # see map_to_canonical's `dayfirst` docstring.
    path = _write_cicids2018_fixture(tmp_path)
    df = load_and_map_2018(path)
    assert (df["timestamp"].dt.month == 3).all()
    assert (df["timestamp"].dt.day == 1).all()


def test_load_and_map_2018_has_no_source_ip_column(tmp_path):
    # CSE-CIC-IDS2018's AWS release has no Source IP column at all --
    # map_to_canonical already treats it as optional, so it should simply
    # be absent rather than raising or being filled with a placeholder.
    path = _write_cicids2018_fixture(tmp_path)
    df = load_and_map_2018(path)
    assert "src_ip" not in df.columns
