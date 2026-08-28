from pathlib import Path

import pandas as pd
import pytest

from ids_ml.data import load_and_map
from ids_ml.split import leave_one_family_out, random_split, rotate_holdout, time_based_split, time_window_split

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


def test_time_based_split_is_chronological_with_no_overlap():
    df = load_and_map(FIXTURE_PATH)
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)

    assert len(train) + len(val) + len(test) == len(df)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_time_based_split_fractions_are_approximately_respected():
    df = load_and_map(FIXTURE_PATH)
    train, val, test = time_based_split(df, train_frac=0.6, val_frac=0.2)

    n = len(df)
    assert abs(len(train) / n - 0.6) < 0.02
    assert abs(len(val) / n - 0.2) < 0.02


def test_random_split_does_not_guarantee_chronological_order():
    df = load_and_map(FIXTURE_PATH)
    train, val, test = random_split(df, train_frac=0.7, val_frac=0.15, random_state=0)

    assert len(train) + len(val) + len(test) == len(df)
    # A random split over two days of data should, essentially always, mix
    # both days into the training set -- unlike the time-based split.
    assert train["timestamp"].dt.date.nunique() > 1


def test_time_window_split_separates_by_cutoff():
    df = load_and_map(FIXTURE_PATH)
    cutoff = pd.Timestamp("2017-07-04")
    before, after = time_window_split(df, cutoff)

    assert len(before) + len(after) == len(df)
    assert (before["timestamp"] < cutoff).all()
    assert (after["timestamp"] >= cutoff).all()


def test_leave_one_family_out_excludes_family_from_train_entirely():
    df = load_and_map(FIXTURE_PATH)
    train, test = leave_one_family_out(df, "Botnet")

    assert "Botnet" not in set(train["attack_category"])
    assert (test["attack_category"] == "Botnet").sum() == (df["attack_category"] == "Botnet").sum()


def test_leave_one_family_out_test_set_covers_known_tail_plus_all_unknown():
    df = load_and_map(FIXTURE_PATH)
    train, test = leave_one_family_out(df, "PortScan", train_frac=0.7)

    known = df[df["attack_category"] != "PortScan"]
    unknown = df[df["attack_category"] == "PortScan"]
    assert len(train) + len(test) == len(df)
    assert len(train) == int(len(known) * 0.7)
    assert (test["attack_category"] == "PortScan").sum() == len(unknown)


def test_leave_one_family_out_refuses_benign():
    df = load_and_map(FIXTURE_PATH)
    with pytest.raises(ValueError):
        leave_one_family_out(df, "BENIGN")


def test_leave_one_family_out_rejects_unknown_family_name():
    df = load_and_map(FIXTURE_PATH)
    with pytest.raises(ValueError):
        leave_one_family_out(df, "NotARealFamily")


def test_rotate_holdout_covers_every_non_benign_family_exactly_once():
    df = load_and_map(FIXTURE_PATH)
    expected_families = sorted(f for f in df["attack_category"].unique() if f != "BENIGN")

    seen = []
    for family, train, test in rotate_holdout(df):
        seen.append(family)
        assert family not in set(train["attack_category"])
        assert (test["attack_category"] == family).sum() == (df["attack_category"] == family).sum()

    assert sorted(seen) == expected_families
