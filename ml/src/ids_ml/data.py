"""Loading and normalizing CICIDS2017 / CIC-IDS2018 flow CSVs.

These datasets are notorious for inconsistent column whitespace across the
per-day CSV files (e.g. `' Flow Duration'` in one file, `'Flow Duration'` in
another) -- see https://www.unb.ca/cic/datasets/ids-2017.html. Every column
name is stripped on load before anything else touches it.

This module maps the raw CICFlowMeter columns onto
`ids_ml.features.CANONICAL_FEATURE_COLUMNS` so the same feature space is
used for training (from these CSVs) and inference (from live flow events).

Two loader paths, for two differently-shaped releases sharing the same
target feature space:

- `load_and_map` / `load_cicids_csv` -- CICIDS2017 (CICFlowMeter-v2 column
  names, e.g. "Total Fwd Packets").
- `load_and_map_2018` / `load_cicids2018_csv` -- CSE-CIC-IDS2018's AWS
  Open Data release (https://registry.opendata.aws/cse-cic-ids2018/),
  which used CICFlowMeter-V3's renamed/abbreviated columns (e.g. "Tot Fwd
  Pkts") and has its own verified data-quality quirks -- see the comment
  block above `CICIDS2018_TO_2017_RENAME`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Union

import pandas as pd

from .features import CANONICAL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Raw CICFlowMeter column name (after whitespace-stripping) -> canonical name.
# "CWE Flag Count" is not a typo on our side: CICFlowMeter itself mislabels
# the CWR flag count column this way in both CICIDS2017 and CIC-IDS2018.
CICIDS_COLUMN_MAP = {
    "Flow Duration": "flow_duration_us",  # microseconds; converted to seconds below
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Fwd Packet Length Min": "fwd_packet_length_min",
    "Fwd Packet Length Max": "fwd_packet_length_max",
    "Fwd Packet Length Mean": "fwd_packet_length_mean",
    "Fwd Packet Length Std": "fwd_packet_length_std",
    "Bwd Packet Length Min": "bwd_packet_length_min",
    "Bwd Packet Length Max": "bwd_packet_length_max",
    "Bwd Packet Length Mean": "bwd_packet_length_mean",
    "Bwd Packet Length Std": "bwd_packet_length_std",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    "Flow IAT Mean": "flow_iat_mean_us",
    "Flow IAT Std": "flow_iat_std_us",
    "Flow IAT Min": "flow_iat_min_us",
    "Flow IAT Max": "flow_iat_max_us",
    "Fwd IAT Mean": "fwd_iat_mean_us",
    "Fwd IAT Std": "fwd_iat_std_us",
    "Fwd IAT Min": "fwd_iat_min_us",
    "Fwd IAT Max": "fwd_iat_max_us",
    "Bwd IAT Mean": "bwd_iat_mean_us",
    "Bwd IAT Std": "bwd_iat_std_us",
    "Bwd IAT Min": "bwd_iat_min_us",
    "Bwd IAT Max": "bwd_iat_max_us",
    "SYN Flag Count": "syn_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "FIN Flag Count": "fin_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "URG Flag Count": "urg_flag_count",
    "ECE Flag Count": "ece_flag_count",
    "CWE Flag Count": "cwr_flag_count",  # CICFlowMeter's naming quirk, see above
}

# Microsecond-valued CICIDS columns; ids_ingestion.features reports these in
# seconds, so they're converted at load time to keep train/serve consistent.
_MICROSECOND_COLUMNS = [c for c in CICIDS_COLUMN_MAP.values() if c.endswith("_us")]

TIMESTAMP_COLUMN = "Timestamp"
LABEL_COLUMN = "Label"
SOURCE_IP_COLUMN = "Source IP"

# CICIDS2017/2018 raw labels grouped into families for per-category
# reporting. Neither dataset uses NSL-KDD's R2L/U2R terms; Infiltration and
# Web Attack are this dataset's closest analogues -- rare, subtle,
# app-layer-driven classes -- and are called out as such in evaluation.py
# and the README rather than mislabeled as literal R2L/U2R.
ATTACK_CATEGORY_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "DDoS": "DoS/DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Infiltration": "Infiltration",
    "Bot": "Botnet",
    "Heartbleed": "Heartbleed",
}


def attack_category(raw_label: str) -> str:
    """Map a raw CICIDS label to its attack family, defaulting to 'Other'."""
    return ATTACK_CATEGORY_MAP.get(raw_label.strip(), "Other" if raw_label.strip() != "BENIGN" else "BENIGN")


# --- CIC-IDS2018 (AWS Open Data "Processed Traffic Data for ML
# Algorithms" release, https://registry.opendata.aws/cse-cic-ids2018/)
# support --------------------------------------------------------------
#
# This release was generated with CICFlowMeter-V3, which renamed and
# abbreviated every column relative to CICIDS2017/CICFlowMeter-v2's names
# above (e.g. "Tot Fwd Pkts" vs "Total Fwd Packets") and dropped the Flow
# ID/Source IP/Destination IP/Source Port identity columns entirely.
# Verified directly against the 8 downloaded per-day CSVs, not assumed
# from documentation -- "CWE Flag Count"'s typo (see above) carries over
# unchanged, but "Flow Duration"/"Flow IAT ..."/"Fwd IAT ..."/"Bwd IAT
# ..." already match CICIDS_COLUMN_MAP's names verbatim, so only the
# columns actually renamed appear below.
CICIDS2018_TO_2017_RENAME: Dict[str, str] = {
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "SYN Flag Cnt": "SYN Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "FIN Flag Cnt": "FIN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
}

# Raw labels observed across the 8 per-day CSVs this project has actually
# downloaded (Wed-14-02, Thu-15-02, Fri-16-02, Wed-21-02, Thu-22-02,
# Fri-23-02, Thu-01-03, Fri-02-03), mapped onto the same family vocabulary
# ATTACK_CATEGORY_MAP already uses so both datasets feed identical
# downstream categories. "Infilteration" is the dataset's own misspelling,
# not a typo introduced here -- see the AWS-hosted CSVs directly. Other
# CSE-CIC-IDS2018 days (not downloaded here) may contain additional label
# strings (e.g. PortScan, Heartbleed) not covered by this map; those rows
# would fall through to "Other" via attack_category() rather than raise.
CICIDS2018_ATTACK_CATEGORY_MAP: Dict[str, str] = {
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",
    "DoS attacks-GoldenEye": "DoS/DDoS",
    "DoS attacks-Slowloris": "DoS/DDoS",
    "DoS attacks-Hulk": "DoS/DDoS",
    "DoS attacks-SlowHTTPTest": "DoS/DDoS",
    "DDOS attack-HOIC": "DoS/DDoS",
    "DDOS attack-LOIC-UDP": "DoS/DDoS",
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
    "Infilteration": "Infiltration",
    "Bot": "Botnet",
}
ATTACK_CATEGORY_MAP.update(CICIDS2018_ATTACK_CATEGORY_MAP)

# The complete set of labels this project has verified appear in the 8
# downloaded CSVs (Benign + every key above). Rows whose Label doesn't
# match any of these are dropped by load_cicids2018_csv, not passed
# through -- verified data-quality issue: some files contain stray rows
# where a CSV header line got concatenated in as data (Label == "Label"
# literally, seen in Thursday-01-03-2018 and Friday-16-02-2018) or a
# truncated row (Label == "Be", seen in Friday-23-02-2018). Every column
# in a leaked-header row holds header-name strings rather than real
# values, so keeping these rows would silently corrupt every numeric
# feature in them, not just the label -- dropping outright is safer than
# trying to salvage partial data from a corrupted row.
_CICIDS2018_KNOWN_LABELS = {"Benign"} | set(CICIDS2018_ATTACK_CATEGORY_MAP)


def load_cicids2018_csv(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> pd.DataFrame:
    """Load one or more CSE-CIC-IDS2018 "Processed Traffic Data for ML
    Algorithms" CSVs, renamed onto CICIDS2017's column names so
    `map_to_canonical` handles both datasets identically. See the module
    comment above `CICIDS2018_TO_2017_RENAME` for what's actually
    different between the two releases and how each difference is
    handled.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for path in paths:
        logger.info("Loading %s", path)
        df = pd.read_csv(path, low_memory=False)
        df = _strip_columns(df)

        n_before = len(df)
        df = df[df[LABEL_COLUMN].isin(_CICIDS2018_KNOWN_LABELS)].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning("%s: dropped %d row(s) with an unrecognized/corrupted Label", path, n_dropped)

        df[LABEL_COLUMN] = df[LABEL_COLUMN].replace({"Benign": "BENIGN"})
        df = df.rename(columns=CICIDS2018_TO_2017_RENAME)
        frames.append(df)
    if not frames:
        raise ValueError("no CSV paths given")
    return pd.concat(frames, ignore_index=True)


def load_and_map_2018(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> pd.DataFrame:
    # dayfirst=True: verified against the actual downloaded CSVs by
    # cross-referencing Timestamp values against each file's own filename
    # date (e.g. Thursday-01-03-2018's rows read "01/03/2018", which is
    # only consistent with day-first -- month-first would read that as
    # January 3rd, contradicting the filename's March 1st).
    return map_to_canonical(load_cicids2018_csv(paths), dayfirst=True)


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def load_cicids_csv(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> pd.DataFrame:
    """Load one or more CICIDS-style CSVs, concatenated, with columns stripped."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for path in paths:
        logger.info("Loading %s", path)
        df = pd.read_csv(path, low_memory=False)
        frames.append(_strip_columns(df))
    if not frames:
        raise ValueError("no CSV paths given")
    return pd.concat(frames, ignore_index=True)


def map_to_canonical(df: pd.DataFrame, dayfirst: bool = False) -> pd.DataFrame:
    """Produce a DataFrame with canonical feature columns, label, category,
    and timestamp (if present), from a raw (whitespace-stripped) CICIDS df.

    `dayfirst` controls how the Timestamp column's ambiguous D/M vs M/D
    dates (e.g. "01/03/2018", where day and month are both <=12) are
    parsed. Get this wrong and dates silently swap month/day instead of
    raising -- corrupting every downstream time-based split without any
    error. CSE-CIC-IDS2018's AWS release is verified day-first (see
    `load_and_map_2018`, which passes `dayfirst=True`); this default
    (`False`) is CICIDS2017's, unchanged from before this parameter
    existed.
    """
    missing = [c for c in CICIDS_COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"input CSV is missing expected CICIDS columns: {missing}")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"input CSV is missing the '{LABEL_COLUMN}' column")

    out = df.rename(columns=CICIDS_COLUMN_MAP)[list(CICIDS_COLUMN_MAP.values())].copy()

    for us_col in _MICROSECOND_COLUMNS:
        s_col = us_col[: -len("_us")]
        out[s_col] = pd.to_numeric(out[us_col], errors="coerce") / 1_000_000.0
        out.drop(columns=[us_col], inplace=True)

    for col in CANONICAL_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[col] = out[col].replace([float("inf"), float("-inf")], 0.0)

    out = out[CANONICAL_FEATURE_COLUMNS]
    out["label"] = df[LABEL_COLUMN].astype(str).str.strip()
    out["attack_category"] = out["label"].map(attack_category)
    out["is_attack"] = out["label"] != "BENIGN"

    if TIMESTAMP_COLUMN in df.columns:
        out["timestamp"] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce", dayfirst=dayfirst)
    if SOURCE_IP_COLUMN in df.columns:
        out["src_ip"] = df[SOURCE_IP_COLUMN]

    return out


def load_and_map(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> pd.DataFrame:
    return map_to_canonical(load_cicids_csv(paths))
