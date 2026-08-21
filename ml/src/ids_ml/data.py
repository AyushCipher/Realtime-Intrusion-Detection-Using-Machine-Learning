"""Loading and normalizing CICIDS2017 / CIC-IDS2018 flow CSVs.

These datasets are notorious for inconsistent column whitespace across the
per-day CSV files (e.g. `' Flow Duration'` in one file, `'Flow Duration'` in
another) -- see https://www.unb.ca/cic/datasets/ids-2017.html. Every column
name is stripped on load before anything else touches it.

This module maps the raw CICFlowMeter columns onto
`ids_ml.features.CANONICAL_FEATURE_COLUMNS` so the same feature space is
used for training (from these CSVs) and inference (from live flow events).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Union

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


def map_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a DataFrame with canonical feature columns, label, category,
    and timestamp (if present), from a raw (whitespace-stripped) CICIDS df.
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
        out["timestamp"] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce", dayfirst=False)
    if SOURCE_IP_COLUMN in df.columns:
        out["src_ip"] = df[SOURCE_IP_COLUMN]

    return out


def load_and_map(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> pd.DataFrame:
    return map_to_canonical(load_cicids_csv(paths))
