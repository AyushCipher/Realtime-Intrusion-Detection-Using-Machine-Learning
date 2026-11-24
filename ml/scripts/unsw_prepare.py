"""Convert CIC-UNSW-NB15 (CICFlowMeter re-extraction of UNSW-NB15) to canonical form.

CIC-UNSW-NB15 (Canadian Institute for Cybersecurity, unb.ca/cic/datasets/cic-unsw-nb15.html;
copy used here: huggingface.co/datasets/bencorn/CIC-UNSW-NB15) runs CICFlowMeter over the
raw UNSW-NB15 captures, which were recorded at UNSW Canberra with the IXIA PerfectStorm
tool: a different network, traffic generator and attack set from the CIC datasets, but the
same flow-feature extractor, so our 35 canonical features exist unchanged.

`Data.csv` holds the features and `Lable.csv` (sic) the numeric labels, decoded with the
dataset's Readme: 0 Benign, 1 Analysis, 2 Backdoor, 3 DoS, 4 Exploits, 5 Fuzzers,
6 Generic, 7 Reconnaissance, 8 Shellcode, 9 Worms. The file has no timestamps, so only
random-split and leave-one-family-out protocols apply; a `timestamp` column holding the
row order is added for code that expects one and must NOT be read as time.

Usage:
    PYTHONPATH=ml/src python ml/scripts/unsw_prepare.py --input-dir C:/nids_data/cic_unsw \
        --output C:/nids_data/unsw_canonical_dedup.csv --dedup
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.data import load_and_map_cicflowmeter  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402

NAMES = {0: "BENIGN", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits", 5: "Fuzzers",
         6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()
    d = Path(args.input_dir)
    X = pd.read_csv(d / "Data.csv", low_memory=False)
    y = pd.read_csv(d / "Lable.csv")["Label"].astype(int)
    assert len(X) == len(y), "Data.csv and Lable.csv row counts differ"
    X["Label"] = y.map(NAMES)
    # This export's CICFlowMeter version names a few columns differently from CICIDS2017.
    X = X.rename(columns={
        "Total Fwd Packet": "Total Fwd Packets", "Total Bwd packets": "Total Backward Packets",
        "Total Length of Fwd Packet": "Total Length of Fwd Packets", "Total Length of Bwd Packet": "Total Length of Bwd Packets",
        "CWR Flag Count": "CWE Flag Count",
    })
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "labelled.csv"
        X.to_csv(p, index=False)
        df = load_and_map_cicflowmeter([p], label_column="Label")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS).reset_index(drop=True)
    n0 = len(df)
    if args.dedup:
        key = pd.util.hash_pandas_object(df[FEATS].round(6), index=False)
        nlab = df.assign(_k=key).groupby("_k")["attack_category"].nunique()
        conflict = set(nlab[nlab > 1].index)
        df = df.assign(_k=key)
        df = df[~df["_k"].isin(conflict)].drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
        print(f"dedup: {n0} -> {len(df)} rows ({len(conflict)} conflicting feature groups removed)")
    df["timestamp"] = pd.Timestamp("2015-01-22") + pd.to_timedelta(np.arange(len(df)), unit="s")  # row order, NOT time
    df["is_attack"] = df["attack_category"] != "BENIGN"
    df.to_csv(args.output, index=False)
    print(df["attack_category"].value_counts().to_string())
    print("wrote", args.output, len(df))


if __name__ == "__main__":
    main()
