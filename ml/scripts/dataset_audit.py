"""Audit a canonical-form dataset for the problems that inflate or distort results.

Reports, per attack family and overall:
  * exact duplicate feature rows (identical 35-feature vectors),
  * conflicting duplicates (identical features, different label),
  * duplicates shared between the chronological train and test splits
    (the leakage a reviewer will ask about),
  * Stage 1 anomaly-score ties, which make conformal thresholds coarse
    (many identical scores let a quantile land on only a few values).
Writes results/paper/<name>/audit.csv, audit_summary.json and manifest_audit.json.

Usage:
    PYTHONPATH=ml/src python ml/scripts/dataset_audit.py --data real.csv --name cicids2018
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402
from ids_ml.split import time_based_split  # noqa: E402
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--name", default="dataset")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--write-dedup", default=None, help="Write a cleaned CSV: drop every row in a conflicting-label group, keep the first of exact duplicates")
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    key = pd.util.hash_pandas_object(df[FEATS].round(6), index=False)
    df = df.assign(_key=key)

    labels_per_key = df.groupby("_key")["attack_category"].nunique()
    conflicting_keys = set(labels_per_key[labels_per_key > 1].index)

    rows = []
    for fam, g in df.groupby("attack_category"):
        dup = g.duplicated("_key", keep="first")
        rows.append({
            "family": fam, "rows": len(g), "unique_rows": int(g["_key"].nunique()),
            "exact_duplicate_rows": int(dup.sum()), "duplicate_share": round(float(dup.mean()), 4),
            "rows_in_conflicting_groups": int(g["_key"].isin(conflicting_keys).sum()),
        })
    audit = pd.DataFrame(rows).sort_values("rows", ascending=False)
    audit.to_csv(out / "audit.csv", index=False)

    train, val, test = time_based_split(df.drop(columns=["_key"]), train_frac=0.7, val_frac=0.15)
    tk = set(pd.util.hash_pandas_object(train[FEATS].round(6), index=False))
    test_keys = pd.util.hash_pandas_object(test[FEATS].round(6), index=False)
    test_in_train = test_keys.isin(tk)
    is_attack = test["attack_category"].to_numpy() != "BENIGN"

    if args.write_dedup:
        clean = df[~df["_key"].isin(conflicting_keys)].drop_duplicates("_key", keep="first").drop(columns=["_key"])
        clean.to_csv(args.write_dedup, index=False)
        summary_extra = {"dedup_rows": len(clean), "dedup_file": args.write_dedup}
    else:
        summary_extra = {}

    stage1 = AnomalyPreFilter(Stage1Config(contamination=0.35)).fit(train[FEATS].to_numpy())
    att = stage1.anomaly_score(df.loc[df["attack_category"] != "BENIGN", FEATS].to_numpy())
    summary = {
        "rows": len(df), "unique_feature_rows": int(df["_key"].nunique()),
        "duplicate_share_overall": round(1 - df["_key"].nunique() / len(df), 4),
        "conflicting_label_groups": len(conflicting_keys),
        "rows_in_conflicting_groups": int(df["_key"].isin(conflicting_keys).sum()),
        "test_rows": len(test),
        "test_rows_also_in_train": int(test_in_train.sum()),
        "test_attack_rows_also_in_train": int((test_in_train.to_numpy() & is_attack).sum()),
        "test_attack_rows": int(is_attack.sum()),
        "stage1_attack_scores": len(att), "stage1_attack_unique_scores": int(len(np.unique(att))),
        "stage1_attack_tie_share": round(1 - len(np.unique(att)) / len(att), 4),
        **summary_extra,
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    write_manifest(out / "manifest_audit.json", inputs=[args.data], outputs=[out / "audit.csv", out / "audit_summary.json"],
                   extra={"name": args.name}, repo_dir=root)
    print(audit.to_string(index=False)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
