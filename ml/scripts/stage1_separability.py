"""How well can a cheap Stage 1 separate attacks from benign on real data?

Compares three cheap pre-filters on the same chronological and random splits:
  iforest_all        Isolation Forest fitted on all training flows (as the cascade does)
  iforest_benign     Isolation Forest fitted on benign training flows only
  xgb_binary_small   40-tree depth-3 supervised benign-vs-attack XGBoost
and reports attack-vs-benign AUROC overall and per family. The chronological
split of CSE-CIC-IDS2018 puts different attack families on different days, so
its test attacks are families unseen in training; the random split tests known
families. Both are reported because they answer different questions.

Writes results/paper/<name>/stage1_separability.json and a manifest.

Usage:
    PYTHONPATH=ml/src python ml/scripts/stage1_separability.py --data real.csv --name cicids2018_dedup
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402
from ids_ml.split import random_split, time_based_split  # noqa: E402
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config  # noqa: E402


def scores(kind: str, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    Xte = te[FEATS].to_numpy()
    if kind == "iforest_all":
        return AnomalyPreFilter(Stage1Config(contamination=0.35)).fit(tr[FEATS].to_numpy()).anomaly_score(Xte)
    if kind == "iforest_benign":
        ben = tr[tr["attack_category"] == "BENIGN"]
        return AnomalyPreFilter(Stage1Config(contamination=0.05)).fit(ben[FEATS].to_numpy()).anomaly_score(Xte)
    y = (tr["attack_category"] != "BENIGN").astype(int)
    m = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.2, n_jobs=4, verbosity=0, random_state=0)
    return m.fit(tr[FEATS].to_numpy(), y).predict_proba(Xte)[:, 1]


def evaluate(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    y = (te["attack_category"] != "BENIGN").to_numpy()
    out = {"train_benign_share": round(float((tr["attack_category"] == "BENIGN").mean()), 3),
           "test_attack_families": sorted(set(te["attack_category"]) - {"BENIGN"})}
    for kind in ("iforest_all", "iforest_benign", "xgb_binary_small"):
        sc = scores(kind, tr, te)
        per = {}
        for fam in out["test_attack_families"]:
            mk = te["attack_category"].isin(["BENIGN", fam]).to_numpy()
            per[fam] = round(float(roc_auc_score((te["attack_category"][mk] == fam).astype(int), sc[mk])), 3)
        out[kind] = {"auroc_overall": round(float(roc_auc_score(y, sc)), 3), "auroc_per_family": per}
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="dataset")
    args = ap.parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    tr, _va, te = time_based_split(df, train_frac=0.7, val_frac=0.15)
    res = {"chronological": evaluate(tr, te)}
    a, _b, c = random_split(df, train_frac=0.7, val_frac=0.15)
    res["random"] = evaluate(a, c)
    out = root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage1_separability.json").write_text(json.dumps(res, indent=2))
    write_manifest(out / "manifest_stage1_separability.json", inputs=[args.data], outputs=[out / "stage1_separability.json"], repo_dir=root)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
