"""Evaluate Stage 1 pre-filters with conformal recall control on real data.

Filters compared (each gets the SAME recall-control calibration):
  iforest_all     Isolation Forest fitted on all training flows (the original design)
  xgb_only        the supervised half of the hybrid
  hybrid          stage1_hybrid.HybridPreFilter
Protocols:
  random          random 70/15/15 split (known families; the guarantee's home turf)
  chronological   70/15/15 by time (CSE-CIC-IDS2018 puts different families on different days)
  lofo            for each family F: train on everything except F, calibrate the recall control
                  on KNOWN-family validation attacks, test on F (a zero-day family: no guarantee)
For each, at miss budgets beta: the realised attack miss rate and the benign flag rate
(1 - benign flag rate = the share of benign traffic the cascade does not have to
classify). The guarantee covers known families under exchangeability only.

Writes results/paper/<name>/stage1_hybrid_eval.csv and a manifest.

Usage:
    PYTHONPATH=ml/src python ml/scripts/stage1_hybrid_eval.py --data real.csv --name cicids2018_dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.conformal_stage1 import calibrate_stage1_threshold  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402
from ids_ml.split import random_split, time_based_split  # noqa: E402
from ids_ml.stage1_hybrid import HybridConfig, HybridPreFilter  # noqa: E402
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config  # noqa: E402

BUDGETS = [0.01, 0.05, 0.10]


class XgbOnly:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit(self, X, y):
        self.m = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.2, n_jobs=4, verbosity=0, random_state=self.seed).fit(X, y)
        return self

    def anomaly_score(self, X):
        return self.m.predict_proba(X)[:, 1]


class IforestAll:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit(self, X, y):
        self.m = AnomalyPreFilter(Stage1Config(contamination=0.35, random_state=self.seed)).fit(X)
        return self

    def anomaly_score(self, X):
        return self.m.anomaly_score(X)


def make(name: str, seed: int):
    if name == "iforest_all":
        return IforestAll(seed)
    if name == "xgb_only":
        return XgbOnly(seed)
    return HybridPreFilter(HybridConfig(random_state=seed))


def run(protocol: str, tr, cal, te, filt: str, seed: int, heldout: str = "-") -> list:
    Xtr, ytr = tr[FEATS].to_numpy(), (tr["attack_category"] != "BENIGN").astype(int).to_numpy()
    model = make(filt, seed).fit(Xtr, ytr)
    cal_att = cal[cal["attack_category"] != "BENIGN"]
    te_att, te_ben = te[te["attack_category"] != "BENIGN"], te[te["attack_category"] == "BENIGN"]
    s_cal = model.anomaly_score(cal_att[FEATS].to_numpy())
    s_att, s_ben = model.anomaly_score(te_att[FEATS].to_numpy()), model.anomaly_score(te_ben[FEATS].to_numpy())
    y = np.r_[np.ones(len(s_att)), np.zeros(len(s_ben))]
    auroc = float(roc_auc_score(y, np.r_[s_att, s_ben])) if len(s_att) and len(s_ben) else float("nan")
    rows = []
    for b in BUDGETS:
        t = calibrate_stage1_threshold(s_cal, b).threshold
        rows.append({"protocol": protocol, "heldout": heldout, "seed": seed, "filter": filt, "budget": b, "auroc": auroc,
                     "attack_miss_rate": float((s_att < t).mean()), "benign_flag_rate": float((s_ben >= t).mean()),
                     "n_attacks": len(s_att), "n_cal_attacks": len(cal_att)})
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cap-per-label", type=int, default=0, help="random per-raw-label cap to bound runtime on very large data (0 = off)")
    args = ap.parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    if args.cap_per_label:
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), args.cap_per_label), random_state=0)).sort_values("timestamp")
    fams = sorted(set(df["attack_category"]) - {"BENIGN"})
    rows = []
    for seed in range(args.seeds):
        for filt in ("iforest_all", "xgb_only", "hybrid"):
            a, b, c = random_split(df, train_frac=0.7, val_frac=0.15, random_state=seed)
            rows += run("random", a, b, c, filt, seed)
            tr, va, te = time_based_split(df, train_frac=0.7, val_frac=0.15)
            rows += run("chronological", tr, va, te, filt, seed)
            for fam in fams:
                rest = df[df["attack_category"] != fam]
                rest_tr, rest_va, _ = random_split(rest, train_frac=0.7, val_frac=0.25, random_state=seed)
                held = df[df["attack_category"] == fam]
                ben_test = rest_va[rest_va["attack_category"] == "BENIGN"]
                te = pd.concat([held, ben_test])
                cal = rest_va[rest_va["attack_category"] != "BENIGN"]
                rows += run("lofo", rest_tr, cal, te, filt, seed, heldout=fam)
        print(f"seed {seed} done", flush=True)
    res = pd.DataFrame(rows)
    out = root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "stage1_hybrid_eval_raw.csv", index=False)
    summ = (res.groupby(["protocol", "filter", "budget"])[["auroc", "attack_miss_rate", "benign_flag_rate"]].mean().round(3).reset_index())
    summ.to_csv(out / "stage1_hybrid_eval.csv", index=False)
    write_manifest(out / "manifest_stage1_hybrid_eval.json", inputs=[args.data], outputs=[out / "stage1_hybrid_eval.csv"], seeds=range(args.seeds), repo_dir=root)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
