"""CLOSR/CLAD-style zero-day protocol on LYCOS-IDS2017.

Follows the experimental procedure of Wilkie et al., "A Novel Contrastive Loss for
Zero-Day Network Intrusion Detection" (arXiv:2601.09902): LYCOS-IDS2017, a 50%/50%
train/test split with each class split evenly, and the Web Attack (SQL injection)
and Heartbleed classes present ONLY in the test set to simulate zero-days.
Detectors are scored by how well their score separates benign flows from each
attack class (AUROC, and FPR at 95% TPR), reported for known classes (mean) and
zero-day classes.

Detectors compared (all are cheap Stage 1 style per-flow scorers; see
stage1_selection.py for their definitions):
  iforest_benign, xgb_d3, xgb_d6, knn_benign, hyb_sup_if, hyb_sup_knn (the shipped
  default of stage1_hybrid.py), hyb3, hyb3_d6.

Differences from the paper that bound comparability: our 35 canonical features
(they use 72 LycoSTand features), our deduplication, a capped training sample
(--cap per class) to bound runtime, no per-model hyperparameter search (they run
200 random-search iterations per model), and a 12-sample Heartbleed / 12-sample
SQL-injection test set after deduplication, so zero-day numbers are very noisy.
Treat results as indicative of the same protocol, not as a reproduction.

Writes results/paper/<name>/closr_protocol.csv (+ _raw) and a manifest.

Usage:
    PYTHONPATH=ml/src python ml/scripts/run_closr_protocol.py --data C:/nids_data/lycos2017_canonical_dedup.csv --name lycos2017
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402

spec = importlib.util.spec_from_file_location("stage1_selection", HERE / "stage1_selection.py")
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)  # reuse score_all()

ZERO_DAY = ["webattack_sql_injection", "heartbleed"]
CANDIDATES = ["iforest_benign", "xgb_d3", "xgb_d6", "knn_benign", "hyb_sup_if", "hyb_sup_knn", "hyb3", "hyb3_d6", "hyb_s6k_w50", "hyb_s6k_w70", "hyb_s6k_w85", "max_s6k", "efc_benign", "max3_s6k_efc"]


def fpr_at_tpr(y: np.ndarray, s: np.ndarray, tpr: float = 0.95) -> float:
    fpr, t, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(t, tpr, side="left")])


def main() -> None:
    root = HERE.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="lycos2017")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cap", type=int, default=30000, help="max training flows per class")
    ap.add_argument("--test-benign", type=int, default=60000)
    args = ap.parse_args()

    df = pd.read_csv(args.data, usecols=FEATS + ["label", "attack_category"])
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        tr_parts, te_parts = [], []
        for lab, g in df.groupby("label"):
            g = g.iloc[rng.permutation(len(g))]
            if lab in ZERO_DAY:
                te_parts.append(g)
            else:
                h = len(g) // 2
                tr_parts.append(g.iloc[:h][: args.cap])
                te_parts.append(g.iloc[h:])
        train, test = pd.concat(tr_parts), pd.concat(te_parts)
        ben_te = test[test["label"] == "benign"]
        test = pd.concat([test[test["label"] != "benign"], ben_te.iloc[: args.test_benign]])
        Xtr, ytr = train[FEATS].to_numpy(), (train["label"] != "benign").astype(int).to_numpy()
        Xte, lab_te = test[FEATS].to_numpy(), test["label"].to_numpy()
        scores = sel.score_all(Xtr, ytr, Xte, seed)
        is_ben = lab_te == "benign"
        for cand in CANDIDATES:
            s = scores[cand]
            for lab in sorted(set(lab_te) - {"benign"}):
                m = lab_te == lab
                y = np.r_[np.ones(m.sum()), np.zeros(is_ben.sum())]
                sc = np.r_[s[m], s[is_ben]]
                rows.append({"seed": seed, "detector": cand, "class": lab, "kind": "zero_day" if lab in ZERO_DAY else "known",
                             "n": int(m.sum()), "auroc": float(roc_auc_score(y, sc)), "fpr_at_95tpr": fpr_at_tpr(y, sc)})
        print(f"seed {seed} done (train {len(train)}, test {len(test)})", flush=True)

    res = pd.DataFrame(rows)
    out = root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "closr_protocol_raw.csv", index=False)
    per_class = res.groupby(["kind", "detector", "class"])[["auroc", "fpr_at_95tpr"]].mean().round(4).reset_index()
    summ = res.groupby(["kind", "detector"])[["auroc", "fpr_at_95tpr"]].mean().round(4).reset_index()
    per_class.to_csv(out / "closr_protocol_per_class.csv", index=False)
    summ.to_csv(out / "closr_protocol.csv", index=False)
    write_manifest(out / "manifest_closr_protocol.json", inputs=[args.data], outputs=[out / "closr_protocol.csv"], seeds=range(args.seeds),
                   extra={"zero_day": ZERO_DAY, "cap": args.cap, "paper_reference_fpr95_clad": {"heartbleed": 0.007264, "web_sql_injection": 0.003331}}, repo_dir=root)
    print(summ.pivot(index="detector", columns="kind", values="auroc").to_string())
    print(summ.pivot(index="detector", columns="kind", values="fpr_at_95tpr").to_string())
    print(per_class[per_class["kind"] == "zero_day"].pivot(index="detector", columns="class", values="auroc").to_string())


if __name__ == "__main__":
    main()
