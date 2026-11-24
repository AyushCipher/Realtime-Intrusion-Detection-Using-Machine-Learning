"""Does training Stage 1 on several networks transfer better to an unseen network?

For each target dataset T in {CSE-CIC-IDS2018, LYCOS-IDS2017, CIC-UNSW-NB15}, Stage 1
candidates are trained on (a) each other dataset alone and (b) both other datasets pooled,
then scored on T without any data from T (benign vs attack AUROC). All three share the
35 canonical CICFlowMeter features. Each source contributes a fixed-size sample so pooling
does not simply mean "more rows". Writes results/paper/multinetwork/stage1_multinetwork.csv.

Usage:
    PYTHONPATH=ml/src python ml/scripts/stage1_multinetwork.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402

spec = importlib.util.spec_from_file_location("stage1_selection", HERE / "stage1_selection.py")
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

DATA = {"cse": "C:/nids_data/real_cicids2018_dedup.csv", "lycos": "C:/nids_data/lycos2017_canonical_dedup.csv",
        "unsw": "C:/nids_data/unsw_canonical_dedup.csv"}
PER_SOURCE, TEST_N, SEEDS = 60000, 40000, 3
CANDS = ["xgb_d6", "knn_benign", "max_s6k"]


def sample(df, n, seed):
    return df.sample(min(len(df), n), random_state=seed)


def main() -> None:
    root = HERE.parent.parent
    dfs = {k: pd.read_csv(v, usecols=FEATS + ["attack_category"]) for k, v in DATA.items()}
    rows = []
    for seed in range(SEEDS):
        for tgt in DATA:
            others = [k for k in DATA if k != tgt]
            te = sample(dfs[tgt], TEST_N, seed)
            yte = (te["attack_category"] != "BENIGN").astype(int).to_numpy()
            for src in [[o] for o in others] + [others]:
                tr = pd.concat([sample(dfs[s], PER_SOURCE, seed) for s in src])
                ytr = (tr["attack_category"] != "BENIGN").astype(int).to_numpy()
                sc = sel.score_all(tr[FEATS].to_numpy(), ytr, te[FEATS].to_numpy(), seed)
                for c in CANDS:
                    rows.append({"seed": seed, "target": tgt, "train_on": "+".join(src), "candidate": c,
                                 "auroc": float(roc_auc_score(yte, sc[c]))})
            print(f"seed {seed} target {tgt} done", flush=True)
    res = pd.DataFrame(rows)
    out = root / "results" / "paper" / "multinetwork"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "stage1_multinetwork_raw.csv", index=False)
    summ = res.groupby(["target", "train_on", "candidate"])["auroc"].agg(["mean", "std"]).round(3).reset_index()
    summ.to_csv(out / "stage1_multinetwork.csv", index=False)
    write_manifest(out / "manifest_stage1_multinetwork.json", inputs=list(DATA.values()), outputs=[out / "stage1_multinetwork.csv"],
                   seeds=range(SEEDS), repo_dir=root)
    print(summ.pivot_table(index=["target", "train_on"], columns="candidate", values="mean").to_string())


if __name__ == "__main__":
    main()
