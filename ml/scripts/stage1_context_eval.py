"""Does per-host context make Stage 1 more capable? (LYCOS-IDS2017, which keeps addresses)

Same protocols and candidates as stage1_selection.py (leave-one-family-out for families never
seen, random split for known ones), run on three feature sets:
  flow         the 35 canonical per-flow features (what Stage 1 uses today)
  flow+ctx     plus the 12 causal per-host context features (context_features.py)
  ctx          the context features alone (a shortcut check: how much do counts alone carry?)

Confound, measured and reported: in CIC-IDS2017/LYCOS every attack family except Botnet comes
from one attacker host (172.16.0.1), which is busy during all attacks, so context features can
partly detect "that host" rather than the attack. Botnet traffic comes from infected internal
hosts and is the confound-free test of the context features; read its row first.

Input: a CSV made by computing add_context_features on the time-ordered LYCOS flows with
endpoints (see the README section in docs/real_data_findings.md). Writes
results/paper/<name>/stage1_context_eval.csv (+ _raw) and a manifest.

Usage:
    PYTHONPATH=ml/src python ml/scripts/stage1_context_eval.py --data C:/nids_data/lycos2017_context.csv --name lycos2017_context
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ids_ml.context_features import context_columns  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402

spec = importlib.util.spec_from_file_location("stage1_selection", HERE / "stage1_selection.py")
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

CANDS = ["xgb_d6", "knn_benign", "max_s6k", "efc_benign", "max3_s6k_efc"]


def main() -> None:
    root = HERE.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="lycos2017_context")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=120000)
    args = ap.parse_args()
    ctx = context_columns()
    df = pd.read_csv(args.data)
    feature_sets = {"flow": FEATS, "flow+ctx": FEATS + ctx, "ctx": ctx}
    fams = [f for f in sorted(set(df["attack_category"]) - {"BENIGN"}) if (df["attack_category"] == f).sum() >= 20]
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        ben = rng.permutation(np.where(df["attack_category"] == "BENIGN")[0])
        ben_tr, ben_te = ben[: len(ben) // 2], ben[len(ben) // 2:]
        splits = []
        for fam in fams:
            fam_idx = np.where(df["attack_category"] == fam)[0]
            other = np.where(~df["attack_category"].isin(["BENIGN", fam]))[0]
            tr = np.concatenate([ben_tr, other])
            tr = rng.choice(tr, min(len(tr), args.max_train), replace=False)
            te = np.concatenate([fam_idx, ben_te[: max(3 * len(fam_idx), 5000)]])
            splits.append(("LOFO_unseen_family", fam, tr, te))
        perm = rng.permutation(len(df))
        cut = int(0.7 * len(df))
        splits.append(("random_known_families", "-", perm[:cut][: args.max_train], perm[cut:]))
        for proto, fam, tr, te in splits:
            ytr = (df["attack_category"].iloc[tr] != "BENIGN").astype(int).to_numpy()
            yte = (df["attack_category"].iloc[te] != "BENIGN").astype(int).to_numpy()
            for fs, cols in feature_sets.items():
                sc = sel.score_all(df.iloc[tr][cols].to_numpy(), ytr, df.iloc[te][cols].to_numpy(), seed)
                for c in CANDS:
                    rows.append({"protocol": proto, "heldout": fam, "seed": seed, "features": fs, "candidate": c,
                                 "auroc": float(roc_auc_score(yte, sc[c]))})
        print(f"seed {seed} done", flush=True)
    res = pd.DataFrame(rows)
    out = root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "stage1_context_eval_raw.csv", index=False)
    summ = res.groupby(["protocol", "heldout", "candidate", "features"])["auroc"].agg(["mean", "std"]).round(3).reset_index()
    summ.to_csv(out / "stage1_context_eval.csv", index=False)
    write_manifest(out / "manifest_stage1_context_eval.json", inputs=[args.data], outputs=[out / "stage1_context_eval.csv"],
                   seeds=range(args.seeds), repo_dir=root)
    piv = res.groupby(["protocol", "heldout", "candidate", "features"])["auroc"].mean().unstack("features").round(3)
    print(piv.to_string())


if __name__ == "__main__":
    main()
