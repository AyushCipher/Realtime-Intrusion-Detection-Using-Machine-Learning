"""End-to-end open-set test through the deployed cascade (Stage 1 -> Stage 2 -> gate).

H1 (run_paper_experiments.py) scores the escalation gates on Stage 2 alone, so it
never asks whether an unknown attack survives Stage 1 to reach a gate. This runs
the whole cascade, leave-one-family-out:

  * the held-out family F is absent from every fit and calibration step;
  * Stage 1 (hybrid pre-filter by default) is fitted on the known training flows and
    its miss budget calibrated on known validation attacks;
  * Stage 2 and each gate are fitted on known training flows; the gate threshold is
    calibrated, at the escalation budget, on the known validation flows Stage 1 flags
    (the only flows a deployed gate ever sees).

Per gate, on the test set:
  f_stage1_pass     share of family-F flows Stage 1 flags (the ceiling for everything after)
  f_escalated       share of F flows flagged AND escalated (reach an analyst as "unknown")
  f_alerted         share of F flows flagged AND escalated-or-classified-as-an-attack
  known_escalated   share of known test flows escalated (the analyst cost; budget applies
                    to flagged known flows, so this is at most the budget)
  benign_alerted    share of benign test flows that raise any alert

Writes results/paper/<name>/cascade_openset.csv (+ _raw) and a manifest.

Usage:
    PYTHONPATH=ml/src python ml/scripts/run_cascade_openset.py --data real.csv --name cicids2018_dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.conformal_gate import calibrate_threshold  # noqa: E402
from ids_ml.conformal_stage1 import calibrate_stage1_dual  # noqa: E402
from ids_ml.evaluation import ALL_GATE_NAMES, _fit_gate, _gate_scores, make_stage1  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402
from ids_ml.split import random_split  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--miss-budget", type=float, default=0.05)
    ap.add_argument("--budget", type=float, default=0.10, help="escalation budget used for the summary table")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20],
                    help="budgets swept for the recall-vs-budget curve (results/paper/<name>/cascade_openset_budgets.csv)")
    ap.add_argument("--cap-per-label", type=int, default=0, help="random per-raw-label cap for very large data (0 = off)")
    ap.add_argument("--stage1-kind", choices=["hybrid", "iforest"], default="hybrid")
    ap.add_argument("--stage1-compute-budget", type=float, default=0.0,
                    help="share of benign traffic Stage 1 may pass on (conformal_stage1.calibrate_stage1_dual); 0 = recall-only")
    ap.add_argument("--stage2-trees", type=int, default=200)
    ap.add_argument("--max-unknown", type=int, default=10000)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.cap_per_label:
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), args.cap_per_label), random_state=0))
    fams = sorted(set(df["attack_category"]) - {"BENIGN"})
    rows = []
    for seed in range(args.seeds):
        for fam in fams:
            known, unk = df[df["attack_category"] != fam], df[df["attack_category"] == fam]
            unk = unk.sample(min(len(unk), args.max_unknown), random_state=seed)
            tr, va, te = random_split(known, train_frac=0.7, val_frac=0.15, random_state=seed)
            s1 = make_stage1(args.stage1_kind, tr, FEATS, seed=seed, contamination=0.35)
            va_att = va[va["attack_category"] != "BENIGN"]
            va_ben = va[va["attack_category"] == "BENIGN"]
            calibrate_stage1_dual(s1.anomaly_score(va_att[FEATS].to_numpy()), s1.anomaly_score(va_ben[FEATS].to_numpy()),
                                  args.miss_budget, args.stage1_compute_budget).apply(s1)

            X_tr, y_tr = tr[FEATS].to_numpy(), tr["attack_category"].tolist()
            s2 = AttackClassifier(Stage2Config(n_estimators=args.stage2_trees, random_state=seed)).fit(X_tr, y_tr)
            Xva = va[FEATS].to_numpy()
            va_flag = s1.flag(Xva)
            Xk, Xu = te[FEATS].to_numpy(), unk[FEATS].to_numpy()
            k_flag, u_flag = s1.flag(Xk), s1.flag(Xu)
            k_pred, u_pred = s2.predict(Xk), s2.predict(Xu)
            is_ben = (te["attack_category"] == "BENIGN").to_numpy()
            for g in ALL_GATE_NAMES:
                gate = _fit_gate(g, s2, X_tr, y_tr, seed=seed)
                s_cal, s_k, s_u = _gate_scores(gate, Xva[va_flag]), _gate_scores(gate, Xk), _gate_scores(gate, Xu)
                for b in sorted(set(args.budgets) | {args.budget}):
                    thr = calibrate_threshold(s_cal, budget=b).threshold
                    k_esc = k_flag & (s_k > thr)
                    u_esc = u_flag & (s_u > thr)
                    rows.append({
                        "seed": seed, "family": fam, "gate": g, "budget": b, "n_unknown": len(unk),
                        "f_stage1_pass": float(u_flag.mean()),
                        "f_escalated": float(u_esc.mean()),
                        "f_alerted": float((u_flag & (u_esc | (u_pred != "BENIGN"))).mean()),
                        "known_escalated": float(k_esc.mean()),
                        "benign_alerted": float((k_flag & (k_esc | (k_pred != "BENIGN")))[is_ben].mean()),
                        "benign_flagged_stage1": float(k_flag[is_ben].mean()),
                    })
            print(f"seed {seed} family {fam} done (stage1 pass {u_flag.mean():.3f})", flush=True)

    allres = pd.DataFrame(rows)
    out = root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)
    allres.to_csv(out / "cascade_openset_raw.csv", index=False)
    cols = ["f_stage1_pass", "f_escalated", "f_alerted", "known_escalated", "benign_alerted", "benign_flagged_stage1"]
    # per-seed means over families, then mean and std over seeds
    per_seed = allres.groupby(["budget", "gate", "seed"])[cols].mean()
    curve = per_seed.groupby(["budget", "gate"]).agg(["mean", "std"]).round(4)
    curve.columns = ["_".join(c) for c in curve.columns]
    curve.reset_index().to_csv(out / "cascade_openset_budgets.csv", index=False)
    res = allres[allres["budget"] == args.budget]
    summ = res.groupby("gate")[cols].mean().round(3).reset_index()
    summ["f_escalated_seed_std"] = summ["gate"].map(res.groupby(["gate", "seed"])["f_escalated"].mean().groupby("gate").std().round(3))
    summ.to_csv(out / "cascade_openset.csv", index=False)
    write_manifest(out / "manifest_cascade_openset.json", inputs=[args.data], outputs=[out / "cascade_openset.csv"],
                   seeds=range(args.seeds), extra=vars(args), repo_dir=root)
    print(summ.to_string(index=False))
    print(res.groupby(["family", "gate"])["f_escalated"].mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
