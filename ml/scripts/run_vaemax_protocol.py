"""VAEMax-style fixed known/unknown protocol on CSE-CIC-IDS2018.

Follows the setup of Qiu et al., "VAEMax: Open-Set Intrusion Detection based
on OpenMax and Variational Autoencoder" (arXiv:2403.04193), Table II:
  known classes    BENIGN, DDOS attack-HOIC, DoS attacks-Hulk, DoS attacks-Slowloris, SSH-Bruteforce
  unknown classes  Bot, DDOS attack-LOIC-UDP, DoS attacks-GoldenEye, Brute Force -Web,
                   Brute Force -XSS, SQL Injection   (present in test only)
Known flows are split 80/20 train/test; unknown flows are added to the test set.
Classes in our 8-day subsample that VAEMax's table does not use (Infilteration,
FTP-BruteForce, DoS attacks-SlowHTTPTest) are left out so the class set matches.

For every escalation gate (softmax, OpenMax, max-logit, energy, Mahalanobis, kNN):
the threshold is the conformal quantile of the gate score on validation BENIGN flows
at a 5% budget (VAEMax reports benign false positives under 5%), then
  benign_fpr           share of test benign flows escalated or predicted as an attack
  known_attack_acc     known-attack flows classified into their true class and not escalated
  unknown_recall       unknown-attack flows escalated
and the same metrics per unknown class.

Important comparability caveats (also written to the manifest): VAEMax uses payload
features from a 1-D CNN and its own preprocessing of raw pcaps; this run uses
CICFlowMeter flow statistics from the deduplicated subsample. Numbers are indicative
of the same protocol, not a reproduction of their results.

Writes results/paper/<name>/vaemax_protocol.csv (+ _raw) and a manifest.

Usage:
    PYTHONPATH=ml/src python ml/scripts/run_vaemax_protocol.py --data real.csv --name cicids2018_dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.conformal_gate import calibrate_threshold  # noqa: E402
from ids_ml.evaluation import ALL_GATE_NAMES, _fit_gate  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config  # noqa: E402

KNOWN = ["BENIGN", "DDOS attack-HOIC", "DoS attacks-Hulk", "DoS attacks-Slowloris", "SSH-Bruteforce"]
UNKNOWN = ["Bot", "DDOS attack-LOIC-UDP", "DoS attacks-GoldenEye", "Brute Force -Web", "Brute Force -XSS", "SQL Injection"]
CAP = 10_000  # per-class cap, as in VAEMax's table


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--stage2-trees", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df[df["label"].isin(KNOWN + UNKNOWN)]
    print(df["label"].value_counts().to_string(), flush=True)

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        parts = {k: [] for k in ("train", "val", "test_known")}
        for lab in KNOWN:
            g = df[df["label"] == lab]
            g = g.iloc[rng.permutation(len(g))[:CAP]]
            n = len(g)
            a, b = int(0.7 * n), int(0.8 * n)  # 70% train, 10% validation (carved from the 80%), 20% test
            parts["train"].append(g.iloc[:a])
            parts["val"].append(g.iloc[a:b])
            parts["test_known"].append(g.iloc[b:])
        train, val, test_known = (pd.concat(parts[k]) for k in ("train", "val", "test_known"))
        unk = pd.concat([df[df["label"] == lab].iloc[rng.permutation((df["label"] == lab).sum())[:CAP]] for lab in UNKNOWN])
        test = pd.concat([test_known, unk])
        X_tr, y_tr = train[FEATS].to_numpy(), train["label"].tolist()
        stage2 = AttackClassifier(Stage2Config(n_estimators=args.stage2_trees)).fit(X_tr, y_tr)
        val_ben = val[val["label"] == "BENIGN"][FEATS].to_numpy()
        X_te = test[FEATS].to_numpy()
        lab_te = test["label"].to_numpy()
        for gate_name in ALL_GATE_NAMES:
            gate = _fit_gate(gate_name, stage2, X_tr, y_tr, seed=seed)
            thr = calibrate_threshold([r.unknown_mass for r in gate.recalibrate_batch(val_ben)], budget=args.budget).threshold
            res = gate.recalibrate_batch(X_te)
            esc = np.array([r.unknown_mass > thr for r in res])
            pred = np.array([r.predicted_class for r in res], dtype=object)
            is_ben = lab_te == "BENIGN"
            is_known_att = np.isin(lab_te, KNOWN) & ~is_ben
            is_unk = np.isin(lab_te, UNKNOWN)
            base = {"seed": seed, "gate": gate_name, "budget": args.budget}
            rows.append({**base, "scope": "ALL", "benign_fpr": float((esc | (pred != "BENIGN"))[is_ben].mean()),
                         "known_attack_acc": float(((pred == lab_te) & ~esc)[is_known_att].mean()),
                         "unknown_recall": float(esc[is_unk].mean())})
            for u in UNKNOWN:
                m = lab_te == u
                if m.any():
                    rows.append({**base, "scope": u, "unknown_recall": float(esc[m].mean())})
        print(f"seed {seed} done", flush=True)

    res = pd.DataFrame(rows)
    out = root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "vaemax_protocol_raw.csv", index=False)
    summ = res.groupby(["scope", "gate"])[["benign_fpr", "known_attack_acc", "unknown_recall"]].agg(["mean", "std"]).round(3)
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "vaemax_protocol.csv", index=False)
    write_manifest(out / "manifest_vaemax_protocol.json", inputs=[args.data], outputs=[out / "vaemax_protocol.csv"], seeds=range(args.seeds),
                   extra={"known": KNOWN, "unknown": UNKNOWN, "cap": CAP, "vaemax_reference": {"unknown_recall_2018": 0.66, "benign_fpr": "<0.05", "known_acc": ">0.95"}}, repo_dir=root)
    print(summ[summ["scope"] == "ALL"].to_string(index=False))
    print(summ[summ["scope"] != "ALL"][["scope", "gate", "unknown_recall_mean"]].pivot(index="scope", columns="gate", values="unknown_recall_mean").to_string())


if __name__ == "__main__":
    main()
