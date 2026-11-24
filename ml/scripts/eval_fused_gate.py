"""Does the fused gate (ood_gates.FusedGate) cover both kinds of unknown?

The best escalation gate depends on the threat model: kNN for entirely new
attack families, softmax/OpenMax for new variants of known families
(docs/real_data_findings.md sections 4 and 8). This compares softmax,
OpenMax, kNN and fused under both protocols on the same data:

  family  leave-one-FAMILY-out (attack_category withheld)
  type    leave-one-attack-TYPE-out (raw label withheld; siblings stay known)

Decision rule, fixed before running: deploy `fused` only if its mean AUROC is
within 0.02 of the best gate's in BOTH protocols. Writes
results/paper/fused_gate/{family,type}.csv and summary.json.

Usage:
    PYTHONPATH=ml/src python ml/scripts/eval_fused_gate.py --data C:/nids_data/real_cicids2018_dedup.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.evaluation import openset_vs_softmax_report, paired_gate_significance  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS  # noqa: E402

GATES = ["softmax", "openset", "knn", "fused"]
METRICS = ["unknown_auroc", "recall_at_matched_fpr"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--family-seeds", type=int, default=3)
    p.add_argument("--type-seeds", type=int, default=2)
    p.add_argument("--min-rows", type=int, default=200)
    p.add_argument("--stage2-n-estimators", type=int, default=200)
    args = p.parse_args(argv)

    out = Path(__file__).resolve().parent.parent.parent / "results" / "paper" / "fused_gate"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=["timestamp"]).dropna(subset=["timestamp"])

    protocols = {
        "family": ("attack_category", sorted(f for f in df["attack_category"].unique() if f != "BENIGN"), args.family_seeds),
        "type": ("label", sorted(l for l, c in df["label"].value_counts().items() if l != "BENIGN" and c >= args.min_rows), args.type_seeds),
    }
    summary = {}
    for name, (label_col, held_out, n_seeds) in protocols.items():
        csv = out / f"{name}.csv"
        if csv.exists():
            report = pd.read_csv(csv)
        else:
            report = openset_vs_softmax_report(
                df, CANONICAL_FEATURE_COLUMNS, families=held_out, seeds=range(n_seeds), label_col=label_col,
                stage2_n_estimators=args.stage2_n_estimators, gate_names=GATES,
            )
            report.to_csv(csv, index=False)
        means = report.groupby("gate_name")[METRICS].mean()
        summary[name] = {
            "held_out": held_out,
            "means": means.round(4).to_dict(),
            "fused_vs": [paired_gate_significance(report, "fused", g, metric=m) for g in GATES if g != "fused" for m in METRICS],
        }
        best = means["unknown_auroc"].max()
        summary[name]["fused_within_0.02_of_best_auroc"] = bool(means.loc["fused", "unknown_auroc"] >= best - 0.02)
        print(name, "\n", means.round(3).to_string())
    summary["deploy_fused"] = all(summary[n]["fused_within_0.02_of_best_auroc"] for n in protocols)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("deploy_fused:", summary["deploy_fused"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
