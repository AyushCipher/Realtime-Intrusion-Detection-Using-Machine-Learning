"""Runs the paper's experiment suite on one canonical-form dataset (the
output of `build_cicids2018_subsample.py`, or any CSV with the canonical
feature columns plus `label`, `attack_category` and `timestamp`).

Parts (select with --parts; each writes its own CSV and is skipped on a
re-run if that CSV exists, unless --force):

  h1      Unknown-attack detection, leave-one-FAMILY-out, every gate in
          evaluation.ALL_GATE_NAMES (softmax, OpenMax, max-logit, energy,
          Mahalanobis, kNN) at one escalation budget.
  h1raw   The same at raw-label granularity (leave-one-LABEL-out, e.g. hold
          out DoS-Hulk while DoS-GoldenEye stays known). More held-out
          classes, so more independent observations -- but a held-out
          variant of a known family is an easier "zero-day" than a whole
          new family. Report the two granularities separately.
  h2      Escalation budget under drift: static vs adaptive (ACI) vs
          detect-then-recalibrate, for the softmax and OpenMax gates.
  stage1  Stage 1 miss budget on the chronological stream: static vs
          adaptive vs detect-then-recalibrate, over budgets, seeds and
          feedback delays.
  mc      Monte Carlo check of the static Stage 1 guarantee under
          exchangeability (pooled real attack scores, random splits).

Writes to --out-dir: one CSV per part plus summary.json with the paired
significance tests. Significance for h1/h1raw is clustered by held-out
class (evaluation.paired_gate_significance): seeds are averaged within a
class first, because seeds sharing one held-out attack are not independent.

Usage:
    PYTHONPATH=ml/src python ml/scripts/run_paper_experiments.py \\
        --data /path/real_cicids2018_subsample.csv --name cicids2018
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.conformal_stage1 import calibrate_stage1_threshold  # noqa: E402
from ids_ml.evaluation import (  # noqa: E402
    ALL_GATE_NAMES,
    drift_method_significance,
    openset_vs_softmax_report,
    paired_gate_significance,
    make_stage1,
    stage1_stream_report,
    static_vs_adaptive_conformal_report,
)
from ids_ml.features import CANONICAL_FEATURE_COLUMNS  # noqa: E402
from ids_ml.split import time_window_split  # noqa: E402

logger = logging.getLogger("run_paper_experiments")

PARTS = ("h1", "h1raw", "h2", "stage1", "mc")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True, help="Dataset name, used for the output folder")
    p.add_argument("--out-dir", default=None, help="Default: results/paper/<name>")
    p.add_argument("--parts", default=",".join(PARTS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--h1-seeds", type=int, default=5)
    p.add_argument("--h1raw-seeds", type=int, default=3)
    p.add_argument("--h1raw-min-rows", type=int, default=200, help="Skip raw labels with fewer rows than this")
    p.add_argument("--h2-seeds", type=int, default=10)
    p.add_argument("--stage1-seeds", type=int, default=5)
    p.add_argument("--stage1-budgets", default="0.05,0.1,0.2")
    p.add_argument("--stage1-delays", default="0,1000,10000")
    p.add_argument("--stage2-n-estimators", type=int, default=200)
    p.add_argument("--mc-trials", type=int, default=1000)
    p.add_argument("--stage1-kind", choices=["hybrid", "iforest"], default="hybrid",
                   help="Stage 1 used by the stage1 and mc parts: hybrid (current system) or iforest (original)")
    return p.parse_args(argv)


def _floats(s: str):
    return [float(x) for x in s.split(",") if x]


def _ints(s: str):
    return [int(x) for x in s.split(",") if x]


def run_h1(df, args, label_col, families, seeds, out_csv):
    logger.info("H1 (%s): %d held-out classes x %d seeds x %d gates", label_col, len(families), len(seeds), len(ALL_GATE_NAMES))
    report = openset_vs_softmax_report(
        df, CANONICAL_FEATURE_COLUMNS, families=families, seeds=seeds, budget=args.budget,
        label_col=label_col, stage2_n_estimators=args.stage2_n_estimators, gate_names=ALL_GATE_NAMES,
    )
    report.to_csv(out_csv, index=False)
    return report


def h1_summary(report):
    out = {"per_gate_mean": report.groupby("gate_name")[["unknown_recall", "recall_at_matched_fpr", "unknown_auroc", "open_auc_score", "escalation_rate_on_known"]].mean().round(4).to_dict()}
    tests = []
    for baseline in [g for g in ALL_GATE_NAMES if g != "openset"]:
        for metric in ("unknown_recall", "recall_at_matched_fpr", "unknown_auroc"):
            if metric not in report.columns:
                continue
            tests.append(paired_gate_significance(report, "openset", baseline, metric=metric))
    out["openset_vs_each_baseline"] = tests
    return out


def run_h2(df, args, out_csv):
    cutoff = df["timestamp"].quantile(0.5, interpolation="nearest")
    window_a, window_b = time_window_split(df, cutoff)
    logger.info("H2: cutoff=%s pre=%d post=%d", cutoff, len(window_a), len(window_b))
    frames = []
    for gate_name in ("softmax", "openset"):
        r = static_vs_adaptive_conformal_report(
            window_a, window_b, CANONICAL_FEATURE_COLUMNS, budget=args.budget, seeds=range(args.h2_seeds),
            stage2_n_estimators=args.stage2_n_estimators, gate_name=gate_name,
        )
        r["gate_name"] = gate_name
        frames.append(r)
    report = pd.concat(frames, ignore_index=True)
    report.to_csv(out_csv, index=False)
    return report


def h2_summary(report):
    out = {}
    for gate_name, g in report.groupby("gate_name"):
        out[gate_name] = {
            "means": g.groupby("segment")[["static_escalation_rate", "adaptive_escalation_rate", "recalibrated_escalation_rate",
                                           "static_error", "adaptive_error", "recalibrated_error", "n_recalibrations"]].mean().round(5).to_dict(),
            "tests": [drift_method_significance(g, a, b, segment="post_drift")
                      for a, b in (("static", "adaptive"), ("recalibrated", "adaptive"), ("static", "recalibrated"))],
        }
    return out


def stage1_summary(report):
    overall = report[report["segment"].isin(["overall", "seen_families", "new_families"])]
    means = overall.groupby(["miss_budget", "feedback_delay", "segment", "method"])[["miss_rate", "miss_error", "benign_flag_rate"]].mean().round(4)
    tests = []
    for (budget, delay), g in report[report["segment"] == "overall"].groupby(["miss_budget", "feedback_delay"]):
        piv = g.pivot_table(index="seed", columns="method", values="miss_error")
        for a, b in (("static", "adaptive"), ("recalibrated", "adaptive")):
            tests.append({"miss_budget": budget, "feedback_delay": delay, "a": a, "b": b,
                          "a_error_mean": float(piv[a].mean()), "b_error_mean": float(piv[b].mean()),
                          "b_better_in_seeds": int((piv[b] < piv[a]).sum()), "n_seeds": int(len(piv))})
    return {"means": {str(k): v for k, v in means.to_dict(orient="index").items()}, "comparisons": tests}


def run_mc(df, args, out_csv):
    """Exchangeable validity check: fit Stage 1 on the chronological first
    30%, pool every later attack's score, and repeatedly split the pool at
    random into calibration and test halves."""
    ordered = df.sort_values("timestamp", kind="mergesort")
    fit_end = int(len(ordered) * 0.3)
    stage1 = make_stage1(args.stage1_kind, ordered.iloc[:fit_end], CANONICAL_FEATURE_COLUMNS, seed=0)
    rest = ordered.iloc[fit_end:]
    pool = stage1.anomaly_score(rest[rest["attack_category"] != "BENIGN"][CANONICAL_FEATURE_COLUMNS].to_numpy())
    rng = np.random.default_rng(0)
    rows = []
    for b in _floats(args.stage1_budgets):
        misses = []
        for _ in range(args.mc_trials):
            idx = rng.permutation(len(pool))
            half = len(pool) // 2
            g = calibrate_stage1_threshold(pool[idx[:half]], b)
            misses.append(float((pool[idx[half:]] < g.threshold).mean()))
        rows.append({"miss_budget": b, "stage1_kind": args.stage1_kind, "n_pooled_attacks": int(len(pool)), "trials": args.mc_trials,
                     "mean_observed_miss_rate": float(np.mean(misses)), "p95_observed_miss_rate": float(np.quantile(misses, 0.95))})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else root / "results" / "paper" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [p for p in args.parts.split(",") if p]
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown parts: {sorted(unknown)}")

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    logger.info("%s: %d rows\n%s", args.name, len(df), df["attack_category"].value_counts().to_string())

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["dataset"] = {"name": args.name, "rows": int(len(df)), "budget": args.budget,
                          "categories": df["attack_category"].value_counts().to_dict(),
                          "raw_labels": df["label"].value_counts().to_dict()}

    def todo(part):
        csv = out_dir / f"{part}.csv"
        return part in parts and (args.force or not csv.exists()), csv

    for part in parts:
        t0 = time.time()
        run, csv = todo(part)
        if part == "h1":
            families = sorted(f for f in df["attack_category"].unique() if f != "BENIGN")
            report = run_h1(df, args, "attack_category", families, list(range(args.h1_seeds)), csv) if run else pd.read_csv(csv)
            summary["h1"] = h1_summary(report)
        elif part == "h1raw":
            counts = df["label"].value_counts()
            labels = sorted(l for l, c in counts.items() if l != "BENIGN" and c >= args.h1raw_min_rows)
            report = run_h1(df, args, "label", labels, list(range(args.h1raw_seeds)), csv) if run else pd.read_csv(csv)
            summary["h1raw"] = h1_summary(report)
        elif part == "h2":
            report = run_h2(df, args, csv) if run else pd.read_csv(csv)
            summary["h2"] = h2_summary(report)
        elif part == "stage1":
            if run:
                report = stage1_stream_report(
                    df, CANONICAL_FEATURE_COLUMNS, miss_budgets=_floats(args.stage1_budgets),
                    seeds=range(args.stage1_seeds), feedback_delays=_ints(args.stage1_delays),
                    stage1_kind=args.stage1_kind,
                )
                report.to_csv(csv, index=False)
            else:
                report = pd.read_csv(csv)
            summary["stage1"] = stage1_summary(report)
        elif part == "mc":
            report = run_mc(df, args, csv) if run else pd.read_csv(csv)
            summary["mc"] = report.to_dict(orient="records")
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("part %s done in %.0fs (%s)", part, time.time() - t0, "ran" if run else "loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
