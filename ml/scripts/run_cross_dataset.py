"""Cross-dataset transfer: train and calibrate on one canonical-form
dataset, deploy on another (see evaluation.cross_dataset_trial).

Both inputs must be canonical-form CSVs (build_cicids2018_subsample.py's
output, or data.load_and_map_cicflowmeter(...) written to CSV for the
CICFlowMeter re-extractions of UNSW-NB15 / ToN-IoT). `--family-map` maps
target attack names to source families, e.g. for CSE-CIC-IDS2018 ->
CIC-ToN-IoT:

    '{"ddos": "DoS/DDoS", "dos": "DoS/DDoS", "password": "Brute Force",
      "xss": "Web Attack", "injection": "Web Attack"}'

Unmapped target attacks are treated as novel families. Writes
results/paper/cross_<source>_to_<target>.csv.

Usage:
    PYTHONPATH=ml/src python ml/scripts/run_cross_dataset.py \\
        --source a.csv --source-name cicids2018 --target b.csv --target-name ton_iot \\
        --family-map '{"ddos": "DoS/DDoS"}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.evaluation import ALL_GATE_NAMES, cross_dataset_report  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--source-name", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--target-name", required=True)
    p.add_argument("--family-map", default="{}", help="JSON object: target attack name -> source family")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--miss-budget", type=float, default=0.1)
    p.add_argument("--stage2-n-estimators", type=int, default=200)
    p.add_argument("--stage1-kind", choices=["hybrid", "iforest"], default="hybrid")
    p.add_argument("--max-target-rows", type=int, default=300000, help="Seeded subsample of the target, for memory")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    source = pd.read_csv(args.source, parse_dates=["timestamp"])
    target = pd.read_csv(args.target, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(args.target, nrows=0).columns else None)
    if len(target) > args.max_target_rows:
        target = target.sample(args.max_target_rows, random_state=0)
    logging.info("source %s: %d rows; target %s: %d rows", args.source_name, len(source), args.target_name, len(target))

    report = cross_dataset_report(
        source, target, CANONICAL_FEATURE_COLUMNS, seeds=range(args.seeds), family_map=json.loads(args.family_map),
        budget=args.budget, miss_budget=args.miss_budget, stage2_n_estimators=args.stage2_n_estimators,
        gate_names=ALL_GATE_NAMES, stage1_kind=args.stage1_kind,
    )
    out = Path(__file__).resolve().parent.parent.parent / "results" / "paper" / f"cross_{args.source_name}_to_{args.target_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    logging.info("wrote %s\n%s", out, report.groupby("gate_name").mean(numeric_only=True).round(4).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
