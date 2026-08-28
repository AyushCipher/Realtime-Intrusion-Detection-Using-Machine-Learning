"""Re-runs the H1 (open-set vs. softmax, LOFO) and H2 (static vs. adaptive
conformal, drift) evaluations from `ids_ml.evaluation` against a real
CICIDS2018 subsample instead of the synthetic test fixture -- the same
functions, real data, produced by `build_cicids2018_subsample.py`.

Usage:
    python -m scripts.run_real_data_experiments --data real_cicids2018_subsample.csv
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ids_ml.evaluation import (
    openset_vs_softmax_report,
    openset_vs_softmax_significance,
    static_vs_adaptive_conformal_report,
    static_vs_adaptive_conformal_significance,
)
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.split import time_window_split

logger = logging.getLogger("ids_ml.scripts.run_real_data_experiments")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to the subsampled canonical-form CSV")
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--h1-seeds", type=int, default=5)
    parser.add_argument("--h2-seeds", type=int, default=10)
    parser.add_argument("--drift-cutoff", default=None, help="ISO date splitting H2's pre/post-drift windows (default: midpoint of the data's date range)")
    parser.add_argument("--stage2-n-estimators", type=int, default=200)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    logger.info("Total rows: %d", len(df))
    logger.info("%s", df["attack_category"].value_counts().to_string())

    families = sorted(f for f in df["attack_category"].unique() if f != "BENIGN")
    logger.info("Non-benign families for LOFO: %s", families)

    logger.info("=" * 70)
    logger.info("H1: OpenMax vs softmax, LOFO x %d seeds, budget=%.2f", args.h1_seeds, args.budget)
    logger.info("=" * 70)
    h1_report = openset_vs_softmax_report(
        df, CANONICAL_FEATURE_COLUMNS, families=families, seeds=range(args.h1_seeds),
        budget=args.budget, stage2_n_estimators=args.stage2_n_estimators,
    )
    summary = h1_report.groupby(["family", "gate_name"])[
        ["unknown_recall", "unknown_auroc", "ece", "brier", "escalation_rate_on_known"]
    ].mean().round(3)
    logger.info("\n%s", summary.to_string())
    for metric in ("unknown_recall", "unknown_auroc"):
        logger.info("%s: %s", metric, openset_vs_softmax_significance(h1_report, metric=metric))

    logger.info("=" * 70)
    logger.info("H2: static vs adaptive conformal calibration under drift")
    logger.info("=" * 70)
    cutoff = pd.Timestamp(args.drift_cutoff) if args.drift_cutoff else df["timestamp"].quantile(0.5, interpolation="nearest")
    window_a, window_b = time_window_split(df, cutoff)
    logger.info("cutoff=%s: pre=%d rows, post=%d rows", cutoff, len(window_a), len(window_b))

    h2_report = static_vs_adaptive_conformal_report(
        window_a, window_b, CANONICAL_FEATURE_COLUMNS, budget=args.budget, seeds=range(args.h2_seeds),
        stage2_n_estimators=args.stage2_n_estimators,
    )
    h2_summary = h2_report.groupby("segment")[
        ["static_escalation_rate", "adaptive_escalation_rate", "static_error", "adaptive_error"]
    ].agg(["mean", "std"])
    logger.info("\n%s", h2_summary.to_string())
    for seg in ("pre_drift", "post_drift", None):
        logger.info("%s: %s", seg, static_vs_adaptive_conformal_significance(h2_report, segment=seg))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
