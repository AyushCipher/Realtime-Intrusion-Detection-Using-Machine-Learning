"""CLI: train the two-stage detector from CICIDS2017/CIC-IDS2018 CSVs.

    python -m ids_ml.train --data path/to/Monday.csv path/to/Tuesday.csv --model-dir models

Always evaluates with a time-based split (see split.py) and additionally
runs the leakage comparison against a random split, and a low-and-slow
robustness probe, printing both so a bad split choice or a fragile detector
shows up before a model ever gets deployed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .data import load_and_map
from .evaluation import adversarial_robustness_report, leakage_comparison_report, per_category_report
from .features import CANONICAL_FEATURE_COLUMNS
from .pipeline import TwoStageDetector
from .split import time_based_split
from .stage1_iforest import AnomalyPreFilter, Stage1Config
from .stage2_xgboost import AttackClassifier, Stage2Config

logger = logging.getLogger("ids_ml.train")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ids_ml two-stage detector")
    parser.add_argument("--data", nargs="+", required=True, help="CICIDS2017/2018 CSV file(s)")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--stage1-n-estimators", type=int, default=200)
    parser.add_argument("--stage1-contamination", default="auto")
    parser.add_argument("--stage2-n-estimators", type=int, default=300)
    parser.add_argument("--skip-leakage-check", action="store_true")
    parser.add_argument("--skip-robustness-probe", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = load_and_map(args.data)
    logger.info("Loaded %d rows from %d file(s)", len(df), len(args.data))

    train, val, test = time_based_split(df, train_frac=args.train_frac, val_frac=args.val_frac)
    logger.info("Time-based split: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()

    contamination = args.stage1_contamination if args.stage1_contamination == "auto" else float(args.stage1_contamination)
    stage1_cfg = Stage1Config(n_estimators=args.stage1_n_estimators, contamination=contamination)
    stage1 = AnomalyPreFilter(stage1_cfg).fit(X_train)

    stage2_cfg = Stage2Config(n_estimators=args.stage2_n_estimators)
    stage2 = AttackClassifier(stage2_cfg).fit(X_train, train["attack_category"].tolist())

    detector = TwoStageDetector(stage1, stage2)

    X_test = test[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_test = test["attack_category"].tolist()

    standalone_report = per_category_report(y_test, list(stage2.predict(X_test)))
    logger.info("Stage-2 standalone per-category report (time-based test split):\n%s", standalone_report.to_string(index=False))

    cascade_preds = [r.stage2_predicted_class for r in detector.score(X_test)]
    cascade_report = per_category_report(y_test, cascade_preds)
    logger.info(
        "End-to-end cascade per-category report -- stage2 only runs on stage1-flagged flows "
        "(time-based test split):\n%s",
        cascade_report.to_string(index=False),
    )

    if not args.skip_leakage_check:
        leakage = leakage_comparison_report(df, CANONICAL_FEATURE_COLUMNS, stage2_config=stage2_cfg)
        logger.info(
            "Split-leakage check: time-based macro-F1=%.4f, random-split macro-F1=%.4f, inflation=%.4f "
            "(inflation should be treated as evidence of leakage, not model quality)",
            leakage["time_based_macro_f1"],
            leakage["random_split_macro_f1"],
            leakage["inflation"],
        )

    if not args.skip_robustness_probe:
        attack_test = test[test["is_attack"]]
        if len(attack_test) > 0:
            robustness = adversarial_robustness_report(detector, attack_test, CANONICAL_FEATURE_COLUMNS)
            logger.info(
                "Low-and-slow probe (%.1fx slowdown, n=%d attack flows): stage1 flag rate %.1f%% -> %.1f%% "
                "(evasion delta %.1f pts)",
                robustness["slowdown_factor"],
                robustness["n_flows"],
                robustness["original_stage1_flag_rate"] * 100,
                robustness["low_and_slow_stage1_flag_rate"] * 100,
                robustness["evasion_delta"] * 100,
            )
        else:
            logger.info("No attack flows in the test split; skipping the low-and-slow robustness probe")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    stage1.save(model_dir / "stage1_iforest.joblib")
    stage2.save(model_dir / "stage2_xgboost.joblib")
    logger.info("Saved models to %s", model_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
