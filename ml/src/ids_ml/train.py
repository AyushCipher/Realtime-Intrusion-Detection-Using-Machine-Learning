"""CLI: train the two-stage detector from CICIDS2017/CIC-IDS2018 CSVs.

    python -m ids_ml.train --data path/to/Monday.csv path/to/Tuesday.csv --model-dir models

Always evaluates with a time-based split (see split.py) and additionally
runs the leakage comparison against a random split, and a low-and-slow
robustness probe, printing both so a bad split choice or a fragile detector
shows up before a model ever gets deployed.

`--gate {none,softmax,openset}` (default `none`, preserving the original
closed-set behavior) optionally fits an open-set escalation gate and
calibrates its escalation threshold against `--escalation-budget`, then
saves both to `--model-dir` for `ids_ml.serve` to load automatically. The
calibration set is the *validation* split (`val`, from the same
`time_based_split` call as `train`/`test`) -- already held out from
stage1/stage2's own fitting and otherwise unused by this script, so
calibrating there doesn't read the model's own training-set
overconfidence as evidence the threshold can be strict (see
`evaluation.run_openset_trial`'s docstring for why that distinction
matters; this is the same fix applied here at train time instead of only
in the evaluation harness).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .adaptive_conformal_gate import AdaptiveConformalGate
from .conformal_gate import calibrate_threshold
from .data import load_and_map
from .evaluation import adversarial_robustness_report, leakage_comparison_report, per_category_report
from .features import CANONICAL_FEATURE_COLUMNS
from .openset_head import OpenMaxConfig, OpenMaxHead
from .pipeline import TwoStageDetector
from .softmax_gate import SoftmaxGate
from .split import time_based_split
from .stage1_iforest import AnomalyPreFilter, Stage1Config
from .stage2_xgboost import AttackClassifier, Stage2Config

logger = logging.getLogger("ids_ml.train")

GATE_TYPE_FILENAME = "gate_type.txt"
OPENSET_GATE_FILENAME = "openset_gate.joblib"
ESCALATION_GATE_FILENAME = "escalation_gate.joblib"
ESCALATION_KIND_FILENAME = "escalation_kind.txt"  # "static" | "adaptive"; absent means "static" (pre-dates this file)


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
    parser.add_argument(
        "--gate",
        choices=["none", "softmax", "openset"],
        default="none",
        help="Fit an open-set escalation gate and save it alongside the models (default: none, closed-set only)",
    )
    parser.add_argument("--escalation-budget", type=float, default=0.1, help="Target escalation rate for --gate calibration")
    parser.add_argument(
        "--adaptive-escalation",
        action="store_true",
        help=(
            "Use adaptive_conformal_gate.AdaptiveConformalGate (online-updating threshold, seeded from the "
            "validation split) instead of a static conformal_gate.ConformalGate. Requires --gate != none. "
            "This opts into the production ground-truth-proxy documented in adaptive_conformal_gate.py's "
            "module docstring -- read that before using this outside a demo/prototype deployment."
        ),
    )
    parser.add_argument("--adaptive-gamma", type=float, default=0.01, help="AdaptiveConformalGate step size")
    parser.add_argument("--adaptive-window-size", type=int, default=200, help="AdaptiveConformalGate sliding-window size")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.adaptive_escalation and args.gate == "none":
        raise ValueError("--adaptive-escalation requires --gate softmax or --gate openset")
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

    if args.gate != "none":
        X_val = val[CANONICAL_FEATURE_COLUMNS].to_numpy()
        if len(X_val) == 0:
            raise ValueError("--gate requires a non-empty validation split -- increase --val-frac or the input data size")

        if args.gate == "openset":
            gate = OpenMaxHead(stage2, OpenMaxConfig()).fit(X_train, train["attack_category"].tolist())
            gate.save(model_dir / OPENSET_GATE_FILENAME)
        else:
            gate = SoftmaxGate(stage2)  # no fitted state of its own; reconstructed fresh in serve.py

        calib_scores = [r.unknown_mass for r in gate.recalibrate_batch(X_val)]

        if args.adaptive_escalation:
            escalation_gate = AdaptiveConformalGate(
                budget=args.escalation_budget, gamma=args.adaptive_gamma, window_size=args.adaptive_window_size
            ).seed(calib_scores)
            escalation_gate.save(model_dir / ESCALATION_GATE_FILENAME)
            (model_dir / ESCALATION_KIND_FILENAME).write_text("adaptive")
            logger.info(
                "Fit %s gate, seeded adaptive escalation gate from %d validation rows: "
                "initial threshold=%.4f, budget=%.2f, gamma=%.4f, window_size=%d -- see "
                "adaptive_conformal_gate.py's module docstring for the production ground-truth-proxy "
                "assumption this online-updating gate is deployed under",
                args.gate,
                len(calib_scores),
                escalation_gate.threshold,
                args.escalation_budget,
                args.adaptive_gamma,
                args.adaptive_window_size,
            )
        else:
            escalation_gate = calibrate_threshold(calib_scores, budget=args.escalation_budget)
            escalation_gate.save(model_dir / ESCALATION_GATE_FILENAME)
            (model_dir / ESCALATION_KIND_FILENAME).write_text("static")
            logger.info(
                "Fit %s gate, calibrated on %d validation rows: escalation threshold=%.4f for budget=%.2f",
                args.gate,
                escalation_gate.n_calibration,
                escalation_gate.threshold,
                args.escalation_budget,
            )
        (model_dir / GATE_TYPE_FILENAME).write_text(args.gate)

    return 0


if __name__ == "__main__":
    sys.exit(main())
