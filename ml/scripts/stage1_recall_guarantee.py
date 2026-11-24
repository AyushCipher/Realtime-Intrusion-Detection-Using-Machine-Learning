"""Evaluate conformal recall control for Stage 1 (see ids_ml/conformal_stage1.py).

Two analyses on the local synthetic fixture (use real CSVs via --data for a
real-data run):

 1. chronological: calibrate the Stage 1 flag threshold on the validation
    split's attack flows for a range of miss budgets, then measure on the
    held-out test split the observed attack miss rate, the share of traffic
    flagged (load passed to Stage 2) and end-to-end cascade accuracy / macro
    F1. The default IsolationForest boundary is the baseline.
 2. monte_carlo: pool validation + test attack scores and repeat many random
    calibration/test splits, so the observed miss rate can be compared with
    the budget under the exchangeability the guarantee assumes. This is a
    validity check of the bound, not a drift experiment.

Writes results/stage1_recall_guarantee.csv and
results/stage1_recall_guarantee_montecarlo.csv.

Usage:
    PYTHONPATH=ml/src python ml/scripts/stage1_recall_guarantee.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.conformal_stage1 import calibrate_stage1_threshold  # noqa: E402
from ids_ml.data import load_and_map  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS  # noqa: E402
from ids_ml.split import time_based_split  # noqa: E402
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage1_recall_guarantee")

BUDGETS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def cascade_metrics(stage1, stage2, X, y):
    y = np.array(y)
    flagged = stage1.flag(X)
    pred = np.full(len(X), "BENIGN", dtype=object)
    if flagged.any():
        pred[flagged] = stage2.predict(X[flagged])
    is_attack = y != "BENIGN"
    labels = sorted(set(y) | set(pred.tolist()))
    _p, _r, f1, _s = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    return {
        "attack_miss_rate": round(float((~flagged & is_attack).sum() / max(is_attack.sum(), 1)), 4),
        "flag_rate": round(float(flagged.mean()), 4),
        "benign_flagged": int((flagged & ~is_attack).sum()),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1), 4),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", nargs="+", default=[str(root / "ml" / "tests" / "fixtures" / "synthetic_cicids_sample.csv")])
    parser.add_argument("--contamination", type=float, default=0.35)
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo splits")
    args = parser.parse_args()

    df = load_and_map(args.data)
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    feats = lambda d: d[CANONICAL_FEATURE_COLUMNS].to_numpy()  # noqa: E731
    X_train, y_train = feats(train), train["attack_category"].tolist()
    X_test, y_test = feats(test), test["attack_category"].tolist()
    val_attacks, test_attacks = feats(val[val["is_attack"]]), feats(test[test["is_attack"]])
    logger.info("train=%d val=%d (attacks %d) test=%d (attacks %d)", len(train), len(val), len(val_attacks), len(test), len(test_attacks))

    stage2 = AttackClassifier(Stage2Config()).fit(X_train, y_train)
    stage1 = AnomalyPreFilter(Stage1Config(contamination=args.contamination)).fit(X_train)

    rows = [{"setting": "baseline (IsolationForest boundary)", "miss_budget": None, "guaranteed_miss_rate": None, **cascade_metrics(stage1, stage2, X_test, y_test)}]
    val_scores = stage1.anomaly_score(val_attacks)
    for b in BUDGETS:
        g = calibrate_stage1_threshold(val_scores, b)
        pinned = AnomalyPreFilter(Stage1Config(contamination=args.contamination, flag_threshold=g.threshold))
        pinned.scaler, pinned.model, pinned._fitted = stage1.scaler, stage1.model, True
        rows.append({"setting": "conformal", "miss_budget": b, "guaranteed_miss_rate": round(g.guaranteed_miss_rate, 4), **cascade_metrics(pinned, stage2, X_test, y_test)})
    chrono = pd.DataFrame(rows)
    chrono.to_csv(root / "results" / "stage1_recall_guarantee.csv", index=False)
    print(chrono.to_string(index=False))

    pool = np.concatenate([val_scores, stage1.anomaly_score(test_attacks)])
    rng = np.random.default_rng(0)
    mc = []
    for b in BUDGETS:
        misses = []
        for _ in range(args.trials):
            idx = rng.permutation(len(pool))
            cal, hold = pool[idx[: len(pool) // 2]], pool[idx[len(pool) // 2 :]]
            g = calibrate_stage1_threshold(cal, b)
            misses.append(float((hold < g.threshold).mean()))
        mc.append({"miss_budget": b, "n_pooled_attacks": len(pool), "trials": args.trials, "mean_observed_miss_rate": round(float(np.mean(misses)), 4)})
    mc_df = pd.DataFrame(mc)
    mc_df.to_csv(root / "results" / "stage1_recall_guarantee_montecarlo.csv", index=False)
    print(mc_df.to_string(index=False))


if __name__ == "__main__":
    main()
