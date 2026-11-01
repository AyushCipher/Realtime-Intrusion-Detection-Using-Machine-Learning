"""Sweep Stage 1 (Isolation Forest) sensitivity and report the recall/speed trade-off.

The cascade's weakest point is Stage 1: any attack it passes as benign never
reaches Stage 2 (on the 185-flow test split, 2 PortScan and 1 Infiltration
flows are lost this way). This script measures how Stage 1's sensitivity
setting changes that, on the same chronological split used by
`evaluate_and_compare.py`.

Two sweeps are run:
  * contamination: IsolationForest's own boundary (predict() == -1)
  * threshold_quantile: contamination fixed at 0.35, with `flag_threshold` set
    to a quantile of the training anomaly scores (lower quantile = flags more)

For each setting it reports the share of traffic Stage 1 flags (the load
passed on to Stage 2), how many attack flows it lets through as benign, and
the end-to-end cascade accuracy / macro recall / macro F1 when Stage 2 only
sees flagged flows. Writes results/stage1_threshold_sweep.csv.

Usage:
    PYTHONPATH=ml/src python ml/scripts/stage1_threshold_sweep.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.data import load_and_map  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS  # noqa: E402
from ids_ml.split import time_based_split  # noqa: E402
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage1_threshold_sweep")

BENIGN = "BENIGN"
CONTAMINATIONS = [0.05, 0.10, 0.20, 0.35, 0.50]
THRESHOLD_QUANTILES = [0.50, 0.60, 0.70, 0.80, 0.90]


def _evaluate(stage1: AnomalyPreFilter, stage2: AttackClassifier, X: np.ndarray, y: list) -> dict:
    y_arr = np.array(y)
    flagged = stage1.flag(X)

    pred = np.full(len(X), BENIGN, dtype=object)
    if flagged.any():
        pred[flagged] = stage2.predict(X[flagged])

    is_attack = y_arr != BENIGN
    labels = sorted(set(y) | set(pred.tolist()))
    _p, recall, f1, _s = precision_recall_fscore_support(
        y_arr, pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "flag_rate": round(float(flagged.mean()), 4),
        "benign_flagged": int((flagged & ~is_attack).sum()),
        "attacks_missed_by_stage1": int((~flagged & is_attack).sum()),
        "attacks_total": int(is_attack.sum()),
        "stage1_attack_recall": round(float(flagged[is_attack].mean()), 4) if is_attack.any() else 0.0,
        "accuracy": round(float(accuracy_score(y_arr, pred)), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "ml" / "tests" / "fixtures" / "synthetic_cicids_sample.csv"
    out_path = repo_root / "results" / "stage1_threshold_sweep.csv"

    df = load_and_map([str(data_path)])
    train_df, _val_df, test_df = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train = train_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_train = train_df["attack_category"].tolist()
    X_test = test_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_test = test_df["attack_category"].tolist()
    logger.info("Train %d flows, test %d flows", len(X_train), len(X_test))

    stage2 = AttackClassifier(Stage2Config()).fit(X_train, y_train)

    rows = []
    for c in CONTAMINATIONS:
        stage1 = AnomalyPreFilter(Stage1Config(contamination=c)).fit(X_train)
        rows.append({"sweep": "contamination", "setting": c, **_evaluate(stage1, stage2, X_test, y_test)})

    base = AnomalyPreFilter(Stage1Config(contamination=0.35)).fit(X_train)
    train_scores = base.anomaly_score(X_train)
    for q in THRESHOLD_QUANTILES:
        threshold = float(np.quantile(train_scores, q))
        stage1 = AnomalyPreFilter(Stage1Config(contamination=0.35, flag_threshold=threshold))
        stage1.scaler, stage1.model, stage1._fitted = base.scaler, base.model, True
        rows.append({"sweep": "threshold_quantile", "setting": q, **_evaluate(stage1, stage2, X_test, y_test)})

    table = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
