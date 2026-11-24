"""Detection metrics of a TRAINED, DEPLOYED model directory on its own held-out
test split: does each attack raise any alert, and how much benign traffic does?

train.py's per-category reports score exact-class accuracy with Stage 1's
default threshold. What an operator needs is the deployed decision: with the
saved thresholds (joint cascade calibration, gate, escalation threshold), does
an attack flow end as an alert (known-attack or escalated), and how often does
a benign flow? This rebuilds the same split train.py used (same data, split
kind and seed) and scores its test part through serve.py's loading path.

Usage (inside the ml image, the training data mounted at /data):
    python /work/scripts/eval_deployed_model.py --data /data/site_train_combined.csv \\
        --model-dir /app/site_models --split random --out /data/deployed_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.features import CANONICAL_FEATURE_COLUMNS  # noqa: E402
from ids_ml.pipeline import TwoStageDetector  # noqa: E402
from ids_ml.serve import load_gate  # noqa: E402
from ids_ml.split import random_split, time_based_split  # noqa: E402
from ids_ml.stage1_hybrid import load_stage1  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier  # noqa: E402
from ids_ml.train import CASCADE_CALIBRATION_FILENAME, _load_training_data  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--split", choices=["time", "random"], default="random")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    df = _load_training_data(args.data)
    if args.split == "random":
        _, _, test = random_split(df, train_frac=args.train_frac, val_frac=args.val_frac, random_state=0)
    else:
        _, _, test = time_based_split(df, train_frac=args.train_frac, val_frac=args.val_frac)

    model_dir = Path(args.model_dir)
    stage1 = load_stage1(model_dir / "stage1_iforest.joblib")
    stage2 = AttackClassifier.load(model_dir / "stage2_xgboost.joblib")
    gate, escalation_gate, name = load_gate(model_dir, stage2)
    cal_path = model_dir / CASCADE_CALIBRATION_FILENAME
    silence = json.loads(cal_path.read_text())["benign_silence_threshold"] if cal_path.exists() else None
    # A static copy of the escalation decision: the adaptive gate would adapt
    # while scoring a shuffled test split, which is not how it runs live.
    if hasattr(escalation_gate, "alpha_t"):
        from ids_ml.conformal_gate import ConformalGate

        escalation_gate = ConformalGate(threshold=escalation_gate.threshold, budget=escalation_gate.budget, n_calibration=0)
    detector = TwoStageDetector(stage1, stage2, gate=gate, escalation_gate=escalation_gate,
                                escalation_trigger_name=name, benign_silence_threshold=silence)

    X = test[CANONICAL_FEATURE_COLUMNS].to_numpy()
    decisions, escalated = [], []
    for start in range(0, len(X), 5000):
        for s in detector.score(X[start : start + 5000]):
            decisions.append(s.decision)
            escalated.append(s.escalated)
    decisions = np.array(decisions)
    alerted = decisions != "known-benign"
    is_attack = test["is_attack"].to_numpy()
    fam = test["attack_category"].to_numpy()

    per_family = {}
    for f in sorted(set(fam[is_attack])):
        m = fam == f
        per_family[f] = {"n": int(m.sum()), "detected": round(float(alerted[m].mean()), 4),
                         "escalated": round(float(np.array(escalated)[m].mean()), 4)}
    summary = {
        "model_dir": str(model_dir),
        "split": args.split,
        "n_test": int(len(test)),
        "n_attacks": int(is_attack.sum()),
        "attack_detection_rate": round(float(alerted[is_attack].mean()), 4),
        "attack_miss_rate": round(float((~alerted[is_attack]).mean()), 4),
        "benign_alert_rate": round(float(alerted[~is_attack].mean()), 4),
        "benign_escalation_rate": round(float(np.array(escalated)[~is_attack].mean()), 4),
        "per_family": per_family,
        "per_dataset": {d: {"miss": round(float((~alerted[(test["dataset"] == d).to_numpy() & is_attack]).mean()), 4),
                            "benign_alert": round(float(alerted[(test["dataset"] == d).to_numpy() & ~is_attack].mean()), 4)}
                        for d in sorted(test["dataset"].unique())} if "dataset" in test.columns else {},
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
