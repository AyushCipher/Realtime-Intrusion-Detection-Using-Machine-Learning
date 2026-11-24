"""End-to-end miss-rate control for the cascade on real data (ids_ml/cascade_crc.py).

For each dataset, seed and split protocol, the deployed Tier 1 is fitted on the
train split (hybrid Stage 1, XGBoost Stage 2, kNN escalation gate whose
threshold is fixed at the escalation budget on SHAPING benign flows). Then four
calibrations of the end-to-end known-attack miss budget alpha are compared on
the same scores:

  separate        Stage 1 conformal at alpha, Stage 2 argmax (current practice)
  bonferroni      alpha/2 per stage
  joint_crc       conformal risk control along lockstep quantile maps
  joint_crc_cost  conformal risk control along a cost-aware monotone path

Splits: train 50% / shaping 15% / calibration 15% / test 20%, either a seeded
random split ("random": exchangeable, where the guarantee applies) or in time
order ("chronological": drift, outside the guarantee). Writes one row per
(dataset, protocol, seed, alpha, method) to results/paper/cascade_crc/.

Usage:
    PYTHONPATH=ml/src python ml/scripts/run_cascade_crc.py \\
        --data cse=C:/nids_data/real_cicids2018_dedup.csv unsw=C:/nids_data/unsw_canonical_dedup.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.cascade_crc import (  # noqa: E402
    CascadeScores,
    CostAwareJointCRC,
    JointCRC,
    calibrate_bonferroni,
    calibrate_separate,
    evaluate,
)
from ids_ml.conformal_gate import calibrate_threshold  # noqa: E402
from ids_ml.evaluation import make_stage1  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.ood_gates import KNNGate  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config  # noqa: E402

logger = logging.getLogger("run_cascade_crc")
FRACS = (0.50, 0.15, 0.15)  # train, shaping, calibration; the rest is test


def split(df: pd.DataFrame, protocol: str, seed: int):
    if protocol == "random":
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    a = int(n * FRACS[0])
    b = a + int(n * FRACS[1])
    c = b + int(n * FRACS[2])
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:c], df.iloc[c:]


def scores_for(part: pd.DataFrame, stage1, stage2, gate, esc_threshold: float, benign_idx: int) -> CascadeScores:
    X = part[FEATS].to_numpy()
    proba = stage2.predict_proba(X)
    return CascadeScores(
        s1=stage1.anomaly_score(X),
        p_benign=proba[:, benign_idx],
        argmax_benign=proba.argmax(axis=1) == benign_idx,
        escalated=gate.scores(X) > esc_threshold,
    )


def run_one(df, dataset, protocol, seed, alphas, args):
    train, shaping, calib, test = split(df, protocol, seed)
    if (calib["attack_category"] == "BENIGN").all() or (test["attack_category"] == "BENIGN").all():
        logger.warning("%s/%s seed %d: no attacks in calibration or test; skipped", dataset, protocol, seed)
        return []
    stage1 = make_stage1("hybrid", train, FEATS, seed=seed)
    X_train, y_train = train[FEATS].to_numpy(), train["attack_category"].tolist()
    stage2 = AttackClassifier(Stage2Config(n_estimators=args.stage2_n_estimators, random_state=seed)).fit(X_train, y_train)
    if "BENIGN" not in stage2.classes_:
        return []
    benign_idx = stage2.classes_.index("BENIGN")
    gate = KNNGate(stage2, random_state=seed).fit(X_train, y_train)
    shaping_benign_X = shaping[shaping["attack_category"] == "BENIGN"][FEATS].to_numpy()
    esc_threshold = calibrate_threshold(gate.scores(shaping_benign_X), budget=args.escalation_budget).threshold

    def sc(part):
        return scores_for(part, stage1, stage2, gate, esc_threshold, benign_idx)

    sh_att = sc(shaping[shaping["attack_category"] != "BENIGN"])
    sh_ben = sc(shaping[shaping["attack_category"] == "BENIGN"])
    cal_att = sc(calib[calib["attack_category"] != "BENIGN"])
    test_sc = sc(test)
    is_attack = (test["attack_category"] != "BENIGN").to_numpy()

    joint = JointCRC(sh_att)
    cost_aware = CostAwareJointCRC(sh_att, sh_ben)
    rows = []
    for alpha in alphas:
        for d in (calibrate_separate(cal_att, alpha), calibrate_bonferroni(cal_att, alpha),
                  joint.calibrate(cal_att, alpha), cost_aware.calibrate(cal_att, alpha)):
            r = evaluate(d, test_sc, is_attack)
            r.update(dataset=dataset, protocol=protocol, seed=seed, alpha=alpha,
                     n_calib_attacks=int(len(cal_att.s1)), n_test_attacks=int(is_attack.sum()),
                     n_test_benign=int((~is_attack).sum()), violated=bool(r["end_to_end_miss"] > alpha))
            rows.append(r)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="name=path pairs of canonical-form CSVs")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--protocols", default="random,chronological")
    p.add_argument("--alphas", default="0.05,0.1")
    p.add_argument("--escalation-budget", type=float, default=0.1)
    p.add_argument("--stage2-n-estimators", type=int, default=200)
    p.add_argument("--max-rows", type=int, default=300000, help="seeded subsample per dataset, for memory")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent / "results" / "paper" / "cascade_crc" / "cascade_crc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    alphas = [float(a) for a in args.alphas.split(",")]

    for spec in args.data:
        name, path = spec.split("=", 1)
        df = pd.read_csv(path, parse_dates=["timestamp"]).dropna(subset=["timestamp"])
        if len(df) > args.max_rows:
            df = df.sample(args.max_rows, random_state=0)
        logger.info("%s: %d rows, %d attacks", name, len(df), int((df["attack_category"] != "BENIGN").sum()))
        for protocol in args.protocols.split(","):
            for seed in range(args.seeds):
                if len(done) and ((done["dataset"] == name) & (done["protocol"] == protocol) & (done["seed"] == seed)).any():
                    continue  # resumable: this (dataset, protocol, seed) is already in the CSV
                t0 = time.time()
                rows = run_one(df, name, protocol, seed, alphas, args)
                if rows:
                    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                    done.to_csv(out, index=False)
                logger.info("%s/%s seed %d done in %.0fs", name, protocol, seed, time.time() - t0)

    summary = done.groupby(["dataset", "protocol", "alpha", "method"]).agg(
        miss_mean=("end_to_end_miss", "mean"), miss_max=("end_to_end_miss", "max"),
        violation_rate=("violated", "mean"), benign_alert=("benign_alert_rate", "mean"),
        benign_flag=("benign_flag_rate", "mean"), seeds=("seed", "nunique")).round(4)
    summary.to_csv(out.with_name("cascade_crc_summary.csv"))
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
