"""Does the Stage 1 recall guarantee survive an adaptive attacker, and does
traffic-aware calibration fix it?

Threat model (feature-space, black-box to the threshold): after calibration, an
attacker pads or paces its own flows -- changing features it controls
(packet lengths, byte totals, timing) without changing the attack's action.
Two perturbations are used, both heuristic, not packet-level realisable:
  slow      evaluation.simulate_low_and_slow with factor s (timing stretched)
  pad       packet-length and byte columns scaled by (1 + p) (payload padding)
  slowpad   both together
  mimic     controllable features moved a fraction lam toward a random real benign flow (benign-aware)

Two calibrations of the same Stage 1 miss-budget threshold are compared:
  standard        calibrated on clean validation attacks (what conformal_stage1 does)
  traffic_aware   calibrated on validation attacks perturbed by the SAME law the
                  attacker is expected to use (a mixture over the settings above),
                  following the traffic-aware calibration idea of Robust Conformal
                  Intrusion Detection (arXiv:2609.19241), here applied to Stage 1.

For every perturbation setting it reports, on chronological test attacks, the
realised miss rate against the budget and the share of benign flows flagged
(the cost of the extra sensitivity). Stage 1 is the current hybrid pre-filter by default (--stage1-kind iforest for the original).
Writes results/paper/<name>/adversarial_calibration.csv
and a manifest. The guarantee for traffic_aware holds for the calibration LAW;
a setting outside it (e.g. slow x32) is reported but not covered.

Usage:
    PYTHONPATH=ml/src python ml/scripts/run_adversarial_calibration.py --data real.csv --name cicids2018_dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.conformal_stage1 import calibrate_stage1_threshold  # noqa: E402
from ids_ml.evaluation import simulate_low_and_slow  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402
from ids_ml.split import random_split, time_based_split  # noqa: E402
from ids_ml.evaluation import make_stage1  # noqa: E402

_PAD_COLS = [
    "total_fwd_bytes", "total_bwd_bytes", "fwd_packet_length_min", "fwd_packet_length_max", "fwd_packet_length_mean",
    "bwd_packet_length_min", "bwd_packet_length_max", "bwd_packet_length_mean", "flow_bytes_per_sec",
]


def pad(df: pd.DataFrame, p: float) -> pd.DataFrame:
    out = df.copy()
    for c in _PAD_COLS:
        out[c] = out[c] * (1.0 + p)
    return out


# Features an attacker controls by padding, pacing or splitting its own traffic. Flag counts
# stay fixed (they follow from the protocol exchange itself).
_MIMIC_COLS = [c for c in FEATS if not c.endswith("_flag_count")]
BENIGN_POOL: dict = {}


def mimic(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Move each attack flow's controllable features a fraction `lam` of the way toward a
    randomly drawn REAL benign training flow. Interpolating toward a real flow (not a
    per-feature median, which is not a flow any network produces) is a benign-aware
    attacker stronger than pad/slow; at lam -> 1 the flow becomes that benign flow."""
    ref = BENIGN_POOL["X"]
    rng = np.random.default_rng(BENIGN_POOL["seed"])
    target = ref.iloc[rng.integers(0, len(ref), size=len(df))].to_numpy()
    out = df.copy()
    cols = [FEATS.index(c) for c in _MIMIC_COLS]
    vals = out[FEATS].to_numpy(dtype=float)
    vals[:, cols] = (1.0 - lam) * vals[:, cols] + lam * target[:, cols]
    out[FEATS] = vals
    return out


def perturb(df: pd.DataFrame, kind: str, strength: float) -> pd.DataFrame:
    if kind == "mimic":
        return mimic(df, strength)
    if kind == "none":
        return df
    if kind == "slow":
        return simulate_low_and_slow(df, strength)
    if kind == "pad":
        return pad(df, strength)
    if kind == "slowpad":
        return pad(simulate_low_and_slow(df, strength), 1.0)
    raise ValueError(kind)


SETTINGS = [("none", 1.0), ("slow", 2.0), ("slow", 4.0), ("slow", 8.0), ("pad", 0.5), ("pad", 1.0), ("slowpad", 4.0),
            ("mimic", 0.25), ("mimic", 0.5), ("slow", 32.0), ("mimic", 0.75), ("mimic", 0.9)]
LAW = SETTINGS[:9]  # calibration law; slow x32 and mimicry at 0.75 / 0.9 are deliberately outside it


def law_sample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Each attack row independently gets one setting drawn uniformly from LAW."""
    idx = rng.integers(0, len(LAW), size=len(df))
    parts = []
    for k, (kind, s) in enumerate(LAW):
        sub = df.iloc[np.where(idx == k)[0]]
        if len(sub):
            parts.append(perturb(sub, kind, s))
    return pd.concat(parts)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--stage1-kind", choices=["hybrid", "iforest"], default="hybrid")
    ap.add_argument("--split", choices=["chronological", "random"], default="chronological",
                    help="random = known families, where Stage 1 actually filters and evasion is measurable")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    if args.split == "random":
        train, val, test = random_split(df, train_frac=0.7, val_frac=0.15, random_state=0)
    else:
        train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    BENIGN_POOL.update(X=train[train["attack_category"] == "BENIGN"][FEATS], seed=12345)
    val_att = val[val["attack_category"] != "BENIGN"]
    test_att = test[test["attack_category"] != "BENIGN"]
    test_ben = test[test["attack_category"] == "BENIGN"]
    print(f"train={len(train)} val_attacks={len(val_att)} test_attacks={len(test_att)} test_benign={len(test_ben)}", flush=True)

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        s1 = make_stage1(args.stage1_kind, train, FEATS, seed=seed, contamination=0.35)
        std_scores = s1.anomaly_score(val_att[FEATS].to_numpy())
        ta_scores = s1.anomaly_score(law_sample(val_att, rng)[FEATS].to_numpy())
        ben_scores = s1.anomaly_score(test_ben[FEATS].to_numpy())
        for budget in args.budgets:
            thr = {"standard": calibrate_stage1_threshold(std_scores, budget).threshold,
                   "traffic_aware": calibrate_stage1_threshold(ta_scores, budget).threshold}
            for kind, s in SETTINGS:
                att_scores = s1.anomaly_score(perturb(test_att, kind, s)[FEATS].to_numpy())
                for calib, t in thr.items():
                    rows.append({
                        "seed": seed, "budget": budget, "calibration": calib, "perturbation": kind, "strength": s,
                        "in_calibration_law": (kind, s) in LAW,
                        "miss_rate": float((att_scores < t).mean()), "benign_flag_rate": float((ben_scores >= t).mean()),
                    })
        print(f"seed {seed} done", flush=True)

    res = pd.DataFrame(rows)
    out_dir = root / "results" / "paper" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "adversarial_calibration_raw.csv", index=False)
    agg = (res.groupby(["budget", "calibration", "perturbation", "strength", "in_calibration_law"])[["miss_rate", "benign_flag_rate"]]
           .agg(["mean", "std"]).round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "adversarial_calibration.csv", index=False)
    write_manifest(out_dir / "manifest_adversarial.json", inputs=[args.data], seeds=range(args.seeds),
                   outputs=[out_dir / "adversarial_calibration.csv"], extra={"settings": SETTINGS, "law": LAW, "stage1_kind": args.stage1_kind, "split": args.split}, repo_dir=root)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
