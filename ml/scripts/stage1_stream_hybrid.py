"""Stage 1 recall control over a time-ordered stream, with the hybrid pre-filter.

The stream is the data in time order: the first 70% fits the hybrid pre-filter,
the next 15% calibrates, and the last 15% is scored flow by flow. On
CSE-CIC-IDS2018 the families differ by day, so the test families are new.

Calibration methods compared at the same miss budget:
  pooled        one conformal threshold over all calibration attacks (conformal_stage1)
  per_family    per-family (Mondrian) thresholds, lowest kept; families whose threshold
                would flag more than --benign-cap of calibration benign flows are waived
                and reported (conformal_stage1.calibrate_stage1_per_family)
  adaptive_dD   ACI on confirmed-attack feedback (adaptive_stage1), feedback for an
                attack arriving D flows after it was scored (triage delay)

Reported per method: realised miss rate overall and per test family, the share of
benign flows flagged, and for per_family which families were waived. The adaptive
method bounds the long-run miss rate on the attacks it gets feedback for, new
families included, without exchangeability; that is the price of needing labels.

Writes results/paper/<name>/stage1_stream_hybrid.csv and a manifest.

Usage:
    PYTHONPATH=ml/src python ml/scripts/stage1_stream_hybrid.py --data real.csv --name cicids2018_dedup
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.adaptive_stage1 import AdaptiveStage1Threshold  # noqa: E402
from ids_ml.conformal_stage1 import calibrate_stage1_per_family, calibrate_stage1_threshold  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402
from ids_ml.split import time_based_split  # noqa: E402
from ids_ml.stage1_hybrid import HybridConfig, HybridPreFilter  # noqa: E402


def summarise(method, budget, seed, flagged, te, extra=None):
    att = (te["attack_category"] != "BENIGN").to_numpy()
    row = {"method": method, "budget": budget, "seed": seed,
           "miss_rate": float((~flagged[att]).mean()) if att.any() else float("nan"),
           "benign_flag_rate": float(flagged[~att].mean())}
    for fam in sorted(set(te["attack_category"]) - {"BENIGN"}):
        m = (te["attack_category"] == fam).to_numpy()
        row[f"miss_{fam}"] = float((~flagged[m]).mean())
    row.update(extra or {})
    return row


def run_adaptive(scores, is_att, seed_scores, budget, delay, gamma, window):
    ctl = AdaptiveStage1Threshold(miss_budget=budget, gamma=gamma, window_size=window).seed(seed_scores)
    pending = deque()  # (arrival_index, score, threshold_at_decision)
    flagged = np.zeros(len(scores), dtype=bool)
    for i, s in enumerate(scores):
        while pending and pending[0][0] <= i:
            _, ps, pt = pending.popleft()
            ctl.update(ps, pt)
        t = ctl.threshold
        flagged[i] = s >= t
        if is_att[i]:
            pending.append((i + delay, s, t))
    return flagged


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--benign-cap", type=float, default=0.30)
    ap.add_argument("--delays", type=int, nargs="+", default=[0, 1000])
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--cap-per-label", type=int, default=0)
    ap.add_argument("--timeseries-window", type=int, default=500, help="attacks per point in the miss-rate-over-time series (seed 0)")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    if args.cap_per_label:
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), args.cap_per_label), random_state=0))
    df = df.sort_values("timestamp").reset_index(drop=True)
    tr, va, te = time_based_split(df, train_frac=0.7, val_frac=0.15)
    print("val families", va["attack_category"].value_counts().to_dict(), "| test families", te["attack_category"].value_counts().to_dict(), flush=True)

    rows, waived_log, ts_rows = [], [], []
    for seed in range(args.seeds):
        h = HybridPreFilter(HybridConfig(random_state=seed)).fit(tr[FEATS].to_numpy(), (tr["attack_category"] != "BENIGN").to_numpy())
        s_te = h.anomaly_score(te[FEATS].to_numpy())
        is_att = (te["attack_category"] != "BENIGN").to_numpy()
        va_att = va[va["attack_category"] != "BENIGN"]
        s_va_att = h.anomaly_score(va_att[FEATS].to_numpy())
        s_va_ben = h.anomaly_score(va[va["attack_category"] == "BENIGN"][FEATS].to_numpy())
        by_fam = {f: s_va_att[(va_att["attack_category"] == f).to_numpy()] for f in sorted(set(va_att["attack_category"]))}
        for b in args.budgets:
            t = calibrate_stage1_threshold(s_va_att, b).threshold
            rows.append(summarise("pooled", b, seed, s_te >= t, te))
            g = calibrate_stage1_per_family(by_fam, b, benign_scores=s_va_ben, max_benign_flag_rate=args.benign_cap)
            rows.append(summarise("per_family", b, seed, s_te >= g.threshold, te,
                                  {"guaranteed": "|".join(g.guaranteed_families), "waived": "|".join(g.waived_families)}))
            waived_log.append({"budget": b, "seed": seed, "guaranteed": g.guaranteed_families, "waived": g.waived_families,
                               "uncertifiable": g.uncertifiable_families, "calib_benign_flag_rate": g.benign_flag_rate})
            flags_for_ts = {"pooled": s_te >= t, "per_family": s_te >= g.threshold}
            for d in args.delays:
                fl = run_adaptive(s_te, is_att, s_va_att, b, d, args.gamma, args.window)
                rows.append(summarise(f"adaptive_d{d}", b, seed, fl, te))
                flags_for_ts[f"adaptive_d{d}"] = fl
            if seed == 0:
                att_idx = np.where(is_att)[0]
                w = args.timeseries_window
                for m, fl in flags_for_ts.items():
                    for k in range(0, len(att_idx) - w + 1, w):
                        blk = att_idx[k : k + w]
                        ts_rows.append({"budget": b, "method": m, "attack_index": int(k + w),
                                        "family": te["attack_category"].iloc[blk].mode().iloc[0],
                                        "window_miss_rate": float((~fl[blk]).mean())})
        print(f"seed {seed} done", flush=True)

    res = pd.DataFrame(rows)
    out = root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "stage1_stream_hybrid_raw.csv", index=False)
    num = [c for c in res.columns if c.startswith("miss") or c == "benign_flag_rate"]
    summ = res.groupby(["budget", "method"])[num].mean().round(3).reset_index()
    summ.to_csv(out / "stage1_stream_hybrid.csv", index=False)
    (out / "stage1_stream_hybrid_waived.json").write_text(json.dumps(waived_log, indent=2))
    pd.DataFrame(ts_rows).to_csv(out / "stage1_stream_hybrid_timeseries.csv", index=False)
    write_manifest(out / "manifest_stage1_stream_hybrid.json", inputs=[args.data], outputs=[out / "stage1_stream_hybrid.csv"],
                   seeds=range(args.seeds), extra=vars(args), repo_dir=root)
    print(summ.to_string(index=False))
    print("waived/guaranteed (seed 0):", [w for w in waived_log if w["seed"] == 0])


if __name__ == "__main__":
    main()
