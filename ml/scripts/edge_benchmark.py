"""Tier 1 cost on a constrained machine: a CONSTRAINED-CONTAINER PROXY for an
edge gateway, not a measurement on edge hardware.

Run the `bench` step in a container limited to, e.g., one CPU and 1 GB of
memory (docker run --cpus=1 --memory=1g). That bounds the core count and the
memory, but each core is still a laptop core: a Raspberry Pi 4 (Cortex-A72)
core is several times slower, so latency on real edge hardware will be
higher. Report the result as "1 CPU / 1 GB container", never as a Pi.

Two steps, so the timed process never trains:

  fit    Train the deployed Tier 1 configuration -- hybrid Stage 1, XGBoost
         Stage 2, kNN escalation gate with a conformal threshold -- on a
         time-based split of a canonical-form CSV, and save the models plus
         a sample of held-out flows. Run it in the same image as `bench`
         so the pickled models never cross library versions.
  bench  Load the models and time TwoStageDetector.score: one flow per call
         (how the scoring service runs) and batches; record peak memory,
         model size on disk, and the cgroup CPU and memory limits seen.

Usage (inside the ml image, repo's ml/ mounted at /work):
    python /work/scripts/edge_benchmark.py fit   --data /data/real_cicids2018_dedup.csv --out /models
    python /work/scripts/edge_benchmark.py bench --models /models --out /models/edge_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.conformal_gate import ConformalGate, calibrate_threshold  # noqa: E402
from ids_ml.evaluation import latency_report  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS  # noqa: E402
from ids_ml.ood_gates import KNNGate  # noqa: E402
from ids_ml.pipeline import TwoStageDetector  # noqa: E402
from ids_ml.stage1_hybrid import HybridConfig, HybridPreFilter, load_stage1  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config  # noqa: E402


def fit(args) -> int:
    import pandas as pd

    from ids_ml.split import time_based_split

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    if len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=0)
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_train = train["attack_category"].tolist()

    stage1 = HybridPreFilter(HybridConfig()).fit(X_train, train["is_attack"].to_numpy())
    stage2 = AttackClassifier(Stage2Config(n_estimators=args.stage2_n_estimators)).fit(X_train, y_train)
    gate = KNNGate(stage2).fit(X_train, y_train)
    X_val = val[CANONICAL_FEATURE_COLUMNS].to_numpy()
    flagged_val = X_val[stage1.flag(X_val)]
    conformal = calibrate_threshold(gate.scores(flagged_val), budget=args.budget)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage1.save(out / "stage1.joblib")
    stage2.save(out / "stage2_xgboost.joblib")
    gate.save(out / "knn_gate.joblib")
    conformal.save(out / "escalation_gate.joblib")
    sample = test.sample(min(args.n_sample, len(test)), random_state=0)
    np.save(out / "bench_flows.npy", sample[CANONICAL_FEATURE_COLUMNS].to_numpy())
    print(f"fit: train={len(train)} val_flagged={len(flagged_val)} threshold={conformal.threshold:.4f} -> {out}")
    return 0


def _cgroup_limits() -> dict:
    limits = {}
    for key, path in (("cpu.max", "/sys/fs/cgroup/cpu.max"), ("memory.max", "/sys/fs/cgroup/memory.max")):
        try:
            limits[key] = Path(path).read_text().strip()
        except OSError:
            limits[key] = None
    return limits


def _peak_rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB on Linux
    except ImportError:
        return float("nan")


def bench(args) -> int:
    models = Path(args.models)
    t0 = time.perf_counter()
    stage1 = load_stage1(models / "stage1.joblib")
    stage2 = AttackClassifier.load(models / "stage2_xgboost.joblib")
    gate = KNNGate.load(models / "knn_gate.joblib", stage2)
    escalation = ConformalGate.load(models / "escalation_gate.joblib")
    load_s = time.perf_counter() - t0
    detector = TwoStageDetector(stage1, stage2, gate=gate, escalation_gate=escalation, escalation_trigger_name="knn")
    X = np.load(models / "bench_flows.npy")

    results = {}
    for batch_size in (1, 32, 128):
        rep = latency_report(detector, X, warmup=50, n_trials=args.n_trials, batch_size=max(batch_size, 1))
        results[f"batch_{batch_size}"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in vars(rep).items()}

    summary = {
        "label": "constrained-container proxy (not edge hardware)",
        "cgroup_limits": _cgroup_limits(),
        "os_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model_load_seconds": round(load_s, 3),
        "model_bytes_on_disk": {p.name: p.stat().st_size for p in sorted(models.glob("*.joblib"))},
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "n_bench_flows": int(len(X)),
        "latency": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--max-train", type=int, default=100000)
    f.add_argument("--stage2-n-estimators", type=int, default=300)
    f.add_argument("--budget", type=float, default=0.1)
    f.add_argument("--n-sample", type=int, default=5000)
    b = sub.add_parser("bench")
    b.add_argument("--models", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-trials", type=int, default=500)
    args = p.parse_args(argv)
    return fit(args) if args.cmd == "fit" else bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
