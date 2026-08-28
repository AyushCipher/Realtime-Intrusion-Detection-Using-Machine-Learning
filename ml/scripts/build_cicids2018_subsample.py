"""Builds a manageable, documented subsample of CSE-CIC-IDS2018's "Processed
Traffic Data for ML Algorithms" per-day CSVs (the AWS Open Data release:
https://registry.opendata.aws/cse-cic-ids2018/), for `ids_ml.evaluation`'s
real-data H1 (open-set vs. softmax) and H2 (static vs. adaptive conformal)
experiments.

Why this exists rather than just calling `data.load_and_map_2018` directly
on the raw files:

1. **Memory.** The 8 per-day CSVs this project has verified column/label
   compatibility with (see `ids_ml.data`'s CICIDS2018 support) total
   ~7.5M rows. Concatenating them raw before mapping OOM'd in the
   environment this was built in. This script maps and subsamples one
   file at a time, only ever holding one day's data (plus the running
   subsample) in memory.
2. **Runtime.** Every downstream experiment (LOFO rotation x seeds,
   drift x seeds) refits a classifier per trial -- tractable on ~300k
   rows in a single session, not on 7.5M.

Subsampling rule, applied per raw label within each file: keep every row
if there are <= `--cap` of that label in that file (default 20,000), else
take a uniform random sample of `--cap` (seeded). This bounds flood-heavy
categories (e.g. DDOS-HOIC's 686k rows in one file) to a manageable size
without under-representing genuinely rare ones (e.g. Web Attack's low
hundreds per file), and caps BENIGN the same way per file rather than
keeping all of it.

Usage:
    python -m scripts.build_cicids2018_subsample \\
        --input-dir /path/to/downloaded/cicids2018/csvs \\
        --output real_cicids2018_subsample.csv

The 8 files this was built and verified against (attack scenario in
parentheses): Wednesday-14-02-2018 (Brute Force), Thursday-15-02-2018
(DoS), Friday-16-02-2018 (DoS), Wednesday-21-02-2018 (DDoS),
Thursday-22-02-2018 (Web Attack), Friday-23-02-2018 (Web Attack),
Thursday-01-03-2018 (Infiltration), Friday-02-03-2018 (Botnet). Download
these yourself from the AWS bucket above -- this repo does not ship them
(multi-GB, separately licensed, same policy as the rest of this module's
CICIDS2017/2018 handling).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ids_ml.data import load_cicids2018_csv, map_to_canonical

logger = logging.getLogger("ids_ml.scripts.build_cicids2018_subsample")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory containing the downloaded per-day CSVs")
    parser.add_argument("--output", required=True, help="Path to write the subsampled, canonical-form CSV to")
    parser.add_argument("--cap", type=int, default=20_000, help="Max rows kept per (file, raw label) pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def build_subsample(input_dir: Path, cap: int, seed: int) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"no CSV files found in {input_dir}")
    logger.info("Processing %d files: %s", len(files), [f.name for f in files])

    rng = np.random.default_rng(seed)
    pieces = []
    for f in files:
        logger.info("Loading %s ...", f.name)
        raw = load_cicids2018_csv(f)
        mapped = map_to_canonical(raw, dayfirst=True)  # see ids_ml.data.load_and_map_2018 for why
        del raw

        kept_parts = []
        for label, group in mapped.groupby("label"):
            if len(group) > cap:
                idx = rng.choice(len(group), size=cap, replace=False)
                kept_parts.append(group.iloc[idx])
            else:
                kept_parts.append(group)
        sub = pd.concat(kept_parts, ignore_index=True)
        del mapped
        logger.info("  -> kept %d rows: %s", len(sub), dict(sub["label"].value_counts()))
        pieces.append(sub)

    full = pd.concat(pieces, ignore_index=True)
    return full.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    full = build_subsample(Path(args.input_dir), cap=args.cap, seed=args.seed)
    logger.info("Total rows: %d", len(full))
    logger.info("%s", full["attack_category"].value_counts().to_string())

    out_path = Path(args.output)
    full.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
