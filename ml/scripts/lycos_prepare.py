"""Convert labelled LYCOS-IDS2017 CSVs into this project's canonical form.

LYCOS-IDS2017 (Rosay et al., "From CIC-IDS2017 to LYCOS-IDS2017: A corrected
dataset for better performance", WI-IAT 2021) re-extracts flow features from the
CIC-IDS2017 pcaps with LycoSTand and relabels them. Input here is the output of
the authors' own `labelling.py` (one CSV per weekday). This script:

  * maps LycoSTand's columns onto the 35 canonical features (durations and
    inter-arrival times are microseconds in LYCOS and seconds in the canonical form),
  * maps the 15 LYCOS labels onto the project's attack families, keeping the raw label,
  * drops rows with non-finite values, and optionally exact duplicates and
    conflicting-label groups (same procedure as dataset_audit.py --write-dedup),
  * writes a canonical CSV with `timestamp`, `label`, `attack_category`, `is_attack`.

Usage:
    python ml/scripts/lycos_prepare.py --input-dir C:/nids_data/lycos/lycos-ids2017/lycos-ids2017 \
        --output C:/nids_data/lycos2017_canonical.csv --dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402

US = 1e-6
# canonical column -> (LYCOS column, scale)
MAP = {
    "flow_duration": ("flow_duration", US),
    "total_fwd_packets": ("fwd_pkt_cnt", 1), "total_bwd_packets": ("bwd_pkt_cnt", 1),
    "total_fwd_bytes": ("fwd_pkt_len_tot", 1), "total_bwd_bytes": ("bwd_pkt_len_tot", 1),
    "fwd_packet_length_min": ("fwd_pkt_len_min", 1), "fwd_packet_length_max": ("fwd_pkt_len_max", 1),
    "fwd_packet_length_mean": ("fwd_pkt_len_mean", 1), "fwd_packet_length_std": ("fwd_pkt_len_std", 1),
    "bwd_packet_length_min": ("bwd_pkt_len_min", 1), "bwd_packet_length_max": ("bwd_pkt_len_max", 1),
    "bwd_packet_length_mean": ("bwd_pkt_len_mean", 1), "bwd_packet_length_std": ("bwd_pkt_len_std", 1),
    "flow_bytes_per_sec": ("bytes_per_s", 1), "flow_packets_per_sec": ("pkt_per_s", 1),
    "flow_iat_mean": ("iat_mean", US), "flow_iat_std": ("iat_std", US), "flow_iat_min": ("iat_min", US), "flow_iat_max": ("iat_max", US),
    "fwd_iat_mean": ("fwd_iat_mean", US), "fwd_iat_std": ("fwd_iat_std", US), "fwd_iat_min": ("fwd_iat_min", US), "fwd_iat_max": ("fwd_iat_max", US),
    "bwd_iat_mean": ("bwd_iat_mean", US), "bwd_iat_std": ("bwd_iat_std", US), "bwd_iat_min": ("bwd_iat_min", US), "bwd_iat_max": ("bwd_iat_max", US),
    "syn_flag_count": ("flag_SYN", 1), "ack_flag_count": ("flag_ack", 1), "fin_flag_count": ("flag_fin", 1), "rst_flag_count": ("flag_rst", 1),
    "psh_flag_count": ("flag_psh", 1), "urg_flag_count": ("flag_urg", 1), "ece_flag_count": ("flag_ece", 1), "cwr_flag_count": ("flag_cwr", 1),
}
CATEGORY = {
    "benign": "BENIGN", "ftp_patator": "Brute Force", "ssh_patator": "Brute Force", "heartbleed": "Heartbleed",
    "dos_slowloris": "DoS/DDoS", "dos_slowhttptest": "DoS/DDoS", "dos_hulk": "DoS/DDoS", "dos_goldeneye": "DoS/DDoS", "ddos": "DoS/DDoS",
    "webattack_bruteforce": "Web Attack", "webattack_xss": "Web Attack", "webattack_sql_injection": "Web Attack",
    "bot": "Botnet", "portscan": "PortScan",
}


def convert(df: pd.DataFrame, keep_endpoints: bool = False) -> pd.DataFrame:
    assert set(c for c, _ in MAP.values()) <= set(df.columns), "unexpected LYCOS schema"
    out = pd.DataFrame({k: pd.to_numeric(df[c], errors="coerce") * s for k, (c, s) in MAP.items()})
    out["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="us")
    out["label"] = df["label"].astype(str)
    if keep_endpoints:
        out["src_ip"], out["dst_ip"] = df["src_addr"].astype(str), df["dst_addr"].astype(str)
        out["src_port"], out["dst_port"] = pd.to_numeric(df["src_port"], errors="coerce"), pd.to_numeric(df["dst_port"], errors="coerce")
        out["protocol"] = pd.to_numeric(df["ip_prot"], errors="coerce")
    out["attack_category"] = out["label"].map(CATEGORY)
    unknown = sorted(set(out.loc[out["attack_category"].isna(), "label"]))
    if unknown:
        print("WARNING: unmapped labels dropped:", unknown)
    out = out.dropna(subset=["attack_category", "timestamp"])
    out = out[np.isfinite(out[FEATS].to_numpy()).all(axis=1)]
    out["is_attack"] = out["attack_category"] != "BENIGN"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--cap", type=int, default=0, help="optional per-label cap (random sample) to bound size; 0 = keep all")
    ap.add_argument("--keep-endpoints", action="store_true",
                    help="also keep src/dst address, ports and protocol (needed for context_features.py)")
    args = ap.parse_args()
    frames = []
    for f in sorted(Path(args.input_dir).glob("*.pcap_lycos.csv")):
        d = convert(pd.read_csv(f, low_memory=False), keep_endpoints=args.keep_endpoints)
        print(f.name, len(d), d["label"].value_counts().to_dict(), flush=True)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    if args.dedup:
        key = pd.util.hash_pandas_object(df[FEATS].round(6), index=False)
        nlab = df.assign(_k=key).groupby("_k")["attack_category"].nunique()
        conflict = set(nlab[nlab > 1].index)
        df = df.assign(_k=key)
        df = df[~df["_k"].isin(conflict)].drop_duplicates("_k").drop(columns="_k")
        print(f"dedup: {n0} -> {len(df)} rows ({len(conflict)} conflicting feature groups removed)")
    if args.cap:
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), args.cap), random_state=0))
    df = df.sort_values("timestamp")
    df.to_csv(args.output, index=False)
    print(df["label"].value_counts().to_string())
    print("wrote", args.output, len(df))


if __name__ == "__main__":
    main()
