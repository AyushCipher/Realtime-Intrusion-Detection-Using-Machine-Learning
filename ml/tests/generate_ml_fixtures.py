"""Builds tests/fixtures/synthetic_cicids_sample.csv and
tests/fixtures/synthetic_sequence_sample.csv.

This is a SYNTHETIC dataset shaped like a CICIDS2017 CSV (same column names,
including the dataset's well-known inconsistent leading-whitespace quirk),
used only to exercise this module's data loading, splitting, training, and
evaluation code paths in tests without needing the real multi-gigabyte
CICIDS2017/CIC-IDS2018 download. It is NOT real network traffic and any
metrics produced from it are not representative of real-world detector
performance -- see the README's "Known limitations" section.

Two synthetic days are generated. Day 2 deliberately shifts each attack
category's distribution from Day 1, so a model trained on Day 1 and
evaluated on Day 2 shows genuine (if synthetic) concept-drift degradation,
exercising evaluation.concept_drift_report end-to-end.

Run manually with `python tests/generate_ml_fixtures.py` to regenerate.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RNG_SEED = 1337

# (category, raw_label, day1_count, day2_count)
CATEGORY_SPEC = [
    ("BENIGN", "BENIGN", 420, 380),
    ("DoS/DDoS", "DoS Hulk", 110, 90),
    ("PortScan", "PortScan", 55, 45),
    ("Brute Force", "FTP-Patator", 30, 30),
    ("Web Attack", "Web Attack - Brute Force", 10, 10),
    ("Infiltration", "Infiltration", 5, 5),
    ("Botnet", "Bot", 16, 14),
    ("Heartbleed", "Heartbleed", 2, 2),
]

# Per-category base distributions: (duration_us, fwd_pkts, bwd_pkts,
# fwd_bytes_mean, bwd_bytes_mean, iat_us_mean, syn, ack, fin, rst, psh, urg).
# Deliberately caricatured so categories are separable -- this is a fixture
# for exercising code paths, not a realistic traffic model.
BASE_PARAMS = {
    "BENIGN": dict(dur=2_000_000, fwd_n=12, bwd_n=14, fwd_b=300, bwd_b=800, iat=150_000,
                   syn=1, ack=20, fin=1, rst=0, psh=6, urg=0),
    "DoS/DDoS": dict(dur=50_000, fwd_n=200, bwd_n=2, fwd_b=60, bwd_b=60, iat=250,
                      syn=180, ack=20, fin=0, rst=0, psh=0, urg=0),
    "PortScan": dict(dur=20_000, fwd_n=1, bwd_n=1, fwd_b=40, bwd_b=0, iat=500,
                      syn=1, ack=0, fin=0, rst=1, psh=0, urg=0),
    "Brute Force": dict(dur=800_000, fwd_n=8, bwd_n=8, fwd_b=50, bwd_b=50, iat=90_000,
                         syn=1, ack=14, fin=1, rst=1, psh=2, urg=0),
    "Web Attack": dict(dur=1_500_000, fwd_n=10, bwd_n=10, fwd_b=400, bwd_b=1200, iat=140_000,
                        syn=1, ack=18, fin=1, rst=0, psh=5, urg=0),
    "Infiltration": dict(dur=3_000_000, fwd_n=14, bwd_n=16, fwd_b=350, bwd_b=900, iat=180_000,
                          syn=1, ack=22, fin=1, rst=0, psh=6, urg=0),
    "Botnet": dict(dur=5_000_000, fwd_n=6, bwd_n=6, fwd_b=120, bwd_b=120, iat=800_000,
                    syn=1, ack=10, fin=0, rst=0, psh=1, urg=0),
    "Heartbleed": dict(dur=10_000_000, fwd_n=4, bwd_n=200, fwd_b=60, bwd_b=16000, iat=40_000,
                        syn=1, ack=200, fin=1, rst=0, psh=3, urg=0),
}

# Fraction each parameter's mean shifts by on day 2 (drift simulation).
DRIFT_FACTOR = {
    "BENIGN": 1.0,
    "DoS/DDoS": 1.35,       # attackers ramp up rate
    "PortScan": 0.7,        # scans get slower/stealthier
    "Brute Force": 1.2,
    "Web Attack": 1.0,
    "Infiltration": 1.15,
    "Botnet": 1.4,
    "Heartbleed": 1.0,
}


def _jitter(rng, mean, rel_std=0.25, low=0.0):
    val = rng.normal(mean, mean * rel_std + 1e-6)
    return max(val, low)


def _rows_for(rng, category, raw_label, n, day, day_start_epoch):
    p = dict(BASE_PARAMS[category])
    if day == 2:
        factor = DRIFT_FACTOR[category]
        for k in ("dur", "fwd_n", "bwd_n", "fwd_b", "bwd_b", "iat"):
            p[k] = p[k] * factor

    rows = []
    for i in range(n):
        dur = _jitter(rng, p["dur"], low=1)
        fwd_n = max(1, int(round(_jitter(rng, p["fwd_n"], low=1))))
        bwd_n = max(0, int(round(_jitter(rng, p["bwd_n"], low=0))))
        fwd_b_mean = _jitter(rng, p["fwd_b"], low=1)
        bwd_b_mean = _jitter(rng, p["bwd_b"], low=0)
        iat_mean = _jitter(rng, p["iat"], low=1)

        offset_s = rng.uniform(0, 86_400)
        timestamp = pd.Timestamp(day_start_epoch, unit="s") + pd.Timedelta(seconds=offset_s)
        src_ip = f"10.{day}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        rows.append(_build_row(dur, fwd_n, bwd_n, fwd_b_mean, bwd_b_mean, iat_mean, p, timestamp, src_ip, raw_label))
    return rows


def _build_row(dur, fwd_n, bwd_n, fwd_b_mean, bwd_b_mean, iat_mean, p, timestamp, src_ip, raw_label):
    total_fwd_bytes = fwd_b_mean * fwd_n
    total_bwd_bytes = bwd_b_mean * bwd_n
    flow_bytes_per_sec = (total_fwd_bytes + total_bwd_bytes) / max(dur / 1_000_000.0, 1e-6)
    flow_packets_per_sec = (fwd_n + bwd_n) / max(dur / 1_000_000.0, 1e-6)
    return {
        " Flow Duration": dur,
        " Total Fwd Packets": fwd_n,
        " Total Backward Packets": bwd_n,
        "Total Length of Fwd Packets": total_fwd_bytes,
        " Total Length of Bwd Packets": total_bwd_bytes,
        " Fwd Packet Length Min": max(fwd_b_mean * 0.6, 1),
        " Fwd Packet Length Max": fwd_b_mean * 1.6,
        " Fwd Packet Length Mean": fwd_b_mean,
        " Fwd Packet Length Std": fwd_b_mean * 0.2,
        "Bwd Packet Length Min": max(bwd_b_mean * 0.6, 0),
        " Bwd Packet Length Max": bwd_b_mean * 1.6,
        " Bwd Packet Length Mean": bwd_b_mean,
        " Bwd Packet Length Std": bwd_b_mean * 0.2,
        "Flow Bytes/s": flow_bytes_per_sec,
        " Flow Packets/s": flow_packets_per_sec,
        " Flow IAT Mean": iat_mean,
        " Flow IAT Std": iat_mean * 0.3,
        " Flow IAT Max": iat_mean * 2.5,
        " Flow IAT Min": max(iat_mean * 0.1, 1),
        " Fwd IAT Mean": iat_mean,
        " Fwd IAT Std": iat_mean * 0.3,
        " Fwd IAT Max": iat_mean * 2.5,
        " Fwd IAT Min": max(iat_mean * 0.1, 1),
        "Bwd IAT Mean": iat_mean * 1.1,
        " Bwd IAT Std": iat_mean * 0.3,
        " Bwd IAT Max": iat_mean * 2.5,
        " Bwd IAT Min": max(iat_mean * 0.1, 1),
        "SYN Flag Count": p["syn"],
        " ACK Flag Count": p["ack"],
        "FIN Flag Count": p["fin"],
        " RST Flag Count": p["rst"],
        " PSH Flag Count": p["psh"],
        " URG Flag Count": p["urg"],
        " ECE Flag Count": 0,
        " CWE Flag Count": 0,
        " Timestamp": timestamp,
        " Source IP": src_ip,
        " Label": raw_label,
    }


# --- A second fixture: per-source-IP multi-stage campaigns ----------------
# sequence_model.py needs several flows from the *same* source IP to have
# any history to look back on. The main fixture above assigns each row a
# near-unique random IP (realistic for a broad traffic sample, useless for
# sequence modeling), so this generates a small, separate dataset: a few
# "attacker" IPs each running a recon -> brute-force -> DoS/exfiltration
# campaign in order, plus several "benign" IPs each with a longer run of
# ordinary flows.

CAMPAIGN_STAGES = [
    ("PortScan", "PortScan", 2),
    ("Brute Force", "FTP-Patator", 2),
    ("DoS/DDoS", "DoS Hulk", 2),
]
N_ATTACKER_IPS = 5
N_BENIGN_IPS = 15
BENIGN_FLOWS_PER_IP_RANGE = (6, 10)


def _row_for_category(rng, category, raw_label, timestamp, src_ip):
    p = dict(BASE_PARAMS[category])
    dur = _jitter(rng, p["dur"], low=1)
    fwd_n = max(1, int(round(_jitter(rng, p["fwd_n"], low=1))))
    bwd_n = max(0, int(round(_jitter(rng, p["bwd_n"], low=0))))
    fwd_b_mean = _jitter(rng, p["fwd_b"], low=1)
    bwd_b_mean = _jitter(rng, p["bwd_b"], low=0)
    iat_mean = _jitter(rng, p["iat"], low=1)
    return _build_row(dur, fwd_n, bwd_n, fwd_b_mean, bwd_b_mean, iat_mean, p, timestamp, src_ip, raw_label)


def build_sequence_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 1)
    day_start = pd.Timestamp("2017-07-05").timestamp()
    rows = []

    for i in range(N_ATTACKER_IPS):
        src_ip = f"10.9.{i}.{rng.integers(1, 255)}"
        t = rng.uniform(0, 20_000)  # campaign start, early in the day
        for category, raw_label, n_flows in CAMPAIGN_STAGES:
            for _ in range(n_flows):
                t += rng.uniform(5, 30)  # stages progress close together in time
                timestamp = pd.Timestamp(day_start, unit="s") + pd.Timedelta(seconds=t)
                rows.append(_row_for_category(rng, category, raw_label, timestamp, src_ip))

    for i in range(N_BENIGN_IPS):
        src_ip = f"10.8.{i}.{rng.integers(1, 255)}"
        n_flows = rng.integers(*BENIGN_FLOWS_PER_IP_RANGE)
        t = rng.uniform(0, 80_000)
        for _ in range(n_flows):
            t += rng.uniform(30, 600)
            timestamp = pd.Timestamp(day_start, unit="s") + pd.Timedelta(seconds=t)
            rows.append(_row_for_category(rng, "BENIGN", "BENIGN", timestamp, src_ip))

    return pd.DataFrame(rows)


def build_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    day1_start = pd.Timestamp("2017-07-03").timestamp()
    day2_start = pd.Timestamp("2017-07-04").timestamp()

    all_rows = []
    for category, raw_label, n1, n2 in CATEGORY_SPEC:
        all_rows.extend(_rows_for(rng, category, raw_label, n1, 1, day1_start))
        all_rows.extend(_rows_for(rng, category, raw_label, n2, 2, day2_start))

    df = pd.DataFrame(all_rows)
    # Shuffle row order (real captures aren't grouped by label) but keep
    # timestamps as the source of ordering truth for time-based splitting.
    df = df.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
    return df


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = FIXTURES_DIR / "synthetic_cicids_sample.csv"
    build_dataframe().to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    seq_out_path = FIXTURES_DIR / "synthetic_sequence_sample.csv"
    build_sequence_dataframe().to_csv(seq_out_path, index=False)
    print(f"Wrote {seq_out_path}")


if __name__ == "__main__":
    main()
