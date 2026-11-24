"""Case study: our cascade vs Suricata on the SAME captured pcap.

This is a small, controlled illustration, not a benchmark. The pcap comes from
integration/live_traffic_gen.py run against a local web server (ordinary GETs,
a rapid GET burst, and connect() probes across a port range), captured with
tcpdump on loopback inside a container. Ground truth is known from the
generator's phases:
    normal  first 10 HTTP GET flows to the server port
    burst   the next 300 HTTP GET flows to the server port
    probe   connections to ports other than the server port
Suricata alerts come from `suricata -r test.pcap -S all.rules` with the full
Emerging Threats Open ruleset (see docs/suricata_comparison.md for the commands).

Our side: the cascade is trained on the real deduplicated CSE-CIC-IDS2018
subsample (hybrid supervised/density Stage 1 with conformal recall control, XGBoost
Stage 2, OpenMax escalation gate), flows are extracted from the pcap with the
ingestion module's own feature extractor, and each flow is scored. "Flagged" means
Stage 1 flagged it; "alert" means flagged and predicted an attack class or escalated.

Caveats that bound what this can say: one short synthetic capture; the model was
trained on a different lab network's CICFlowMeter features, so loopback flows are
out of distribution; Suricata rules target known signatures and are not built to
flag connect scans or request bursts on their own.

Usage:
    PYTHONPATH=ml/src:ingestion/src python integration/suricata_compare.py \
        --pcap C:/nids_data/suri/test.pcap --eve C:/nids_data/suri/logs2/eve.json \
        --data C:/nids_data/real_cicids2018_dedup.csv --server-port 8000 --out results/paper/suricata_case_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "ml" / "src", ROOT / "ingestion" / "src"):
    sys.path.insert(0, str(p))

from ids_ingestion.features import FlowFeatureExtractor, WindowConfig  # noqa: E402
from ids_ingestion.replay import PcapReplay  # noqa: E402
from ids_ml.conformal_gate import calibrate_threshold  # noqa: E402
from ids_ml.conformal_stage1 import calibrate_stage1_threshold  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402
from ids_ml.openset_head import OpenMaxConfig, OpenMaxHead  # noqa: E402
from ids_ml.split import random_split  # noqa: E402
from ids_ml.stage1_hybrid import HybridConfig, HybridPreFilter  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config  # noqa: E402


def train_cascade(data: str, miss_budget: float, esc_budget: float, weight: float):
    df = pd.read_csv(data)
    tr, va, _ = random_split(df, train_frac=0.7, val_frac=0.2, random_state=0)
    s1 = HybridPreFilter(HybridConfig(supervised_weight=weight)).fit(tr[FEATS].to_numpy(), (tr["attack_category"] != "BENIGN").to_numpy())
    va_att = va[va["attack_category"] != "BENIGN"]
    calibrate_stage1_threshold(s1.anomaly_score(va_att[FEATS].to_numpy()), miss_budget).apply(s1)
    s2 = AttackClassifier(Stage2Config(n_estimators=150)).fit(tr[FEATS].to_numpy(), tr["attack_category"].tolist())
    gate = OpenMaxHead(s2, OpenMaxConfig()).fit(tr[FEATS].to_numpy(), tr["attack_category"].tolist())
    vf = va[va["attack_category"] != "BENIGN"][FEATS].to_numpy()
    thr = calibrate_threshold([r.unknown_mass for r in gate.recalibrate_batch(vf)], esc_budget).threshold
    return s1, s2, gate, thr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--eve", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--server-port", type=int, default=8000)
    ap.add_argument("--miss-budget", type=float, default=0.05)
    ap.add_argument("--esc-budget", type=float, default=0.10)
    ap.add_argument("--supervised-weight", type=float, default=0.5)
    ap.add_argument("--benign-pcap", default=None, help="Benign-only capture from the SAME environment; if given, thresholds are re-calibrated on it (site calibration)")
    ap.add_argument("--site-flag-rate", type=float, default=0.05, help="Share of site-benign flows Stage 1 may flag")
    ap.add_argument("--out", default="results/paper/suricata_case_study.json")
    args = ap.parse_args()

    def extract(path):
        ex = FlowFeatureExtractor(WindowConfig(active_timeout=15.0, idle_timeout=5.0))
        out_flows = []
        for pkt in PcapReplay(path, speed=0).packets():
            out_flows.extend(ex.process_packet(pkt))
        out_flows.extend(ex.flush())
        return pd.DataFrame(out_flows)

    f = extract(args.pcap)
    server = args.server_port
    # a flow's server side is the endpoint on the server port, whichever direction was seen first
    f["server_side_port"] = np.where((f["dst_port"] == server) | (f["src_port"] == server), server, f["dst_port"])
    f = f.sort_values("flow_start_time").reset_index(drop=True)
    http = f["server_side_port"] == server
    phase = np.array(["probe"] * len(f), dtype=object)
    order = np.cumsum(http.to_numpy()) * http.to_numpy()
    phase[http.to_numpy() & (order <= 10)] = "normal"
    phase[http.to_numpy() & (order > 10)] = "burst"
    f["phase"] = phase

    s1, s2, gate, thr = train_cascade(args.data, args.miss_budget, args.esc_budget, args.supervised_weight)
    site = None
    if args.benign_pcap:
        # Site calibration uses only benign flows from this environment. Half of them calibrate,
        # the other half are held out to measure the false-alarm rate on unseen normal traffic.
        b = extract(args.benign_pcap).sample(frac=1.0, random_state=0).reset_index(drop=True)
        half = len(b) // 2
        cal, held = b.iloc[:half], b.iloc[half:]
        Xc = cal[FEATS].to_numpy()
        s1.config.flag_threshold = float(np.quantile(s1.anomaly_score(Xc), 1.0 - args.site_flag_rate))
        fl_c = s1.flag(Xc)
        thr = calibrate_threshold([r.unknown_mass for r in gate.recalibrate_batch(Xc[fl_c])] or [0.0], args.esc_budget).threshold
        Xh = held[FEATS].to_numpy()
        fh = s1.flag(Xh)
        eh = np.zeros(len(held), dtype=bool)
        if fh.any():
            eh[fh] = [r.unknown_mass > thr for r in gate.recalibrate_batch(Xh[fh])]
        site = {"benign_calibration_flows": int(len(cal)), "benign_heldout_flows": int(len(held)),
                "heldout_stage1_flagged": int(fh.sum()), "heldout_escalated": int(eh.sum())}
    X = f[FEATS].to_numpy()
    flagged = s1.flag(X)
    pred = np.array(["BENIGN"] * len(f), dtype=object)
    esc = np.zeros(len(f), dtype=bool)
    if flagged.any():
        res = gate.recalibrate_batch(X[flagged])
        pred[flagged] = [r.predicted_class for r in res]
        esc[flagged] = [r.unknown_mass > thr for r in res]
    ours_alert = flagged & ((pred != "BENIGN") | esc)

    suri_alerts = [json.loads(l) for l in open(args.eve) if '"event_type":"alert"' in l or '"event_type": "alert"' in l]
    suri_flow_ids = {a["flow_id"] for a in suri_alerts}
    out = {
        "pcap_flows_extracted_by_ingestion": len(f),
        "ground_truth_flows": f["phase"].value_counts().to_dict(),
        "suricata": {"rules": "Emerging Threats Open (all categories)", "alerts": len(suri_alerts), "flows_alerted": len(suri_flow_ids),
                     "signatures": sorted({a["alert"]["signature"] for a in suri_alerts})},
        "ours": {},
        "site_calibration": site,
    }
    for ph in ("normal", "burst", "probe"):
        m = (f["phase"] == ph).to_numpy()
        out["ours"][ph] = {"flows": int(m.sum()), "stage1_flagged": int(flagged[m].sum()), "alerts": int(ours_alert[m].sum()),
                           "predicted_classes": pd.Series(pred[m & ours_alert]).value_counts().to_dict()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=int))
    write_manifest(Path(args.out).with_suffix(".manifest.json"), inputs=[args.pcap, args.eve, args.data], outputs=[args.out],
                   extra={"miss_budget": args.miss_budget, "esc_budget": args.esc_budget, "supervised_weight": args.supervised_weight}, repo_dir=ROOT)
    print(json.dumps(out, indent=2, default=int))


if __name__ == "__main__":
    main()
