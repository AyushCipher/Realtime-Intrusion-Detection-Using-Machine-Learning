"""Measure sliding-window feature-extraction throughput on the sample pcap.

Replays tests/fixtures/sample_tcp.pcap with no pacing (speed=0), repeats it
`--repeat` times with shifted timestamps so flows stay distinct, and reports
packets/s and flows/s for FlowFeatureExtractor. Numbers are machine-dependent;
run it on the same machine before and after a change to compare.

Usage:
    python ingestion/scripts/benchmark_extraction.py --repeat 2000
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ids_ingestion.features import FlowFeatureExtractor, WindowConfig  # noqa: E402
from ids_ingestion.replay import PcapReplay  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "sample_tcp.pcap"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=1000, help="times to replay the pcap")
    args = parser.parse_args()

    base = list(PcapReplay(str(FIXTURE), speed=0).packets())
    span = max(p.timestamp for p in base) - min(p.timestamp for p in base) + 1000.0

    packets = []
    for i in range(args.repeat):
        packets.extend(dataclasses.replace(p, timestamp=p.timestamp + i * span) for p in base)

    extractor = FlowFeatureExtractor(WindowConfig())
    flows = 0
    start = time.perf_counter()
    for pkt in packets:
        flows += len(extractor.process_packet(pkt))
    flows += len(extractor.flush())
    elapsed = time.perf_counter() - start

    print(f"packets processed : {len(packets)}")
    print(f"flows emitted     : {flows}")
    print(f"elapsed           : {elapsed:.3f} s")
    print(f"throughput        : {len(packets) / elapsed:,.0f} packets/s, {flows / elapsed:,.0f} flows/s")


if __name__ == "__main__":
    main()
