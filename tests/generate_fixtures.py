"""Regenerates tests/fixtures/sample_tcp.pcap.

Run manually with `python tests/generate_fixtures.py` whenever the fixture
needs to change. The pcap is committed to the repo so tests don't depend on
regenerating it, but this script documents exactly how it was built and lets
the exact per-packet timestamps/flags/sizes asserted on in
tests/test_features.py be traced back to their source.
"""

from pathlib import Path

from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def build_packets():
    packets = []
    t0 = 1_700_000_000.0  # fixed epoch base so the fixture is reproducible

    # --- TCP flow: client 10.0.0.1:5555 -> server 10.0.0.2:443 -----------
    # Handshake, one request, one response, graceful close on a single FIN.
    tcp_spec = [
        # (offset_seconds, src, sport, dst, dport, flags, payload_len)
        (0.00, "10.0.0.1", 5555, "10.0.0.2", 443, "S", 0),
        (0.01, "10.0.0.2", 443, "10.0.0.1", 5555, "SA", 0),
        (0.02, "10.0.0.1", 5555, "10.0.0.2", 443, "A", 0),
        (0.05, "10.0.0.1", 5555, "10.0.0.2", 443, "PA", 480),
        (0.08, "10.0.0.2", 443, "10.0.0.1", 5555, "A", 0),
        (0.10, "10.0.0.2", 443, "10.0.0.1", 5555, "PA", 980),
        (0.20, "10.0.0.1", 5555, "10.0.0.2", 443, "FA", 0),
    ]
    for offset, src, sport, dst, dport, flags, payload_len in tcp_spec:
        pkt = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
        if payload_len:
            pkt = pkt / Raw(load=b"x" * payload_len)
        pkt.time = t0 + offset
        packets.append(pkt)

    # --- UDP flow: 10.0.0.5:6000 -> 10.0.0.6:53, three DNS-ish packets ----
    udp_spec = [
        (1.00, "10.0.0.5", 6000, "10.0.0.6", 53, 40),
        (1.05, "10.0.0.6", 53, "10.0.0.5", 6000, 120),
        (1.10, "10.0.0.5", 6000, "10.0.0.6", 53, 40),
    ]
    for offset, src, sport, dst, dport, payload_len in udp_spec:
        pkt = Ether() / IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / Raw(load=b"y" * payload_len)
        pkt.time = t0 + offset
        packets.append(pkt)

    return packets


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES_DIR / "sample_tcp.pcap"
    wrpcap(str(out_path), build_packets())
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
