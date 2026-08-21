"""Feature-extraction correctness tests against a known pcap sample.

tests/fixtures/sample_tcp.pcap is built by tests/generate_fixtures.py from an
explicit per-packet spec (timestamps, flags, payload sizes), so every value
asserted here can be traced back to that spec rather than to a black box.
"""

from pathlib import Path

import pytest

from ids_ingestion.features import FlowFeatureExtractor, WindowConfig
from ids_ingestion.packet_source import PacketMeta
from ids_ingestion.replay import PcapReplay

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_tcp.pcap"


def _mean(values):
    return sum(values) / len(values)


def _pstd(values):
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def _iat(timestamps):
    ordered = sorted(timestamps)
    return [b - a for a, b in zip(ordered, ordered[1:])]


def _load_fixture_packets():
    replay = PcapReplay(str(FIXTURE_PATH), speed=0)  # speed=0 -> no sleeping
    return list(replay.packets())


def test_fixture_has_expected_raw_packets():
    packets = _load_fixture_packets()
    assert len(packets) == 10
    tcp = [p for p in packets if p.protocol == 6]
    udp = [p for p in packets if p.protocol == 17]
    assert len(tcp) == 7
    assert len(udp) == 3


def test_tcp_flow_feature_extraction():
    packets = [p for p in _load_fixture_packets() if p.protocol == 6]
    extractor = FlowFeatureExtractor(WindowConfig(active_timeout=120.0, idle_timeout=60.0))

    closed = []
    for pkt in packets:
        closed.extend(extractor.process_packet(pkt))

    # The flow's last packet carries FIN, so it closes on its own -- no flush needed.
    assert len(closed) == 1
    feat = closed[0]

    assert feat["close_reason"] == "fin"
    assert feat["src_ip"] == "10.0.0.1"
    assert feat["src_port"] == 5555
    assert feat["dst_ip"] == "10.0.0.2"
    assert feat["dst_port"] == 443
    assert feat["protocol"] == 6

    assert feat["total_fwd_packets"] == 4
    assert feat["total_bwd_packets"] == 3
    assert feat["total_fwd_bytes"] == 696
    assert feat["total_bwd_bytes"] == 1142

    assert feat["flow_duration"] == pytest.approx(0.20, abs=1e-6)
    assert feat["flow_bytes_per_sec"] == pytest.approx(1838 / 0.20, rel=1e-6)
    assert feat["flow_packets_per_sec"] == pytest.approx(7 / 0.20, rel=1e-6)

    assert feat["syn_flag_count"] == 2
    assert feat["ack_flag_count"] == 6
    assert feat["fin_flag_count"] == 1
    assert feat["rst_flag_count"] == 0
    assert feat["psh_flag_count"] == 2
    assert feat["urg_flag_count"] == 0

    fwd_lengths = [54, 54, 534, 54]
    bwd_lengths = [54, 54, 1034]
    assert feat["fwd_packet_length_min"] == min(fwd_lengths)
    assert feat["fwd_packet_length_max"] == max(fwd_lengths)
    assert feat["fwd_packet_length_mean"] == pytest.approx(_mean(fwd_lengths))
    assert feat["fwd_packet_length_std"] == pytest.approx(_pstd(fwd_lengths))
    assert feat["bwd_packet_length_min"] == min(bwd_lengths)
    assert feat["bwd_packet_length_max"] == max(bwd_lengths)
    assert feat["bwd_packet_length_mean"] == pytest.approx(_mean(bwd_lengths))
    assert feat["bwd_packet_length_std"] == pytest.approx(_pstd(bwd_lengths))

    fwd_ts = [0.00, 0.02, 0.05, 0.20]
    bwd_ts = [0.01, 0.08, 0.10]
    all_ts = sorted(fwd_ts + bwd_ts)
    # abs tolerance (not just rel) because these timestamps come off packets
    # carrying a large epoch base, so subtraction loses a bit of precision.
    assert feat["flow_iat_mean"] == pytest.approx(_mean(_iat(all_ts)), abs=1e-6)
    assert feat["flow_iat_std"] == pytest.approx(_pstd(_iat(all_ts)), abs=1e-6)
    assert feat["fwd_iat_mean"] == pytest.approx(_mean(_iat(fwd_ts)), abs=1e-6)
    assert feat["bwd_iat_mean"] == pytest.approx(_mean(_iat(bwd_ts)), abs=1e-6)


def test_udp_flow_requires_flush_and_has_no_tcp_flags():
    packets = [p for p in _load_fixture_packets() if p.protocol == 17]
    extractor = FlowFeatureExtractor(WindowConfig(active_timeout=120.0, idle_timeout=60.0))

    closed = []
    for pkt in packets:
        closed.extend(extractor.process_packet(pkt))
    # UDP has no FIN/RST, so nothing should close until we flush.
    assert closed == []

    flushed = extractor.flush()
    assert len(flushed) == 1
    feat = flushed[0]

    assert feat["close_reason"] == "flush"
    assert feat["protocol"] == 17
    assert feat["src_ip"] == "10.0.0.5"
    assert feat["dst_ip"] == "10.0.0.6"
    assert feat["total_fwd_packets"] == 2
    assert feat["total_bwd_packets"] == 1
    assert feat["total_fwd_bytes"] == 164
    assert feat["total_bwd_bytes"] == 162
    for flag in ("syn", "ack", "fin", "rst", "psh", "urg", "ece", "cwr"):
        assert feat[f"{flag}_flag_count"] == 0


def test_idle_timeout_closes_flow_without_fin():
    extractor = FlowFeatureExtractor(WindowConfig(active_timeout=120.0, idle_timeout=5.0))

    p1 = PacketMeta(timestamp=100.0, src_ip="10.0.0.9", dst_ip="10.0.0.10",
                     src_port=1111, dst_port=2222, protocol=17, length=100)
    p2 = PacketMeta(timestamp=100.5, src_ip="10.0.0.10", dst_ip="10.0.0.9",
                     src_port=2222, dst_port=1111, protocol=17, length=200)
    assert extractor.process_packet(p1) == []
    assert extractor.process_packet(p2) == []

    # A later, unrelated packet drives the tracker's clock forward and should
    # sweep the first flow out on idle timeout (last activity at t=100.5,
    # idle_timeout=5s, new packet arrives at t=110).
    other = PacketMeta(timestamp=110.0, src_ip="10.0.0.20", dst_ip="10.0.0.21",
                        src_port=3333, dst_port=4444, protocol=6, length=60)
    closed = extractor.process_packet(other)

    assert len(closed) == 1
    assert closed[0]["close_reason"] == "idle_timeout"
    assert closed[0]["src_ip"] == "10.0.0.9"
    assert closed[0]["total_fwd_packets"] == 1
    assert closed[0]["total_bwd_packets"] == 1


def test_active_timeout_splits_long_lived_flow_into_windows():
    extractor = FlowFeatureExtractor(WindowConfig(active_timeout=10.0, idle_timeout=1000.0))

    def pkt(t):
        return PacketMeta(timestamp=t, src_ip="10.0.0.1", dst_ip="10.0.0.2",
                           src_port=1000, dst_port=2000, protocol=17, length=50)

    closed = []
    closed.extend(extractor.process_packet(pkt(0.0)))
    closed.extend(extractor.process_packet(pkt(2.0)))
    # This packet arrives 11s after the window started (> active_timeout=10s),
    # so it should force-close the first window and start a new one.
    closed.extend(extractor.process_packet(pkt(11.0)))

    assert len(closed) == 1
    assert closed[0]["close_reason"] == "active_timeout"
    assert closed[0]["total_fwd_packets"] == 2

    remaining = extractor.flush()
    assert len(remaining) == 1
    assert remaining[0]["total_fwd_packets"] == 1


def test_flow_key_is_direction_independent():
    from ids_ingestion.flow import make_flow_key

    a = PacketMeta(timestamp=0.0, src_ip="1.1.1.1", dst_ip="2.2.2.2",
                    src_port=111, dst_port=222, protocol=6, length=1)
    b = PacketMeta(timestamp=0.0, src_ip="2.2.2.2", dst_ip="1.1.1.1",
                    src_port=222, dst_port=111, protocol=6, length=1)
    assert make_flow_key(a) == make_flow_key(b)


def test_stats_helper_handles_empty_and_single_values():
    from ids_ingestion.features import _stats

    assert _stats([]) == {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    single = _stats([42.0])
    assert single["min"] == single["max"] == single["mean"] == 42.0
    assert single["std"] == 0.0
