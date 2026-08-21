"""Sliding-window flow tracking and feature extraction.

`FlowFeatureExtractor` groups packets into bidirectional flows and, when a
flow closes (FIN/RST observed, idle timeout, or a window boundary on a
long-lived flow), computes a flat dict of features: duration, packet/byte
counts, inter-arrival time statistics, and TCP flag counts. This mirrors the
kind of features CICIDS2017 ships pre-computed, but every value here is
derived directly from packets -- nothing is loaded from a feature CSV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .flow import FlowState, make_flow_key
from .packet_source import PacketMeta

logger = logging.getLogger(__name__)


def _stats(values: List[float]) -> Dict[str, float]:
    """Return min/max/mean/std for a list of numbers, 0.0 for an empty list."""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance**0.5
    else:
        std = 0.0
    return {"min": float(min(values)), "max": float(max(values)), "mean": float(mean), "std": float(std)}


def _inter_arrival_times(timestamps: List[float]) -> List[float]:
    ordered = sorted(timestamps)
    return [b - a for a, b in zip(ordered, ordered[1:])]


@dataclass
class WindowConfig:
    """Tunable sliding-window parameters.

    active_timeout: force-close and emit a flow that has been open this long,
        even with no idle gap, so long-lived connections still produce
        periodic feature windows instead of one giant flow at teardown.
    idle_timeout: close a flow after this many seconds with no packets.
    """

    active_timeout: float = 120.0
    idle_timeout: float = 60.0


class FlowFeatureExtractor:
    """Stateful tracker: feed packets in, get closed-flow feature dicts out.

    Time is driven entirely by packet timestamps (not wall clock), so this
    class behaves identically whether packets arrive live, are replayed with
    realistic pacing, or are fed directly from a test fixture.
    """

    def __init__(self, config: Optional[WindowConfig] = None) -> None:
        self.config = config or WindowConfig()
        self._flows: Dict[tuple, FlowState] = {}
        self._max_timestamp: float = 0.0

    def process_packet(self, pkt: PacketMeta) -> List[dict]:
        """Add one packet to its flow; return feature dicts for any flows this closes."""
        self._max_timestamp = max(self._max_timestamp, pkt.timestamp)
        closed = self._expire_idle_flows(self._max_timestamp)

        key = make_flow_key(pkt)
        state = self._flows.get(key)

        if state is not None and (pkt.timestamp - state.start_time) > self.config.active_timeout:
            closed.append(self._close(key, reason="active_timeout"))
            state = None

        if state is None:
            state = FlowState.start(pkt, key)
            self._flows[key] = state

        state.add_packet(pkt)

        if state.saw_fin or state.saw_rst:
            reason = "fin" if state.saw_fin else "rst"
            closed.append(self._close(key, reason=reason))

        return closed

    def flush(self) -> List[dict]:
        """Close and emit all remaining in-progress flows (e.g. at shutdown)."""
        return [self._close(key, reason="flush") for key in list(self._flows.keys())]

    def _expire_idle_flows(self, now: float) -> List[dict]:
        expired_keys = [
            key
            for key, state in self._flows.items()
            if (now - state.last_time) > self.config.idle_timeout
        ]
        return [self._close(key, reason="idle_timeout") for key in expired_keys]

    def _close(self, key: tuple, reason: str) -> dict:
        state = self._flows.pop(key)
        return extract_features(state, close_reason=reason)


def extract_features(state: FlowState, close_reason: str = "unknown") -> dict:
    """Compute the flat feature dict for one closed/expired flow."""
    duration = max(state.last_time - state.start_time, 0.0)

    fwd_len_stats = _stats([float(v) for v in state.fwd_lengths])
    bwd_len_stats = _stats([float(v) for v in state.bwd_lengths])

    all_timestamps = state.fwd_timestamps + state.bwd_timestamps
    flow_iat_stats = _stats(_inter_arrival_times(all_timestamps))
    fwd_iat_stats = _stats(_inter_arrival_times(state.fwd_timestamps))
    bwd_iat_stats = _stats(_inter_arrival_times(state.bwd_timestamps))

    total_fwd_bytes = sum(state.fwd_lengths)
    total_bwd_bytes = sum(state.bwd_lengths)
    total_packets = state.packet_count
    total_bytes = total_fwd_bytes + total_bwd_bytes

    return {
        "flow_id": "-".join(str(part) for part in state.key),
        "src_ip": state.forward_src_ip,
        "src_port": state.forward_src_port,
        "dst_ip": state.forward_dst_ip,
        "dst_port": state.forward_dst_port,
        "protocol": state.protocol,
        "flow_start_time": state.start_time,
        "flow_end_time": state.last_time,
        "flow_duration": duration,
        "close_reason": close_reason,
        "total_fwd_packets": len(state.fwd_timestamps),
        "total_bwd_packets": len(state.bwd_timestamps),
        "total_fwd_bytes": total_fwd_bytes,
        "total_bwd_bytes": total_bwd_bytes,
        "fwd_packet_length_min": fwd_len_stats["min"],
        "fwd_packet_length_max": fwd_len_stats["max"],
        "fwd_packet_length_mean": fwd_len_stats["mean"],
        "fwd_packet_length_std": fwd_len_stats["std"],
        "bwd_packet_length_min": bwd_len_stats["min"],
        "bwd_packet_length_max": bwd_len_stats["max"],
        "bwd_packet_length_mean": bwd_len_stats["mean"],
        "bwd_packet_length_std": bwd_len_stats["std"],
        "flow_bytes_per_sec": (total_bytes / duration) if duration > 0 else 0.0,
        "flow_packets_per_sec": (total_packets / duration) if duration > 0 else 0.0,
        "flow_iat_mean": flow_iat_stats["mean"],
        "flow_iat_std": flow_iat_stats["std"],
        "flow_iat_min": flow_iat_stats["min"],
        "flow_iat_max": flow_iat_stats["max"],
        "fwd_iat_mean": fwd_iat_stats["mean"],
        "fwd_iat_std": fwd_iat_stats["std"],
        "fwd_iat_min": fwd_iat_stats["min"],
        "fwd_iat_max": fwd_iat_stats["max"],
        "bwd_iat_mean": bwd_iat_stats["mean"],
        "bwd_iat_std": bwd_iat_stats["std"],
        "bwd_iat_min": bwd_iat_stats["min"],
        "bwd_iat_max": bwd_iat_stats["max"],
        "syn_flag_count": state.flag_counts["syn"],
        "ack_flag_count": state.flag_counts["ack"],
        "fin_flag_count": state.flag_counts["fin"],
        "rst_flag_count": state.flag_counts["rst"],
        "psh_flag_count": state.flag_counts["psh"],
        "urg_flag_count": state.flag_counts["urg"],
        "ece_flag_count": state.flag_counts["ece"],
        "cwr_flag_count": state.flag_counts["cwr"],
    }
