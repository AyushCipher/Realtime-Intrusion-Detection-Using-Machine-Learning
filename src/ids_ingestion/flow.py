"""Flow identification and per-flow state tracked across a sliding window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .packet_source import (
    TCP_FLAG_ACK,
    TCP_FLAG_CWR,
    TCP_FLAG_ECE,
    TCP_FLAG_FIN,
    TCP_FLAG_PSH,
    TCP_FLAG_RST,
    TCP_FLAG_SYN,
    TCP_FLAG_URG,
    PacketMeta,
)

# Canonical flow key: direction-independent so both legs of a connection map
# to the same flow regardless of which endpoint sent the first packet we saw.
FlowKey = Tuple[str, int, str, int, int]

_FLAG_NAMES = (
    ("syn", TCP_FLAG_SYN),
    ("ack", TCP_FLAG_ACK),
    ("fin", TCP_FLAG_FIN),
    ("rst", TCP_FLAG_RST),
    ("psh", TCP_FLAG_PSH),
    ("urg", TCP_FLAG_URG),
    ("ece", TCP_FLAG_ECE),
    ("cwr", TCP_FLAG_CWR),
)


def make_flow_key(pkt: PacketMeta) -> FlowKey:
    """Build a direction-independent key so both legs of a flow collide."""
    endpoint_a = (pkt.src_ip, pkt.src_port)
    endpoint_b = (pkt.dst_ip, pkt.dst_port)
    if endpoint_a <= endpoint_b:
        lo, hi = endpoint_a, endpoint_b
    else:
        lo, hi = endpoint_b, endpoint_a
    return (lo[0], lo[1], hi[0], hi[1], pkt.protocol)


@dataclass
class FlowState:
    """Mutable, in-progress state for one flow within the current window."""

    key: FlowKey
    protocol: int
    forward_src_ip: str
    forward_src_port: int
    forward_dst_ip: str
    forward_dst_port: int
    start_time: float
    last_time: float
    fwd_timestamps: List[float] = field(default_factory=list)
    fwd_lengths: List[int] = field(default_factory=list)
    bwd_timestamps: List[float] = field(default_factory=list)
    bwd_lengths: List[int] = field(default_factory=list)
    flag_counts: Dict[str, int] = field(default_factory=lambda: {name: 0 for name, _ in _FLAG_NAMES})
    saw_fin: bool = False
    saw_rst: bool = False

    @classmethod
    def start(cls, pkt: PacketMeta, key: FlowKey) -> "FlowState":
        return cls(
            key=key,
            protocol=pkt.protocol,
            forward_src_ip=pkt.src_ip,
            forward_src_port=pkt.src_port,
            forward_dst_ip=pkt.dst_ip,
            forward_dst_port=pkt.dst_port,
            start_time=pkt.timestamp,
            last_time=pkt.timestamp,
        )

    def is_forward(self, pkt: PacketMeta) -> bool:
        return pkt.src_ip == self.forward_src_ip and pkt.src_port == self.forward_src_port

    def add_packet(self, pkt: PacketMeta) -> None:
        if self.is_forward(pkt):
            self.fwd_timestamps.append(pkt.timestamp)
            self.fwd_lengths.append(pkt.length)
        else:
            self.bwd_timestamps.append(pkt.timestamp)
            self.bwd_lengths.append(pkt.length)

        for name, bit in _FLAG_NAMES:
            if pkt.tcp_flags & bit:
                self.flag_counts[name] += 1

        if pkt.tcp_flags & TCP_FLAG_FIN:
            self.saw_fin = True
        if pkt.tcp_flags & TCP_FLAG_RST:
            self.saw_rst = True

        self.last_time = max(self.last_time, pkt.timestamp)

    @property
    def packet_count(self) -> int:
        return len(self.fwd_timestamps) + len(self.bwd_timestamps)
