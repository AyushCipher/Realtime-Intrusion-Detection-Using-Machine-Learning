"""Common packet representation and source interface.

Both live capture and pcap replay normalize packets into `PacketMeta` before
handing them to the flow tracker, so the rest of the pipeline never needs to
know whether a packet came off a live interface or a replayed file.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator, Optional


# TCP flag bit values, as exposed by Scapy's `flags` field on a TCP layer.
TCP_FLAG_FIN = 0x01
TCP_FLAG_SYN = 0x02
TCP_FLAG_RST = 0x04
TCP_FLAG_PSH = 0x08
TCP_FLAG_ACK = 0x10
TCP_FLAG_URG = 0x20
TCP_FLAG_ECE = 0x40
TCP_FLAG_CWR = 0x80


@dataclass(frozen=True)
class PacketMeta:
    """Normalized fields extracted from a single captured/replayed packet.

    `timestamp` is a float Unix epoch (seconds, sub-second precision as
    provided by the capture source). `protocol` is the IANA transport
    protocol number (6 = TCP, 17 = UDP, 1 = ICMP, ...).
    """

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    length: int
    tcp_flags: int = 0

    @property
    def is_tcp(self) -> bool:
        return self.protocol == 6

    def has_flag(self, flag: int) -> bool:
        return bool(self.tcp_flags & flag)


def packet_to_meta(pkt) -> Optional[PacketMeta]:
    """Convert a Scapy packet into a `PacketMeta`, or None if it has no IP layer.

    Only IPv4/IPv6 packets with a TCP or UDP transport layer are turned into
    flow-relevant metadata; other protocols (ARP, pure ICMP without a useful
    port pair, etc.) are skipped by returning None so callers can filter them
    out without special-casing packet types themselves.
    """
    # Imported lazily so environments that only replay pre-captured metadata
    # (e.g. some unit tests) do not pay Scapy's import cost/privileges.
    # `scapy.all` (rather than the individual `scapy.layers.*` submodules) is
    # used deliberately: it registers the linktype -> layer bindings (e.g.
    # Ethernet) that PcapReader/AsyncSniffer need to decode captures at all.
    from scapy.all import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6

    if IP in pkt:
        ip_layer = pkt[IP]
        protocol = ip_layer.proto
    elif IPv6 in pkt:
        ip_layer = pkt[IPv6]
        protocol = ip_layer.nh
    else:
        return None

    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    length = len(pkt)
    timestamp = float(pkt.time)

    src_port = 0
    dst_port = 0
    tcp_flags = 0

    if TCP in pkt:
        tcp_layer = pkt[TCP]
        src_port = int(tcp_layer.sport)
        dst_port = int(tcp_layer.dport)
        tcp_flags = int(tcp_layer.flags)
        protocol = 6
    elif UDP in pkt:
        udp_layer = pkt[UDP]
        src_port = int(udp_layer.sport)
        dst_port = int(udp_layer.dport)
        protocol = 17

    return PacketMeta(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        length=length,
        tcp_flags=tcp_flags,
    )


class PacketSource(abc.ABC):
    """Common interface for anything that yields a stream of `PacketMeta`."""

    @abc.abstractmethod
    def packets(self) -> Iterator[PacketMeta]:
        """Yield packets in arrival order until the source is exhausted or stopped."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any underlying resources (sockets, file handles)."""
        return None
