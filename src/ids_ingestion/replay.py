"""Pcap replay with realistic inter-packet timing.

Unlike simply iterating a pcap file (which yields packets as fast as they can
be parsed), `PcapReplay` sleeps between packets to reproduce the original
capture's timing, so the rest of the pipeline (sliding-window feature
extraction in particular) sees the same arrival pattern it would see live.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

from .packet_source import PacketMeta, PacketSource, packet_to_meta

logger = logging.getLogger(__name__)


class PcapReplay(PacketSource):
    """Replays a pcap file, sleeping between packets to simulate real timing.

    Args:
        path: Path to a pcap/pcapng file.
        speed: Playback speed multiplier. 1.0 reproduces original timing,
            2.0 replays twice as fast, 0 (or a falsy value) disables sleeping
            entirely for instant playback (useful in tests).
        max_delay: Upper bound (seconds) on any single inter-packet sleep, so
            a large idle gap in the capture doesn't stall replay for minutes.
    """

    def __init__(self, path: str, speed: float = 1.0, max_delay: float = 5.0) -> None:
        self.path = path
        self.speed = speed
        self.max_delay = max_delay
        self._reader = None

    def packets(self) -> Iterator[PacketMeta]:
        from scapy.all import PcapReader

        last_pkt_time: Optional[float] = None
        with PcapReader(self.path) as reader:
            self._reader = reader
            for pkt in reader:
                meta = packet_to_meta(pkt)
                if meta is None:
                    continue

                if self.speed and last_pkt_time is not None:
                    gap = (meta.timestamp - last_pkt_time) / self.speed
                    if gap > 0:
                        time.sleep(min(gap, self.max_delay))
                last_pkt_time = meta.timestamp

                yield meta
        self._reader = None
        logger.info("Pcap replay of %s complete", self.path)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
