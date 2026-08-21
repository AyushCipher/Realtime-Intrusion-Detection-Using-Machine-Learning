"""Live packet capture via Scapy.

Scapy's `sniff()` is callback-driven, not a generator, so `LiveCapture`
bridges it to the `PacketSource` generator interface with a bounded queue fed
by a background sniffer thread. This keeps `LiveCapture` and `PcapReplay`
(see `replay.py`) interchangeable from the pipeline's point of view.
"""

from __future__ import annotations

import logging
import queue
from typing import Iterator, Optional

from .packet_source import PacketMeta, PacketSource, packet_to_meta

logger = logging.getLogger(__name__)

# Sentinel placed on the queue by the sniffer thread's stop callback so the
# consuming generator can tell "stopped" apart from "queue momentarily empty".
_STOP = object()


class LiveCapture(PacketSource):
    """Captures packets from a live network interface using Scapy.

    Requires the process to have packet capture privileges (root/Administrator,
    or an interface configured for unprivileged capture). Capture runs on a
    background thread so the consuming generator is never blocked inside
    Scapy's own capture loop.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        bpf_filter: str = "tcp or udp",
        queue_size: int = 10_000,
    ) -> None:
        self.interface = interface
        self.bpf_filter = bpf_filter
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)
        self._sniffer = None

    def _on_packet(self, pkt) -> None:
        try:
            meta = packet_to_meta(pkt)
        except Exception:  # noqa: BLE001 - a single malformed packet must not kill capture
            logger.exception("Failed to parse captured packet; dropping it")
            return
        if meta is None:
            return
        try:
            self._queue.put_nowait(meta)
        except queue.Full:
            # Backpressure at the capture boundary: drop the newest packet
            # rather than blocking Scapy's capture thread indefinitely.
            logger.warning("Live capture queue full; dropping packet")

    def packets(self) -> Iterator[PacketMeta]:
        from scapy.all import AsyncSniffer

        self._sniffer = AsyncSniffer(
            iface=self.interface,
            filter=self.bpf_filter,
            prn=self._on_packet,
            store=False,
        )
        self._sniffer.start()
        logger.info("Live capture started on interface=%s filter=%r", self.interface, self.bpf_filter)
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                yield item
        finally:
            self.close()

    def close(self) -> None:
        if self._sniffer is not None and getattr(self._sniffer, "running", False):
            self._sniffer.stop()
        # Unblock a generator that is parked in queue.get().
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
