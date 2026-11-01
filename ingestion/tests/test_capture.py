"""LiveCapture must fail loudly when the sniffer thread dies, not hang silently."""

import threading

import pytest

import ids_ingestion.capture as capture_mod
from ids_ingestion.capture import LiveCapture


class _DeadSniffer:
    """Stands in for scapy's AsyncSniffer after its thread died on a setup error."""

    running = False

    def __init__(self, **_kwargs):
        self.exception = RuntimeError("libpcap is not available. Cannot compile filter !")
        self.thread = threading.Thread(target=lambda: None)
        self.thread.start()
        self.thread.join()

    def start(self):
        pass

    def stop(self):
        pass


def test_dead_sniffer_raises_instead_of_hanging(monkeypatch):
    scapy_all = pytest.importorskip("scapy.all")
    monkeypatch.setattr(scapy_all, "AsyncSniffer", _DeadSniffer)
    monkeypatch.setattr(capture_mod, "_POLL_SECONDS", 0.05)

    with pytest.raises(RuntimeError, match="libpcap"):
        next(LiveCapture(interface="eth0").packets())
