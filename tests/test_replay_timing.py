"""Verifies PcapReplay actually paces packets rather than dumping them instantly."""

import time
from pathlib import Path

from ids_ingestion.replay import PcapReplay

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_tcp.pcap"


def test_speed_zero_replays_without_sleeping():
    replay = PcapReplay(str(FIXTURE_PATH), speed=0)
    start = time.monotonic()
    packets = list(replay.packets())
    elapsed = time.monotonic() - start

    assert len(packets) == 10
    # The fixture spans ~1.1s of capture time; instant playback must be far
    # faster than that, with generous slack for slow CI machines.
    assert elapsed < 0.5


def test_realistic_speed_paces_playback_to_roughly_capture_duration():
    # The fixture's packets span timestamps 0.0 -> 1.1s. At speed=1 this
    # should take roughly that long to fully replay (minus the very last
    # packet, which never needs a trailing sleep).
    replay = PcapReplay(str(FIXTURE_PATH), speed=1.0)
    start = time.monotonic()
    packets = list(replay.packets())
    elapsed = time.monotonic() - start

    assert len(packets) == 10
    assert 0.9 <= elapsed <= 2.0


def test_higher_speed_replays_faster():
    fast = PcapReplay(str(FIXTURE_PATH), speed=10.0)
    start = time.monotonic()
    list(fast.packets())
    elapsed = time.monotonic() - start

    # ~1.1s of capture time at 10x should take roughly ~0.11s, well under a
    # speed=1 replay of the same fixture.
    assert elapsed < 0.5
