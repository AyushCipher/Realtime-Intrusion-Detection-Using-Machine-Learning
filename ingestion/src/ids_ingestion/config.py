"""Pipeline configuration and the factory that wires a pipeline together."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .features import WindowConfig
from .schema import DEFAULT_TOPIC


@dataclass
class PipelineConfig:
    # Packet source: exactly one of (interface) or (pcap_path) is expected.
    interface: Optional[str] = None
    bpf_filter: str = "tcp or udp"
    pcap_path: Optional[str] = None
    replay_speed: float = 1.0

    # Sliding window.
    active_timeout_s: float = 120.0
    idle_timeout_s: float = 60.0

    # Kafka producer. use_stub_producer=True publishes to an in-memory stub
    # instead of a real broker -- useful for local development/testing of
    # everything upstream of Kafka without standing up a cluster.
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    topic: str = DEFAULT_TOPIC
    use_stub_producer: bool = False

    # Backpressure.
    queue_size: int = 10_000
    max_retries: int = 5
    retry_backoff_s: float = 0.5
    drop_policy: str = "drop_oldest"

    @property
    def window_config(self) -> WindowConfig:
        return WindowConfig(active_timeout=self.active_timeout_s, idle_timeout=self.idle_timeout_s)
