"""Wires a packet source, the flow feature extractor, and a producer together.

This is the only place that needs to know about all three pieces; capture,
feature extraction, and publishing otherwise stay independent of each other,
which is what makes each one unit-testable on its own (see tests/).
"""

from __future__ import annotations

import logging
from typing import Optional

from .capture import LiveCapture
from .config import PipelineConfig
from .features import FlowFeatureExtractor
from .packet_source import PacketSource
from .producer import BufferedProducer, FlowEventProducer, KafkaFlowProducer, StubFlowProducer
from .replay import PcapReplay
from .schema import build_event

logger = logging.getLogger(__name__)


def build_packet_source(config: PipelineConfig) -> PacketSource:
    if config.pcap_path:
        return PcapReplay(config.pcap_path, speed=config.replay_speed)
    return LiveCapture(interface=config.interface, bpf_filter=config.bpf_filter)


def build_producer(config: PipelineConfig) -> BufferedProducer:
    inner: FlowEventProducer
    if config.use_stub_producer:
        inner = StubFlowProducer()
    else:
        inner = KafkaFlowProducer(bootstrap_servers=config.bootstrap_servers, topic=config.topic)
    return BufferedProducer(
        inner,
        queue_size=config.queue_size,
        max_retries=config.max_retries,
        retry_backoff_s=config.retry_backoff_s,
        drop_policy=config.drop_policy,
    )


class IngestionPipeline:
    """Consumes packets from `source`, tracks flows, and publishes closed
    flows' features through `producer` until the source is exhausted or
    `stop()` is called from another thread (e.g. on shutdown signal)."""

    def __init__(
        self,
        source: PacketSource,
        extractor: FlowFeatureExtractor,
        producer: BufferedProducer,
    ) -> None:
        self.source = source
        self.extractor = extractor
        self.producer = producer
        self.processed_packets = 0
        self.published_flows = 0

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "IngestionPipeline":
        return cls(
            source=build_packet_source(config),
            extractor=FlowFeatureExtractor(config.window_config),
            producer=build_producer(config),
        )

    def run(self) -> None:
        try:
            for pkt in self.source.packets():
                self.processed_packets += 1
                for feature_dict in self.extractor.process_packet(pkt):
                    self._publish(feature_dict)
        finally:
            for feature_dict in self.extractor.flush():
                self._publish(feature_dict)
            self.source.close()
            self.producer.close()

    def _publish(self, feature_dict: dict) -> None:
        self.producer.publish(build_event(feature_dict))
        self.published_flows += 1

    def stop(self) -> None:
        """Signal a live-capture source to stop; safe to call from another thread."""
        self.source.close()
