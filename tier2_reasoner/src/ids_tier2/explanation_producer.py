"""Publishing explanation events to this module's output Kafka topic.

Simpler than `ids_ml.alert_producer`'s `BufferedAlertProducer` (no
bounded-queue/retry/backpressure wrapper) -- a deliberate scope
reduction, not an oversight: this module's throughput ceiling is the LLM
call itself (hundreds of ms to seconds per explanation, see the README's
latency section), several orders of magnitude slower than a Kafka
publish, so a slow/unavailable broker is not the bottleneck backpressure
handling exists to protect against here the way it is for the
per-flow-rate `ml`/`ingestion` producers. Add that wrapper if this module
is ever deployed somewhere that assumption doesn't hold.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict

from .schema import EXPLANATION_TOPIC, event_to_json

logger = logging.getLogger(__name__)


class ExplanationProducer(abc.ABC):
    @abc.abstractmethod
    def publish(self, explanation: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StubExplanationProducer(ExplanationProducer):
    """In-memory producer for tests and for downstream modules
    (dashboard-api) to develop against without a Kafka broker."""

    def __init__(self) -> None:
        self.published: list[Dict[str, Any]] = []
        self.closed = False

    def publish(self, explanation: Dict[str, Any]) -> None:
        self.published.append(explanation)

    def close(self) -> None:
        self.closed = True


class KafkaExplanationProducer(ExplanationProducer):
    """Publishes explanations to Kafka as JSON, keyed by alert_id so a
    consumer can join them back to the originating alert. Tears down the
    client on any publish failure so the next publish() reconnects fresh."""

    def __init__(
        self,
        bootstrap_servers,
        topic: str = EXPLANATION_TOPIC,
        request_timeout_s: float = 10.0,
        **client_kwargs,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.request_timeout_s = request_timeout_s
        self._client_kwargs = client_kwargs
        self._producer = None

    def _ensure_connected(self):
        if self._producer is not None:
            return self._producer
        from kafka import KafkaProducer

        logger.info("Connecting KafkaExplanationProducer to %s", self.bootstrap_servers)
        self._producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: event_to_json(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
            acks="all",
            **self._client_kwargs,
        )
        return self._producer

    def publish(self, explanation: Dict[str, Any]) -> None:
        from kafka.errors import KafkaError

        producer = self._ensure_connected()
        try:
            future = producer.send(self.topic, key=explanation.get("alert_id"), value=explanation)
            future.get(timeout=self.request_timeout_s)
        except KafkaError:
            logger.warning("Kafka explanation publish failed; will reconnect on next attempt", exc_info=True)
            self._producer = None
            raise

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush(timeout=self.request_timeout_s)
            self._producer.close(timeout=self.request_timeout_s)
            self._producer = None
