"""Reading alerts off the ML module's Kafka topic.

Mirrors the shape of `ids_ml.flow_consumer` (which itself mirrors
`ids_ingestion.consumer_contract`): a small ABC, an in-memory stub for
tests/local development, and a real Kafka implementation that tears itself
down on failure so the next iteration reconnects from scratch.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Iterable, Iterator, Optional

from .schema import ALERT_TOPIC, event_from_json, validate_alert_event

logger = logging.getLogger(__name__)


class AlertEventSource(abc.ABC):
    """Yields validated alert events."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StubAlertEventSource(AlertEventSource):
    """Reads pre-published events from an in-memory list -- e.g. the
    `published` list from ids_ml's StubAlertProducer in an end-to-end test."""

    def __init__(self, events: Iterable[Dict[str, Any]] = ()) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for event in self._events:
            validate_alert_event(event)
            yield event


class KafkaAlertEventSource(AlertEventSource):
    """Reads and validates alerts from the real Kafka topic.

    `consumer_timeout_ms` (default: kafka-python's own default of blocking
    forever) can be set so the iterator periodically raises StopIteration,
    letting a caller running this on a background thread check a stop flag
    between alerts instead of only when the next one arrives -- see
    ingest_service.py's known-limitations note on shutdown timing.
    """

    def __init__(
        self,
        bootstrap_servers,
        topic: str = ALERT_TOPIC,
        group_id: Optional[str] = "ids-dashboard",
        **client_kwargs,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self._client_kwargs = client_kwargs
        self._consumer = None

    def _ensure_connected(self):
        if self._consumer is not None:
            return self._consumer
        from kafka import KafkaConsumer

        logger.info("Connecting KafkaAlertEventSource to %s (topic=%s)", self.bootstrap_servers, self.topic)
        self._consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda raw: event_from_json(raw.decode("utf-8")),
            auto_offset_reset="latest",
            **self._client_kwargs,
        )
        return self._consumer

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        from kafka.errors import KafkaError

        consumer = self._ensure_connected()
        try:
            for record in consumer:
                event = record.value
                validate_alert_event(event)
                yield event
        except KafkaError:
            logger.warning("Kafka consume failed; will reconnect on next iteration", exc_info=True)
            self._consumer = None
            raise

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
