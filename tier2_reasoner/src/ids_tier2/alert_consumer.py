"""Reading alert events off the `ml` module's Kafka topic, filtering for
escalated ones client-side -- see `schema.py`'s module docstring for why
that's a deliberate simplification instead of a dedicated escalations
topic. Mirrors `ids_ml.flow_consumer`'s shape (abstract source, stub,
reconnect-on-failure Kafka implementation) rather than importing it, per
this project's module-boundary convention.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Iterable, Iterator, Optional

from .schema import ALERT_TOPIC, event_from_json, validate_alert_event

logger = logging.getLogger(__name__)


class EscalatedAlertSource(abc.ABC):
    """Yields validated alert events that were escalated (`escalated: true`)."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StubEscalatedAlertSource(EscalatedAlertSource):
    """Reads pre-published alerts from an in-memory list -- e.g. the
    `published` list from `ids_ml`'s `StubAlertProducer` in an end-to-end
    test, or any hand-built list of alert events."""

    def __init__(self, alerts: Iterable[Dict[str, Any]]) -> None:
        self._alerts = list(alerts)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for alert in self._alerts:
            validate_alert_event(alert)
            if alert.get("escalated"):
                yield alert


class KafkaEscalatedAlertSource(EscalatedAlertSource):
    """Reads and validates alert events from the real Kafka topic,
    yielding only the escalated ones. On an iteration error the underlying
    consumer is torn down so the next `__iter__` call reconnects fresh."""

    def __init__(
        self,
        bootstrap_servers,
        topic: str = ALERT_TOPIC,
        group_id: Optional[str] = "ids-tier2-reasoner",
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

        logger.info("Connecting KafkaEscalatedAlertSource to %s (topic=%s)", self.bootstrap_servers, self.topic)
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
                alert = record.value
                validate_alert_event(alert)
                if alert.get("escalated"):
                    yield alert
        except KafkaError:
            logger.warning("Kafka consume failed; will reconnect on next iteration", exc_info=True)
            self._consumer = None
            raise

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
