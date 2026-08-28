"""Reading Tier 2 explanations off tier2_reasoner's Kafka topic.

Mirrors alert_consumer.py's shape exactly (ABC, in-memory stub, reconnect-
on-failure Kafka implementation) -- a second, independent consumer for a
second, independent topic, not a variant of the alert consumer. Alerts and
explanations arrive on separate topics from separate producers with no
ordering guarantee between them (see store.py's insert_explanation
docstring), so this has to be its own consumer group, not a filter bolted
onto AlertEventSource.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Iterable, Iterator, Optional

from .schema import EXPLANATION_TOPIC, event_from_json, validate_explanation_event

logger = logging.getLogger(__name__)


class ExplanationEventSource(abc.ABC):
    """Yields validated explanation events."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StubExplanationEventSource(ExplanationEventSource):
    """Reads pre-published events from an in-memory list -- e.g. the
    `published` list from tier2_reasoner's StubExplanationProducer in an
    end-to-end test."""

    def __init__(self, events: Iterable[Dict[str, Any]] = ()) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for event in self._events:
            validate_explanation_event(event)
            yield event


class KafkaExplanationEventSource(ExplanationEventSource):
    """Reads and validates explanation events from the real Kafka topic.

    On an iteration error the underlying consumer is torn down so the next
    `__iter__` call reconnects from scratch.
    """

    def __init__(
        self,
        bootstrap_servers,
        topic: str = EXPLANATION_TOPIC,
        group_id: Optional[str] = "ids-dashboard-explanations",
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

        logger.info("Connecting KafkaExplanationEventSource to %s (topic=%s)", self.bootstrap_servers, self.topic)
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
                validate_explanation_event(event)
                yield event
        except KafkaError:
            logger.warning("Kafka consume failed; will reconnect on next iteration", exc_info=True)
            self._consumer = None
            raise

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
