"""The read side of the schema contract in schema.py.

This module is intentionally thin: it exists so the ML module (or anything
else downstream) has a documented, testable way to consume flow-feature
events without needing this ingestion module's internals or a live Kafka
broker. `KafkaFlowEventConsumer` is a real Kafka consumer for production use;
`StubFlowEventConsumer` reads from an in-memory list, e.g. `StubFlowProducer.
published`, for integration tests that only need this module's stubs, not
Kafka itself. No detection/ML logic belongs here -- see the project README
for module boundaries.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .schema import DEFAULT_TOPIC, event_from_json, validate_event


class FlowEventConsumer(abc.ABC):
    """Yields validated flow-feature events published to the contract topic."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StubFlowEventConsumer(FlowEventConsumer):
    """Reads pre-published events from an in-memory list (e.g. the
    `published` list on a `StubFlowProducer`), validating each one against
    the schema contract exactly like a real consumer would."""

    def __init__(self, events: Iterable[Dict[str, Any]]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for event in self._events:
            validate_event(event)
            yield event


class KafkaFlowEventConsumer(FlowEventConsumer):
    """Reads and validates events from the real Kafka topic.

    This is the reference implementation the ML module can copy or depend on
    directly; it deliberately does nothing with the events beyond
    deserializing and validating them.
    """

    def __init__(
        self,
        bootstrap_servers,
        topic: str = DEFAULT_TOPIC,
        group_id: Optional[str] = "ids-ingestion-flow-features",
        **client_kwargs,
    ) -> None:
        from kafka import KafkaConsumer

        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda raw: event_from_json(raw.decode("utf-8")),
            auto_offset_reset="latest",
            **client_kwargs,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for record in self._consumer:
            event = record.value
            validate_event(event)
            yield event

    def close(self) -> None:
        self._consumer.close()
