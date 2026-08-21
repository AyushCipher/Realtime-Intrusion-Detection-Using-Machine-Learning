"""Reading flow-feature events off the ingestion module's Kafka topic.

This is the input side of the module boundary: everything here depends only
on the documented schema in schema.py (FLOW_TOPIC / FLOW_EVENT_FIELDS), not
on the ingestion module's code.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Iterable, Iterator, Optional

from .schema import FLOW_TOPIC, event_from_json, validate_flow_event

logger = logging.getLogger(__name__)


class FlowEventSource(abc.ABC):
    """Yields validated flow-feature events."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StubFlowEventSource(FlowEventSource):
    """Reads pre-published events from an in-memory list -- e.g. the
    `published` list from ids_ingestion's StubFlowProducer in an end-to-end
    test, or any hand-built list of flow events."""

    def __init__(self, events: Iterable[Dict[str, Any]]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for event in self._events:
            validate_flow_event(event)
            yield event


class KafkaFlowEventSource(FlowEventSource):
    """Reads and validates flow-feature events from the real Kafka topic.

    On an iteration error the underlying consumer is torn down so the next
    `__iter__` call reconnects from scratch, rather than continuing to pull
    from a client stuck against a dead broker connection.
    """

    def __init__(
        self,
        bootstrap_servers,
        topic: str = FLOW_TOPIC,
        group_id: Optional[str] = "ids-ml-scoring",
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

        logger.info("Connecting KafkaFlowEventSource to %s (topic=%s)", self.bootstrap_servers, self.topic)
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
                validate_flow_event(event)
                yield event
        except KafkaError:
            logger.warning("Kafka consume failed; will reconnect on next iteration", exc_info=True)
            self._consumer = None
            raise

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
