"""Publishing alerts to the output Kafka topic, with backpressure/reconnection.

Deliberately mirrors the shape of the ingestion module's producer.py
(bounded queue, retry-with-backoff, drop-oldest backpressure, reconnect on
failure) rather than importing it, so this module has no code dependency on
`ids_ingestion` -- only the documented topic contract in schema.py.
"""

from __future__ import annotations

import abc
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

from .schema import ALERT_TOPIC, event_to_json

logger = logging.getLogger(__name__)


class AlertEventProducer(abc.ABC):
    @abc.abstractmethod
    def publish(self, alert: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StubAlertProducer(AlertEventProducer):
    """In-memory producer for tests and for downstream modules (dashboard/API)
    to develop against without a Kafka broker."""

    def __init__(self, fail_next: int = 0) -> None:
        self.published: list[Dict[str, Any]] = []
        self._fail_next = fail_next
        self.closed = False

    def publish(self, alert: Dict[str, Any]) -> None:
        if self._fail_next > 0:
            self._fail_next -= 1
            raise ConnectionError("StubAlertProducer: simulated transient failure")
        self.published.append(alert)

    def close(self) -> None:
        self.closed = True


class KafkaAlertProducer(AlertEventProducer):
    """Publishes alerts to Kafka as JSON, keyed by flow_id. Tears down the
    client on any publish failure so the next publish() reconnects fresh."""

    def __init__(
        self,
        bootstrap_servers,
        topic: str = ALERT_TOPIC,
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

        logger.info("Connecting KafkaAlertProducer to %s", self.bootstrap_servers)
        self._producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: event_to_json(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
            acks="all",
            **self._client_kwargs,
        )
        return self._producer

    def publish(self, alert: Dict[str, Any]) -> None:
        from kafka.errors import KafkaError

        producer = self._ensure_connected()
        try:
            future = producer.send(self.topic, key=alert.get("flow_id"), value=alert)
            future.get(timeout=self.request_timeout_s)
        except KafkaError:
            logger.warning("Kafka alert publish failed; will reconnect on next attempt", exc_info=True)
            self._producer = None
            raise

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush(timeout=self.request_timeout_s)
            self._producer.close(timeout=self.request_timeout_s)
            self._producer = None


class BufferedAlertProducer:
    """Bounded queue + background worker in front of an AlertEventProducer,
    for backpressure and retry/reconnection -- see producer.py's counterpart
    in the ingestion module for the identical design rationale."""

    def __init__(
        self,
        inner: AlertEventProducer,
        queue_size: int = 10_000,
        max_retries: int = 5,
        retry_backoff_s: float = 0.5,
        max_backoff_s: float = 30.0,
        drop_policy: str = "drop_oldest",
    ) -> None:
        if drop_policy not in ("drop_oldest", "block"):
            raise ValueError(f"unknown drop_policy: {drop_policy!r}")

        self._inner = inner
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=queue_size)
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._max_backoff_s = max_backoff_s
        self._drop_policy = drop_policy

        self._lock = threading.Lock()
        self._published = 0
        self._dropped = 0
        self._retries = 0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="BufferedAlertProducer", daemon=True)
        self._thread.start()

    def publish(self, alert: Dict[str, Any]) -> None:
        if self._drop_policy == "block":
            self._queue.put(alert)
            return
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            try:
                self._queue.get_nowait()
                with self._lock:
                    self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(alert)
            except queue.Full:
                with self._lock:
                    self._dropped += 1

    def _run(self) -> None:
        while not (self._stop_event.is_set() and self._queue.empty()):
            try:
                alert = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._publish_with_retry(alert)

    def _publish_with_retry(self, alert: Dict[str, Any]) -> None:
        attempt = 0
        while True:
            try:
                self._inner.publish(alert)
                with self._lock:
                    self._published += 1
                return
            except Exception:  # noqa: BLE001 - any transport failure is retryable here
                attempt += 1
                with self._lock:
                    self._retries += 1
                if attempt > self._max_retries:
                    logger.error(
                        "Dropping alert %s after %d failed publish attempts",
                        alert.get("alert_id"),
                        attempt,
                        exc_info=True,
                    )
                    with self._lock:
                        self._dropped += 1
                    return
                backoff = min(self._retry_backoff_s * (2 ** (attempt - 1)), self._max_backoff_s)
                logger.warning(
                    "Alert publish attempt %d/%d failed for %s; retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    alert.get("alert_id"),
                    backoff,
                )
                time.sleep(backoff)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "published": self._published,
                "dropped": self._dropped,
                "retries": self._retries,
                "queue_depth": self._queue.qsize(),
            }

    def close(self, timeout: Optional[float] = 10.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._inner.close()
