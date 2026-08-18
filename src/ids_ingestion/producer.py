"""Publishing flow-feature events to a stream, plus backpressure/reconnection.

`FlowEventProducer` is the interface the rest of the pipeline depends on.
`KafkaFlowProducer` is the real implementation; `StubFlowProducer` is an
in-memory stand-in used by tests and by anything (including the ML module)
that wants to exercise this pipeline without a live Kafka broker.
`BufferedProducer` wraps either one with a bounded queue, a background
publisher thread, retry-with-backoff, and a configurable drop policy so a
slow or unreachable broker degrades instead of blocking packet processing
indefinitely.
"""

from __future__ import annotations

import abc
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

from .schema import DEFAULT_TOPIC, event_to_json

logger = logging.getLogger(__name__)


class FlowEventProducer(abc.ABC):
    """Publishes one flow-feature event at a time. Implementations may raise
    on transient failure -- callers (typically `BufferedProducer`) are
    expected to retry."""

    @abc.abstractmethod
    def publish(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class StubFlowProducer(FlowEventProducer):
    """In-memory producer for tests and for downstream modules (e.g. the ML
    pipeline) to develop against without a Kafka broker.

    `fail_next` lets a test simulate N consecutive transient publish
    failures before the stub starts succeeding, to exercise retry logic.
    """

    def __init__(self, fail_next: int = 0) -> None:
        self.published: list[Dict[str, Any]] = []
        self._fail_next = fail_next
        self.closed = False

    def publish(self, event: Dict[str, Any]) -> None:
        if self._fail_next > 0:
            self._fail_next -= 1
            raise ConnectionError("StubFlowProducer: simulated transient failure")
        self.published.append(event)

    def close(self) -> None:
        self.closed = True


class KafkaFlowProducer(FlowEventProducer):
    """Publishes events to a Kafka topic as JSON, keyed by flow_id so all
    records for one flow land on the same partition (ordering per flow).

    On any publish failure the underlying client is torn down so the next
    `publish()` call reconnects from scratch, rather than continuing to use
    a client stuck against a dead broker connection.
    """

    def __init__(
        self,
        bootstrap_servers,
        topic: str = DEFAULT_TOPIC,
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

        logger.info("Connecting KafkaFlowProducer to %s", self.bootstrap_servers)
        self._producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: event_to_json(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
            acks="all",
            **self._client_kwargs,
        )
        return self._producer

    def publish(self, event: Dict[str, Any]) -> None:
        from kafka.errors import KafkaError

        producer = self._ensure_connected()
        try:
            future = producer.send(self.topic, key=event.get("flow_id"), value=event)
            future.get(timeout=self.request_timeout_s)
        except KafkaError:
            logger.warning("Kafka publish failed; will reconnect on next attempt", exc_info=True)
            self._producer = None
            raise

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush(timeout=self.request_timeout_s)
            self._producer.close(timeout=self.request_timeout_s)
            self._producer = None


class BufferedProducer:
    """Bounded queue + background worker in front of a `FlowEventProducer`.

    This is where backpressure and retry/reconnection policy live, kept
    separate from the transport (`KafkaFlowProducer`) so both can be tested
    independently: transport correctness needs a broker (or a mock of one),
    backpressure/retry behavior does not.

    drop_policy:
        "drop_oldest" (default) -- when the queue is full, discard the
            oldest buffered event to make room for the new one. Bounds
            memory and staleness at the cost of losing data under sustained
            overload.
        "block" -- publish() blocks the caller until space is available.
            Never drops, but propagates backpressure upstream into the
            feature-extraction loop.
    """

    def __init__(
        self,
        inner: FlowEventProducer,
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
        self._thread = threading.Thread(target=self._run, name="BufferedProducer", daemon=True)
        self._thread.start()

    def publish(self, event: Dict[str, Any]) -> None:
        if self._drop_policy == "block":
            self._queue.put(event)
            return

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()
                with self._lock:
                    self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._lock:
                    self._dropped += 1

    def _run(self) -> None:
        while not (self._stop_event.is_set() and self._queue.empty()):
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._publish_with_retry(event)

    def _publish_with_retry(self, event: Dict[str, Any]) -> None:
        attempt = 0
        while True:
            try:
                self._inner.publish(event)
                with self._lock:
                    self._published += 1
                return
            except Exception:  # noqa: BLE001 - any transport failure is retryable here
                attempt += 1
                with self._lock:
                    self._retries += 1
                if attempt > self._max_retries:
                    logger.error(
                        "Dropping event %s after %d failed publish attempts",
                        event.get("flow_id"),
                        attempt,
                        exc_info=True,
                    )
                    with self._lock:
                        self._dropped += 1
                    return
                backoff = min(self._retry_backoff_s * (2 ** (attempt - 1)), self._max_backoff_s)
                logger.warning(
                    "Publish attempt %d/%d failed for %s; retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    event.get("flow_id"),
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
