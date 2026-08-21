"""Tests for BufferedProducer's retry/reconnection and backpressure behavior.

These exercise BufferedProducer against StubFlowProducer / small hand-written
fakes only -- no live Kafka broker is needed, matching the "stub the
downstream Kafka topic" scoping for this module.
"""

import threading
import time

from ids_ingestion.producer import BufferedProducer, FlowEventProducer, StubFlowProducer


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_buffered_producer_publishes_events_to_inner():
    stub = StubFlowProducer()
    bp = BufferedProducer(stub, queue_size=10, max_retries=3, retry_backoff_s=0.01)

    bp.publish({"flow_id": "a"})
    bp.publish({"flow_id": "b"})

    assert _wait_until(lambda: len(stub.published) == 2)
    stats = bp.stats()
    assert stats["published"] == 2
    assert stats["dropped"] == 0

    bp.close()
    assert stub.closed


def test_buffered_producer_retries_transient_failures_then_succeeds():
    stub = StubFlowProducer(fail_next=2)
    bp = BufferedProducer(stub, queue_size=10, max_retries=5, retry_backoff_s=0.01)

    bp.publish({"flow_id": "retry-me"})

    assert _wait_until(lambda: bp.stats()["published"] == 1)
    stats = bp.stats()
    assert stats["retries"] == 2
    assert stats["dropped"] == 0
    assert stub.published == [{"flow_id": "retry-me"}]

    bp.close()


def test_buffered_producer_drops_after_exhausting_retries():
    stub = StubFlowProducer(fail_next=1000)  # always fails
    bp = BufferedProducer(stub, queue_size=10, max_retries=2, retry_backoff_s=0.01)

    bp.publish({"flow_id": "doomed"})

    assert _wait_until(lambda: bp.stats()["dropped"] == 1)
    stats = bp.stats()
    assert stats["published"] == 0
    assert stats["retries"] == 3  # attempts 1, 2, and the final one that exceeds max_retries

    bp.close()


class _GatedProducer(FlowEventProducer):
    """Blocks each publish() until the test releases `gate`, so the test can
    control exactly when the background worker drains the queue."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.published = []

    def publish(self, event):
        self.gate.wait()
        self.published.append(event)

    def close(self):
        pass


def test_drop_oldest_backpressure_sheds_the_oldest_buffered_event():
    inner = _GatedProducer()
    bp = BufferedProducer(inner, queue_size=2, max_retries=0, drop_policy="drop_oldest")

    # First publish is picked up by the worker immediately and blocks there,
    # leaving the internal queue empty and ready to fill up.
    bp.publish({"flow_id": "a"})
    _wait_until(lambda: bp.stats()["queue_depth"] == 0)

    bp.publish({"flow_id": "b"})  # queue: [b]
    bp.publish({"flow_id": "c"})  # queue: [b, c] (at capacity)
    bp.publish({"flow_id": "d"})  # queue full -> drop "b", queue: [c, d]

    assert bp.stats()["dropped"] == 1
    assert bp.stats()["queue_depth"] == 2

    inner.gate.set()  # let the worker finish "a" and drain the rest

    assert _wait_until(lambda: len(inner.published) == 3)
    assert [e["flow_id"] for e in inner.published] == ["a", "c", "d"]

    bp.close()


def test_block_policy_never_drops():
    stub = StubFlowProducer()
    bp = BufferedProducer(stub, queue_size=1, max_retries=0, drop_policy="block")

    for i in range(5):
        bp.publish({"flow_id": str(i)})

    assert _wait_until(lambda: bp.stats()["published"] == 5)
    assert bp.stats()["dropped"] == 0

    bp.close()
