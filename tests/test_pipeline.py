"""End-to-end test: pcap fixture -> flow tracking -> producer -> consumer contract.

Runs the whole module against the same known fixture used in
tests/test_features.py, but this time checks the wiring (pipeline.py,
schema.py, producer.py, consumer_contract.py) rather than feature-value
correctness.
"""

from pathlib import Path

from ids_ingestion.config import PipelineConfig
from ids_ingestion.consumer_contract import StubFlowEventConsumer
from ids_ingestion.pipeline import IngestionPipeline

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_tcp.pcap"


def test_pipeline_publishes_valid_events_for_both_flows():
    config = PipelineConfig(
        pcap_path=str(FIXTURE_PATH),
        replay_speed=0,  # instant playback for a fast test
        use_stub_producer=True,
        active_timeout_s=120.0,
        idle_timeout_s=60.0,
    )
    pipeline = IngestionPipeline.from_config(config)
    pipeline.run()

    assert pipeline.processed_packets == 10

    stub = pipeline.producer._inner
    assert stub.closed
    # TCP flow closes on its own FIN; the UDP flow only closes when run()
    # flushes remaining flows at shutdown -- two events total.
    assert len(stub.published) == 2

    close_reasons = {event["close_reason"] for event in stub.published}
    assert close_reasons == {"fin", "flush"}

    # Reading the events back through the documented consumer contract must
    # succeed -- this is the guarantee the ML module depends on.
    consumed = list(StubFlowEventConsumer(stub.published))
    assert len(consumed) == 2
    for event in consumed:
        assert event["schema_version"] == 1
