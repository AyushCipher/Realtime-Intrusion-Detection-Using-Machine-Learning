from ids_ingestion.features import extract_features
from ids_ingestion.flow import FlowState, make_flow_key
from ids_ingestion.packet_source import PacketMeta
from ids_ingestion.schema import (
    SCHEMA_VERSION,
    build_event,
    event_from_json,
    event_to_json,
    validate_event,
)


def _sample_feature_dict():
    pkt = PacketMeta(timestamp=0.0, src_ip="1.1.1.1", dst_ip="2.2.2.2",
                      src_port=111, dst_port=222, protocol=6, length=60, tcp_flags=2)
    state = FlowState.start(pkt, make_flow_key(pkt))
    state.add_packet(pkt)
    return extract_features(state, close_reason="flush")


def test_build_event_adds_schema_version():
    event = build_event(_sample_feature_dict())
    assert event["schema_version"] == SCHEMA_VERSION


def test_extracted_features_satisfy_the_published_contract():
    # Every field extract_features() produces must match the documented
    # schema in FLOW_EVENT_FIELDS -- this is what keeps schema.py honest as
    # features.py evolves.
    event = build_event(_sample_feature_dict())
    validate_event(event)  # raises on mismatch


def test_json_roundtrip_preserves_values():
    event = build_event(_sample_feature_dict())
    payload = event_to_json(event)
    restored = event_from_json(payload)
    assert restored == event
