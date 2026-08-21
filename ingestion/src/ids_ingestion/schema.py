"""The event schema this module publishes -- the contract downstream consumers rely on.

Anything that changes the shape of `FLOW_EVENT_FIELDS` or the meaning of an
existing field is a breaking change and must bump `SCHEMA_VERSION`. See
docs/CONSUMER_CONTRACT.md for the full human-readable contract.
"""

from __future__ import annotations

import json
from typing import Any, Dict

SCHEMA_VERSION = 1
DEFAULT_TOPIC = "network.flow.features"

# name -> JSON type, kept in sync with the dict produced by
# ids_ingestion.features.extract_features(). Used both as documentation and
# by `validate_event` below for a lightweight shape check before publishing.
FLOW_EVENT_FIELDS: Dict[str, str] = {
    "flow_id": "string",
    "src_ip": "string",
    "src_port": "integer",
    "dst_ip": "string",
    "dst_port": "integer",
    "protocol": "integer",
    "flow_start_time": "number",
    "flow_end_time": "number",
    "flow_duration": "number",
    "close_reason": "string",
    "total_fwd_packets": "integer",
    "total_bwd_packets": "integer",
    "total_fwd_bytes": "integer",
    "total_bwd_bytes": "integer",
    "fwd_packet_length_min": "number",
    "fwd_packet_length_max": "number",
    "fwd_packet_length_mean": "number",
    "fwd_packet_length_std": "number",
    "bwd_packet_length_min": "number",
    "bwd_packet_length_max": "number",
    "bwd_packet_length_mean": "number",
    "bwd_packet_length_std": "number",
    "flow_bytes_per_sec": "number",
    "flow_packets_per_sec": "number",
    "flow_iat_mean": "number",
    "flow_iat_std": "number",
    "flow_iat_min": "number",
    "flow_iat_max": "number",
    "fwd_iat_mean": "number",
    "fwd_iat_std": "number",
    "fwd_iat_min": "number",
    "fwd_iat_max": "number",
    "bwd_iat_mean": "number",
    "bwd_iat_std": "number",
    "bwd_iat_min": "number",
    "bwd_iat_max": "number",
    "syn_flag_count": "integer",
    "ack_flag_count": "integer",
    "fin_flag_count": "integer",
    "rst_flag_count": "integer",
    "psh_flag_count": "integer",
    "urg_flag_count": "integer",
    "ece_flag_count": "integer",
    "cwr_flag_count": "integer",
}

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


def build_event(feature_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a raw feature dict with schema/topic envelope metadata."""
    event = dict(feature_dict)
    event["schema_version"] = SCHEMA_VERSION
    return event


def validate_event(event: Dict[str, Any]) -> None:
    """Raise ValueError if `event` doesn't match the documented contract.

    This is a shape check (fields present, roughly-right types), not a
    numeric-correctness check -- that's what tests/test_features.py covers.
    """
    missing = [name for name in FLOW_EVENT_FIELDS if name not in event]
    if missing:
        raise ValueError(f"event missing required fields: {missing}")

    wrong_type = []
    for name, expected_type in FLOW_EVENT_FIELDS.items():
        check = _TYPE_CHECKS[expected_type]
        if not check(event[name]):
            wrong_type.append((name, expected_type, type(event[name]).__name__))
    if wrong_type:
        raise ValueError(f"event fields have unexpected types: {wrong_type}")


def event_to_json(event: Dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True)


def event_from_json(payload: str) -> Dict[str, Any]:
    return json.loads(payload)
