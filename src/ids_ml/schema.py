"""Kafka topic contracts this module depends on.

Two contracts live here:

1. FLOW_EVENT_FIELDS / FLOW_TOPIC -- the *input* contract, published by the
   ingestion module. Duplicated here (rather than imported from
   `ids_ingestion`) so this module has no code dependency on ingestion --
   only a documented schema dependency, per the module boundary in the
   project README. If the ingestion module's schema changes, this copy and
   `ids_ml.features.CANONICAL_FEATURE_COLUMNS` must be updated to match.

2. ALERT_EVENT_FIELDS / ALERT_TOPIC -- the *output* contract this module
   publishes, for the dashboard/API module to consume.
"""

from __future__ import annotations

import json
from typing import Any, Dict

# --- Input contract: flow-feature events from the ingestion module --------

FLOW_TOPIC = "network.flow.features"
FLOW_SCHEMA_VERSION = 1

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
    "schema_version": "integer",
}


def validate_flow_event(event: Dict[str, Any]) -> None:
    """Shape check for an inbound flow-feature event. Raises ValueError."""
    missing = [name for name in FLOW_EVENT_FIELDS if name not in event]
    if missing:
        raise ValueError(f"flow event missing required fields: {missing}")


# --- Output contract: alerts published for the dashboard/API module -------

ALERT_TOPIC = "network.ids.alerts"
ALERT_SCHEMA_VERSION = 1

# severity is derived from stage2's predicted class + confidence; see
# pipeline.py for the exact rule. "info" covers stage1-only flags that
# stage2 resolved back down to benign (kept for dashboard visibility into
# the pre-filter's false-positive rate, not meant to page anyone).
SEVERITY_LEVELS = ("info", "low", "medium", "high", "critical")

ALERT_EVENT_FIELDS: Dict[str, str] = {
    "alert_id": "string",
    "flow_id": "string",
    "src_ip": "string",
    "src_port": "integer",
    "dst_ip": "string",
    "dst_port": "integer",
    "protocol": "integer",
    "flow_start_time": "number",
    "scored_at": "number",
    "stage1_anomaly_score": "number",
    "stage1_flagged": "boolean",
    "stage2_predicted_class": "string",
    "stage2_confidence": "number",
    "stage2_class_probabilities": "object",
    "severity": "string",
    "explanation": "array",
    "model_version": "string",
    "schema_version": "integer",
}

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


def validate_alert_event(event: Dict[str, Any]) -> None:
    missing = [name for name in ALERT_EVENT_FIELDS if name not in event]
    if missing:
        raise ValueError(f"alert event missing required fields: {missing}")
    wrong_type = []
    for name, expected_type in ALERT_EVENT_FIELDS.items():
        if not _TYPE_CHECKS[expected_type](event[name]):
            wrong_type.append((name, expected_type, type(event[name]).__name__))
    if wrong_type:
        raise ValueError(f"alert event fields have unexpected types: {wrong_type}")
    if event["severity"] not in SEVERITY_LEVELS:
        raise ValueError(f"unknown severity: {event['severity']!r}")


def event_to_json(event: Dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True)


def event_from_json(payload: str) -> Dict[str, Any]:
    return json.loads(payload)
