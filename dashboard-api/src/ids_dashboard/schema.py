"""The input Kafka contract this module depends on: alerts published by the
ML module (`ids_ml`). Duplicated here (rather than imported from `ids_ml`)
so this module has no code dependency on it -- only a documented schema
dependency, same pattern as `ids_ml`'s own duplicated copy of the ingestion
module's flow-event schema. If the ML module's alert schema changes, this
copy must be updated to match.
"""

from __future__ import annotations

import json
from typing import Any, Dict

ALERT_TOPIC = "network.ids.alerts"
ALERT_SCHEMA_VERSION = 1

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
    # Open-set fields (see ids_ml.pipeline's module docstring). Added here
    # once this module actually started storing/serving them -- previously
    # these passed through unvalidated since they weren't declared
    # required (see SCHEMA.md's reconciliation note on this).
    "unknown_mass": "number",
    "escalated": "boolean",
    "escalation_trigger": "string",
    "model_version": "string",
    "schema_version": "integer",
}

# --- Output-side contract this module also depends on: tier2_reasoner's
# explanations, joined onto alerts by alert_id. Duplicated from
# tier2_reasoner's own schema.py, same "each module keeps its own copy"
# boundary as the alert contract above -- no code dependency on
# tier2_reasoner either. -------------------------------------------------

EXPLANATION_TOPIC = "network.ids.explanations"
EXPLANATION_SCHEMA_VERSION = 1

EXPLANATION_EVENT_FIELDS: Dict[str, str] = {
    "explanation_id": "string",
    "alert_id": "string",
    "flow_id": "string",
    "generated_at": "number",
    "suspected_technique_id": "string",
    "suspected_technique_name": "string",
    "risk_explanation": "string",
    "recommended_action": "string",
    "retrieved_technique_ids": "array",
    "rag_enabled": "boolean",
    "llm_latency_ms": "number",
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


def _validate_event(event: Dict[str, Any], fields: Dict[str, str]) -> None:
    missing = [name for name in fields if name not in event]
    if missing:
        raise ValueError(f"event missing required fields: {missing}")
    wrong_type = []
    for name, expected_type in fields.items():
        if not _TYPE_CHECKS[expected_type](event[name]):
            wrong_type.append((name, expected_type, type(event[name]).__name__))
    if wrong_type:
        raise ValueError(f"event fields have unexpected types: {wrong_type}")


def validate_alert_event(event: Dict[str, Any]) -> None:
    _validate_event(event, ALERT_EVENT_FIELDS)
    if event["severity"] not in SEVERITY_LEVELS:
        raise ValueError(f"unknown severity: {event['severity']!r}")


def validate_explanation_event(event: Dict[str, Any]) -> None:
    _validate_event(event, EXPLANATION_EVENT_FIELDS)


def event_from_json(payload: str) -> Dict[str, Any]:
    return json.loads(payload)
