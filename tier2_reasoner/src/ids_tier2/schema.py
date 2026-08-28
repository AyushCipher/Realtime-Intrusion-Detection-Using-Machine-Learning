"""Kafka topic contracts this module depends on.

Two contracts live here:

1. The *input* contract -- this module consumes `ml`'s existing
   `network.ids.alerts` topic (schema owned by `ids_ml.schema`; duplicated
   here per the same "each module keeps its own copy" boundary the other
   three modules already use, so this module has no code dependency on
   `ids_ml`). Only the fields this module actually reads are declared as
   required here; `ids_ml`'s alert carries more fields than that.
2. The *output* contract -- `network.ids.explanations`, published by this
   module, consumed by `dashboard-api` (once wired up there; see the
   top-level README's known-limitations section for what that still
   needs).

Deliberate simplification from the original plan sketch: rather than
`ml` publishing a second, pre-filtered `network.ids.escalations` topic,
this module subscribes to the existing `network.ids.alerts` topic and
filters client-side for `escalated == true` (see `alert_consumer.py`).
That keeps `ml`'s producer side completely unchanged -- no new Kafka
producer, no new topic for it to manage -- at the cost of this consumer
group seeing (and cheaply discarding) every alert, not just the escalated
ones. Since discarding a JSON message client-side is orders of magnitude
cheaper than an LLM call, this doesn't reintroduce the "LLM on the hot
path" problem the escalation gate exists to avoid; it would only matter
at alert volumes far beyond what this project's demo pipeline produces.
"""

from __future__ import annotations

import json
from typing import Any, Dict

# --- Input contract: alerts from the ml module -----------------------

ALERT_TOPIC = "network.ids.alerts"

# Only the fields this module actually reads. ids_ml.schema.ALERT_EVENT_FIELDS
# is the authoritative full contract; alerts carry more fields than this.
REQUIRED_ALERT_FIELDS = (
    "alert_id",
    "flow_id",
    "src_ip",
    "dst_ip",
    "dst_port",
    "protocol",
    "stage2_predicted_class",
    "stage2_confidence",
    "severity",
    "explanation",
    "unknown_mass",
    "escalated",
    "escalation_trigger",
)


def validate_alert_event(event: Dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_ALERT_FIELDS if name not in event]
    if missing:
        raise ValueError(f"alert event missing required fields: {missing}")


# --- Output contract: explanations published by this module ----------

EXPLANATION_TOPIC = "network.ids.explanations"
EXPLANATION_SCHEMA_VERSION = 1

EXPLANATION_EVENT_FIELDS: Dict[str, str] = {
    "explanation_id": "string",  # UUID, generated per explanation
    "alert_id": "string",  # joins back to the originating alert
    "flow_id": "string",
    "generated_at": "number",  # Unix epoch seconds
    "suspected_technique_id": "string",  # e.g. "T1110" ; "" if retrieval found nothing above threshold
    "suspected_technique_name": "string",
    "risk_explanation": "string",  # the LLM's free-text reasoning
    "recommended_action": "string",
    "retrieved_technique_ids": "array",  # every technique_id retrieval surfaced, not just the top one
    "rag_enabled": "boolean",  # False when run with --no-rag (the H3 ablation switch)
    "llm_latency_ms": "number",
    "model_version": "string",  # e.g. "stub-v1" or the real LLM client's model name
    "schema_version": "integer",
}

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
}


def validate_explanation_event(event: Dict[str, Any]) -> None:
    missing = [name for name in EXPLANATION_EVENT_FIELDS if name not in event]
    if missing:
        raise ValueError(f"explanation event missing required fields: {missing}")
    wrong_type = []
    for name, expected_type in EXPLANATION_EVENT_FIELDS.items():
        if not _TYPE_CHECKS[expected_type](event[name]):
            wrong_type.append((name, expected_type, type(event[name]).__name__))
    if wrong_type:
        raise ValueError(f"explanation event fields have unexpected types: {wrong_type}")


def event_to_json(event: Dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True)


def event_from_json(payload: str) -> Dict[str, Any]:
    return json.loads(payload)
