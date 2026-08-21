# Kafka Topic Schemas (authoritative)

This file is the single source of truth for the two Kafka topics that
connect the three modules. It supersedes the schema tables in each module's
own README (`ingestion/README.md`, `ml/README.md`, `dashboard-api/README.md`)
-- those still describe the same contract for readers working inside one
module, but if anything ever disagrees with this file, this file wins.

Each module independently implements its own copy of these two schemas
(`ingestion/src/ids_ingestion/schema.py`, `ml/src/ids_ml/schema.py`,
`dashboard-api/src/ids_dashboard/schema.py`) rather than importing a shared
package -- a deliberate design choice from the original per-module work, so
each module stays buildable and testable without the others installed. That
means the three copies can drift, which is exactly what this reconciliation
pass checked for.

## Reconciliation findings

The three independently-built schema copies were compared field-by-field,
type-by-type, both against each other and against `ids_ml.features.
CANONICAL_FEATURE_COLUMNS` (the live feature vector the ML module actually
builds from a consumed flow event -- the place a silent field-name mismatch
would be most dangerous, since a typo there fails silently: an unrecognized
feature name just defaults to `0.0` instead of raising). Verified by
`python -c` cross-checking the three `FLOW_EVENT_FIELDS`/`ALERT_EVENT_FIELDS`
dicts and `CANONICAL_FEATURE_COLUMNS` directly (not by reading the docs) --
see the integration commit for the exact check.

**Result: no runtime-breaking mismatch was found.** Topic names, field
names, and field types agree exactly on both hops (ingestion -> ML, ML ->
dashboard), and all 35 `CANONICAL_FEATURE_COLUMNS` entries have a matching
field in the ingestion module's actual `FLOW_EVENT_FIELDS`. This isn't
surprising in hindsight -- the ML and dashboard modules' schema copies were
originally written by directly mirroring the upstream module's schema at
the time -- but it was verified rather than assumed.

Two minor, non-breaking documentation gaps were found and are noted (not
"fixed" with an adapter, since nothing at runtime is actually broken):

1. **`ingestion`'s own `FLOW_EVENT_FIELDS` doesn't declare `schema_version`**,
   even though `ingestion`'s `schema.build_event()` always adds it before
   publishing, and both downstream modules' schema copies do declare it as
   required. The field is always present on the wire; only ingestion's own
   internal validation dict doesn't request it. No adapter is needed because
   there's nothing to reconcile between modules -- this is purely an
   omission in one module's self-check. Recorded here so it isn't
   rediscovered as a false alarm.
2. **`ingestion/src/ids_ingestion/schema.py`'s docstring references
   `docs/CONSUMER_CONTRACT.md`**, which was never actually created -- the
   real documentation ended up in `ingestion/README.md`'s "Consumer
   contract" section instead. This file (`SCHEMA.md`) is now the actual
   answer to that reference.

No code inside any module folder was changed to address either point --
per the integration scope, this file plus the "Known quirks for
integrators" section below are the fix.

## Topic: `network.flow.features`

Published by `ingestion`, consumed by `ml`.

| Field | Type | Notes |
|---|---|---|
| `flow_id` | string | Direction-independent flow key |
| `src_ip`, `dst_ip` | string | Forward direction = the flow's first observed packet |
| `src_port`, `dst_port` | integer | |
| `protocol` | integer | IANA transport protocol number (6=TCP, 17=UDP) |
| `flow_start_time`, `flow_end_time` | number | Unix epoch seconds (see "Timestamp semantics" below) |
| `flow_duration` | number | Seconds |
| `close_reason` | string | `fin`, `rst`, `idle_timeout`, `active_timeout`, or `flush` |
| `total_fwd_packets`, `total_bwd_packets` | integer | |
| `total_fwd_bytes`, `total_bwd_bytes` | integer | |
| `fwd_packet_length_{min,max,mean,std}` | number | |
| `bwd_packet_length_{min,max,mean,std}` | number | |
| `flow_bytes_per_sec`, `flow_packets_per_sec` | number | |
| `flow_iat_{mean,std,min,max}` | number | Inter-arrival time across both directions |
| `fwd_iat_{mean,std,min,max}` | number | Inter-arrival time, forward direction only |
| `bwd_iat_{mean,std,min,max}` | number | Inter-arrival time, backward direction only |
| `{syn,ack,fin,rst,psh,urg,ece,cwr}_flag_count` | integer | TCP flag counts observed in the flow (0 for UDP) |
| `schema_version` | integer | Currently `1`. Always present on the wire (see finding #1 above). |

**Consumed subset**: `ml` only reads the 35 numeric feature fields (everything
above except `flow_id`, `src_ip`/`dst_ip`/`src_port`/`dst_port`/`protocol`,
`flow_start_time`/`flow_end_time`, `close_reason`, and `schema_version`) into
its feature vector, via `ids_ml.features.CANONICAL_FEATURE_COLUMNS`. The
identity/metadata fields pass through unread by the model but are carried
into the alert event (see below).

## Topic: `network.ids.alerts`

Published by `ml`, consumed by `dashboard-api`.

| Field | Type | Notes |
|---|---|---|
| `alert_id` | string | UUID, generated per alert by `ml` |
| `flow_id`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `protocol`, `flow_start_time` | -- | Carried through unchanged from the source flow event |
| `scored_at` | number | Unix epoch seconds -- when `ml` scored the flow (see "Timestamp semantics") |
| `stage1_anomaly_score` | number | Higher = more anomalous |
| `stage1_flagged` | boolean | Whether the Isolation Forest pre-filter flagged the flow |
| `stage2_predicted_class` | string | `BENIGN` if stage 1 didn't flag it (stage 2 never ran), else the predicted attack category |
| `stage2_confidence` | number | Max class probability from stage 2; `1.0` when stage 2 didn't run |
| `stage2_class_probabilities` | object | Full per-class probability map; `{}` when stage 2 didn't run |
| `severity` | string | One of `info`/`low`/`medium`/`high`/`critical` |
| `explanation` | array | Top-k `{feature, value, shap_value}` TreeSHAP contributions; `[]` if stage 2 didn't run |
| `model_version` | string | e.g. `two-stage-v1` |
| `schema_version` | integer | Currently `1` |

By default `ml` only publishes alerts where `stage2_predicted_class !=
"BENIGN"`. Pass `--alert-on-stage1-flag-only` to `ml`'s `serve.py` to also
publish stage-1-flagged-but-benign events (useful for the dashboard's
stage-1 false-positive-rate stat -- see `dashboard-api/README.md`).

## Timestamp semantics (three distinct clocks, all Unix epoch seconds float)

Easy to conflate, so spelled out explicitly:

- **`flow_start_time` / `flow_end_time`** (flow event) -- when the actual
  packets were captured/replayed, taken from the packet timestamps
  themselves (`ingestion`).
- **`scored_at`** (alert event) -- wall-clock time `ml` finished scoring
  the flow. Can lag `flow_end_time` by however long the flow sat in Kafka
  plus scoring latency.
- **`received_at`** (dashboard-api's SQLite row, not on the wire) -- wall-
  clock time `dashboard-api`'s ingest thread wrote the alert to its store.
  Can lag `scored_at` by network/queue latency between `ml` and
  `dashboard-api`.

None of these are interchangeable; a query filtering "alerts in the last
hour" via the dashboard API's `start_time`/`end_time` params filters on
`scored_at`, not on when the underlying traffic occurred.

## Known quirks for integrators

- **CICFlowMeter's flag-count column naming**: if you ever feed real
  CICIDS2017/CIC-IDS2018 CSVs through `ml`'s training path
  (`ml/src/ids_ml/data.py`), note its `CWE Flag Count` column (a
  long-documented typo in the original CICFlowMeter tool) maps to
  `cwr_flag_count` -- unrelated to this topic's wire format, but a common
  point of confusion when cross-referencing `ml`'s training data against
  its live-serving schema.
- **`explanation` is only populated when stage 2 actually ran.** A
  `stage1_flagged: false` alert (only possible via
  `--alert-on-stage1-flag-only`) always has `explanation: []` -- there's no
  SHAP output for a prediction stage 2 never made.
- **Field presence is validated; field values mostly aren't**, beyond type
  and (for alerts) `severity` being a known level. `close_reason` and
  `stage2_predicted_class`, for example, are plain strings on the wire with
  no enum enforcement at the schema layer.
