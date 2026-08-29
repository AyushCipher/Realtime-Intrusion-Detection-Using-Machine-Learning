# Alert Dashboard & API Layer (`ids_dashboard`)

FastAPI backend and React frontend for triaging alerts from the ML module,
including open-set escalation status and Tier 2 (LLM/RAG) explanations.
Consumes `network.ids.alerts` (schema in `schema.py`, duplicated from
`ids_ml`'s own copy) and `network.ids.explanations` (duplicated from
`tier2_reasoner`'s own copy) -- two independent consumers, since the two
topics come from two different producers with no delivery-order guarantee
between them (see `explanation_consumer.py`'s module docstring). Stores
both for historical query/filter, pushes both live over one WebSocket
connection, and serves REST endpoints for the dashboard. This package
does not implement ingestion or any detection/reasoning model -- see
[Out of scope](#out-of-scope).

## Architecture

```
Kafka: network.ids.alerts        IngestService                  FastAPI
(from ids_ml)                -->  (background thread       -->  REST  /api/alerts, /api/alerts/summary,
                                    consuming Kafka,               /api/alerts/{id}/triage,
                                    writing to SQLite,              /api/alerts/{id}/explanation,
                                    broadcasting)                   /api/ws-token
                                          |                   -->  WebSocket  /ws/alerts (token-gated,
Kafka: network.ids.explanations  ExplanationIngestService          carries both alert and explanation
(from tier2_reasoner)        -->  (independent background          broadcasts -- see routes_ws.py)
                                    thread, same store +
                                    broadcaster)
                                          |
                                          v
                                   SQLite (alerts.db)
                                   alerts + explanations + triage state
```

- `schema.py` -- the alert *and* explanation topic contracts (both duplicated from their producer module's own copy).
- `store.py` -- SQLite-backed alert + explanation history, filtering (including by `escalated`), and analyst triage state.
- `alert_consumer.py` / `explanation_consumer.py` -- the two independent Kafka sources (real + in-memory stub each), mirroring `ids_ml.flow_consumer`'s shape.
- `broadcaster.py` -- in-process WebSocket fan-out, shared by both ingest services (explanation broadcasts carry a `__type: "explanation"` marker; alert broadcasts are unmarked, unchanged from before -- see `explanation_ingest_service.py`).
- `ingest_service.py` / `explanation_ingest_service.py` -- the two background threads tying each consumer -> store -> broadcaster.
- `auth.py` -- HTTP Basic auth + short-lived WebSocket tokens.
- `routes_alerts.py` / `routes_ws.py` / `app.py` -- the FastAPI surface.
- `frontend/` (top-level, sibling to `src/`) -- the React dashboard: live feed and triage view show an ESCALATED badge; the alert detail panel shows `unknown_mass`/`escalation_trigger` and, for escalated alerts, fetches (and live-updates) the Tier 2 explanation.

## Running

Backend:

```
pip install -r requirements-dashboard.txt
export IDS_DASHBOARD_USERNAME=analyst
export IDS_DASHBOARD_PASSWORD=<a real password>
export IDS_DASHBOARD_BOOTSTRAP_SERVERS=localhost:9092   # or IDS_DASHBOARD_USE_STUB=true for no Kafka
python -m ids_dashboard
```

Frontend: see `frontend/README.md`.

Environment variables (all optional except the auth pair, which are
required -- the process refuses to start without them):

| Variable | Default | Purpose |
|---|---|---|
| `IDS_DASHBOARD_USERNAME` / `IDS_DASHBOARD_PASSWORD` | *(required)* | The single shared Basic-auth credential pair |
| `IDS_DASHBOARD_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka bootstrap servers, comma-separated |
| `IDS_DASHBOARD_ALERT_TOPIC` | `network.ids.alerts` | Alert topic to consume |
| `IDS_DASHBOARD_EXPLANATION_TOPIC` | `network.ids.explanations` | Tier 2 explanation topic to consume |
| `IDS_DASHBOARD_USE_STUB` | `false` | Run against empty in-memory sources (both alerts and explanations) instead of Kafka |
| `IDS_DASHBOARD_DB_PATH` | `alerts.db` | SQLite file path (`:memory:` for ephemeral) |
| `IDS_DASHBOARD_WS_TOKEN_TTL` | `300` | WebSocket token lifetime, seconds |
| `IDS_DASHBOARD_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `IDS_DASHBOARD_HOST` / `IDS_DASHBOARD_PORT` | `0.0.0.0` / `8000` | uvicorn bind address |

## REST API

All routes below except `/healthz` require HTTP Basic auth.

- `GET /api/alerts?severity=&attack_type=&start_time=&end_time=&triage_status=&escalated=&limit=&offset=`
  -- filtered, paginated alert history, newest first. `escalated=true`/`false`
  filters to alerts the open-set gate did/didn't escalate.
- `GET /api/alerts/{alert_id}` -- one alert (including `unknown_mass`/`escalated`/`escalation_trigger`), 404 if unknown.
- `GET /api/alerts/{alert_id}/explanation` -- Tier 2's explanation for this alert.
  404 both when the alert itself doesn't exist and when it exists but Tier 2
  hasn't produced an explanation yet (or never will -- not every alert is
  escalated) -- callers already know an alert's `escalated` flag from the
  alert itself and can use that to distinguish the two 404 cases if needed.
- `PATCH /api/alerts/{alert_id}/triage` -- body `{"status": "...", "note": "..."}`,
  status one of `new`/`acknowledged`/`confirmed`/`false_positive`.
- `GET /api/alerts/summary?start_time=&end_time=` -- volume by severity/attack
  type/day, plus both false-positive-rate signals (see `store.AlertStore.summary`'s
  docstring for what each one means and requires).
- `POST /api/ws-token` -- issues a short-lived token for the WebSocket endpoint.
- `GET /healthz` -- liveness probe; intentionally unauthenticated. Reports
  `alerts_processed` and `explanations_processed` separately.

## WebSocket

`GET /ws/alerts?token=...` pushes both alert and explanation events as JSON
frames to every connected client, over one connection -- alert frames are
the original, unmarked shape (a raw alert object); explanation frames carry
a `__type: "explanation"` field so the client can tell the two apart
without breaking the existing alert frame format (see
`explanation_ingest_service.py` and the frontend's `api.ts::connectAlertStream`).
Browsers cannot attach an `Authorization` header to a WebSocket handshake,
so auth here is a short-lived opaque token fetched via the Basic-auth-protected
`POST /api/ws-token` first -- not the Basic credentials themselves.

## Testing

```
pip install -r requirements-dashboard.txt
PYTHONPATH=src pytest
```

56 tests, backend only. Uses `fastapi.testclient.TestClient` with injected
stub alert *and* explanation sources and an in-memory SQLite database --
no live Kafka broker needed, matching the pattern used throughout this
project. The frontend has no automated test suite; `npm run build` (`tsc
-b && vite build`) passing is the only verification the new
escalation/explanation UI has -- see [Known
limitations](#known-limitations).

## Known limitations

- **Basic auth is a deliberately minimal, documented gap**, not a
  production-grade auth system: a single shared username/password pair (no
  per-user accounts, no roles), no rate limiting or lockout on failed
  attempts, and no built-in TLS -- credentials are base64-encoded, not
  encrypted, unless a TLS-terminating proxy sits in front of this service.
  A real deployment needs at least: per-user credentials or an identity
  provider (OAuth2/OIDC), TLS termination, and login rate limiting.
- **WebSocket tokens are single-process, in-memory, and unrevocable**
  (`auth.TokenStore`): they don't survive a restart, aren't shared across
  multiple API replicas, and there's no way to invalidate one before it
  expires short of restarting the process.
- **The WebSocket broadcaster is single-process** (`broadcaster.py`):
  running multiple API replicas behind a load balancer means each replica
  only broadcasts to the clients connected to *it*, not all clients
  globally. A production multi-replica deployment needs a shared pub/sub
  layer (e.g. Redis) behind the broadcaster instead.
- **SQLite is a single-writer store.** Fine for one API process at the
  alert volumes a triage dashboard sees; not a fit for multiple concurrent
  writers without moving to a real client/server database.
- **Kafka consumer shutdown timing isn't guaranteed** (`ingest_service.py`
  `IngestService.stop`'s docstring): if the consumer is blocked awaiting
  the next record, closing it from another thread is expected to unblock
  it but isn't documented by kafka-python as officially thread-safe for
  that exact sequence.
- **The two false-positive-rate signals are proxies, not ground truth**
  (see `store.AlertStore.summary`'s docstring): the stage-1 proxy requires
  the ML module to be run with `alert_on_stage1_flag_only=True` or it's
  simply unavailable; the analyst-reviewed rate is only as good as triage
  coverage, which the summary reports alongside the rate so a low-coverage
  number isn't mistaken for a reliable one.
- **CORS defaults to `*`** (`IDS_DASHBOARD_CORS_ORIGINS`); tighten this to
  the actual dashboard origin(s) before deploying anywhere reachable by
  untrusted clients.
- **The frontend keeps credentials in memory only** (component state, not
  local/session storage), so a page reload requires signing in again --
  intentional, to avoid persisting a shared Basic-auth password in browser
  storage, but worth knowing if a smoother reload experience is wanted
  later (that would need a proper session/token mechanism, not Basic auth).
- **The escalation/explanation UI has now been checked in a real browser
  against the live stack -- and doing so found a real crash bug `npm run
  build` and the 56 backend tests couldn't have caught.** A headless-
  browser session against a running `docker compose up -d` stack logged
  in, opened the live feed, opened Triage, and clicked into an escalated
  alert to confirm the explainability panel (SHAP feature bars, class
  probabilities) and the Tier 2 analysis section both render real data.
  Along the way it hit `Cannot read properties of undefined (reading
  'replace')`, a hard crash: `AlertRow.tsx` called
  `alert.triage_status.replace(...)` unconditionally, but a `triage_status`
  field is a store/DB-only concept `ml`'s Kafka alert schema never
  includes -- a REST-fetched alert always has it, but a *live*
  WebSocket-pushed alert (broadcast as `ingest_service.py` received it
  from Kafka, before the DB round-trip that adds the default) never does.
  Every automated test uses fixture data with `triage_status` already
  present, so nothing caught this until an actual live alert was pushed
  to an actual open browser tab. Fixed by defaulting a missing
  `triage_status` to `"new"` at render time (true by construction -- a
  freshly-pushed alert has never been triaged) and correcting `types.ts`'s
  `Alert` interface to mark `triage_status`/`triage_note`/
  `triage_updated_at` as the optional fields they actually are on the live
  path, not falsely required.
- **A real integration bug was found and fixed via this store**: a stale
  `dashboard_data` Docker volume from before Tier 2 existed crashed
  `AlertStore.__init__` with `sqlite3.OperationalError: no such column:
  escalated`, because `CREATE TABLE IF NOT EXISTS` is a no-op against an
  already-existing `alerts` table. `store.py` now migrates missing columns
  (`PRAGMA table_info` + `ALTER TABLE ... ADD COLUMN`) before creating any
  index that references them -- see the top-level README's "What
  integration found" section and
  `test_dash_store.py::test_opening_a_pre_tier2_database_migrates_the_alerts_table`.
- **`ExplanationIngestService` assumes Kafka delivery order between
  `network.ids.alerts` and `network.ids.explanations` is not guaranteed**
  (see `store.insert_explanation`'s docstring) and is written to tolerate
  an explanation arriving before its alert -- exercised in
  `test_dash_store.py::test_insert_explanation_does_not_require_the_alert_to_exist_yet`,
  but only at the store layer, not as a live-timing race condition.
- **Tier 2 explanations can take 5-40+ seconds to arrive** (see
  `ml/README.md`'s Latency section) -- the UI's "pending" state for an
  escalated alert with no explanation yet is the normal, expected case,
  not an error state, but it does mean an analyst opening an alert right
  after it's escalated will usually see "pending" first.

## Out of scope

Ingestion, the ML detection/open-set gate, and Tier 2's reasoning itself
are not implemented here -- this module only consumes the alerts and
explanations topics and serves them via the REST/WebSocket API and
dashboard described above.
