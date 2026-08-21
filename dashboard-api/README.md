# Alert Dashboard & API Layer (`ids_dashboard`)

FastAPI backend and React frontend for triaging alerts from the ML module.
Consumes `network.ids.alerts` (schema in `schema.py`, duplicated from the ML
module's own copy so this module has no code dependency on it), stores
alerts for historical query/filter, pushes live alerts over a WebSocket,
and serves REST endpoints for the dashboard. This package does not
implement ingestion or any detection model -- see
[Out of scope](#out-of-scope).

## Architecture

```
Kafka: network.ids.alerts     IngestService              FastAPI
(from ids_ml)            -->  (background thread    -->  REST  /api/alerts, /api/alerts/summary,
                                consuming Kafka,           /api/alerts/{id}/triage, /api/ws-token
                                writing to SQLite,    -->  WebSocket  /ws/alerts (token-gated)
                                broadcasting to
                                connected clients)
                                     |
                                     v
                               SQLite (alerts.db)
                               history + triage state
```

- `schema.py` -- the input alert contract (duplicated from `ids_ml.schema`).
- `store.py` -- SQLite-backed alert history, filtering, and analyst triage state.
- `alert_consumer.py` -- Kafka alert source (real + in-memory stub), mirrors `ids_ml.flow_consumer`.
- `broadcaster.py` -- in-process WebSocket fan-out.
- `ingest_service.py` -- background thread tying consumer -> store -> broadcaster.
- `auth.py` -- HTTP Basic auth + short-lived WebSocket tokens.
- `routes_alerts.py` / `routes_ws.py` / `app.py` -- the FastAPI surface.
- `frontend/` (top-level, sibling to `src/`) -- the React dashboard.

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
| `IDS_DASHBOARD_ALERT_TOPIC` | `network.ids.alerts` | Topic to consume |
| `IDS_DASHBOARD_USE_STUB` | `false` | Run against an empty in-memory source instead of Kafka |
| `IDS_DASHBOARD_DB_PATH` | `alerts.db` | SQLite file path (`:memory:` for ephemeral) |
| `IDS_DASHBOARD_WS_TOKEN_TTL` | `300` | WebSocket token lifetime, seconds |
| `IDS_DASHBOARD_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `IDS_DASHBOARD_HOST` / `IDS_DASHBOARD_PORT` | `0.0.0.0` / `8000` | uvicorn bind address |

## REST API

All routes below except `/healthz` require HTTP Basic auth.

- `GET /api/alerts?severity=&attack_type=&start_time=&end_time=&triage_status=&limit=&offset=`
  -- filtered, paginated alert history, newest first.
- `GET /api/alerts/{alert_id}` -- one alert, 404 if unknown.
- `PATCH /api/alerts/{alert_id}/triage` -- body `{"status": "...", "note": "..."}`,
  status one of `new`/`acknowledged`/`confirmed`/`false_positive`.
- `GET /api/alerts/summary?start_time=&end_time=` -- volume by severity/attack
  type/day, plus both false-positive-rate signals (see `store.AlertStore.summary`'s
  docstring for what each one means and requires).
- `POST /api/ws-token` -- issues a short-lived token for the WebSocket endpoint.
- `GET /healthz` -- liveness probe; intentionally unauthenticated.

## WebSocket

`GET /ws/alerts?token=...` pushes each newly-stored alert as a JSON frame to
every connected client. Browsers cannot attach an `Authorization` header to
a WebSocket handshake, so auth here is a short-lived opaque token fetched
via the Basic-auth-protected `POST /api/ws-token` first -- not the Basic
credentials themselves.

## Testing

```
pip install -r requirements-dashboard.txt
PYTHONPATH=src pytest
```

Backend tests use `fastapi.testclient.TestClient` with an injected stub
alert source and an in-memory SQLite database -- no live Kafka broker
needed, matching the pattern used throughout this project.

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

## Out of scope

Ingestion and the ML detection models are not implemented here -- this
module only consumes the alerts topic and serves it via the REST/WebSocket
API and dashboard described above.
