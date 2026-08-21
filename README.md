# Realtime Intrusion Detection Using Machine Learning

A real-time network intrusion detection system built as three independently
authored, independently testable modules, wired together here into one
running system.

| Module | Folder | What it does |
|---|---|---|
| Ingestion | [`ingestion/`](ingestion/README.md) | Live/pcap-replay packet capture -> sliding-window flow feature extraction -> publishes to Kafka |
| ML detection | [`ml/`](ml/README.md) | Consumes flow features -> two-stage detector (Isolation Forest + XGBoost) -> SHAP explanations -> publishes alerts to Kafka |
| Dashboard/API | [`dashboard-api/`](dashboard-api/README.md) | Consumes alerts -> FastAPI REST/WebSocket API -> React triage dashboard |

Each module folder has its own README with that module's design decisions
and a "known limitations" section written from inside that module. This
file is the integration layer on top: how the three actually run together,
the true (verified, not assumed) schema they agree on, and what broke and
got fixed only once they were wired up for real.

## Architecture

```
                         docker-compose network
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │   ┌────────────┐   network.flow.features   ┌────────────┐         │
   │   │ ingestion  │ ─────────────────────────► │            │        │
   │   │ (pcap      │                            │            │        │
   │   │  replay,   │                            │   kafka    │        │
   │   │  looping)  │                            │ (1 broker, │        │
   │   └────────────┘                            │  KRaft)    │        │
   │                                              │            │        │
   │   ┌────────────┐   network.flow.features     │            │        │
   │   │            │ ◄────────────────────────── │            │        │
   │   │     ml     │                             │            │        │
   │   │ (two-stage │   network.ids.alerts        │            │        │
   │   │  detector) │ ─────────────────────────►  │            │        │
   │   └────────────┘                             └─────┬──────┘        │
   │                                                     │               │
   │   ┌────────────────────┐  network.ids.alerts        │               │
   │   │   dashboard-api     │ ◄──────────────────────────┘              │
   │   │ (FastAPI: REST +    │                                          │
   │   │  WebSocket, SQLite) │                                          │
   │   └──────────┬──────────┘                                          │
   │              │ REST (8000) + WebSocket                             │
   └──────────────┼──────────────────────────────────────────────────┘
                  │
   ┌──────────────┴──────────────┐
   │  dashboard-frontend (5173)   │   <- served to your browser (host machine)
   │  React dashboard             │
   └───────────────────────────────┘
```

All four application services (`ingestion`, `ml`, `dashboard-api`,
`dashboard-frontend`) point at the **one** `kafka` service defined in
`docker-compose.yml` -- none of them runs or connects to a broker of its
own. `dashboard-frontend` is the only piece your browser talks to directly
(over the published `8000`/`5173` ports); everything else communicates
over the compose-internal network by service name.

## Running it

```
docker compose up -d --build
```

Then open **http://localhost:5173** and sign in with the demo credentials
below (or `docker-compose.yml`'s `IDS_DASHBOARD_USERNAME`/`PASSWORD`, if
you changed them). The REST API is at `http://localhost:8000` (see
`dashboard-api/README.md` for the full endpoint list); Kafka itself is not
published to the host, since nothing outside the compose network needs it.

| | |
|---|---|
| Dashboard | http://localhost:5173 |
| API | http://localhost:8000 (docs at `/docs`) |
| Demo credentials | `analyst` / `changeme123` (set in `docker-compose.yml`) |

`ingestion` replays `ingestion/tests/fixtures/sample_tcp.pcap` (the same
fixture that module's own tests assert exact feature values against) on a
loop, so the dashboard keeps receiving fresh alerts rather than a single
one-shot burst. `ml` trains its demo model at Docker build time from the ML
module's own synthetic test fixture -- **not real CICIDS2017 data** -- see
[Known limitations](#known-limitations).

To verify the whole pipeline programmatically instead of watching the UI:

```
docker compose up -d --build
python integration/test_e2e.py
```

This polls the dashboard API until an alert traceable back to the demo
pcap's actual flows appears, checks it carries a SHAP explanation, and
checks it's individually retrievable -- i.e. it proves the full chain
(replay -> ingestion -> Kafka -> ml -> Kafka -> dashboard-api -> REST)
actually ran, not just that the containers started.

Tear down with `docker compose down` (add `-v` to also drop the Kafka and
alert-history volumes).

## How the modules communicate

Exactly two Kafka topics, both JSON, both documented field-by-field with
their true (verified) schema in **[SCHEMA.md](SCHEMA.md)**:

- `network.flow.features` -- ingestion -> ml
- `network.ids.alerts` -- ml -> dashboard-api

No module imports another's code. Each module maintains its own copy of
the schema it depends on (see each module's `schema.py`); `SCHEMA.md` is
the reconciled, cross-checked source of truth for what those copies
actually agree on, superseding the schema tables in each module's own
README where the two differ.

## What integration found

Three independently-built modules that had never actually talked to a real
broker together surfaced two real bugs that no module's own test suite
(which stub out Kafka) could have caught:

1. **Single-broker Kafka defaults break consumer groups silently.**
   `offsets.topic.replication.factor` (and the transaction-log equivalents)
   default to values that assume a 3-broker cluster. With one broker, the
   internal `__consumer_offsets` topic gets stuck retrying its own creation
   forever, with no client-visible error -- both `ml` and `dashboard-api`
   would connect fine and then simply never receive a message. Fixed in
   `docker-compose.yml` by setting these to `1` for the single-broker demo
   cluster; see the comment there. Nothing in any module's code was wrong
   -- this is purely a broker configuration issue that only exists in a
   single-node setup like this demo's.
2. **Docs vs. actual schema drift, not breaking mismatches** -- see
   `SCHEMA.md`'s "Reconciliation findings" section for the two minor
   documentation gaps found (and why neither needed an adapter).

## Repo layout

```
ingestion/        packet capture, flow extraction, Kafka producer (+ its own README/tests)
ml/                two-stage detector, SHAP, Kafka consumer+producer (+ its own README/tests)
dashboard-api/     FastAPI backend + React frontend (+ its own README/tests)
integration/       test_e2e.py -- the full-stack check described above
docker-compose.yml the one shared broker + all four services
SCHEMA.md          reconciled, verified Kafka topic schemas
```

Everything at the top level (`docker-compose.yml`, per-service
`Dockerfile`s and `docker-entrypoint.sh` scripts, `SCHEMA.md`,
`integration/`, this file) is integration glue added on top of the three
modules. No code inside `ingestion/src`, `ml/src`, `dashboard-api/src`, or
`dashboard-api/frontend/src` was changed to make the system integrate --
where a module's own file needed touching at all (see "What integration
found" above), it was a docker-compose environment setting, not application
code.

## Known limitations

Each module's own README has a fuller "known limitations" section written
from inside that module; this is the consolidated, cross-cutting view.

**Model quality (ml)**: the demo model is trained at Docker build time on
`ml`'s own synthetic test fixture, not real CICIDS2017/2018 data -- expect
plausible-looking but not meaningful predictions (e.g. the demo pcap's
plain TCP handshake sometimes gets classified as "Heartbleed" by the toy
model). Stage 1's flag threshold is tuned for that synthetic fixture's ~35%
attack prevalence, nowhere close to real traffic's <<1%. See `ml/README.md`
for the honest per-category evaluation results this claim is based on, and
for the training-side split-leakage/concept-drift/adversarial-robustness
findings.

**Auth (dashboard-api)**: a single shared Basic-auth credential pair, no
TLS, no per-user accounts, no rate limiting. WebSocket auth is a short-lived
token issued over that same Basic-auth REST call. Fine for this demo;
not for anything internet-facing. See `dashboard-api/README.md`.

**Single points of failure (this integration layer)**: one Kafka broker,
one `dashboard-api` process (its in-memory WebSocket broadcaster and
WebSocket-token store don't survive a restart or scale past one replica),
one SQLite file for alert history. None of this is hidden behind retry
logic that would mask data loss -- see `dashboard-api/README.md`'s
known-limitations section for specifics.

**Ingestion (live capture)**: this demo only exercises pcap replay.
`ingestion` also supports live packet capture, but that needs
Npcap/libpcap and elevated privileges neither of which are set up in these
containers -- see `ingestion/README.md`.

**No detection-quality guarantee, adversarial or otherwise**: see
`ml/README.md`'s adversarial-robustness section for why "low-and-slow"
evasion's effect on this specific model went the *opposite* direction from
the usual intuition, and why that result doesn't generalize on its own.

**This demo's pcap is tiny (2 flows).** It's enough to prove the pipeline
wiring end-to-end (which is what `integration/test_e2e.py` checks), not
enough to say anything about detection quality or the dashboard's
volume/severity views at any real scale.
