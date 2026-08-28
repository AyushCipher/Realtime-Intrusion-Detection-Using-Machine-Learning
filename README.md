# Realtime Intrusion Detection Using Machine Learning

A real-time network intrusion detection system built as four
independently authored, independently testable modules, wired together
into one running system in `docker-compose.yml` -- though "wired
together" for Tier 2 specifically means "unit-tested and live-LLM-tested
in isolation, and the compose/schema/dashboard plumbing connects it,"
**not** "verified end-to-end via a real `docker compose up`" -- see
"What integration found" below and the Known limitations section for
exactly what has and hasn't been confirmed running together for real.

| Module | Folder | What it does |
|---|---|---|
| Ingestion | [`ingestion/`](ingestion/README.md) | Live/pcap-replay packet capture -> sliding-window flow feature extraction -> publishes to Kafka |
| ML detection | [`ml/`](ml/README.md) | Consumes flow features -> two-stage detector (Isolation Forest + XGBoost) + open-set escalation gate -> SHAP explanations -> publishes alerts to Kafka |
| Dashboard/API | [`dashboard-api/`](dashboard-api/README.md) | Consumes alerts + Tier 2 explanations -> FastAPI REST/WebSocket API -> React triage dashboard (severity, escalation, Tier 2 analysis) |
| Tier 2 reasoning | [`tier2_reasoner/`](tier2_reasoner/README.md) | Consumes `ml`'s escalated alerts -> RAG over MITRE ATT&CK -> LLM explanation -> publishes to its own topic |

Each module folder has its own README with that module's design decisions
and a "known limitations" section written from inside that module. This
file is the integration layer on top: how the four actually run together,
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
ml/                two-stage detector, SHAP, open-set escalation gate, Kafka consumer+producer (+ its own README/tests)
dashboard-api/     FastAPI backend + React frontend (+ its own README/tests)
tier2_reasoner/    RAG + LLM escalation reasoning (+ its own README/tests)
integration/       test_e2e.py -- the full-stack check described above (covers the first three modules only -- see Known limitations)
docker-compose.yml the shared broker + all four services
SCHEMA.md          reconciled, verified Kafka topic schemas, including network.ids.explanations
```

Unlike the original three-module integration (where no module's own code
needed changing -- see "What integration found" above), wiring Tier 2 in
*did* require real application-code changes, not just compose/environment
glue: `ml/src` gained the ability to fit, calibrate, save, and load an
open-set gate (`ids_ml.train --gate` / `ids_ml.serve` auto-loading it) so
the demo model actually produces `escalated: true` alerts instead of only
being capable of it in principle; `dashboard-api/src` and
`dashboard-api/frontend/src` gained a second Kafka consumer, storage
table, REST endpoint, and UI panel to actually consume and display
`network.ids.explanations`. This is called out explicitly because it's a
deliberate departure from the "integration never touches module code"
principle the first three modules established -- Tier 2's existence
*is* new capability those modules didn't have before, not a
config-only wiring exercise.

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

**Tier 2 reasoning (`tier2_reasoner/`) is now wired into this system --
`docker-compose.yml` includes it, `ml`'s demo model is trained with the
`openset` gate so it actually produces `escalated: true` alerts, and
`dashboard-api` consumes `network.ids.explanations` and surfaces
`unknown_mass`/`escalated`/Tier 2 explanations in the dashboard (REST +
live WebSocket) -- but its own live testing found it is not currently
reliable or fast enough to call real-time.** Its LLM path has been
verified live against two real API keys: 9/9 completed RAG-enabled calls
correctly named the right MITRE technique (a real, consistent, positive
signal), but a real batch run also surfaced the honest problem -- 2 of 6
calls in one batch failed outright even after retries, all 6 no-RAG calls
in that same batch failed too, and the calls that did succeed took
23.5-60.4 seconds each (vs. 2.6s for an earlier isolated call) -- see
`tier2_reasoner/README.md`'s "Latency (live-verified -- and a reliability
problem)" section. The batch-measured amortized latency is ~4,056ms/flow
(~420x Tier 1's own latency), not the ~584ms an earlier single fast call
suggested. `tier2_reasoner` now has retry/backoff logic
(`RetryingLLMClient`), built directly from that failure, but it has not
yet been re-tested against the same live failure conditions that exposed
the need for it.

**What "wired in" does and doesn't mean here:** every piece above was
built, unit-tested (241 tests passing across all four modules combined),
and the `ids_ml.train --gate openset` / `docker compose config` steps
were verified to actually run and produce correct output. **A full
`docker compose up --build` of the whole system, and a real browser
check of the new dashboard UI, were not performed** -- the Docker daemon
was not running in the environment this was built in (the CLI is
present; `docker compose build` failed to connect to it). `docker compose
config` (which only parses/resolves the compose file, no daemon needed)
did succeed. Run `docker compose up -d --build` yourself to get the
end-to-end verification this repo's own philosophy says matters more
than any individual module's tests -- see "What integration found" above
for why.
