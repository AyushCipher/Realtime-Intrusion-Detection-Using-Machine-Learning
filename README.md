# Realtime Intrusion Detection Using Machine Learning

A real-time network intrusion detection system built as four
independently authored, independently testable modules, wired together
into one running system in `docker-compose.yml`. **All six containers
(kafka, ingestion, ml, tier2-reasoner, dashboard-api, dashboard-frontend)
have been verified running together for real**: `docker compose up -d
--build` built all five application images and started the full stack,
`integration/test_e2e.py` passed against it (pcap replay -> ingestion ->
Kafka -> ml -> Kafka -> dashboard-api -> REST), and `dashboard-api`'s
`/healthz` showed both `network.ids.alerts` and `network.ids.explanations`
being consumed live (`alerts_processed`/`explanations_processed` both >0)
-- i.e. Tier 2 actually escalated, reasoned, and published back through
the whole chain, not just in isolated unit tests. **A human has since
also driven the live dashboard UI itself** (a headless-browser session
against the running stack: signed in, opened the live feed and Triage
view, clicked into an escalated alert, confirmed the SHAP explainability
panel and Tier 2 analysis section both render) -- see "What integration
found" below for the three real bugs (one Docker/OneDrive integration
issue, one stale-schema-volume crash, one frontend crash on live
WebSocket data) that surfaced doing all of this, and Known limitations
for the host-resource caveat that verification also surfaced.

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

Independently-built modules that had never actually talked to a real
broker (or, for the two bugs below, never actually run inside real Docker
containers against each other) surfaced four real bugs that no module's
own test suite (which stub out Kafka, or stub out Docker entirely) could
have caught:

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
3. **OneDrive Files On-Demand corrupts Docker Desktop's build-context
   transfer, on a machine where the repo lives inside a OneDrive-synced
   folder.** `docker compose build` failed for 3-5 of the 5 services with
   a cryptic `ERROR: invalid file request Dockerfile` (or the same error
   against other context files, unpredictably, on a later attempt) --
   nothing to do with the Dockerfiles themselves. Root cause: files under
   OneDrive's Files On-Demand feature carry a `ReparsePoint` attribute
   (Windows' cloud-placeholder mechanism) even once fully downloaded/
   pinned locally, and Docker Desktop's Windows file-sharing layer cannot
   reliably read reparse-point-tagged files -- confirmed by comparing
   `Get-Item`'s `Attributes` on the failing vs. succeeding services'
   Dockerfiles (`Archive, ReparsePoint` vs. plain `Archive`), and by the
   failure following files unpredictably between build attempts as
   OneDrive re-tagged them. Neither pinning the files (`attrib +P -U`) nor
   restarting the OneDrive sync host resolved it. **Fix**: copy each
   service's build context to a location outside any OneDrive-synced
   folder (this session used the scratch directory) and build from there
   via a `docker-compose.override.yml`-style context override; the
   resulting images are identical and `docker compose up` afterward needs
   no further workaround. This is an environment/OS-integration issue, not
   a bug in this repo's Dockerfiles or compose config -- it will only
   affect a machine where the repo sits inside a OneDrive (or similar
   cloud-sync) folder with Files On-Demand enabled.
4. **A stale `dashboard-api` SQLite volume from before Tier 2 existed
   crashed the container on startup** with `sqlite3.OperationalError: no
   such column: escalated`. `store.py`'s schema used `CREATE TABLE IF NOT
   EXISTS`, which is a no-op against a database file that already has an
   `alerts` table -- so a `dashboard_data` Docker volume created by an
   earlier partial `docker compose up` attempt (before `unknown_mass`/
   `escalated`/`escalation_trigger` were added) kept missing those
   columns forever, and the `CREATE INDEX ... ON alerts(escalated)`
   statement right after it then failed outright. **Fixed** with a real
   migration step in `AlertStore.__init__` (`PRAGMA table_info` + `ALTER
   TABLE ... ADD COLUMN` for whatever's missing, before the index-creation
   statements run) -- see `dashboard-api/src/ids_dashboard/store.py` and
   its regression test
   (`test_dash_store.py::test_opening_a_pre_tier2_database_migrates_the_alerts_table`).
   This is a genuine gap the demo's own test suite couldn't catch, since
   every test there starts from a fresh `:memory:` database -- only a real
   persistent volume that outlived a schema change surfaced it.
5. **The dashboard frontend crashed on a live-pushed alert** with `Cannot
   read properties of undefined (reading 'replace')`. `AlertRow.tsx`
   called `alert.triage_status.replace(...)` unconditionally, but
   `triage_status` is a `dashboard-api`/DB-only concept that `ml`'s Kafka
   alert schema never includes -- a REST-fetched alert always has it (the
   store adds it on insert), a *live* WebSocket-pushed alert (broadcast
   as received from Kafka, before any DB round-trip) never does. Found by
   actually opening the dashboard in a headless browser against the live
   stack, not by any of the 56 backend tests or `npm run build` -- every
   fixture those use already has `triage_status` set, so nothing
   exercised the field's absence until a real alert arrived over a real
   WebSocket connection. Fixed by defaulting a missing `triage_status` to
   `"new"` at render time and correcting `types.ts` to mark it (and
   `triage_note`/`triage_updated_at`) optional, matching what's actually
   guaranteed on each path -- see `dashboard-api/README.md`'s known-
   limitations section and `AlertRow.tsx`.

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

**Tier 2 reasoning (`tier2_reasoner/`) is wired into this system and
verified running end-to-end for real -- `docker-compose.yml` includes it,
`ml`'s demo model is trained with the `openset` gate so it actually
produces `escalated: true` alerts, `dashboard-api` consumes
`network.ids.explanations` and surfaces `unknown_mass`/`escalated`/Tier 2
explanations in the dashboard (REST + live WebSocket), and a live
`docker compose up -d --build` run showed `dashboard-api`'s `/healthz`
reporting both `alerts_processed` and `explanations_processed` > 0 -- but
its own live testing found it is not currently reliable or fast enough to
call real-time.** Its LLM path has been verified live against two real
API keys: 9/9 completed RAG-enabled calls correctly named the right MITRE
technique (a real, consistent, positive signal), but a real batch run
also surfaced the honest problem -- 2 of 6 calls in one batch failed
outright even after retries, all 6 no-RAG calls in that same batch failed
too, and the calls that did succeed took 23.5-60.4 seconds each (vs. 2.6s
for an earlier isolated call) -- see `tier2_reasoner/README.md`'s
"Latency (live-verified -- and a reliability problem)" section. The
batch-measured amortized latency is ~4,056ms/flow (~420x Tier 1's own
latency), not the ~584ms an earlier single fast call suggested.
`tier2_reasoner` has retry/backoff logic (`RetryingLLMClient`), built
directly from that failure and since live-retested with a real 36-call
paired RAG-ablation batch: it correctly recovered from transient
per-minute rate limiting, but 16 of 36 calls still failed once a daily
quota ceiling was hit partway through -- a different failure mode
retry/backoff can't fix by construction. The completed calls produced a
real, statistically significant H3 result: RAG-grounded calls named the
correct MITRE technique in 10/10 completed cases vs. 0/10 for no-RAG,
paired McNemar p=0.0039 -- see `tier2_reasoner/README.md`'s H3 section
for the full breakdown. In this
session's `docker compose` run, `tier2_reasoner` had no API key configured
(so it ran on the deterministic stub LLM client, not a real one) -- the
end-to-end wiring verification above is real; the live-LLM latency/
reliability numbers above are still from the isolated tests described in
`tier2_reasoner/README.md`, not from this compose run.

**What "wired in" does and doesn't mean here:** every piece above was
built, unit-tested (260 tests passing across all four modules combined as
of this update: 115 ml + 70 tier2_reasoner + 56 dashboard-api + 19
ingestion -- the frontend crash below was caught by the live browser
check, not by an automated test, since this frontend has no test runner
set up; see `dashboard-api/README.md`'s known-limitations section), and
`docker compose up -d --build` was run for real: all
five application images built, all six containers started, and
`integration/test_e2e.py` passed against the live stack (see "What
integration found" above for all five real bugs -- one Docker/OneDrive
integration issue, one stale-schema-volume crash, and one live-alert
frontend crash among them -- that surfaced doing this and how each was
fixed). **A human has since also driven the live dashboard UI itself**:
a headless-browser session (Playwright against the system's installed
Edge, run against the actual running compose stack, not a mock) signed
in, opened the live feed (WebSocket connects and shows "OPEN"), opened
the Triage view (rendered 100 real, DB-backed escalated alerts with
correct severity/class/ESC badges), and clicked into one to confirm the
SHAP explainability panel and the "Tier 2 analysis (LLM/RAG)" section
both render real backend data -- exactly the check flagged as missing in
an earlier version of this README, and the one that found bug #5 above.
This machine's RAM constraint (below) made getting a stable enough
window to run this check take several Docker Desktop restarts, but it
did get done, not skipped.

**This demo's `docker compose` stack needs more RAM than this
development machine reliably has free.** The host has 7.7GB total RAM;
Docker Desktop's backend was observed crashing outright (not just losing
the daemon connection -- the process disappeared entirely) multiple times
while this stack was running, correlating with free memory dropping to
under 1GB. The full stack (a JVM-based Kafka broker, three Python
services each pulling in numpy/pandas/scikit-learn/xgboost, and a Node
dev server, all running simultaneously) is a meaningfully heavier
footprint than most of this demo's design decisions accounted for. This
is a host-resource limitation, not a code defect -- the stack itself ran
and passed integration verification every time Docker Desktop's backend
was actually up. If running this continuously (not just for one-off
verification) on similarly constrained hardware, consider: lowering
Kafka's JVM heap (`KAFKA_HEAP_OPTS`, unset in `docker-compose.yml`
today), adding `mem_limit`s per service, or running on a machine/VM with
more headroom.
