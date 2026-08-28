# Tier 2 Reasoning Layer (`ids_tier2`)

The budget-gated LLM/RAG escalation tier for this project's open-set IDS.
It consumes alerts the `ml` module marked `escalated: true` (the fraction
`ml`'s conformal-calibrated open-set gate decided looks unlike anything in
the training set -- see `ml/README.md`'s "Open-set escalation" section),
retrieves relevant MITRE ATT&CK technique descriptions, asks an LLM to
reason about what the traffic might be and what to do about it, and
publishes the result. This package does not implement ingestion, ML
scoring, the escalation decision itself, or the dashboard -- see
[Out of scope](#out-of-scope).

`ids_tier2` has no code dependency on `ids_ml`: the only coupling is the
documented Kafka topic contract in `schema.py`, duplicated from `ids_ml`'s
schema (same pattern the other three modules already use).

## Why this exists

The literature check that motivated this whole open-set upgrade (see
`ml/README.md`'s open-set section) found real, current work combining
LLM reasoning with intrusion detection, but consistently either (a) runs
the LLM on every flow -- MA-IDS's Traffic Classification Agent *is* the
classifier, called on every sample, with no gating at all -- or (b) names
LLM-gated novel-traffic reasoning as wanted future work without building
it (FIRCE's own limitations section: *"a practical next step is to
couple CE rejections with human-in-the-loop or LLM-assisted labeling
workflows"*). Real LLM latency is 3-5+ seconds per call by published
industry estimates -- running that per-flow is a non-starter for
real-time IDS at any meaningful traffic rate. Keeping the LLM off the hot
path by construction (it only ever sees the fraction `ml`'s gate already
decided is uncertain) is the whole point of this module being separate
from, and only conditionally invoked by, the fast closed-set cascade.

## Architecture

```
Kafka: network.ids.alerts          Tier2Reasoner                Kafka: network.ids.explanations
(from ids_ml, ALL alerts)   --->    filter: escalated==true --->  (for dashboard-api, once wired up)
                                     retrieve (TF-IDF over
                                      MITRE ATT&CK subset)
                                     LLM call (stub or real)
                                     parse structured response
```

- `schema.py` -- the input (alert) and output (explanation) topic contracts.
- `knowledge_base.py` -- a small, curated MITRE ATT&CK technique set covering this project's attack families.
- `retrieval.py` -- TF-IDF retrieval over the knowledge base (the "R" in RAG).
- `llm_client.py` -- `StubLLMClient` (deterministic, offline) and `AnthropicLLMClient` (real, untested live -- see [Known limitations](#known-limitations)).
- `reasoner.py` -- `Tier2Reasoner`: retrieval + prompt + LLM call + defensive response parsing.
- `alert_consumer.py` / `explanation_producer.py` / `service.py` -- the live Kafka-facing pipeline.
- `serve.py` -- CLI.

## Deliberate simplification: no dedicated escalations topic

The original design sketch called for `ml` to publish a second, pre-filtered
`network.ids.escalations` topic. This module instead subscribes to `ml`'s
existing `network.ids.alerts` topic and filters client-side for
`escalated == true` (see `alert_consumer.py`). That keeps `ml`'s producer
side completely unchanged -- no new producer, no new topic for it to
manage -- at the cost of this consumer group seeing (and cheaply
discarding) every alert, not just escalated ones. Discarding a JSON
message client-side is orders of magnitude cheaper than an LLM call, so
this doesn't reintroduce "LLM on the hot path" -- it would only matter at
alert volumes far beyond this project's demo pipeline.

## Installation

```
pip install -r requirements.txt
```

## Retrieval: why TF-IDF, not embeddings

`ml/README.md` already documents that PyTorch is unusable in the
environment this whole project was built in (blocked by the host's
Application Control/WDAC policy at DLL load time). Embedding-based RAG
(e.g. MA-IDS's all-MiniLM-L6-v2 + FAISS) depends on it. `retrieval.py`
uses scikit-learn's TF-IDF vectorizer + cosine similarity instead --
needs only numpy/scipy, already dependencies elsewhere in this project.
For a knowledge base this small (single digits of entries), the retrieval
quality gap between TF-IDF and dense embeddings is not the binding
constraint; it would matter far more at real ATT&CK-corpus scale
(~200+ techniques), where semantic-but-not-lexical matches become common.

**The knowledge base itself is a curated subset, not the full ATT&CK
corpus** -- 7 entries covering the 6 non-benign attack families this
project's own `ATTACK_CATEGORY_MAP` models (DoS/DDoS, Brute Force, Web
Attack, PortScan, Infiltration, Botnet). Building or fetching the full
ATT&CK STIX bundle was out of scope for this pass; `retrieval.py` doesn't
care how many entries the knowledge base has, so extending it is
mechanical. The category -> technique mapping is this project's own
reasonable association, not a verified ground truth -- CICIDS2017/2018
predate widespread ATT&CK adoption and don't ship with technique labels.

Verified (see `tests/test_tier2_retrieval.py`'s parametrized test): every
one of the 6 real attack categories retrieves its intended top-1
technique when queried with the alert's exact `stage2_predicted_class`
string. This needed a real fix during development, not just a passing
test written around correct-by-luck behavior: the knowledge base's
indexed text originally excluded the `category` field, built only from
each entry's `name`/`description` -- querying "PortScan" (the exact alert
label) against T1046's description ("...a *port scan*...", two separate
words) scored a *different* entry higher, because "PortScan" doesn't
appear as a literal token anywhere in T1046's prose. Fixed by indexing
`category` alongside `name`/`description`, since the category label
itself, verbatim, is the dominant real query shape.

## LLM client: stub vs. real

`reasoner.py` depends only on the `LLMClient` interface (`llm_client.py`),
not on which implementation is behind it -- the same `--use-stub` pattern
`ids_ml.serve` already uses for Kafka:

- `StubLLMClient` -- deterministic, no network call, no API key. Extracts
  whatever ATT&CK technique ID retrieval embedded in the prompt and
  echoes a validly-structured JSON response around it. This is what every
  test in this module's test suite runs against, and what `serve.py`
  defaults to.
- `AnthropicLLMClient` -- calls a real hosted model via the `anthropic`
  package. **No API key or live network access to the API was available
  in the environment this module was built in.** Its request/response
  wiring is unit-tested by mocking the SDK client (`test_tier2_llm_client.py`
  verifies the exact `messages.create` call shape and response parsing),
  but it has not been exercised against a real API call. Treat it as
  implemented-but-live-untested until you've run it yourself with real
  credentials -- see [Known limitations](#known-limitations).

The response-parsing contract: the system prompt asks for a JSON object
with exactly `suspected_technique_id`, `suspected_technique_name`,
`risk_explanation`, `recommended_action`. `Tier2Reasoner._parse_response`
never raises on a malformed response -- an unparseable reply falls back
to the raw text as `risk_explanation` with empty technique fields, since
a degraded explanation is still more useful to an analyst than a dropped
alert (verified in `test_tier2_reasoner.py`).

## `--no-rag`: the RAG ablation switch

`Tier2Reasoner(use_rag=False)` (or `serve.py --no-rag`) skips retrieval
entirely -- the LLM gets no grounding context, just the alert's own
fields. This exists for the H3 comparison the original research plan
named (does RAG grounding actually improve explanation quality/accuracy
vs. a bare LLM call) -- **not yet run**. Running it requires a real LLM
client and some way to score explanation quality (human rating, or an
LLM-judge setup), neither of which exists in this repo yet; the switch is
built and tested (`test_tier2_reasoner.py::test_explain_without_rag_retrieves_nothing`),
the actual comparison isn't.

## Latency

Orchestration overhead (retrieval + prompt construction, everything in
this module *except* the LLM call itself), measured with
`StubLLMClient(simulated_latency_ms=0.0)`, 200 calls after warmup:
**0.581 ms/call**. Negligible, as expected -- TF-IDF cosine similarity
over 7 short documents is not where the cost lives.

**The real LLM call is the entire latency story here, and it is not
measured in this repo** -- no API access was available to measure it
live. Published estimates for LLM-based SOC alert triage put per-call
latency at 3-5+ seconds (see the literature check in `ml/README.md`'s
open-set section). Combined with `ml/README.md`'s measured Tier 1 latency
(9.67ms median single-flow) and this project's 10% target escalation
budget, `ids_ml.evaluation.amortized_latency_ms` gives:

```
amortized = 9.67 + 0.10 * tier2_latency_ms
```

At a 3-5s real LLM call, that's **~309-509ms amortized per flow** --
roughly 30-50x Tier 1's own latency, dominated entirely by the escalated
minority's LLM cost despite that minority being only 10% of traffic. This
is the number a real deployment needs to measure and accept (or push the
budget down, or use a faster/smaller model) before calling this
architecture "real-time" without qualification -- run
`ids_tier2.serve --llm anthropic` yourself with real credentials and feed
the measured latency back into `amortized_latency_ms` to get a real
number instead of this estimate.

## Serving

```
python -m ids_tier2.serve --bootstrap-servers localhost:9092 --llm anthropic
```

`--use-stub` runs against in-memory stubs instead of a real Kafka broker.
`--llm stub` (the default) needs no API key. `--no-rag` disables
retrieval. `--top-k` controls how many techniques retrieval returns per
alert (default 3).

## Testing

```
pip install -r requirements.txt
PYTHONPATH=src pytest
```

49 tests, all passing, all against `StubLLMClient` (no network calls, no
API key required) -- see [Known limitations](#known-limitations) for what
that does and doesn't cover.

## Known limitations

- **`AnthropicLLMClient` has not been exercised against a real API
  call.** Unit-tested via mocking, not integration-tested live -- no
  credentials were available in the build environment. Test it yourself
  before trusting it in any real deployment.
- **Real LLM latency is estimated from published literature, not
  measured in this repo.** See [Latency](#latency) -- the 3-5s figure is
  someone else's number, not this project's own measurement.
- **The knowledge base is 7 curated entries, not the full ATT&CK
  corpus**, and its category-to-technique mapping is this project's own
  reasonable association, not a verified ground truth -- see
  [Retrieval](#retrieval-why-tf-idf-not-embeddings).
- **The RAG-vs-no-RAG ablation (H3) is built but not run** -- the
  `--no-rag` switch exists and is tested; no explanation-quality
  comparison between the two modes has actually been performed.
- **No batching, no backpressure/retry wrapper on the explanation
  producer** -- see `explanation_producer.py`'s module docstring for why
  that's a documented scope reduction (the LLM call itself is the
  bottleneck by 2-3 orders of magnitude, not the Kafka publish).
- **No Dockerfile/docker-compose wiring yet.** This module runs
  standalone (`python -m ids_tier2.serve`) but isn't yet integrated into
  the top-level `docker-compose.yml` the way `ingestion`/`ml`/
  `dashboard-api` are -- see the top-level README's known-limitations
  section.
- **`dashboard-api` doesn't consume `network.ids.explanations` yet.**
  This module publishes; nothing downstream reads it.

## Out of scope

Ingestion, ML scoring, the escalation decision itself (which alerts get
`escalated: true` -- that's `ml`'s conformal-calibrated open-set gate),
and the dashboard are not implemented here -- this module only
consumes/publishes via the documented Kafka topic contracts above.
