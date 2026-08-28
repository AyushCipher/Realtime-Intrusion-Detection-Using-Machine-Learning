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
- `llm_client.py` -- `StubLLMClient` (deterministic, offline), `AnthropicLLMClient`, and `GeminiLLMClient` (both real, both live-verified -- see [LLM client: stub vs. real](#llm-client-stub-vs-real)).
- `retry.py` -- `RetryingLLMClient`: retry/backoff and rate-limit-aware pacing, wrapping any `LLMClient`.
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
  package. Connectivity and request wiring were confirmed live (real key,
  real request, real structured error back) -- the account behind that
  key had no credit balance, so no successful completion has been
  observed from this client yet. Unit-tested by mocking the SDK client
  otherwise. See [Known limitations](#known-limitations).
- `GeminiLLMClient` -- calls a real hosted model via the `google-genai`
  package. **Live-verified successfully** -- see
  [Latency](#latency-live-verified) and [the RAG ablation
  section](#--no-rag-the-rag-ablation-switch) below for real results from
  real API calls, not estimates. Lower-friction to test than Anthropic:
  Google AI Studio keys (https://aistudio.google.com/apikey) work on the
  free tier immediately, no payment method required (rate-limited to 5
  requests/minute per model on the free tier for `gemini-2.5-flash`,
  confirmed by hitting that exact limit during testing).

The response-parsing contract: the system prompt asks for a JSON object
with exactly `suspected_technique_id`, `suspected_technique_name`,
`risk_explanation`, `recommended_action`. `Tier2Reasoner._parse_response`
never raises on a malformed response -- an unparseable reply falls back
to the raw text as `risk_explanation` with empty technique fields, since
a degraded explanation is still more useful to an analyst than a dropped
alert (verified in `test_tier2_reasoner.py`).

**A real bug this live testing caught, that mocked tests couldn't have:**
the first live Gemini call returned a perfectly correct JSON response
wrapped in a ` ```json ... ``` ` markdown code fence, despite the system
prompt explicitly asking for raw JSON only. `_parse_response` didn't
anticipate that, so `json.loads` failed and the response fell through to
the degraded (unparsed) fallback -- silently discarding a good structured
response. Fixed by stripping a leading/trailing code fence before
parsing; see `reasoner.py::_parse_response` and the regression tests
`test_explain_strips_markdown_code_fence_around_json` /
`test_explain_strips_bare_code_fence_without_language_tag`. This is
exactly the kind of thing a stub client, by construction, cannot surface.

## `--no-rag`: the RAG ablation switch (informal live result)

`Tier2Reasoner(use_rag=False)` (or `serve.py --no-rag`) skips retrieval
entirely -- the LLM gets no grounding context, just the alert's own
fields. This exists for the H3 comparison the original research plan
named: does RAG grounding actually change/improve the explanation, vs. a
bare LLM call.

**A small, real, live comparison has now been run twice** (not the full
statistically-powered ablation this still needs, but real, growing
evidence, not a guess): first 3 escalated alerts (Brute Force, DoS/DDoS,
PortScan) against `gemini-2.5-flash`, then a 6-category batch (adding Web
Attack, Infiltration, Botnet) against `gemini-3.6-flash`.

| | with RAG | without RAG |
|---|---|---|
| Brute Force | `T1110` (correct) -- confirmed twice, both models | `""` -- explicitly declined, cited "no reference technique retrieved" |
| DoS/DDoS | `T1498` (correct) -- confirmed twice, both models | `""` -- same |
| PortScan | `T1046` (correct) -- confirmed twice, both models | `""` -- same |
| Web Attack | `T1190` (correct) | not attempted (call failed after retries -- see below) |
| Infiltration | call failed after 4 retries (see [Latency](#latency-live-verified--and-a-reliability-problem)) | not attempted |
| Botnet | call failed after 4 retries | not attempted |

**9 of 9 RAG-enabled live calls that actually completed, across two
sessions and two models, correctly named the technique this project's
own knowledge base associates with that category. Zero incorrect
technique attributions observed so far.** All 3 no-RAG calls that
completed (all from the first session) correctly followed the system
prompt's instruction not to invent a technique ID when none was
retrieved, instead reasoning qualitatively from the alert's own fields.
That's the real, observed effect of RAG here: specific, correct
technique attribution vs. none at all -- and it's held up consistently
so far, not just in the first small sample.

**The second session's no-RAG batch (all 6 calls) failed outright**,
and 2 of the 6 RAG calls also failed even after retries -- see
[Latency](#latency-live-verified--and-a-reliability-problem) for why
this is a real reliability finding, not just missing data, and why the
no-RAG evidence above still stands at n=3, not n=9.

This is still n=9 (RAG) / n=3 (no-RAG) alerts, not a rigorous ablation --
no seeds, no significance test, no sample large enough to claim a
statistically supported result, and now a real, observed reliability
ceiling on how large a live batch can even complete. Turning this into
H3 properly means running this same technique-ID-match comparison across
many more escalated alerts (the mapping in `knowledge_base.py` makes
"did it name the right technique" an automatable, countable outcome --
no human evaluation required for this specific proxy metric) with a
paired significance test, the same treatment H1/H2 got in `ml/` --
and, per the finding below, budgeting for a much slower, retry-tolerant
collection process than "just call it in a loop." Not yet built.

## Latency (live-verified -- and a reliability problem)

Orchestration overhead (retrieval + prompt construction, everything in
this module *except* the LLM call itself), measured with
`StubLLMClient(simulated_latency_ms=0.0)`, 200 calls after warmup:
**0.581 ms/call**. Negligible, as expected -- TF-IDF cosine similarity
over 7 short documents is not where the cost lives.

**The real LLM call dominates, and it is now measured across two
sessions -- and the second session changed the honest answer
substantially, for the worse.**

Session 1, `gemini-2.5-flash`, 7 successful RAG calls in a short burst:
4,503 / 5,708 / 5,746 / 5,750 / 5,976 / 6,095 / 6,382 ms -- median ~5,750ms.
3 no-RAG calls: 4,482 / 4,504 / 6,912 ms -- median ~4,504ms.

Session 2, `gemini-3.6-flash` (a different, newer model on a different
account): the very first isolated call was fast -- **2,623 ms**. Then a
6-category batch run immediately after told a very different story: only
**4 of 6 RAG calls succeeded, at 23,536 / 29,693 / 51,235 / 60,371 ms**
-- 10-25x slower than that first isolated call -- and **2 of 6 RAG calls,
plus all 6 no-RAG calls, failed outright** even after 4 retries each with
growing backoff (20s/40s/60s/80s, up to 200s of waiting per call).

**The honest interpretation: a single isolated call being fast tells you
almost nothing about what a real batch of traffic will experience.**
The most likely explanation is a longer-window quota (hourly/daily, not
just the per-minute limit hit in session 1) getting exhausted partway
through the batch, which would explain both the cascading failures *and*
the ballooning latency on the calls that did complete (server-side
queuing/throttling under the same constraint). This wasn't independently
confirmed against Google's own quota dashboard -- it's the most plausible
reading of the evidence, not a verified root cause. Either way, the
practical conclusion doesn't change: **treat single-call latency numbers
as a lower bound, not a representative one**, and build in retry/backoff
(not present anywhere in this module yet -- see [Known
limitations](#known-limitations)) before trusting this pipeline under any
sustained load.

Combined with `ml/README.md`'s measured Tier 1 latency (9.67ms median
single-flow) and this project's 10% target escalation budget,
`ids_ml.evaluation.amortized_latency_ms`, using the session-2 batch
median (a more representative, if worse, number than the isolated first
call):

```
amortized = 9.67 + 0.10 * 40464  =  ~4,056 ms/flow
```

That's roughly **420x** Tier 1's own latency, not the ~60x the first
(unrepresentative) fast call suggested. This is the real number this
project has actually measured, and it should be read as a warning, not a
target: at this budget and this observed batch behavior, "real-time" is
not an honest description of the current pipeline without either a much
lower escalation budget, a faster/more reliable model, or real
retry/backoff and rate-limit-aware request pacing that doesn't exist yet.

Failure modes observed live across both sessions, worth planning around
rather than treating as edge cases: a transient `503 UNAVAILABLE`
("model experiencing high demand"), a hard `429 RESOURCE_EXHAUSTED`
after 5 requests/minute on the free tier (session 1), and unexplained
`ClientError`/`ConnectError` failures that persisted through 4 retries
with up to 200s of backoff each (session 2's batch). At the time both
sessions were run, none of this was handled anywhere in the module's
actual code -- the ad-hoc retry loop that gave up after 4 attempts on 2
of 6 alerts lived in the throwaway batch script, not in `tier2_reasoner`
itself. `retry.py`'s `RetryingLLMClient` (see [Known
limitations](#known-limitations)) now exists specifically to close this
gap, with `serve.py` wrapping real clients in it by default -- but it
was built and unit-tested after these two sessions, not re-verified
against the same live failure conditions yet.

## Serving

```
python -m ids_tier2.serve --bootstrap-servers localhost:9092 --llm gemini
```

`--use-stub` runs against in-memory stubs instead of a real Kafka broker.
`--llm stub` (the default) needs no API key; `--llm gemini` (live-verified
on the free tier, no payment method needed) and `--llm anthropic`
(live-verified for connectivity, no successful completion observed yet --
needs a funded account) are the real options. Both real clients are
wrapped in `retry.RetryingLLMClient` by default (`--no-retry` disables
this; `--max-retries`/`--base-backoff-s`/`--min-interval-s` tune it -- see
[Known limitations](#known-limitations)). `--no-rag` disables retrieval.
`--top-k` controls how many techniques retrieval returns per alert
(default 3).

## Testing

```
pip install -r requirements.txt
PYTHONPATH=src pytest
```

70 tests, all passing, all against `StubLLMClient` or a fake in-process
client (no network calls, no API key required, deterministic) -- separate
from, and in addition to, the live Gemini calls described above under
[Latency](#latency-live-verified--and-a-reliability-problem) and the [RAG
ablation](#--no-rag-the-rag-ablation-switch-informal-live-result), which
were run manually and aren't part of the automated suite (no API key is
available in CI).

## Known limitations

- **`AnthropicLLMClient` has connectivity confirmed but no successful
  completion.** A real key reached the API and got a real structured
  error back (insufficient credit balance) -- the request/response
  wiring works, but this specific client's actual output has never been
  observed. `GeminiLLMClient` has: see [Latency](#latency-live-verified).
- **The H3 RAG ablation has a real n=3 live result, not the statistically
  powered comparison H1/H2 got.** See [the ablation
  section](#--no-rag-the-rag-ablation-switch-informal-live-result) for
  what was actually run and what a proper version needs -- the
  technique-ID-match proxy metric described there is buildable now and
  doesn't need human evaluation, just more live calls.
- **The knowledge base is 7 curated entries, not the full ATT&CK
  corpus**, and its category-to-technique mapping is this project's own
  reasonable association, not a verified ground truth -- see
  [Retrieval](#retrieval-why-tf-idf-not-embeddings).
- **Retry/backoff now exists (`retry.py`'s `RetryingLLMClient`), built
  directly from the confirmed real failures below -- but hasn't been
  live-retested yet.** `serve.py` wraps real clients in it by default
  (proactive pacing at a 12s/call default, matching the 5 requests/minute
  free-tier limit observed live for Gemini, plus reactive retry that
  parses the server's own suggested delay when available -- verified
  against Gemini's real 429 response shape). `--no-retry` disables it;
  `--max-retries`/`--base-backoff-s`/`--min-interval-s` tune it. 9 unit
  tests (`test_tier2_retry.py`) cover the retry/backoff/pacing logic
  against a fake client; it has not yet been run against the real batch
  scenario that originally exposed the need for it (2/6 RAG + 6/6 no-RAG
  failures -- see below), so whether it actually fixes that specific
  failure pattern is still an open question, not a verified fix.
- **The ~584ms amortized-latency estimate from a single fast call was
  wrong -- the real, batch-measured number is ~4,056ms/flow (~420x Tier
  1), not ~60x.** A single isolated LLM call is not representative of
  latency under any real load; see
  [Latency](#latency-live-verified--and-a-reliability-problem) for both
  numbers and why the gap is this large.
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
