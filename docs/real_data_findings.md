# Real-data findings (CSE-CIC-IDS2018) and what they mean for the claims

Everything here was produced by scripts in `ml/scripts/` and `integration/` from
the real CSE-CIC-IDS2018 per-day CSVs (8 days, subsampled; see
`build_cicids2018_subsample.py`). Each result file under `results/paper/<name>/`
has a `manifest_*.json` (git commit, input hashes, seeds). Negative results are
reported as they came out.

## 1. The data has duplicate and label-conflict problems

`ml/scripts/dataset_audit.py` on the 333,648-row subsample:

| Check | Result |
|---|---|
| Exact duplicate feature rows | 31% overall; Brute Force 75%, Botnet 48%, DoS/DDoS 38% |
| Feature vectors with conflicting labels | 2,410 vectors (83,753 rows) |
| Chronological test rows also present in train | 13,263 of 50,048 (26%); 43% of test attack rows |
| Stage 1 attack-score ties | 78% of attack scores are ties (conformal thresholds become coarse) |

`--write-dedup` writes a cleaned set (227,219 rows: conflicting groups dropped,
first copy kept). After cleaning, FTP-BruteForce falls from 20,000 to 29 rows
and DoS-SlowHTTPTest from 20,000 to 13. Headline numbers computed on the
uncleaned subsample are partly inflated by duplicates and should be re-run on
the cleaned data (`--name cicids2018_dedup`).

## 2. Stage 1: the Isolation Forest filtered nothing on real data; the new hybrid does

On the cleaned CSE-CIC-IDS2018 data, attack-vs-benign AUROC of the original Isolation
Forest is 0.51 (chronological) and 0.40 (random split); it flags 91-100% of benign flows
at any useful miss budget (`stage1_separability.py`, `stage1_hybrid_eval.py`). The "cheap
pre-filter drops almost all benign traffic" premise held only on the synthetic fixture.

Leave-one-family-out selection (`stage1_selection.py`; no use of any test split),
CSE-CIC-IDS2018 cleaned, 2 seeds. AUROC for an attack family the model never saw,
and for known families (random split):

| Candidate | unseen family | known families |
|---|---|---|
| Isolation Forest (benign only) | 0.73 | 0.76 |
| Supervised XGBoost, depth 3 / depth 6 | 0.74 / 0.73 | 0.97 / 0.98 |
| Nearest-benign-neighbour distance (kNN) | 0.85 | 0.88 |
| Earlier hybrid: depth-3 XGBoost + Isolation Forest | 0.79 | 0.91 |
| **Weighted hybrid: depth-6 XGBoost + kNN, equal weights** | **0.84** | **0.96** |
| **Max fusion of the same two (current default, see 2b)** | **0.835** | **0.975** |
| Same, supervised weight 0.7 / 0.85 | 0.80 / 0.76 | 0.97 / 0.98 |

`ids_ml/stage1_hybrid.py` implements the hybrid per flow: each score is turned
into a percentile against a held-out benign reference sample and the two are combined
(maximum by default, or a weighted average)
(`--stage1-kind hybrid` in `train.py`; `--stage1-supervised-weight` slides between
supervised-only and balanced). It costs about 9 ms per flow and 9,400 flows/s in batches
of 128, against 10 ms and 11,400 for the Isolation Forest.

Conformal recall control on top of each filter (`stage1_hybrid_eval.py`, 3 seeds, 5%
miss budget). Each cell: realised attack miss rate / share of benign flows still flagged
(the compute the cascade still has to spend; lower is better):

| Data / protocol | Isolation Forest | supervised only (depth 3) | weighted hybrid |
|---|---|---|---|
| CSE-CIC-2018, random split | 4.7% / 99.9% | 3.6% / 31% | 4.8% / 23% |
| CSE-CIC-2018, unseen family | not meaningful | 55% / 26% | **19%** / 27% |
| CSE-CIC-2018, chronological | 1.7% / 95% | 1.2% / 92% | 1.1% / 91% |
| LYCOS-IDS2017, random split | 5.1% / 99.9% | 4.2% / 0.2% | 5.0% / 2.1% |
| LYCOS-IDS2017, unseen family | not meaningful | 99% / 0.1% | **44%** / 8.8% |
| LYCOS-IDS2017, chronological | 1.0% / 98% | 0% / 13% | 0% / 16% |

Reading: on correctly labelled LYCOS data the cascade premise holds (the filter drops
about 98% of benign flows at a 5% miss budget on known families, and the recall guarantee
holds). On CSE-CIC-IDS2018 it holds for known families only; on its chronological split,
where each day is a different attack family, no filter removes much benign traffic.
For unseen families the recall guarantee does not apply, and the hybrid roughly halves to
thirds the misses of a supervised filter at the cost of flagging more benign traffic.

### 2a. Why the chronological CSE-CIC-IDS2018 split fails, and what fixes it

Diagnosis (hybrid filter, chronological split): benign traffic does NOT drift much
between days (median score 0.63 on validation, 0.64 on test), and the test attacks
(Botnet, Infiltration) score high (median 0.93). The failure is the threshold: the
validation attacks it is calibrated on are 97% Infiltration, and Infiltration is
close to inseparable from benign traffic in this dataset. 29% of Infiltration flows
have feature vectors identical to benign flows, and even a model trained on
Infiltration reaches only 0.88 AUROC against benign. Meeting a 5% miss budget on
Infiltration drags the threshold below the benign median, so about 90% of benign
flows are flagged. Labelling problems in CSE-CIC-IDS2018 are documented by Liu et
al., "Error Prevalence in NIDS datasets" (IEEE CNS 2022); we did not confirm their
Infiltration-specific findings.

Two changes, evaluated over the time-ordered stream (`stage1_stream_hybrid.py`, test
families Botnet and Infiltration are new; miss rate / benign flows flagged at a 5% budget):

| Calibration | Miss rate (Botnet, Infiltration) | Benign flagged |
|---|---|---|
| pooled conformal (before) | 1.3% (0.0%, 5.2%) | 92% |
| per-family with a 30% benign-cost cap | 15% (0.0%, 58%) | 25% |
| adaptive (ACI), feedback immediately | 4.4% (4.1%, 5.0%) | 27% |
| adaptive (ACI), feedback after 1,000 flows | 12% (12%, 13%) | 29% |

* `conformal_stage1.calibrate_stage1_per_family` gives each family its own conformal
  threshold and, under a benign-cost cap, waives (and names) families whose budget would
  breach it. It makes the trade-off explicit: here it waives Infiltration and Web Attack
  and keeps the cost at 25%, at the price of missing most Infiltration flows.
* The adaptive control (`adaptive_stage1.py`, ACI on confirmed-attack feedback) meets the
  budget on both NEW families while flagging 27% of benign flows instead of 92%. It needs
  analyst-confirmed attack labels; with a 1,000-flow feedback delay the realised miss rate
  rises to 12% at a 5% budget. This is the only mechanism here that bounds misses on new
  families, and only over the long run.
* On LYCOS-IDS2017 (chronological, test = DoS/DDoS, 150,000 per-label cap) the adaptive
  control holds 5.0% misses at 0.3% benign flagged.

### 2b. Removing the threat-model choice: max fusion

The weighted hybrid forces a choice between near-known zero-days and entirely new
families. Taking the MAX of the supervised and kNN percentiles (`fusion="max"`, now the
default) ranks near the best single detector on both: AUROC 0.835 on unseen families and
0.975 on known ones (CSE-CIC-IDS2018), 0.999 known and 0.992 zero-day with FPR-at-95%-TPR
0.017 on LYCOS. Caveat: with a one-time calibration on known families, max fusion misses
MORE new-family attacks than the weighted hybrid (29% against 19% on CSE, 87% against 44% on
LYCOS, at a 5% budget), because its high scores on known attacks push the threshold up. It
should be paired with the adaptive control (2a), where it is the best option measured
(4.4% misses, 27% benign flagged on CSE). `--stage1-fusion weighted` restores the older
behaviour for static-only deployments.

### 2c. Main suite re-run with the hybrid Stage 1 (`run_paper_experiments.py --stage1-kind hybrid`)

Cleaned CSE-CIC-IDS2018. The `stage1` part fits Stage 1 on the first 30% of the
time-ordered stream, calibrates on the next 15% and streams the remaining 55%;
3 seeds. Overall miss rate / benign flagged:

| Miss budget 5% | static | adaptive (ACI) | detect-then-recalibrate (KS) |
|---|---|---|---|
| feedback delay 0 | 25.5% / 39% | **4.8%** / 52% | 2.5% / 51% |
| feedback delay 1,000 | 25.5% / 39% | 14.8% / 48% | **4.5%** / 52% |
| feedback delay 10,000 | 25.5% / 39% | 21.7% / 44% | 17.3% / 54% |

With the Isolation Forest the same experiment flagged 87-96% of benign flows for every
method; with the hybrid it is 36-54%. A static threshold breaks under drift (25.5% misses
at a 5% budget). Adaptive control meets the budget with prompt feedback; the KS
detect-then-recalibrate baseline is more tolerant of feedback delay (4.5% at a
1,000-flow delay, against 14.8% for ACI). Neither holds with a 10,000-flow delay.

Monte Carlo check of the static guarantee under exchangeability (`mc` part, 1,000 random
splits): observed miss 5.0% / 10.0% / 17.9% at budgets 5% / 10% / 20%. With the Isolation
Forest it was 0.3% / 9.8% / 12.6%, made loose by score ties; the hybrid's scores have far
fewer ties, so the bound is now tight.

H2 on the cleaned data (`h2` part, 10 seeds, escalation budget 10%; this part does not use
Stage 1): post-drift budget error static 0.128 (softmax) / 0.119 (OpenMax), KS
detect-then-recalibrate 0.017 / 0.015, adaptive 0.0009 / 0.0009. On the cleaned data the
static gate OVER-escalates (22-33% of known flows against the 10% budget), the opposite
direction of the near-zero escalation seen on the duplicated data; the adaptive gate
holds about 10.1% in both cases. The adaptive-gate conclusion is unchanged.

### 2d. What else was tried to make Stage 1 more capable

* **Per-host context features** (`context_features.py`, causal counts over 2 s and 30 s
  windows: flows, distinct ports and hosts, RST share, per source and per destination;
  `stage1_context_eval.py`, LYCOS-IDS2017, 3 seeds). In LYCOS every family except Botnet
  comes from one attacker host (172.16.0.1, 100% of those flows against 0.3% of benign), so
  context can detect "that host" instead of the attack. On Botnet, the only family without
  that shortcut, context gave the max-fusion filter +0.016 AUROC (0.914 to 0.930) and made
  the supervised model much WORSE (0.929 to 0.698): it learned the attacker host. Gains on
  the other families appeared exactly where the shortcut is available. Not shipped.
* **EFC as a third view of the max fusion** (`max3_s6k_efc`, 3 seeds): +0.013 on CSE unseen
  families, -0.004 to -0.005 elsewhere, and higher zero-day FPR on LYCOS (0.025 vs 0.017).
  Not shipped.
* **Training on other networks** (`stage1_multinetwork.py`, 3 seeds, no data from the target
  network): benign-vs-attack AUROC on the unseen network was 0.34-0.80 when trained on one
  other dataset and 0.51-0.72 when trained on both others pooled, with several results below
  0.5. Pooling helps in some directions and hurts in others. A Stage 1 does not transfer
  between networks, even with identical CICFlowMeter features; it needs data (at minimum,
  benign traffic for calibration) from the network it protects.
* Across all 17 candidates and three datasets, the max-fusion hybrid stayed within about
  0.01 AUROC of the best candidate on each protocol and never fell far behind.

## 3. Adversarial calibration (padding, slowing, mimicry), with the hybrid Stage 1

`run_adversarial_calibration.py --split random` (known families, where Stage 1 filters
about 80% of benign traffic, so evasion is measurable), CSE-CIC-IDS2018 cleaned, 3 seeds,
5% miss budget. "Mimic lam" moves every attacker-controllable feature (all except TCP flag
counts) a fraction lam of the way toward a randomly drawn REAL benign training flow.

| Perturbation | Miss rate, standard calibration | Miss rate, traffic-aware calibration |
|---|---|---|
| none | 5.0% | 6.0% |
| pad 0.5 / 1.0 | 4.9% / 2.9% | 8.3% / 4.8% |
| slow x2 / x8 / x32 | 5.9% / 4.3% / 3.3% | 7.3% / 5.3% / 3.8% |
| mimic 0.25 / 0.5 | 1.2% / 1.0% | 1.6% / 1.6% |
| mimic 0.75 / 0.9 (outside the calibration law) | 3.4% / **7.3%** | 4.3% / **10.8%** |
| benign flows flagged | 20% | 15% |

Reading: padding and slowing do not evade the hybrid; they often make attacks easier to
catch, because the density view flags unusual shapes. A benign-aware attacker that moves a
flow 90% of the way to a real benign flow pushes misses to 7-11% against a 5% budget:
degradation, not collapse. Traffic-aware calibration (calibrating on attacks perturbed by the
expected law) lowers the benign cost from 20% to 15% and meets the budget on average over
that law (4.98% across the nine in-law settings), but individual settings exceed it (up to
8.3%), as the guarantee only covers the mixture. On the chronological split the same test is
uninformative because Stage 1 still flags 79-94% of benign traffic under pooled calibration
(see 2a). These are feature-space perturbations, not packet-level realisable attacks; a
gradient-based or query-based attacker was not tested.

## 4. Open-set gates: OpenMax is not the best gate; kNN is, end to end

Three protocols on real data:

* **Stage 2 only** (`run_paper_experiments.py` H1, leave-one-family-out, uncleaned data,
  5 families x 5 seeds, budget 0.10): mean unknown recall kNN 0.48, Mahalanobis 0.28,
  energy 0.12, max-logit 0.11, OpenMax 0.11, softmax 0.08; ranking AUROC softmax 0.611,
  OpenMax 0.610. kNN escalated 37% of known traffic against the 10% budget there, so its
  recall in this protocol is not a fair comparison. Clustered significance (seeds averaged
  within each held-out family, Wilcoxon over 5 families): OpenMax vs softmax recall 4-1,
  p = 0.31; AUROC 1-4, p = 0.63. The previously documented p = 0.00012 treated 25 seed x
  family pairs as independent (pseudo-replication) and should not be reported.
  Re-run on the CLEANED data (5 families x 5 seeds): unknown recall at a matched 10%
  false-escalation rate on test known flows: kNN 0.58, Mahalanobis 0.42, softmax 0.20,
  OpenMax 0.18, energy 0.17, max-logit 0.17; AUROC kNN 0.77, Mahalanobis 0.72, OpenMax
  0.64, softmax 0.63. OpenMax vs softmax, clustered: recall at matched rate 4-1, p = 0.63;
  AUROC 3-2, p = 0.63. No difference between OpenMax and softmax; kNN is best.
* **VAEMax-style fixed known/unknown classes** (`run_vaemax_protocol.py`, cleaned data, 5%
  benign false-positive budget, 3 seeds): unknown recall softmax 0.64, OpenMax 0.62, energy
  0.51, kNN 0.24, Mahalanobis 0.11; kNN has the best known-attack accuracy (0.89 vs 0.66).
* **End to end through the deployed cascade** (`run_cascade_openset.py`, cleaned data,
  hybrid Stage 1 at a 5% miss budget, gate calibrated on the known flows Stage 1 flags,
  5 families x 2 seeds). Every gate escalates 4.4-4.8% of known traffic, so the cost is
  matched:

| Gate | Unknown-family flows escalated | Benign flows raising any alert |
|---|---|---|
| kNN | **0.64** | 4.9% |
| Mahalanobis | 0.32 | 3.8% |
| OpenMax | 0.32 | 4.5% |
| energy | 0.32 | 4.6% |
| max-logit | 0.30 | 4.2% |
| softmax | 0.19 | 4.0% |

  Stage 1 passes 71% of unknown-family flows on average (100% of Botnet and Brute Force,
  77% of DoS/DDoS and Web Attack, 2% of Infiltration), which caps every gate. Per family,
  kNN is best or tied on four of five held-out families; nothing catches Infiltration.

Conclusion: the "OpenMax on tree margins" contribution is not supported as a clear win. It
beats softmax in the cascade but ties energy and max-logit, and a kNN gate on the input
features escalates twice as many unknown-family flows at the same cost. The deployed gate
should be kNN: `train.py --gate knn` now fits, calibrates (on the validation flows Stage 1
flags, as deployed) and saves it, and `serve.py` loads it. The previously documented headline (OpenMax 0.166 vs softmax
0.094 recall) does not reproduce with the current code and data; another session re-ran
H2 with the committed code and got the same values as the current code, so the old numbers
came from a different environment.

## 5. Suricata case study (small, illustrative)

`integration/suricata_compare.py` scores the same captured pcap with both
systems. The pcap comes from `live_traffic_gen.py` against a local web server on
loopback (10 ordinary GETs, a request burst, connection probes), captured with tcpdump
in a container. Suricata 8.0.8 ran offline with the full Emerging Threats Open
ruleset (53,050 rules; the container could not fetch rules itself, so they were
downloaded on the host and passed with `-S`).

* Suricata: 0 alerts on 420 flows.
* Our cascade, with thresholds from the CSE-CIC-IDS2018 lab: flagged and escalated
  every flow, including the normal ones (out-of-distribution flow statistics).
* Our cascade after site calibration on a separate benign-only capture: false alarms
  on held-out normal flows were 5% flagged and 0.5% escalated, but none of the burst
  or probe flows were flagged.

So neither system detected this traffic. The test shows the importance of
calibrating on the deployment network, and that per-flow features cannot see
a burst or scan that is spread over many short flows. It is one short capture
of generated traffic, not a benchmark; ground-truth phases are approximate
because the ingestion extractor emits more flows than Suricata counts
(1,042 against 420).

## 6. LYCOS-IDS2017 (corrected CIC-IDS2017) with a CLOSR-style protocol

Sources: the authors' repository (maupiti-git.univ-lemans.fr/lycos/lycos-ids2017)
ships the original CIC-IDS2017 CSVs and the LycoSTand-extracted flows; their own
`labelling.py` was run unmodified (2 worker processes) to produce the labelled files,
and `ml/scripts/lycos_prepare.py` maps them to the 35 canonical features. 1,837,498
labelled flows reduce to 1,540,331 after removing exact duplicates and conflicting
groups (23 groups). After this cleaning PortScan falls from 160,106 to 1,429 rows (its
flows were near-identical), Heartbleed has 11 rows and SQL injection 12.

`run_closr_protocol.py` follows Wilkie et al. (arXiv:2601.09902): a 50/50 split with
each class split evenly, SQL injection and Heartbleed in the test set only as
zero-days, benign-vs-attack AUROC. 3 seeds, training capped at 30,000 flows per class:

| Detector | AUROC known | AUROC zero-day | FPR at 95% TPR, zero-day |
|---|---|---|---|
| Isolation Forest (benign only) | 0.858 | 0.874 | 0.182 |
| kNN distance to benign | 0.822 | 0.934 | 0.143 |
| Supervised depth-3 XGBoost | 0.999 | 0.907 | 0.126 |
| Supervised depth-6 XGBoost | **1.000** | **0.990** | **0.015** |
| Weighted hybrid (depth 6 + kNN, equal) | 0.928 | 0.982 | 0.050 |
| **Max-fusion hybrid (current default)** | **0.999** | **0.992** | **0.017** |
| Hybrid, supervised weight 0.7 / 0.85 | 0.958 / 0.982 | 0.989 / 0.991 | 0.027 / 0.017 |

On this protocol a plain depth-6 XGBoost is best, because the held-out zero-day classes
(web SQL injection, Heartbleed) sit close to known web and DoS families. The hybrid is
chosen for families that are entirely new (section 2). The right Stage 1 depends on the
threat model, and `--stage1-supervised-weight` lets an operator choose. For reference,
the CLOSR paper reports FPR-at-95% of 0.0073 (Heartbleed) and 0.0033 (SQL injection)
for its own model; comparability is limited by its 72 features, its hyperparameter
search, and the 11 and 12 flow zero-day classes here.

## 7. H3: retrieval grounding, larger and fairer

`tier2_reasoner/scripts/run_h3_v2.py`, model `gemini-3.1-flash-lite` (the code's default
`gemini-3.6-flash` was unavailable to new users and the larger Flash models returned
"high demand"), 6 categories x 20 paired trials, alert fields varied per trial, no failed calls:

| Condition | Correct technique |
|---|---|
| Retrieval-grounded (as built) | 101 / 120 (84%, 95% CI 77-90%) |
| Ungrounded, allowed to guess (fair baseline) | 66 / 120 (55%, CI 46-64%) |
| Ungrounded, told not to guess (the original pilot's baseline) | 0 / 120 |

Exact McNemar, grounded vs fair baseline: 49 vs 14 discordant pairs, p = 1.1e-5. The
original pilot's 9/9 vs 0/9 compared against a baseline that was told not to answer;
against a baseline allowed to guess, the gain is real but smaller (84% vs 55%).

Per category, grounded answers were 20/20 everywhere except Web Attack (1/20): retrieval
ranked the right technique (T1190, score 0.24) first but also passed weak distractors
(T1499 at 0.08), and the model chose T1499 in 19 of 20 trials. `Tier2Reasoner` now drops
retrieved techniques below half the top score (`min_relative_score=0.5`). This change was
made after seeing the result, so it was re-tested on fresh trials (20-39). That run
stopped at 31 paired trials when the free-tier daily quota ran out: grounded 31/31,
ungrounded-with-guessing 17/31, told-not-to-guess 0/31, no failed calls; Web Attack is now
5/5 with grounding (was 1/20). Exact McNemar grounded vs guessing: 14 vs 0 discordant,
p = 1.2e-4. The re-test is small (about 5 trials per category).

Limits: the knowledge base both grounds the model and defines the correct answer, so
this measures whether the model uses retrieved context, not its ATT&CK knowledge in
general; alerts are synthetic; one small model.

### 7a. H3 on REAL escalated alerts: grounding fails when the predicted class is wrong

`ml/scripts/export_escalated_alerts.py` trains the deployed cascade (hybrid Stage 1, kNN gate)
without one attack family at a time and exports what it escalates, as the scoring service
would publish it, with real TreeSHAP attributions; 20 alerts per true family (100 total, all
from the fold where that family was withheld). `tier2_reasoner/scripts/run_h3_real.py`, same
model, judged against the TRUE family's technique:

| Condition | Correct |
|---|---|
| Retrieval-grounded (as built) | **0 / 100** (95% CI 0-4%) |
| Ungrounded, allowed to guess | 12 / 100 (7-20%), 11 of them on Web Attack |

McNemar: 0 vs 12 discordant, p = 0.0005, against grounding. Every one of these alerts carries
a wrong predicted class (84 of 100 are predicted BENIGN: Stage 2 has no class for the novel
family and the kNN gate escalated them anyway). Retrieval is keyed on the predicted class, so
it returns no relevant technique, and the model, as instructed, declines to name one. The
synthetic-alert result (84% vs 55%) only holds when the classifier's label is right; for the
escalated unknown attacks Tier 2 exists for, class-keyed retrieval adds nothing. Correction to the mechanism: class-keyed retrieval on "BENIGN" matches no knowledge-base
entry (TF-IDF score 0 for all), but `TechniqueRetriever.retrieve` keeps zero-score rows, so
for those 84 alerts the model received the LAST three entries (T1071.001, T1210, T1046)
as "reference techniques" -- arbitrary context, which it then correctly declined to use.

**Behaviour-keyed retrieval** (built by a second session; `tier2_reasoner/src/ids_tier2/
behaviour.py`, `retrieval.retrieve_by_behaviour`, `Tier2Reasoner(retrieval_mode=...)`):
each alert's top TreeSHAP features are mapped to fixed-threshold behaviour phrases ("very
short connection", "reset flag", "high packet rate") and matched against a hand-written
`signature` of observable behaviour per technique. Evaluated offline on the same 100 real
alerts (no LLM; `scripts/eval_retrieval_real.py`), counting whether the TRUE family's
technique is ranked first, and whether it is among the techniques actually sent to the
model after the relative-score filter:

| Retrieval | Correct technique first | Correct technique sent |
|---|---|---|
| class-keyed (as built) | 15 / 100 | 31 / 100 |
| behaviour-keyed | **40 / 100** | **70 / 100** (mean 2.4 sent) |
| random expectation | 17.1 | 48.6 (3 sent) |

Per family (first / sent): Botnet 13 / 20, Brute Force 0 / 19, DoS/DDoS 14 / 18, Web Attack
13 / 13, Infiltration 0 / 0. Class-keyed retrieval's 15 Botnet "hits" are the zero-score
artifact above, not a match. Caveats: the signatures were written knowing which five family
names exist (not their data); this measures retrieval only, and whether the LLM then names
the right technique needs a new live run.

## 8. Not done

* The leave-one-label-out `h1raw` part was not re-run on the cleaned data.
* Completing the H3 re-test to 20 trials per category (needs another day of quota).
* An LLM run of behaviour-keyed retrieval on the real alerts (section 7a; retrieval-only results are in).
