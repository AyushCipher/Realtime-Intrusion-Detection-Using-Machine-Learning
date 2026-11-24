# Results summary for the report

All results below come from runs on real network-traffic datasets. Each section names the
result file in `results/paper/` it comes from, and `docs/real_data_findings.md` has the
full details. Numbers that did not hold up are listed in the last section, so the report
does not repeat them.

**Datasets** (all cleaned: exact-duplicate flows and flows with conflicting labels removed)

| Dataset | Flows used | Notes |
|---|---|---|
| CSE-CIC-IDS2018 | 227,219 | 8 days; 31% of the original sample were duplicates |
| UNSW-NB15 | 280,345 | 9 attack families |
| LYCOS-IDS2017 | 1,540,331 (300,000 sampled where memory required) | corrected relabelling of CIC-IDS2017 |

---

## 1. Data quality (CSE-CIC-IDS2018)

| Check | Result |
|---|---|
| Exact duplicate rows | 31% overall (Brute Force 75%) |
| Feature vectors with conflicting labels | 2,410 (83,753 rows) |
| Test rows that also appear in training (time split) | 26% (43% of test attacks) |

Meaning: results on the raw dataset are inflated. Everything below uses the cleaned data.

## 2. Stage 1 pre-filter

| Pre-filter | AUROC known families | AUROC unseen families | Benign flows flagged at a 5% miss budget |
|---|---|---|---|
| Isolation Forest (original) | 0.51 (chronological) | – | 91–100% (filters nothing) |
| **Hybrid (supervised + nearest-benign distance)** | **0.975** | **0.835** | **21%** (CSE-CIC-IDS2018), **0.3%** (LYCOS-IDS2017) |

On LYCOS-IDS2017 under a published zero-day protocol (CLOSR): hybrid AUROC 0.999 on known
attacks, 0.992 on zero-days. Source: `cicids2018_dedup*`, `lycos2017*`.

## 3. Detecting unknown attacks (escalation gates)

Significance is computed over held-out attack families, not over repeated runs.

**Entirely new families** (one family withheld from training at a time; end to end through the cascade, all gates at the same analyst cost):

| Gate | Withheld-family flows escalated |
|---|---|
| **kNN** | **0.64** |
| OpenMax / energy / Mahalanobis | 0.32 |
| Max-logit | 0.30 |
| Softmax | 0.19 |

**New variants of known families** (one attack type withheld, 9 types; `cicids2018_dedup_h1raw`):

| Gate | AUROC |
|---|---|
| **Softmax** | **0.861** |
| OpenMax | 0.857 |
| kNN | 0.704 |

- OpenMax vs softmax: no significant difference in any test (p = 0.63–0.95).
- The best gate depends on the kind of unknown attack.
- A fused gate combining kNN and softmax was a compromise (AUROC 0.744 on new families, 0.814 on variants) and failed its pre-set deployment rule, so kNN was kept (`fused_gate`).
- Infiltration is undetectable by every gate (AUROC 0.47).

## 4. Analyst (escalation) budget under drift, without labels

10% budget, cleaned CSE-CIC-IDS2018, 10 seeds, after drift (`cicids2018_dedup/h2.csv`):

| Threshold | Error from the 10% budget |
|---|---|
| Static (calibrated once) | 0.128 (escalated 22–33% of normal traffic) |
| Detect-then-recalibrate (KS) | 0.017 |
| **Adaptive (ACI)** | **0.0009** |

Adaptive beat both alternatives on every seed (Wilcoxon p = 0.002).

## 5. Miss budget when analyst labels arrive late (Part A)

Stage 1 on a time-ordered stream (37,195 attacks, 87,777 benign), 5% miss budget, 5 seeds (`cicids2018_dedup_delay`):

| Label delay (flows) | Static | Adaptive (published delayed ACI) | Detect-then-recalibrate | **Union (new)** |
|---|---|---|---|---|
| 0 | 25.2% | 4.8% | 2.5% | **1.8%** |
| 1,000 | 25.2% | 14.6% | 4.5% | **3.1%** |
| 10,000 | 25.2% | 22.2% | 17.6% | **7.6%** |
| 100,000 | **25.2%** | 47.3% | 43.7% | 43.7% |

- The union flags a flow if either online method flags it.
- It holds the budget up to a 1,000-flow delay, at a cost of 2–4 extra points of benign traffic flagged.
- At very long delays every adaptive method is worse than not adapting.

## 6. End-to-end miss budget for the whole cascade (Part B)

Random split, 10 seeds, 3 datasets (`cascade_crc`). Each cell shows the end-to-end miss rate, with the number of seeds over budget in brackets:

| Dataset, budget | Stages calibrated separately | Bonferroni | Joint (lockstep) | **Joint (cost-aware)** |
|---|---|---|---|---|
| CSE-CIC-IDS2018, 5% | **10.2%** (10/10) | 4.9% (2) | 4.9% (5) | 4.6% (2) |
| CSE-CIC-IDS2018, 10% | **11.1%** (10/10) | 9.7% (1) | 9.9% (5) | 9.4% (0) |
| UNSW-NB15, 5% | 5.07% (8) | 5.06% (5) | 5.05% (6) | 4.2% (1) |
| LYCOS-IDS2017, 5% | 3.8% (0) | 4.7% (1) | 4.9% (3) | 3.9% (0) |

- Calibrating the stages separately doubled the miss rate on CSE-CIC-IDS2018: Stage 2 silently passed 5.5% of the attacks Stage 1 flagged.
- The cost-aware joint calibration stayed under budget on all three datasets.
- At a 10% budget on CSE-CIC-IDS2018, joint calibration raised half as many benign alerts as Bonferroni (4.8% vs 10.2%).

## 7. The deployed system

Trained on 353,238 flows from all three datasets (15 attack families) with:
- hybrid Stage 1;
- kNN escalation gate with an adaptive threshold;
- joint calibration at a 5% end-to-end miss budget.

Held-out test split, known families (`deployment/deployed_eval.json`):

| Measure | Result |
|---|---|
| **Attacks detected** | **95.0%** (miss rate 4.98%, the 5% budget) |
| Benign flows raising an alert | 4.3% |
| Benign flows sent to an analyst | 2.4% |
| Families detected at ≥ 99% | 12 of 15 |
| Web Attack / PortScan | 94.5% / 93.5% |
| Infiltration | 33.8% |

**Speed and size** (container limited to 1 CPU and 1 GB; a proxy, not edge hardware; `edge`):

| Measure | Result |
|---|---|
| One flow per call | 4.9–6.6 ms median (about 240 flows/s) |
| Batches of 128 | 4,304 flows/s |
| Peak memory | 199 MB |
| Models on disk | 13.6 MB |

## 8. Tier 2 (LLM explanations of escalated alerts)

**Retrieval on 500 real escalated alerts** (no LLM; 97.4% of these alerts carry a wrong predicted class; `tier2/retrieval_real_500.json`):

| Retrieval keyed on | Correct technique ranked first | Correct technique among those sent |
|---|---|---|
| Predicted class (original) | 92 | 163 |
| **Flow behaviour (new)** | **175** | **332 (66%)** |
| Random (expected) | 86 | 243 |

**LLM accuracy on 100 real escalated alerts** (Gemini 3.1 Flash-Lite; `tier2/h3_real_gemini_100.summary.json`):

| Condition | Correct ATT&CK technique |
|---|---|
| Class-keyed retrieval (original) | 0 / 100 |
| No retrieval | 12 / 100 |
| **Behaviour-keyed retrieval** | **34 / 97 (35%)** |

- Behaviour-keyed vs class-keyed: p = 1.2×10⁻¹⁰.
- Behaviour-keyed vs no retrieval: p = 1.0×10⁻⁵.
- On synthetic alerts with correct classes, grounding gave 84% vs 55% (p = 1.1×10⁻⁵).
- Llama 3.2 1B (local, open-weights) on 500 alerts: *pending, run in progress.*

**Verdict:** useful as an analyst aid with behaviour-keyed retrieval, not as detection. The retrieval suggestions alone are the most reliable part. Hosted free tiers were unreliable (503 errors, daily quotas), so real use needs a stronger model.

## 9. Live end-to-end tests (Docker)

| Test | Result |
|---|---|
| Live packet capture → ingestion → Kafka → ML → dashboard | Passed (alerts from sniffed packets) |
| Demo pcap replay end to end | Passed (alert with SHAP explanation) |
| **Real-data system with a real LLM** (Llama 3.2 1B through Ollama) | **Passed**: escalated alerts explained, median 4.2 s, worst 31.6 s |

The live-capture traffic was generated by a script against the project's own service. It tests the pipeline, not detection of real attacks.

## 10. What did not hold up (do not report these)

| Earlier claim | What the tests show |
|---|---|
| OpenMax beats softmax (0.166 vs 0.094 recall, p = 0.00012) | Does not reproduce; p was inflated by counting 25 correlated runs as independent. Properly tested, no difference (p = 0.31–0.95) |
| Static escalation threshold collapses to about 0.05% | Current code gives about 0.45% on the uncleaned data (the conclusion still holds) |
| Isolation Forest is a useful pre-filter | AUROC 0.51 on real data; replaced by the hybrid |
| "Real-time" for the full system | Tier 1 is real-time; LLM explanations take seconds and arrive asynchronously |

## Limitations to state in the report

- **Guarantees cover known families only.** The miss-rate guarantees hold for attack families present in training, under no drift. New families rely on the kNN gate, which caught about 64% of their flows.
- **Adaptive miss control needs prompt labels.** It depends on confirmed labels, and no method held the budget with labels 10,000 or more flows late.
- **A new network needs its own calibration.** The escalation threshold has to be calibrated on that network's own normal traffic; the synthetic demo traffic was all escalated.
- **Data limits.** Public datasets with known label problems, subsampled to fit 8 GB of RAM. Infiltration is close to inseparable from benign traffic.
