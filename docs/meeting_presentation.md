# Faculty Review Meeting: Complete Presentation & Defense Guide

This document is your complete, section-by-section script and defense playbook for the faculty review meeting. Every section details **What to Display**, **What to Say to Sir**, **Which File Proves It**, **Potential Questions Sir May Ask**, and **Strong, Technically Accurate Answers**.

---

## Section 1: Project Title and Objective

- **What to Display:** Title slide or [`docs/project_overview.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/project_overview.md).
- **What to Say to Sir:**
  > "Respected Sir, our project is titled **Real-Time Network Intrusion Detection Using a Two-Stage Machine Learning Cascade and Conformal Open-Set Triage**. I am Ayush, presenting Module 1 of Group 1. Our objective is to design, implement, and benchmark an end-to-end intrusion detection system that achieves high multi-class attack recall, operates within real-time latency budgets, mathematically flags unseen zero-day attacks, and delivers local feature explanations for security analysts."
- **File / Proof:** [`docs/project_overview.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/project_overview.md)
- **Potential Question:** *"Is this a standard classification problem or anomaly detection?"*
- **Strong Answer:**
  > "It is a hybrid system, Sir. We deliberately combine unsupervised anomaly detection (Stage 1) to pre-filter normal traffic, supervised multi-class classification (Stage 2) to categorize known attack types, and statistical open-set recognition (OpenMax + Conformal Prediction) to detect zero-day anomalies that do not belong to any known class."

---

## Section 2: Problem Statement

- **What to Display:** Problem Statement section of [`docs/project_overview.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/project_overview.md).
- **What to Say to Sir:**
  > "The primary problem in existing NIDS literature is the closed-set assumption: models assume that all future network traffic will match the attack signatures present in the training data. When an attacker deploys a zero-day exploit, closed-set classifiers cannot say 'I don't know'—they silently and overconfidently misclassify the threat. Furthermore, evaluating heavy deep learning models on 100% of packets causes queuing bloat on high-speed backbones. We solve both challenges."
- **File / Proof:** [`paper/paper.html`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/paper/paper.html) (Section 1: Introduction).
- **Potential Question:** *"Why not just use a deep neural network with a softmax output?"*
- **Strong Answer:**
  > "Because standard softmax probabilities are notoriously overconfident on out-of-distribution inputs, Sir. Softmax normalizes exponential logits ($\sum p_i = 1$), forcing points that are far away from all training clusters into high-probability regions. In contrast, our approach adapts OpenMax to gradient-boosted tree margins, fitting Weibull extreme-value tails to explicitly quantify unknown mass."

---

## Section 3: Dataset Description

- **What to Display:** [`docs/dataset_and_features.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/dataset_and_features.md) (Class Distribution Table).
- **What to Say to Sir:**
  > "Our system is designed around the industry-standard Canadian Institute for Cybersecurity benchmarks: CICIDS2017 and CSE-CIC-IDS2018. For our local verification and test coverage, we utilize a 1,224-row dataset structured with 38 raw columns representing 8 distinct classes: BENIGN, DoS/DDoS, PortScan, Brute Force, Botnet, Web Attack, Infiltration, and Heartbleed. The dataset exhibits severe class imbalance, with benign traffic accounting for 65.4% and rare attacks like Infiltration representing under 1%."
- **File / Proof:** [`ml/tests/fixtures/synthetic_cicids_sample.csv`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/tests/fixtures/synthetic_cicids_sample.csv) & [`ml/src/ids_ml/data.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/data.py).
- **Potential Question:** *"Why not train on the classic KDD Cup 99 or NSL-KDD dataset?"*
- **Strong Answer:**
  > "KDD99 and NSL-KDD are over 25 years old and suffer from severe documented synthetic artifacts and redundant records. More importantly, their feature space does not reflect modern network protocols (such as IPv6, modern TLS handshakes, or realistic HTTP attack variants). CICIDS2017/2018 provides realistic modern PCAP captures generated using the CICFlowMeter feature extractor."

---

## Section 4: Input Features and Expected Output

- **What to Display:** Feature list and sample alert JSON in [`docs/dataset_and_features.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/dataset_and_features.md).
- **What to Say to Sir:**
  > "Our model receives an input vector of 35 canonical numerical features. These capture temporal characteristics such as flow duration and inter-arrival times; volumetric metrics such as forward/backward packets and bytes; packet length statistics; and TCP control flag counts. The model outputs a structured JSON alert specifying the predicted class, confidence, Stage 1 anomaly score, unknown mass, a boolean escalation decision, and top-3 TreeSHAP feature attributions."
- **File / Proof:** [`ml/src/ids_ml/features.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/features.py) & [`ml/src/ids_ml/schema.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/schema.py).
- **Potential Question:** *"Does the model consume raw packet bytes or flow statistics?"*
- **Strong Answer:**
  > "It consumes bidirectional flow statistics, Sir. Processing raw packet bytes requires packet payload inspection, which fails under encrypted traffic (HTTPS/TLS 1.3) and incurs extreme computational cost. Flow statistics computed over packet headers provide protocol-agnostic visibility even across encrypted traffic."

---

## Section 5: Preprocessing Workflow

- **What to Display:** Diagram in [`docs/training_testing_evaluation.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/training_testing_evaluation.md).
- **What to Say to Sir:**
  > "Our preprocessing pipeline executes three key steps: First, column name whitespace stripping to handle well-known CICFlowMeter formatting discrepancies. Second, temporal unit standardization, converting microsecond duration and IAT metrics to seconds. Third, feature normalization using a Scikit-Learn StandardScaler fitted strictly on the training partition and transformed across validation and testing."
- **File / Proof:** [`ml/src/ids_ml/data.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/data.py#L39-L79) and [`ml/src/ids_ml/stage1_iforest.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/stage1_iforest.py#L46-L50).
- **Potential Question:** *"How did you prevent data leakage during preprocessing?"*
- **Strong Answer:**
  > "We applied two strict safeguards, Sir: First, all data was split chronologically before any scaling, meaning `StandardScaler.fit()` was only executed on `X_train`. Second, in `train.py`, we execute an automated `leakage_comparison_report` comparing our time-based split against a random split; our report measured zero inflation ($0.0000$), confirming that no future packets or burst duplicates leaked into training."

---

## Section 6: Model Selection and Working Principle

- **What to Display:** [`docs/model_selection.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/model_selection.md) (Mathematical intuition & cascade diagram).
- **What to Say to Sir:**
  > "We selected a Two-Stage Cascade. Stage 1 is an unsupervised Isolation Forest containing 200 isolation trees. It scores anomalies based on average tree path length: anomalies require fewer random splits to isolate, resulting in shorter path lengths and higher anomaly scores. Stage 2 is an XGBoost ensemble of 300 decision trees trained with second-order gradient boosting on multi-class log-loss. Stage 2 executes only on flows flagged by Stage 1. Finally, an OpenMax head recalibrates margins using Weibull extreme value distributions, and a Conformal Gate enforces our escalation budget."
- **File / Proof:** [`ml/src/ids_ml/stage1_iforest.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/stage1_iforest.py) & [`ml/src/ids_ml/stage2_xgboost.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/stage2_xgboost.py).
- **Potential Question:** *"Why not just run XGBoost by itself without Isolation Forest?"*
- **Strong Answer:**
  > "Because in live networks, over 99% of traffic is normal. Running a 300-tree ensemble on every flow creates severe queuing bottlenecks. Isolation Forest acts as an ultra-fast sieve, dropping benign traffic with minimal computation while preserving Stage 2 capacity for complex attack classification."

---

## Section 7: Comparison with Alternative Models

- **What to Display:** Model Comparison Table from [`results/model_comparison.csv`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/model_comparison.csv).
- **What to Say to Sir:**
  > "To rigorously defend our model choice, we benchmarked our Two-Stage Cascade against four alternatives under identical train/test splits and canonical features: Standalone XGBoost, Random Forest, Logistic Regression, and a Decision Tree. While standalone XGBoost and Decision Trees achieved 100% closed-set accuracy on this fixture, they are completely closed-set and incapable of detecting zero-day attacks. Random Forest was 3.2 times slower in inference latency (59.8 ms). Logistic Regression suffered a poor Macro F1 of 85.59%. Our Two-Stage Cascade achieved 98.38% accuracy and 94.26% Macro F1, while uniquely providing open-set escalation and TreeSHAP explainability."
- **File / Proof:** [`results/model_comparison.csv`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/model_comparison.csv) and [`ml/scripts/evaluate_and_compare.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/scripts/evaluate_and_compare.py).
- **Potential Question:** *"Why did the Decision Tree get 100% accuracy?"*
- **Strong Answer:**
  > "On a synthetic fixture with caricatured features, a single decision tree can construct orthogonal axis-aligned splits that separate classes perfectly. However, in real networks, single decision trees suffer from severe variance and overfit aggressively to training noise. Our XGBoost ensemble with gradient shrinkage and depth constraints generalizes far more robustly."

---

## Section 8: Training Pipeline

- **What to Display:** [`ml/src/ids_ml/train.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/train.py) and output logs.
- **What to Say to Sir:**
  > "The training pipeline is executed via `ids_ml.train`. It loads the dataset, applies the 70/15/15 chronological split, trains the Isolation Forest pre-filter, fits the 300-tree XGBoost classifier, fits the OpenMax Weibull centroids, and calibrates the conformal escalation threshold on the held-out validation set. All serialized artifacts are saved into the `models/` directory."
- **File / Proof:** Saved models in [`models/`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/models/) (`stage1_iforest.joblib`, `stage2_xgboost.joblib`, `openset_gate.joblib`, `escalation_gate.joblib`).
- **Potential Question:** *"Why calibrate the conformal threshold on validation data rather than training data?"*
- **Strong Answer:**
  > "Because machine learning models are naturally overconfident on their training data. If we calibrated the threshold on training data, the model would produce artificially low unknown mass scores, leading to an overly strict threshold that would fail to escalate genuine zero-day attacks during testing. Calibrating on held-out validation data guarantees a valid, unbiased split-conformal quantile."

---

## Section 9: Testing Pipeline

- **What to Display:** [`ml/scripts/evaluate_and_compare.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/scripts/evaluate_and_compare.py).
- **What to Say to Sir:**
  > "Our testing pipeline runs completely independently of training. The script `evaluate_and_compare.py` loads the pre-fitted model artifacts directly from `models/` and evaluates them against the 185 held-out test flows. The test data was never seen during Stage 1 training, Stage 2 fitting, or conformal calibration."
- **File / Proof:** [`results/classification_report.txt`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/classification_report.txt).
- **Potential Question:** *"Can you run the testing script right now in front of us?"*
- **Strong Answer:**
  > "Yes, Sir! I can run `$env:PYTHONPATH='ml/src'; python ml/scripts/evaluate_and_compare.py` right now. It takes approximately 10 seconds to evaluate all models, verify predictions, and update the results folder."

---

## Section 10: Evaluation Metrics and Their Meaning

- **What to Display:** Evaluation Metrics Table in [`docs/training_testing_evaluation.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/training_testing_evaluation.md).
- **What to Say to Sir:**
  > "We evaluate eight metrics: Overall Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Single-Flow Latency, Throughput, and Model Size. In intrusion detection, Macro Precision is critical to avoid alert fatigue among SOC analysts, while Macro Recall ensures that rare zero-day threats are not missed. Accuracy alone is insufficient due to benign class dominance."
- **File / Proof:** [`results/metrics.json`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/metrics.json).
- **Potential Question:** *"What is the difference between Macro F1 and Weighted F1?"*
- **Strong Answer:**
  > "Macro F1 computes the unweighted arithmetic mean of F1-scores across all classes, giving equal importance to rare attacks like Infiltration (2 samples) as to large floods like DoS (36 samples). Weighted F1 weights each class by its support. In our results, Weighted F1 is 98.27% while Macro F1 is 94.26%, reflecting our honest accounting of rare classes."

---

## Section 11: Actual Experimental Results

- **What to Display:** [`docs/experimental_results.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/experimental_results.md) & [`results/metrics.json`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/metrics.json).
- **What to Say to Sir:**
  > "Our local test run achieved 98.38% accuracy (182 / 185 test samples correct), 99.64% Macro Precision, 91.35% Macro Recall, and 94.26% Macro F1. Furthermore, in our research paper on real CSE-CIC-IDS2018 traffic across 8 days, our OpenMax gate significantly outperformed the softmax baseline in zero-day recall ($p=0.00012$), and our online adaptive conformal gate held target budgets 100 times more accurately under drift ($p=1.9 \times 10^{-6}$)."
- **File / Proof:** [`results/classification_report.txt`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/classification_report.txt) and [`paper/paper.html`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/paper/paper.html) (Table I & Table II).
- **Potential Question:** *"Why is your Macro Recall 91.35% instead of 100%?"*
- **Strong Answer:**
  > "Because of the fundamental cascade trade-off: Stage 1 (Isolation Forest) operates in an unsupervised manner with contamination=0.35. Two single-packet PortScan flows and one low-volume Infiltration flow had feature values close to normal traffic, so Stage 1 did not flag them. This is an honest, mathematically realistic outcome of using a pre-filter."

---

## Section 12: Confusion Matrix and Interpretation

- **What to Display:** [`results/confusion_matrix.png`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/confusion_matrix.png).
- **What to Say to Sir:**
  > "Here is our confusion matrix on the held-out test split. The diagonal reflects strong detection: 116/116 BENIGN flows, 36/36 DoS/DDoS flows, 6/6 Brute Force flows, 4/4 Botnet flows, and 2/2 Web Attack flows were classified with 100% precision. The only off-diagonal entries are 2 stealthy PortScan flows and 1 Infiltration flow that slipped past the Stage 1 pre-filter into BENIGN."
- **File / Proof:** [`results/confusion_matrix.png`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/confusion_matrix.png).
- **Potential Question:** *"Why are there no false positives on BENIGN traffic?"*
- **Strong Answer:**
  > "Because when Stage 1 flags a flow, Stage 2 (XGBoost) re-evaluates it with supervised decision boundaries. If Stage 2 determines the flow is actually benign, it resolves the classification back to BENIGN, effectively canceling Stage 1 false alarms."

---

## Section 13: Overall Architecture

- **What to Display:** Mermaid diagram in [`docs/architecture.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/architecture.md).
- **What to Say to Sir:**
  > "Our system is structured as four independently deployable microservices orchestrated via Docker Compose: Ingestion replays PCAPs into Kafka topic `network.flow.features`; our ML Detection service scores flows and emits alerts to `network.ids.alerts`; Tier 2 Reasoner queries MITRE ATT&CK via RAG for escalated alerts; and FastAPI + React Dashboard visualizes alerts and TreeSHAP feature attributions in real time."
- **File / Proof:** [`docker-compose.yml`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docker-compose.yml) & [`SCHEMA.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/SCHEMA.md).
- **Potential Question:** *"Do the microservices share Python code or databases?"*
- **Strong Answer:**
  > "No, Sir. Each microservice is strictly decoupled. They share no Python code and communicate exclusively over validated JSON schemas on Kafka topics (`network.flow.features` and `network.ids.alerts`). This adheres to production cloud-native design principles."

---

## Section 14: Live Demonstration Commands

- **What to Display:** [`docs/demo_runbook.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/demo_runbook.md).
- **What to Say to Sir:**
  > "Sir, I can demonstrate the pipeline live. First, we run `evaluate_and_compare.py` to demonstrate testing and benchmark validation in 10 seconds. Second, we can run `docker compose up -d` to show the full stack running across Kafka, ML detection, and the live React dashboard at localhost:5173."
- **File / Proof:** [`docs/demo_runbook.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/demo_runbook.md) & [`Start.txt`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/Start.txt).

---

## Section 15: Limitations

- **What to Display:** Limitations section of [`docs/project_overview.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/project_overview.md) or [`paper/paper.html`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/paper/paper.html) (Section 6).
- **What to Say to Sir:**
  > "We believe in honest academic rigor. We document four genuine limitations: First, rare attack categories like Infiltration have small sample sizes in the public release. Second, the Tier-2 LLM reasoning service relies on external API quotas that cannot sustain real-time line rates on a free tier. Third, live packet capture requires administrative promiscuous interface privileges, so our containerized demo uses reproducible PCAP replay. Fourth, the cascade's attack recall is inherently bounded by Stage 1's sensitivity."
- **File / Proof:** [`paper/paper.html`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/paper/paper.html) (Section 6: Limitations).

---

## Section 16: Future Improvements

- **What to Display:** Section 3 of [`docs/architecture.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/architecture.md).
- **What to Say to Sir:**
  > "For future work, we plan to: (1) Replace the external cloud LLM with a local quantized model (such as Llama-3-8B-Instruct) served via vLLM for zero API latency; (2) Integrate hardware acceleration using eBPF or DPDK for kernel-bypass packet ingestion; and (3) Implement closed-loop active learning, where SOC analyst triage confirmations dynamically update the adaptive conformal step size."
- **File / Proof:** [`docs/architecture.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/architecture.md).

---

## Section 17: Conclusion

- **What to Display:** Summary slide or [`results/model_comparison.csv`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/model_comparison.csv).
- **What to Say to Sir:**
  > "In conclusion, Sir, we have developed a mathematically sound, empirically verified intrusion detection system. By combining an Isolation Forest pre-filter with an XGBoost classifier, an OpenMax zero-day recalibration gate, and conformal budget guarantees, we achieve 98.38% test accuracy, 99.64% precision, sub-20ms latency, and full TreeSHAP explainability. Every result presented is fully reproducible from our repository. Thank you, and I look forward to your questions."
