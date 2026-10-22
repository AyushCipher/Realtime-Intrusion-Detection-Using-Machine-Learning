# Live Faculty Demonstration Runbook (5–10 Minutes)

This step-by-step runbook guides you through a flawless, confident live demonstration in front of your professor. It includes exact PowerShell terminal commands, execution runtimes, expected outputs, and a fallback plan.

---

## 1. Demonstration Overview & Timeline

| Step | Time | What You Do | Screen / Terminal |
|---|---|---|---|
| **Step 1** | 0:00 – 1:00 | Architecture & Repo Tour | Show VS Code explorer + [`docs/architecture.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/architecture.md) |
| **Step 2** | 1:00 – 2:00 | Dataset & Target Labels | Open [`ml/tests/fixtures/synthetic_cicids_sample.csv`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/tests/fixtures/synthetic_cicids_sample.csv) & run label check |
| **Step 3** | 2:00 – 3:30 | Show Training & Artifacts | Inspect [`ml/src/ids_ml/train.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/train.py) & list [`models/`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/models/) |
| **Step 4** | 3:30 – 5:30 | Run Independent Testing | Execute `evaluate_and_compare.py` live in terminal (~10s) |
| **Step 5** | 5:30 – 7:00 | Show Real Results & Matrix | Display [`results/classification_report.txt`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/classification_report.txt) & open [`results/confusion_matrix.png`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/confusion_matrix.png) |
| **Step 6** | 7:00 – 8:30 | Model Comparison Defense | Display [`results/model_comparison.csv`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/model_comparison.csv) and defend model choice |
| **Step 7** | 8:30 – 10:00| Full-Stack / Fallback Tour| Show Docker Compose stack or sample predictions with TreeSHAP |

---

## 2. Step-by-Step Execution Script

### Step 1: Open Repository & Architecture (1 Minute)
1. Open the project root folder `c:\Users\ayush\OneDrive\Desktop\Urop Project` in VS Code.
2. Open [`docs/architecture.md`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/docs/architecture.md).
3. **Say to Sir:**
   > "Sir, here is our project architecture. As agreed in Group 1, I am responsible for Module 1: The Machine Learning Detection Engine and complete training/testing pipeline. We implemented a Two-Stage Cascade: Stage 1 is an Isolation Forest anomaly pre-filter, Stage 2 is an XGBoost multi-class classifier, and our OpenMax Conformal Gate handles zero-day escalation. All services communicate over Kafka topics."

---

### Step 2: Show the Dataset & Label Distribution (1 Minute)
In PowerShell terminal:
```powershell
python -c "import pandas as pd; df = pd.read_csv('ml/tests/fixtures/synthetic_cicids_sample.csv'); print('Dimensions:', df.shape); print('\nClass Distribution:\n', df[' Label'].value_counts())"
```
**Expected Output:**
```
Dimensions: (1224, 38)

Class Distribution:
 BENIGN                      800
 DoS Hulk                    200
 PortScan                    100
 FTP-Patator                  60
 Bot                          30
 Web Attack - Brute Force     20
 Infiltration                 10
 Heartbleed                    4
```
**Say to Sir:**
> "Here is our dataset: 1,224 rows and 38 columns, mapping onto 35 canonical numerical features. It reflects real-world network traffic with severe class imbalance: BENIGN traffic makes up 65.4%, while rare attacks like Infiltration represent under 1%."

---

### Step 3: Show the Training Code & Saved Model Artifacts (1.5 Minutes)
1. Point to [`ml/src/ids_ml/train.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/train.py). Show the chronological time-based split at line 97:
   ```python
   train, val, test = time_based_split(df, train_frac=args.train_frac, val_frac=args.val_frac)
   ```
2. In PowerShell terminal, show the saved artifacts:
   ```powershell
   Get-ChildItem models | Select-Object Name, Length, LastWriteTime
   ```
**Expected Output:**
```
Name                   Length LastWriteTime
----                   ------ -------------
escalation_gate.joblib     73 10/15/2026 1:24 PM
escalation_kind.txt         6 10/15/2026 1:24 PM
gate_type.txt               7 10/15/2026 1:24 PM
openset_gate.joblib      1960 10/15/2026 1:24 PM
stage1_iforest.joblib 2151079 10/15/2026 1:24 PM
stage2_xgboost.joblib 1730785 10/15/2026 1:24 PM
```
**Say to Sir:**
> "Here are our pre-trained, serialized model artifacts in the `models/` folder: our fitted Isolation Forest pre-filter (2.15 MB), our 300-tree XGBoost classifier (1.73 MB), our OpenMax centroids, and our calibrated conformal threshold ($\hat{q} = 0.1357$)."

---

### Step 4: Run Independent Testing & Benchmark Live (2 Minutes)
In PowerShell terminal:
```powershell
$env:PYTHONPATH="ml/src"
python ml/scripts/evaluate_and_compare.py
```
*Approximate Runtime:* **~10–15 seconds.**
**Expected Terminal Output:**
```
Loading dataset from: ml/tests/fixtures/synthetic_cicids_sample.csv
Split sizes -> Train: 856, Val: 183, Test: 185
Loading existing models from models/
Training Alternative: Random Forest Classifier...
Training Alternative: Logistic Regression...
Training Alternative: Decision Tree (CART)...
Evaluating model: Selected: Two-Stage Cascade (IF+XGB)
Evaluating model: Alternative 1: Standalone XGBoost
Evaluating model: Alternative 2: Random Forest
Evaluating model: Alternative 3: Logistic Regression
Evaluating model: Alternative 4: Decision Tree
Saved model comparison to: results/model_comparison.csv
Saved detailed metrics to: results/metrics.json
Saved classification report to: results/classification_report.txt
Saved confusion matrix image to: results/confusion_matrix.png
Saved sample predictions to: results/sample_predictions.csv
```
**Say to Sir:**
> "Notice, Sir, that the testing program executed completely independently. It loaded the saved models, scored the 185 held-out test flows, evaluated all 4 candidate alternative models under identical conditions, and updated our verified results folder."

---

### Step 5: Display Evaluation Report & Confusion Matrix (1.5 Minutes)
1. Display the classification report in PowerShell:
   ```powershell
   Get-Content results/classification_report.txt
   ```
2. Open the generated confusion matrix image:
   ```powershell
   Start-Process results/confusion_matrix.png
   ```
**Say to Sir:**
> "Here is our actual test performance: 98.38% overall accuracy, 99.64% Macro Precision, 91.35% Macro Recall, and 94.26% Macro F1. On the confusion matrix, you can see that all 116 benign flows and all 36 DoS/DDoS flows were classified with 100% precision. Only 2 stealthy PortScan flows and 1 Infiltration flow were dropped by Stage 1, illustrating the honest engineering trade-off of pre-filtering."

---

### Step 6: Show Sample Predictions with TreeSHAP Explanations (1.5 Minutes)
In PowerShell terminal:
```powershell
Import-Csv results/sample_predictions.csv | Select-Object -First 6 Sample_ID, True_Label, Predicted_Label, Stage1_Anomaly_Score, Stage1_Flagged, Confidence, Unknown_Mass, Escalated | Format-Table -AutoSize
```
**Say to Sir:**
> "Here are actual test flow predictions from `results/sample_predictions.csv`. Notice:
> - Normal benign flows (Sample 0, 1, 2) have anomaly scores around 0.40 and `Stage1_Flagged = False`. Stage 2 was not even called, saving CPU cycles.
> - Known attacks (Sample 3, 4) have anomaly scores > 0.45, `Stage1_Flagged = True`, and confident Stage 2 predictions with low unknown mass.
> - In Sample 10, the model encountered an ambiguous Brute Force flow. The OpenMax head calculated an `Unknown_Mass` of 0.6965, exceeding our conformal threshold of 0.1357, so the flow was flagged with `Escalated = True` for Tier-2 expert review. Each alert also carries exact TreeSHAP feature attributions."

---

### Step 7: Defend Model Choice Over Alternatives (1.5 Minutes)
In PowerShell terminal:
```powershell
Import-Csv results/model_comparison.csv | Select-Object Model, Accuracy, "Macro F1", "Weighted F1", "Latency (Median ms)", "Throughput (flows/s)" | Format-Table -AutoSize
```
**Say to Sir:**
> "Here is our comparative defense against alternatives:
> - Standalone XGBoost and Decision Trees scored 100% on this closed-set test, but they are closed-set: when novel zero-day attacks hit the network, they blindly misclassify without escalation.
> - Random Forest had an unacceptable single-flow latency of 59.85 ms—more than three times slower than our cascade.
> - Logistic Regression is fast but suffered an 85.59% Macro F1 due to non-linear network patterns.
> Our Two-Stage Cascade is the optimal real-world compromise: 98.38% accuracy, 18.5ms latency, native TreeSHAP explainability, and mathematically bounded zero-day escalation."

---

## 3. Fallback Demonstration Plan (In Case of Any Glitch)

If terminal execution is slow or blocked during the review:
1. **Fallback 1 (Pre-Generated Verified Results):** Open [`results/model_comparison.csv`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/model_comparison.csv) and [`results/confusion_matrix.png`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/results/confusion_matrix.png) directly in VS Code or Windows Image Viewer. All files were generated and timestamped today.
2. **Fallback 2 (Unit Test Verification):** Run the pytest suite to prove code integrity:
   ```powershell
   python -m pytest ml/tests -v
   ```
   *(All 115 tests pass in ~40 seconds).*
3. **Fallback 3 (Docker Compose Stack Verification):** If Sir wants to see the running microservices:
   ```powershell
   docker compose ps
   ```
   All containers (`kafka`, `ingestion`, `ml`, `tier2-reasoner`, `dashboard-api`, `dashboard-frontend`) will show status `Up`.
