"""Comprehensive evaluation and alternative model comparison script.

Conducts a fair, rigorous benchmark on the held-out time-based test set:
1. Selected Model: Two-Stage Cascade (Isolation Forest pre-filter + XGBoost classifier + OpenMax Gate)
2. Alternative 1: Random Forest Classifier (Ensemble of bagged trees)
3. Alternative 2: Logistic Regression (L2 regularized with StandardScaler)
4. Alternative 3: Standalone XGBoost (Single-stage gradient boosting)
5. Alternative 4: Decision Tree (Single CART tree baseline)

Generates:
- results/model_comparison.csv
- results/metrics.json
- results/classification_report.txt
- results/confusion_matrix.png
- results/sample_predictions.csv
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Ensure ml/src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.data import load_and_map
from ids_ml.explainability import ShapExplainer
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.pipeline import TwoStageDetector
from ids_ml.serve import load_gate
from ids_ml.split import time_based_split
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_and_compare")


def measure_latency(predict_fn, X: np.ndarray, warmup: int = 50, n_trials: int = 200) -> dict:
    """Measure single-flow inference latency with warmup."""
    n = len(X)
    for i in range(warmup):
        predict_fn(X[i % n : i % n + 1])

    times_ms = np.empty(n_trials)
    for i in range(n_trials):
        idx = (warmup + i) % n
        t0 = time.perf_counter()
        predict_fn(X[idx : idx + 1])
        t1 = time.perf_counter()
        times_ms[i] = (t1 - t0) * 1000.0

    median_ms = float(np.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    p99_ms = float(np.percentile(times_ms, 99))
    throughput = float(1000.0 / median_ms) if median_ms > 0 else 0.0

    return {
        "median_ms": round(median_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "p99_ms": round(p99_ms, 4),
        "throughput_fps": round(throughput, 1),
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "ml" / "tests" / "fixtures" / "synthetic_cicids_sample.csv"
    model_dir = repo_root / "models"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from: %s", data_path)
    df = load_and_map([str(data_path)])
    logger.info("Total rows loaded: %d", len(df))

    # Time-based split: 70% train, 15% val, 15% test
    train_df, val_df, test_df = time_based_split(df, train_frac=0.7, val_frac=0.15)
    logger.info("Split sizes -> Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    X_train = train_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_train = train_df["attack_category"].tolist()
    X_val = val_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_val = val_df["attack_category"].tolist()
    X_test = test_df[CANONICAL_FEATURE_COLUMNS].to_numpy()
    y_test = test_df["attack_category"].tolist()

    all_labels = sorted(list(set(y_train) | set(y_test)))
    logger.info("Evaluation labels: %s", all_labels)

    # 1. Load or train Selected Two-Stage Model
    if (model_dir / "stage1_iforest.joblib").exists() and (model_dir / "stage2_xgboost.joblib").exists():
        logger.info("Loading existing models from %s", model_dir)
        stage1 = AnomalyPreFilter.load(model_dir / "stage1_iforest.joblib")
        stage2 = AttackClassifier.load(model_dir / "stage2_xgboost.joblib")
    else:
        logger.info("Fitting Stage 1 and Stage 2 models...")
        stage1 = AnomalyPreFilter(Stage1Config(contamination=0.35)).fit(X_train)
        stage2 = AttackClassifier(Stage2Config()).fit(X_train, y_train)

    gate, escalation_gate, gate_type = load_gate(model_dir, stage2)
    cascade_detector = TwoStageDetector(
        stage1=stage1,
        stage2=stage2,
        gate=gate,
        escalation_gate=escalation_gate,
        escalation_trigger_name=gate_type,
    )

    # Measure Selected Model training time
    t0 = time.perf_counter()
    _s1_tmp = AnomalyPreFilter(Stage1Config(contamination=0.35)).fit(X_train)
    _s2_tmp = AttackClassifier(Stage2Config()).fit(X_train, y_train)
    selected_train_time = time.perf_counter() - t0

    # Model sizes
    selected_size_kb = (
        (model_dir / "stage1_iforest.joblib").stat().st_size
        + (model_dir / "stage2_xgboost.joblib").stat().st_size
    ) / 1024.0

    # 2. Alternative Models Definition
    models_dict = {}

    # Model 1: Selected Cascade Detector
    def cascade_predict(X):
        results = cascade_detector.score(X)
        return np.array([r.stage2_predicted_class for r in results])

    models_dict["Selected: Two-Stage Cascade (IF+XGB)"] = {
        "train_fn": None,
        "train_time": selected_train_time,
        "predict_fn": cascade_predict,
        "size_kb": selected_size_kb,
        "interpretability": "High (TreeSHAP per-flow attribution + Stage 1 anomaly score)",
        "strengths": "Fast anomaly pre-filtering; bounds costly classifier calls; native open-set escalation & TreeSHAP",
        "limitations": "Cascade attack recall bounded by Stage 1 recall on rare zero-day anomalies",
    }

    # Model 2: Standalone XGBoost (Single stage)
    t0 = time.perf_counter()
    xgb_standalone = AttackClassifier(Stage2Config()).fit(X_train, y_train)
    xgb_standalone_time = time.perf_counter() - t0
    tmp_path = results_dir / "temp_xgb.joblib"
    joblib.dump(xgb_standalone, tmp_path)
    xgb_size_kb = tmp_path.stat().st_size / 1024.0
    tmp_path.unlink()

    models_dict["Alternative 1: Standalone XGBoost"] = {
        "train_fn": None,
        "train_time": xgb_standalone_time,
        "predict_fn": lambda X: xgb_standalone.predict(X),
        "size_kb": xgb_size_kb,
        "interpretability": "High (TreeSHAP feature attributions available)",
        "strengths": "Exceptional gradient-boosted decision tree accuracy on non-linear tabular features",
        "limitations": "Heavier computation per packet (no pre-filter); closed-set prediction without gate",
    }

    # Model 3: Random Forest Classifier
    logger.info("Training Alternative: Random Forest Classifier...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - t0
    tmp_path = results_dir / "temp_rf.joblib"
    joblib.dump(rf, tmp_path)
    rf_size_kb = tmp_path.stat().st_size / 1024.0
    tmp_path.unlink()

    models_dict["Alternative 2: Random Forest"] = {
        "train_fn": None,
        "train_time": rf_train_time,
        "predict_fn": lambda X: rf.predict(X),
        "size_kb": rf_size_kb,
        "interpretability": "Moderate (Global Gini feature importances, slower per-instance SHAP)",
        "strengths": "Robust ensemble averaging, low variance, resistant to tabular overfitting",
        "limitations": "Larger memory footprint, slower single-instance inference than boosted trees",
    }

    # Model 4: Logistic Regression (with StandardScaler)
    logger.info("Training Alternative: Logistic Regression...")
    t0 = time.perf_counter()
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=0, C=1.0)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_train_time = time.perf_counter() - t0
    tmp_path = results_dir / "temp_lr.joblib"
    joblib.dump(lr_pipe, tmp_path)
    lr_size_kb = tmp_path.stat().st_size / 1024.0
    tmp_path.unlink()

    models_dict["Alternative 3: Logistic Regression"] = {
        "train_fn": None,
        "train_time": lr_train_time,
        "predict_fn": lambda X: lr_pipe.predict(X),
        "size_kb": lr_size_kb,
        "interpretability": "Very High (Linear coefficients per feature per class)",
        "strengths": "Extremely fast training and low prediction latency, tiny model footprint",
        "limitations": "Linear boundary assumption fails on complex multi-modal and non-linear attack distributions",
    }

    # Model 5: Decision Tree Classifier
    logger.info("Training Alternative: Decision Tree (CART)...")
    t0 = time.perf_counter()
    dt = DecisionTreeClassifier(max_depth=8, random_state=0)
    dt.fit(X_train, y_train)
    dt_train_time = time.perf_counter() - t0
    tmp_path = results_dir / "temp_dt.joblib"
    joblib.dump(dt, tmp_path)
    dt_size_kb = tmp_path.stat().st_size / 1024.0
    tmp_path.unlink()

    models_dict["Alternative 4: Decision Tree"] = {
        "train_fn": None,
        "train_time": dt_train_time,
        "predict_fn": lambda X: dt.predict(X),
        "size_kb": dt_size_kb,
        "interpretability": "High (Direct IF-THEN tree visualization)",
        "strengths": "Ultra-low latency, tiny footprint, completely transparent decision paths",
        "limitations": "Prone to high variance, brittle to small perturbations, lower generalizability",
    }

    # Benchmark Execution
    comparison_rows = []
    detailed_metrics = {}

    for name, meta in models_dict.items():
        logger.info("Evaluating model: %s", name)
        predict_fn = meta["predict_fn"]

        # Latency
        lat = measure_latency(predict_fn, X_test)

        # Predictions on held-out test split
        y_pred = predict_fn(X_test)

        # Overall Metrics
        acc = accuracy_score(y_test, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_test, y_pred, labels=all_labels, zero_division=0
        )
        class_breakdown = {}
        for idx, lbl in enumerate(all_labels):
            class_breakdown[lbl] = {
                "precision": round(float(p_class[idx]), 4),
                "recall": round(float(r_class[idx]), 4),
                "f1_score": round(float(f1_class[idx]), 4),
                "support": int(support[idx]),
            }

        comparison_rows.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Macro Precision": f"{p_macro:.4f}",
            "Macro Recall": f"{r_macro:.4f}",
            "Macro F1": f"{f1_macro:.4f}",
            "Weighted F1": f"{f1_weighted:.4f}",
            "Training Time (s)": f"{meta['train_time']:.3f}s",
            "Latency (Median ms)": f"{lat['median_ms']:.3f} ms",
            "Throughput (flows/s)": f"{lat['throughput_fps']:.1f}",
            "Model Size (KB)": f"{meta['size_kb']:.1f} KB",
            "Interpretability": meta["interpretability"],
            "Key Strengths": meta["strengths"],
            "Key Limitations": meta["limitations"],
        })

        detailed_metrics[name] = {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(p_weighted), 4),
            "weighted_recall": round(float(r_weighted), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "training_time_seconds": round(float(meta["train_time"]), 4),
            "latency": lat,
            "model_size_kb": round(float(meta["size_kb"]), 1),
            "per_class": class_breakdown,
        }

    # Save Comparison CSV
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = results_dir / "model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info("Saved model comparison to: %s", comp_csv_path)

    # Save Detailed Metrics JSON
    metrics_json_path = results_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    logger.info("Saved detailed metrics to: %s", metrics_json_path)

    # 3. Selected Model In-Depth Evaluation (Classification Report & Confusion Matrix)
    selected_preds = cascade_predict(X_test)
    cm = confusion_matrix(y_test, selected_preds, labels=all_labels)

    # Classification Report Text
    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append("REAL-TIME INTRUSION DETECTION SYSTEM: TEST EVALUATION REPORT")
    report_lines.append("=" * 75)
    report_lines.append(f"Evaluation Dataset : {data_path.name}")
    report_lines.append(f"Test Split Size    : {len(X_test)} samples (Time-based chronological holdout)")
    report_lines.append(f"Model Architecture : Two-Stage Cascade (Isolation Forest -> XGBoost + OpenMax)")
    report_lines.append("-" * 75)
    report_lines.append(f"{'Category':<16} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    report_lines.append("-" * 75)
    for lbl in all_labels:
        d = detailed_metrics["Selected: Two-Stage Cascade (IF+XGB)"]["per_class"][lbl]
        report_lines.append(
            f"{lbl:<16} {d['precision']:<10.4f} {d['recall']:<10.4f} {d['f1_score']:<10.4f} {d['support']:<8}"
        )
    report_lines.append("-" * 75)
    m = detailed_metrics["Selected: Two-Stage Cascade (IF+XGB)"]
    report_lines.append(f"{'Accuracy':<16} {'':<10} {'':<10} {m['accuracy']:<10.4f} {len(X_test):<8}")
    report_lines.append(f"{'Macro Avg':<16} {m['macro_precision']:<10.4f} {m['macro_recall']:<10.4f} {m['macro_f1']:<10.4f} {len(X_test):<8}")
    report_lines.append(f"{'Weighted Avg':<16} {m['weighted_precision']:<10.4f} {m['weighted_recall']:<10.4f} {m['weighted_f1']:<10.4f} {len(X_test):<8}")
    report_lines.append("=" * 75)

    report_text = "\n".join(report_lines)
    txt_path = results_dir / "classification_report.txt"
    txt_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved classification report to: %s", txt_path)

    # Confusion Matrix PNG
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Two-Stage Cascade NIDS (Held-Out Test Set)", fontsize=13, pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(all_labels))
    plt.xticks(tick_marks, all_labels, rotation=40, ha="right", fontsize=10)
    plt.yticks(tick_marks, all_labels, fontsize=10)
    plt.xlabel("Predicted Attack Category", fontsize=11, labelpad=10)
    plt.ylabel("True Attack Category", fontsize=11, labelpad=10)

    # Add text annotations inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), horizontalalignment="center", verticalalignment="center", color=color, fontweight="bold")

    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix image to: %s", cm_path)

    # 4. Generate Sample Predictions with SHAP explanations
    explainer = ShapExplainer(stage2)
    sample_rows = []
    detector_scores = cascade_detector.score(X_test)

    # Take 15 diverse samples across classes
    sample_indices = []
    labels_found = set()
    for idx, true_lbl in enumerate(y_test):
        if true_lbl not in labels_found or len(sample_indices) < 15:
            sample_indices.append(idx)
            labels_found.add(true_lbl)
        if len(labels_found) == len(all_labels) and len(sample_indices) >= 12:
            break

    for idx in sample_indices:
        feat_vec = X_test[idx : idx + 1]
        score_res = detector_scores[idx]
        true_label = y_test[idx]

        # Top 3 TreeSHAP features if stage 2 was executed
        top_shap_str = "None (Filtered at Stage 1 as Benign)"
        if score_res.stage1_flagged:
            exp_items = explainer.explain(feat_vec[0], score_res.stage2_predicted_class, top_k=3)
            top_shap_str = " | ".join([f"{item.feature}={item.value:.2f} (SHAP:{item.shap_value:+.2f})" for item in exp_items])

        sample_rows.append({
            "Sample_ID": idx,
            "True_Label": true_label,
            "Predicted_Label": score_res.stage2_predicted_class,
            "Correct": (true_label == score_res.stage2_predicted_class),
            "Stage1_Anomaly_Score": round(float(score_res.stage1_anomaly_score), 4),
            "Stage1_Flagged": score_res.stage1_flagged,
            "Confidence": round(float(score_res.stage2_confidence), 4),
            "Unknown_Mass": round(float(score_res.unknown_mass), 4) if score_res.unknown_mass is not None else 0.0,
            "Escalated": score_res.escalated,
            "Top_SHAP_Contributions": top_shap_str,
        })

    sample_df = pd.DataFrame(sample_rows)
    sample_csv_path = results_dir / "sample_predictions.csv"
    sample_df.to_csv(sample_csv_path, index=False)
    logger.info("Saved sample predictions to: %s", sample_csv_path)

    print("\n" + report_text)
    print(f"\nEvaluation and Benchmark Completed Successfully.")
    print(f"Results saved in {results_dir.resolve()}:")
    print(f"  - {comp_csv_path.name}")
    print(f"  - {metrics_json_path.name}")
    print(f"  - {txt_path.name}")
    print(f"  - {cm_path.name}")
    print(f"  - {sample_csv_path.name}\n")


if __name__ == "__main__":
    main()
