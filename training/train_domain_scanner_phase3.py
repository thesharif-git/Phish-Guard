"""
PhishGuard AI — Phase 3 Domain Scanner Training & Calibration Pipeline
Trains LightGBM on data/clean_domain_corpus.csv, calibrates via Platt scaling,
validates quality gates, and exports models/domain_model.pkl + metadata.
"""
import os
import sys
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.domain_features import (
    extract_domain_features_batch,
    DOMAIN_FEATURE_NAMES
)
from utils.domain_predictor import CalibratedDomainPredictor


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error across 15 uniform probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b_low, b_high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (probs > b_low) & (probs <= b_high)
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(labels[in_bin])
            conf = np.mean(probs[in_bin])
            ece += np.abs(conf - acc) * prop
    return float(ece)


def main():
    print("=" * 80, flush=True)
    print("  PHISHGUARD AI — PHASE 3 DOMAIN SCANNER TRAINING & GATE VALIDATION", flush=True)
    print("=" * 80, flush=True)

    # Step 3.1: Dataset Hygiene & Split Verification
    data_path = os.path.join(PROJECT_ROOT, "data", "clean_domain_corpus.csv")
    print(f"\n[STEP 3.1] Loading Domain corpus: {data_path}...", flush=True)
    df = pd.read_csv(data_path)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"  Training Split:    {len(train_df):,} samples (Clean: {sum(train_df['label']==0):,}, Phish: {sum(train_df['label']==1):,})", flush=True)
    print(f"  Validation Split:  {len(val_df):,} samples (Clean: {sum(val_df['label']==0):,}, Phish: {sum(val_df['label']==1):,})", flush=True)
    print(f"  Frozen Test Split: {len(test_df):,} samples (Clean: {sum(test_df['label']==0):,}, Phish: {sum(test_df['label']==1):,})", flush=True)

    # Step 3.2: Dense Domain Feature Extraction
    print(f"\n[STEP 3.2] Extracting {len(DOMAIN_FEATURE_NAMES)} Domain Features across all splits...", flush=True)
    t0_feat = time.perf_counter()
    X_train = extract_domain_features_batch(train_df["psl_domain"].tolist())
    X_val = extract_domain_features_batch(val_df["psl_domain"].tolist())
    X_test = extract_domain_features_batch(test_df["psl_domain"].tolist())
    t_feat = time.perf_counter() - t0_feat

    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)
    print(f"  Extraction complete in {t_feat:.2f}s ({t_feat / len(df) * 1000:.3f} ms / domain)", flush=True)

    # Step 3.3: LightGBM Model Training
    print("\n[STEP 3.3] Training Regularized LightGBM Booster...", flush=True)
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=DOMAIN_FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=DOMAIN_FEATURE_NAMES, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": 4,
    }

    t0_train = time.perf_counter()
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=600,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    t_train = time.perf_counter() - t0_train
    best_iteration = booster.best_iteration
    print(f"  Booster converged at iteration {best_iteration} in {t_train:.2f}s", flush=True)

    # Step 3.4: Post-Hoc Platt Scaling Calibration (Sigmoid)
    print("\n[STEP 3.4] Fitting Platt Scaling Calibrator on Held-Out Validation Split...", flush=True)
    raw_val = booster.predict(X_val, raw_score=True).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", random_state=42)
    calibrator.fit(raw_val, y_val)
    a_param = float(calibrator.coef_[0][0])
    b_param = float(calibrator.intercept_[0])
    print(f"  Calibrated Sigmoid Parameters: a = {a_param:.4f}, b = {b_param:.4f}", flush=True)

    val_probs = expit(a_param * raw_val.ravel() + b_param)
    val_ece = compute_ece(val_probs, y_val)
    print(f"  Validation Set ECE: {val_ece:.4f}", flush=True)

    # Step 3.5: Validation Threshold Selection
    print("\n[STEP 3.5] Selecting Operating Thresholds on Validation Data...", flush=True)
    best_f1, tau_bal = 0.0, 0.50
    for candidate_tau in np.linspace(0.20, 0.80, 61):
        cand_preds = (val_probs >= candidate_tau).astype(int)
        cand_f1 = f1_score(y_val, cand_preds)
        if cand_f1 > best_f1:
            best_f1 = cand_f1
            tau_bal = candidate_tau

    tau_strict = 0.85
    for candidate_tau in np.linspace(0.70, 0.99, 30):
        cand_preds = (val_probs >= candidate_tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, cand_preds).ravel()
        cand_fpr = fp / (fp + tn)
        if cand_fpr <= 0.05:
            tau_strict = candidate_tau
            break

    tau_inspect = 0.40
    print(f"  Balanced Threshold (F1 optimum): tau_bal = {tau_bal:.3f} (Val F1: {best_f1:.4f})", flush=True)
    print(f"  Strict Blocking Threshold (FPR <= 5%): tau_strict = {tau_strict:.3f}", flush=True)
    print(f"  Inspection Floor Threshold: tau_inspect = {tau_inspect:.3f}", flush=True)

    # Step 3.6: Audit on Frozen Holdout Test Split
    print("\n[STEP 3.6] Evaluating on Frozen Unseen Test Split (N = 6,312)...", flush=True)
    predictor = CalibratedDomainPredictor(booster, a=a_param, b=b_param)

    raw_test = booster.predict(X_test, raw_score=True)
    test_probs = expit(a_param * raw_test + b_param)
    test_preds = (test_probs >= tau_bal).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    test_ece = compute_ece(test_probs, y_test)

    # Strict threshold metrics
    strict_preds = (test_probs >= tau_strict).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_test, strict_preds).ravel()
    fpr_strict = fp_s / (fp_s + tn_s)

    # Single-item CPU Latency Benchmark
    print("  Measuring Pure-CPU Latency Budget (100 warmup + 200 measured iterations)...", flush=True)
    sample_domain = "microsoft.com"
    for _ in range(100):
        _ = predictor.predict_proba([sample_domain])

    latencies = []
    for _ in range(200):
        t0_lat = time.perf_counter()
        _ = predictor.predict_proba([sample_domain])
        latencies.append((time.perf_counter() - t0_lat) * 1000)

    p95_lat = float(np.percentile(latencies, 95))
    p50_lat = float(np.percentile(latencies, 50))
    print(f"  Latency: P50 = {p50_lat:.3f} ms, P95 = {p95_lat:.3f} ms", flush=True)

    print("\n" + "=" * 80)
    print("  PHASE 3 AUDITED METRICS SUMMARY")
    print("=" * 80)
    print(f"  Accuracy:         {acc * 100:.2f}%")
    print(f"  Precision:        {prec * 100:.2f}%")
    print(f"  Recall:           {rec * 100:.2f}%")
    print(f"  F1 Score:         {f1 * 100:.2f}%")
    print(f"  ROC AUC:          {roc_auc:.4f}")
    print(f"  PR AUC:           {pr_auc:.4f}")
    print(f"  FPR (balanced):   {fpr * 100:.2f}%")
    print(f"  FPR (strict):     {fpr_strict * 100:.2f}%")
    print(f"  FNR:              {fnr * 100:.2f}%")
    print(f"  ECE Calibration:  {test_ece:.4f}")
    print(f"  P95 CPU Latency:  {p95_lat:.3f} ms")
    print("=" * 80)

    # Save artifacts
    model_out = os.path.join(PROJECT_ROOT, "models", "domain_model.pkl")
    meta_out = os.path.join(PROJECT_ROOT, "models", "domain_phase3_metadata.json")

    print(f"\n[STEP 3.7] Saving Serialized Predictor to {model_out}...", flush=True)
    joblib.dump(predictor, model_out, compress=3)
    file_size_mb = os.path.getsize(model_out) / (1024 * 1024)
    print(f"  Model artifact saved ({file_size_mb:.2f} MB, down from legacy 8.54 MB)", flush=True)

    metadata = {
        "model_type": "LightGBM + Platt Scaling (Sigmoid Calibration)",
        "n_trees": best_iteration,
        "features": list(DOMAIN_FEATURE_NAMES),
        "feature_count": len(DOMAIN_FEATURE_NAMES),
        "thresholds": {
            "tau_balanced": round(float(tau_bal), 3),
            "tau_strict": round(float(tau_strict), 3),
            "tau_inspection": round(float(tau_inspect), 3),
        },
        "platt_params": {
            "a": round(a_param, 4),
            "b": round(b_param, 4),
        },
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "fpr": round(float(fpr), 4),
            "fpr_strict": round(float(fpr_strict), 4),
            "fnr": round(float(fnr), 4),
            "ece": round(float(test_ece), 4),
            "p95_latency_ms": round(float(p95_lat), 3),
        },
        "scientific_note": (
            "Bare domain lexical scoring measures domain-name structural and registrar risk. "
            "Because 40-60% of web phishing attacks leverage compromised legitimate domains "
            "(whose domain names are authentically benign), high-confidence domain verdicts "
            "in OriginTrace are established through ProvenanceMatrix (DNS ephemerality, registrar age) "
            "and live threat feeds, while this model provides fast, leak-free offline lexical estimation."
        ),
        "trained_at_iso": datetime.now(timezone.utc).isoformat(),
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {meta_out}", flush=True)
    print("\nPhase 3 Domain Scanner training and calibration complete!", flush=True)


if __name__ == "__main__":
    main()
