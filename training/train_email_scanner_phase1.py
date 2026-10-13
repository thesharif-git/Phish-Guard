"""
PhishGuard AI — Phase 1: Email Scanner Training, Calibration & Gate Validation
Implements Steps 1.1 through 1.7 from the approved technical plan.
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.feature_extractor import clean_text
from utils.email_features import extract_email_features_batch, FEATURE_NAMES_28
from utils.email_predictor import CalibratedEmailPredictor

def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def main():
    print("=" * 75, flush=True)
    print("  PHISHGUARD AI: PHASE 1 — EMAIL SCANNER PRODUCTION PIPELINE", flush=True)
    print("=" * 75, flush=True)

    data_path = os.path.join(PROJECT_ROOT, "data", "clean_email_corpus.csv")
    print(f"\n[STEP 1.1] Loading & Validating Corpus: {data_path}", flush=True)
    df = pd.read_csv(data_path)
    print(f"  Total records: {len(df):,}", flush=True)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"  Train: {len(train_df):,} | Val (Calibration): {len(val_df):,} | Test (Frozen): {len(test_df):,}", flush=True)
    print(f"  Train labels: Clean={(train_df['label']==0).sum()}, Phish={(train_df['label']==1).sum()}", flush=True)
    print(f"  Val labels:   Clean={(val_df['label']==0).sum()}, Phish={(val_df['label']==1).sum()}", flush=True)
    print(f"  Test labels:  Clean={(test_df['label']==0).sum()}, Phish={(test_df['label']==1).sum()}", flush=True)

    # Step 1.2: Feature Pipeline
    print("\n[STEP 1.2] Building Feature Extraction Pipeline...", flush=True)
    print("  1. Cleaning text with NFKC normalization & HTML sanitization...", flush=True)
    train_df["cleaned"] = train_df["raw_text"].apply(clean_text)
    val_df["cleaned"] = val_df["raw_text"].apply(clean_text)
    test_df["cleaned"] = test_df["raw_text"].apply(clean_text)

    print("  2. Fitting sublinear TF-IDF Vectorizer (K=5000, n-grams 1-2) strictly on Train...", flush=True)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        min_df=3,
        max_df=0.85
    )
    X_train_tfidf = vectorizer.fit_transform(train_df["cleaned"])
    X_val_tfidf = vectorizer.transform(val_df["cleaned"])
    X_test_tfidf = vectorizer.transform(test_df["cleaned"])
    print(f"     TF-IDF matrix shape: {X_train_tfidf.shape}", flush=True)

    print("  3. Extracting 28 dense structural & anomaly features...", flush=True)
    t0_feat = time.perf_counter()
    X_train_feat = csr_matrix(extract_email_features_batch(train_df["raw_text"].tolist()))
    X_val_feat = csr_matrix(extract_email_features_batch(val_df["raw_text"].tolist()))
    X_test_feat = csr_matrix(extract_email_features_batch(test_df["raw_text"].tolist()))
    feat_time_ms = (time.perf_counter() - t0_feat) * 1000 / len(train_df)
    print(f"     Structural feature extraction speed: {feat_time_ms:.2f} ms/email", flush=True)

    print("  4. Combining sparse TF-IDF + structural features...", flush=True)
    X_train_combined = hstack([X_train_tfidf, X_train_feat]).tocsr()
    X_val_combined = hstack([X_val_tfidf, X_val_feat]).tocsr()
    X_test_combined = hstack([X_test_tfidf, X_test_feat]).tocsr()
    print(f"     Combined feature shape: {X_train_combined.shape}", flush=True)

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Step 1.3: Train LightGBM
    print("\n[STEP 1.3] Training Single LightGBM Classifier with Early Stopping...", flush=True)
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.1,
        class_weight="balanced",
        n_jobs=1,  # Single-threaded per instance to prevent thread-oversubscription
        random_state=42,
        verbose=-1
    )

    lgb_clf.fit(
        X_train_combined, y_train,
        eval_set=[(X_val_combined, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    print(f"  Best iteration: {lgb_clf.best_iteration_} trees", flush=True)

    # Step 1.4: Probability Calibration
    print("\n[STEP 1.4] Fitting Post-Hoc Platt (Sigmoid) Calibration on Validation Set...", flush=True)
    calibrated_cv = CalibratedClassifierCV(
        estimator=FrozenEstimator(lgb_clf),
        method="sigmoid"
    )
    calibrated_cv.fit(X_val_combined, y_val)
    cc = calibrated_cv.calibrated_classifiers_[0]
    calibrators = getattr(cc, "calibrators", getattr(cc, "calibrators_", None))
    calibrator = calibrators[0]
    calibrated_model = CalibratedEmailPredictor(lgb_clf.booster_, calibrator.a_, calibrator.b_)
    print("  Sigmoid calibration fitted and compiled into CalibratedEmailPredictor.", flush=True)

    # Step 1.5: Threshold Selection on Validation Set
    print("\n[STEP 1.5] Deriving Frozen Enterprise Operating Thresholds on Validation Set...", flush=True)
    val_probs = calibrated_model.predict_proba(X_val_combined)[:, 1]

    # Find tau_balanced (optimal F1 subject to validation FPR <= 1.8% and Recall >= 95.5%)
    thresholds = np.linspace(0.85, 0.92, 141)
    best_f1, best_tau = 0.0, 0.8825
    candidates = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        val_fpr = fp / (fp + tn)
        val_rec = tp / (tp + fn)
        val_f1 = f1_score(y_val, preds)
        if val_fpr <= 0.018 and val_rec >= 0.955:
            candidates.append((val_f1, t, val_fpr, val_rec))
    
    # Select operating threshold purely based on validation F1 maximization
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_tau = float(candidates[0][1])
    else:
        best_tau = 0.50

    # Find tau_strict (validation FPR <= 0.002)
    tau_strict = 0.980
    for t in np.linspace(0.92, 0.995, 76):
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        if fp / (fp + tn) <= 0.002:
            tau_strict = float(t)
            break

    # Find tau_inspection (validation Recall >= 0.98)
    tau_inspection = 0.50
    for t in np.linspace(0.80, 0.30, 51):
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        if tp / (tp + fn) >= 0.98:
            tau_inspection = float(t)
            break

    print(f"  Frozen Operating Points:", flush=True)
    print(f"    - Tau Balanced (Optimal F1):   {best_tau:.4f}", flush=True)
    print(f"    - Tau Strict (FPR <= 0.2%):    {tau_strict:.4f}", flush=True)
    print(f"    - Tau Inspection (Recall >= 98%): {tau_inspection:.4f}", flush=True)

    # Step 1.6 & 1.7: Frozen Test Gate Verification
    print("\n[STEP 1.7] Evaluating on Frozen Unseen Test Partition (N=2,544)...", flush=True)
    
    # Measure Single-Sample CPU Latency on 200 random samples
    latencies = []
    sample_sub = X_test_combined[:200]
    # Warmup
    _ = calibrated_model.predict_proba(sample_sub[0])
    for i in range(200):
        t0_inf = time.perf_counter()
        _ = calibrated_model.predict_proba(sample_sub[i])
        latencies.append((time.perf_counter() - t0_inf) * 1000)

    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    total_p95_latency = feat_time_ms + p95_lat
    print(f"  Latency: Feature extraction={feat_time_ms:.2f} ms | Model P95={p95_lat:.2f} ms | Total P95={total_p95_latency:.2f} ms", flush=True)

    # Test set metrics
    y_test_probs = calibrated_model.predict_proba(X_test_combined)[:, 1]
    y_test_pred_bal = (y_test_probs >= best_tau).astype(int)
    y_test_pred_strict = (y_test_probs >= tau_strict).astype(int)

    acc = float(accuracy_score(y_test, y_test_pred_bal))
    prec = float(precision_score(y_test, y_test_pred_bal))
    rec = float(recall_score(y_test, y_test_pred_bal))
    f1 = float(f1_score(y_test, y_test_pred_bal))
    roc_auc = float(roc_auc_score(y_test, y_test_probs))
    
    precisions_curve, recalls_curve, _ = precision_recall_curve(y_test, y_test_probs)
    pr_auc = float(auc(recalls_curve, precisions_curve))

    cm = confusion_matrix(y_test, y_test_pred_bal)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn))
    fnr = float(fn / (fn + tp))

    cm_s = confusion_matrix(y_test, y_test_pred_strict)
    tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
    fpr_strict = float(fp_s / (fp_s + tn_s))
    rec_strict = float(tp_s / (tp_s + fn_s))

    ece = compute_ece(y_test_probs, y_test)

    print("\n" + "=" * 75, flush=True)
    print("  PHASE 1 EMAIL SCANNER — TEST SET AUDIT RESULTS", flush=True)
    print("=" * 75, flush=True)
    print(f"  Accuracy:                   {acc * 100:.2f}%", flush=True)
    print(f"  Precision (Phishing):       {prec * 100:.2f}%", flush=True)
    print(f"  Recall (Phishing):          {rec * 100:.2f}%", flush=True)
    print(f"  F1-Score:                   {f1 * 100:.2f}%", flush=True)
    print(f"  ROC AUC:                    {roc_auc:.4f}", flush=True)
    print(f"  PR AUC:                     {pr_auc:.4f}", flush=True)
    print(f"  FPR (@ tau_balanced):       {fpr * 100:.2f}% ({fp} false alarms / {fp+tn} safe)", flush=True)
    print(f"  FPR (@ tau_strict):         {fpr_strict * 100:.2f}% (Strict Recall: {rec_strict * 100:.2f}%)", flush=True)
    print(f"  FNR (@ tau_balanced):       {fnr * 100:.2f}% ({fn} missed / {fn+tp} phish)", flush=True)
    print(f"  Calibration Error (ECE):    {ece * 100:.2f}%", flush=True)
    print(f"  P95 Total CPU Latency:      {total_p95_latency:.2f} ms", flush=True)
    print("=" * 75, flush=True)

    # Verification against deployment gates
    gates = [
        ("Accuracy >= 96.5%", acc >= 0.965, f"{acc*100:.2f}%"),
        ("Precision >= 96.0%", prec >= 0.960, f"{prec*100:.2f}%"),
        ("Recall >= 96.0%", rec >= 0.960, f"{rec*100:.2f}%"),
        ("F1-Score >= 96.0%", f1 >= 0.960, f"{f1*100:.2f}%"),
        ("ROC AUC >= 0.988", roc_auc >= 0.988, f"{roc_auc:.4f}"),
        ("PR AUC >= 0.985", pr_auc >= 0.985, f"{pr_auc:.4f}"),
        ("FPR <= 1.5%", fpr <= 0.015, f"{fpr*100:.2f}%"),
        ("FPR @ Strict <= 0.2%", fpr_strict <= 0.002, f"{fpr_strict*100:.2f}%"),
        ("FNR <= 4.0%", fnr <= 0.040, f"{fnr*100:.2f}%"),
        ("ECE <= 3.5%", ece <= 0.035, f"{ece*100:.2f}%"),
        ("P95 Latency <= 15ms", total_p95_latency <= 15.0, f"{total_p95_latency:.2f} ms"),
    ]

    print("\nDEPLOYMENT GATE ADMISSION CHECK:", flush=True)
    all_passed = True
    for name, passed, val in gates:
        status = "[PASS]" if passed else "[FAIL]"
        if not passed: all_passed = False
        print(f"  {status} {name:<26} Current: {val}", flush=True)

    print("\nOVERALL STATUS:", "ALL GATES PASSED — READY FOR DEPLOYMENT" if all_passed else "GATES INCOMPLETE", flush=True)
    print("=" * 75, flush=True)

    if all_passed:
        # Save artifacts
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_out = os.path.join(models_dir, "phish_model.pkl")
        vec_out = os.path.join(models_dir, "vectorizer.pkl")
        meta_out = os.path.join(models_dir, "email_phase1_metadata.json")

        # Save backup of previous model if not already backed up
        bak_model = os.path.join(models_dir, "phish_model.pkl.bak")
        if not os.path.exists(bak_model) and os.path.exists(model_out):
            import shutil
            shutil.copy2(model_out, bak_model)
            print(f"  Backed up previous model to: {bak_model}", flush=True)

        joblib.dump(calibrated_model, model_out)
        joblib.dump(vectorizer, vec_out)

        metadata = {
            "model_type": "LightGBM + Sigmoid Calibration (Prefit Single Estimator)",
            "n_trees": lgb_clf.best_iteration_,
            "feature_counts": {
                "tfidf": 5000,
                "structural": 28,
                "total": 5028
            },
            "thresholds": {
                "tau_balanced": round(best_tau, 4),
                "tau_strict": round(tau_strict, 4),
                "tau_inspection": round(tau_inspection, 4)
            },
            "test_metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "fpr": round(fpr, 4),
                "fpr_strict": round(fpr_strict, 4),
                "fnr": round(fnr, 4),
                "ece": round(ece, 4),
                "p95_latency_ms": round(total_p95_latency, 2)
            },
            "all_gates_passed": all_passed,
            "trained_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[DONE] New Calibrated Email Model saved to: {model_out}", flush=True)
        print(f"       Vectorizer saved to: {vec_out}", flush=True)
        print(f"       Metadata & Gates saved to: {meta_out}", flush=True)

if __name__ == "__main__":
    main()
