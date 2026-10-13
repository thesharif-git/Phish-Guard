"""
PhishGuard AI — Production URL Scanner Model Training & Gate Verification (Phase 2)
Full end-to-end training, Platt calibration, threshold optimization, and gate auditing.
Strictly sequential: All 11 gates must pass on the frozen unseen test partition (N=10,132).
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.url_features import (
    extract_url_features_single,
    extract_url_features_batch,
    URL_FEATURE_NAMES
)
from utils.url_predictor import CalibratedUrlPredictor


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
    print("  PHISHGUARD AI — PHASE 2 URL SCANNER TRAINING & GATE VALIDATION", flush=True)
    print("=" * 80, flush=True)

    # Step 2.1: Dataset Hygiene & Split Verification
    data_path = os.path.join(PROJECT_ROOT, "data", "clean_url_corpus.csv")
    print(f"\n[STEP 2.1] Loading URL corpus: {data_path}...", flush=True)
    df = pd.read_csv(data_path)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"  Training Split:    {len(train_df):,} samples (Clean: {sum(train_df['label']==0):,}, Phish: {sum(train_df['label']==1):,})", flush=True)
    print(f"  Validation Split:  {len(val_df):,} samples (Clean: {sum(val_df['label']==0):,}, Phish: {sum(val_df['label']==1):,})", flush=True)
    print(f"  Frozen Test Split: {len(test_df):,} samples (Clean: {sum(test_df['label']==0):,}, Phish: {sum(test_df['label']==1):,})", flush=True)

    # Step 2.2: Feature Pipeline Construction
    print("\n[STEP 2.2] Constructing Dual Representation Pipeline (TF-IDF + Dense)...", flush=True)
    t0_feat = time.perf_counter()

    # Fit TF-IDF char n-grams strictly on train partition
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=2000,
        sublinear_tf=True
    )
    X_tfidf_train = vectorizer.fit_transform(train_df["url"])
    X_tfidf_val = vectorizer.transform(val_df["url"])
    X_tfidf_test = vectorizer.transform(test_df["url"])

    print(f"  TF-IDF Char-Ngrams (K=2000) fit on train split in {(time.perf_counter()-t0_feat):.2f}s", flush=True)

    # Extract 33 dense behavioral, lexical, structural, and obfuscation features
    t0_dense = time.perf_counter()
    X_dense_train = csr_matrix(extract_url_features_batch(train_df["url"].tolist()))
    X_dense_val = csr_matrix(extract_url_features_batch(val_df["url"].tolist()))
    X_dense_test = csr_matrix(extract_url_features_batch(test_df["url"].tolist()))
    print(f"  Dense 33-feature matrix extracted in {(time.perf_counter()-t0_dense):.2f}s", flush=True)

    # Combine into unified float32 sparse matrices
    X_train = hstack([X_tfidf_train, X_dense_train]).tocsr().astype(np.float32)
    X_val = hstack([X_tfidf_val, X_dense_val]).tocsr().astype(np.float32)
    X_test = hstack([X_tfidf_test, X_dense_test]).tocsr().astype(np.float32)

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    total_feat_count = X_train.shape[1]
    print(f"  Combined Feature Space: {total_feat_count} features (2,000 TF-IDF + 33 Dense)", flush=True)

    # Step 2.3: Model Training with Regularization & Early Stopping
    print("\n[STEP 2.3] Training Production LightGBM Classifier...", flush=True)
    t0_train = time.perf_counter()

    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=9,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )

    train_time = time.perf_counter() - t0_train
    print(f"  Model trained in {train_time:.2f}s. Best iteration: {lgb_clf.best_iteration_}", flush=True)

    # Step 2.4: Post-Hoc Platt Scaling Calibration on Held-Out Validation Split
    print("\n[STEP 2.4] Calibrating Probabilities via Platt Scaling (Validation Split)...", flush=True)
    val_raw_preds = lgb_clf.booster_.predict(X_val)

    platt_calibrator = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
    platt_calibrator.fit(val_raw_preds.reshape(-1, 1), y_val)

    # Extract Platt scaling slope (a) and intercept (b): P = sigmoid(-(a * raw + b))
    # In sklearn: P = sigmoid(coef * raw + intercept) = sigmoid(-(-coef * raw - intercept))
    coef = float(platt_calibrator.coef_[0][0])
    intercept = float(platt_calibrator.intercept_[0])
    cal_a = -coef
    cal_b = -intercept

    print(f"  Platt Scaling Parameters: a = {cal_a:.6f}, b = {cal_b:.6f}", flush=True)

    # Assemble Production Calibrated URL Predictor
    calibrated_model = CalibratedUrlPredictor(
        booster=lgb_clf.booster_,
        vectorizer=vectorizer,
        a=cal_a,
        b=cal_b,
        classes=(0, 1)
    )

    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    val_ece = compute_ece(val_probs, y_val)
    print(f"  Validation ECE after calibration: {val_ece * 100:.2f}%", flush=True)

    # Step 2.5: Threshold Selection on Validation Split
    print("\n[STEP 2.5] Algorithmic Threshold Tuning on Held-Out Validation Split...", flush=True)
    thresholds = np.linspace(0.60, 0.98, 191)
    candidates = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        val_fpr = fp / (fp + tn)
        val_rec = tp / (tp + fn)
        val_f1 = f1_score(y_val, preds)
        if val_fpr <= 0.005 and val_rec >= 0.980:
            candidates.append((val_f1, float(t), val_fpr, val_rec))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_tau = float(candidates[0][1])
    else:
        best_tau = 0.85

    # Strict Quarantine Threshold (conservative validation FPR <= 0.0005 to guarantee <= 0.1% generalization)
    tau_strict = 0.992
    for t in np.linspace(0.85, 0.999, 150):
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        if fp / (fp + tn) <= 0.0005:
            tau_strict = float(t)
            break

    # Deep Inspection Threshold (validation Recall >= 0.995)
    tau_inspection = 0.50
    for t in np.linspace(0.80, 0.30, 51):
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        if tp / (tp + fn) >= 0.995:
            tau_inspection = float(t)
            break

    print(f"  Frozen Operating Points:", flush=True)
    print(f"    - Tau Balanced (FPR <= 0.5%, High Recall): {best_tau:.4f}", flush=True)
    print(f"    - Tau Strict (FPR <= 0.1%):                {tau_strict:.4f}", flush=True)
    print(f"    - Tau Inspection (Recall >= 99.5%):        {tau_inspection:.4f}", flush=True)

    # Step 2.6: Latency Benchmarking (P95 CPU Latency on 200 Individual URLs)
    print("\n[STEP 2.6] Measuring Single-URL CPU Inference Latency (N=200)...", flush=True)
    sample_urls = test_df["url"][:200].tolist()
    latencies = []

    # Warmup
    _ = calibrated_model.predict_proba([sample_urls[0]])

    for u in sample_urls:
        t0_inf = time.perf_counter()
        _ = calibrated_model.predict_proba([u])
        latencies.append((time.perf_counter() - t0_inf) * 1000)

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    print(f"  Single-URL CPU Latency: P50 = {p50_latency:.2f} ms | P95 = {p95_latency:.2f} ms", flush=True)

    # Step 2.7: Frozen Test Gate Verification (N=10,132)
    print("\n[STEP 2.7] Evaluating on Frozen Unseen Test Partition (N=10,132)...", flush=True)
    y_test_probs = calibrated_model.predict_proba(X_test)[:, 1]
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

    print("\n" + "=" * 80, flush=True)
    print("  PHASE 2 URL SCANNER — FROZEN TEST SET AUDIT RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"  Accuracy:                   {acc * 100:.2f}%", flush=True)
    print(f"  Precision (Phishing):       {prec * 100:.2f}%", flush=True)
    print(f"  Recall (Phishing):          {rec * 100:.2f}%", flush=True)
    print(f"  F1-Score:                   {f1 * 100:.2f}%", flush=True)
    print(f"  ROC AUC:                    {roc_auc:.4f}", flush=True)
    print(f"  PR AUC:                     {pr_auc:.4f}", flush=True)
    print(f"  FPR (@ tau_balanced):       {fpr * 100:.2f}% ({fp} false alarms / {fp+tn} safe)", flush=True)
    print(f"  FPR (@ tau_strict):         {fpr_strict * 100:.2f}% ({fp_s} false alarms / {fp_s+tn_s} safe, Recall: {rec_strict*100:.2f}%)", flush=True)
    print(f"  FNR (@ tau_balanced):       {fnr * 100:.2f}% ({fn} missed / {fn+tp} phish)", flush=True)
    print(f"  Calibration Error (ECE):    {ece * 100:.2f}%", flush=True)
    print(f"  P95 Total CPU Latency:      {p95_latency:.2f} ms", flush=True)
    print("=" * 80, flush=True)

    # Verification against 11 Deployment Gates
    gates = [
        ("Accuracy >= 97.0%", acc >= 0.970, f"{acc*100:.2f}%"),
        ("Precision >= 97.0%", prec >= 0.970, f"{prec*100:.2f}%"),
        ("Recall >= 96.5%", rec >= 0.965, f"{rec*100:.2f}%"),
        ("F1-Score >= 96.8%", f1 >= 0.968, f"{f1*100:.2f}%"),
        ("ROC AUC >= 0.992", roc_auc >= 0.992, f"{roc_auc:.4f}"),
        ("PR AUC >= 0.990", pr_auc >= 0.990, f"{pr_auc:.4f}"),
        ("FPR <= 1.0%", fpr <= 0.010, f"{fpr*100:.2f}%"),
        ("FPR @ Strict <= 0.1%", fpr_strict <= 0.001, f"{fpr_strict*100:.2f}%"),
        ("FNR <= 3.5%", fnr <= 0.035, f"{fnr*100:.2f}%"),
        ("ECE <= 3.0%", ece <= 0.030, f"{ece*100:.2f}%"),
        ("P95 Latency <= 3.0ms", p95_latency <= 3.0, f"{p95_latency:.2f} ms"),
    ]

    print("\nDEPLOYMENT GATE ADMISSION CHECK:", flush=True)
    all_passed = True
    for name, passed, val in gates:
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"  {status} {name:<26} Current: {val}", flush=True)

    print("\nOVERALL STATUS:", "ALL 11 GATES PASSED — READY FOR DEPLOYMENT" if all_passed else "GATES INCOMPLETE", flush=True)
    print("=" * 80, flush=True)

    if all_passed:
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)

        model_out = os.path.join(models_dir, "url_model.pkl")
        vec_out = os.path.join(models_dir, "url_vectorizer.pkl")
        meta_out = os.path.join(models_dir, "url_phase2_metadata.json")

        bak_model = os.path.join(models_dir, "url_model.pkl.bak")
        if not os.path.exists(bak_model) and os.path.exists(model_out):
            import shutil
            shutil.copy2(model_out, bak_model)
            print(f"  Backed up previous model to: {bak_model}", flush=True)

        joblib.dump(calibrated_model, model_out)
        joblib.dump(vectorizer, vec_out)

        metadata = {
            "model_type": "LightGBM + Sigmoid Calibration (Hybrid Char-Ngram + 33 Dense)",
            "n_trees": lgb_clf.best_iteration_,
            "feature_counts": {
                "tfidf": 2000,
                "dense": 33,
                "total": total_feat_count
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
                "p95_latency_ms": round(p95_latency, 2)
            },
            "all_gates_passed": all_passed,
            "trained_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[DONE] New Calibrated URL Model saved to: {model_out}", flush=True)
        print(f"       URL Vectorizer saved to:          {vec_out}", flush=True)
        print(f"       Metadata & Gates saved to:        {meta_out}", flush=True)


if __name__ == "__main__":
    main()
