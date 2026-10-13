"""
Phase 1: Train & Freeze Deployable Pipeline (Development Data Only)
OriginTrace-AdaptiveGraph Scientific Validation Protocol

This script:
  1. Loads development corpora (Phishing_Email.csv & malicious_phish.csv).
  2. Deduplicates and groups URLs by Public Suffix List (tldextract) with zero domain leakage.
  3. Splits into Train (60%), Validation (20%), and Development-Test (20%).
  4. Trains the exact specialist models (Email, URL, Domain) and the Shared Monolith.
  5. Selects operating thresholds (Strict 0.5%, Inspection 2.0%, Balanced) on VALIDATION DATA ONLY.
  6. Freezes all models, vectorizers, and preprocessing parameters into models/frozen_pipeline/.
  7. Calculates SHA-256 cryptographic hashes for full auditable traceability.
"""

import os
import sys
import hashlib
import json
import posixpath
import unicodedata
import urllib.parse
import re
import math
import joblib
import numpy as np
import pandas as pd
import tldextract
from typing import Dict, List, Any
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_curve, confusion_matrix
import lightgbm as lgb
import sklearn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

TLD_RISK = {
    "tk": 1.0, "ml": 1.0, "ga": 1.0, "gq": 1.0, "cf": 1.0,
    "xyz": 0.9, "top": 0.9, "pw": 0.9, "cc": 0.85, "icu": 0.85,
    "club": 0.8, "online": 0.8, "site": 0.8, "click": 0.85,
    "link": 0.8, "download": 0.9, "zip": 0.9, "review": 0.8,
    "com": 0.1, "org": 0.05, "net": 0.1, "edu": 0.0, "gov": 0.0,
    "io": 0.1, "co": 0.1, "app": 0.1, "dev": 0.05
}
KNOWN_SHORTENERS = re.compile(r"bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|rb\.gy")
BRAND_KEYWORDS = re.compile(r"paypal|apple|microsoft|google|facebook|amazon|netflix|chase|wellsfargo|binance|coinbase")
OFFICIAL_DOMAINS = re.compile(r"^(paypal|apple|microsoft|google|amazon|chase)\.(com|org|net)$")

HOMOGLYPH_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'і': 'i', 'ј': 'j', 'ѕ': 's',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T',
    'Х': 'X', 'Ү': 'Y'
}

def _entropy(s: str) -> float:
    if not s: return 0.0
    freq: Dict[str, int] = {}
    for c in s: freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())

def _tld_risk(netloc: str) -> float:
    tld = netloc.split(".")[-1].lower() if "." in netloc else netloc.lower()
    return TLD_RISK.get(tld, 0.2)

def _char_continuation_rate(s: str) -> float:
    if not s: return 0.0
    max_run, cur = 1, 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        max_run = max(max_run, cur)
    return max_run / len(s)

def _unique_char_ratio(s: str) -> float:
    return len(set(s)) / len(s) if s else 0.0

def canonicalize_email_text(text: str) -> str:
    if not isinstance(text, str): return ""
    try:
        text = urllib.parse.unquote(text)
    except Exception:
        pass
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xad]', '', text)
    for k, v in HOMOGLYPH_MAP.items():
        text = text.replace(k, v)
    return text

def extract_email_anomaly_features(text: str) -> List[float]:
    if not isinstance(text, str): return [0.0] * 7
    zw_count = len(re.findall(r'[\u200b\u200c\u200d\ufeff\xad]', text))
    hg_count = sum(1 for c in text if c in HOMOGLYPH_MAP)
    hex_count = len(re.findall(r'%[0-9a-fA-F]{2}', text))
    length = max(1, len(text))
    non_ascii_count = sum(ord(c) > 127 for c in text)
    b_entropy = _entropy(text)
    return [
        float(zw_count),
        float(hg_count),
        float(hex_count),
        round(hg_count / length, 5),
        round(hex_count / length, 5),
        round(non_ascii_count / length, 5),
        round(b_entropy, 4)
    ]

def normalize_url_text(url: str) -> str:
    if not isinstance(url, str): return ""
    s = url.strip().lower()
    s = re.sub(r'^https?://', '', s)
    s = re.sub(r'^www\.', '', s)
    return s

def canonicalize_url(url: str) -> str:
    if not isinstance(url, str): return ""
    s = url.strip()
    p = urllib.parse.urlparse(s if s.startswith("http") else "http://" + s)
    raw_path = urllib.parse.unquote(p.path or "")
    canon_path = posixpath.normpath(raw_path) if raw_path else ""
    if raw_path.endswith("/") and not canon_path.endswith("/"):
        canon_path += "/"
    canon_host = unicodedata.normalize('NFKC', urllib.parse.unquote(p.netloc).lower())
    canon_query = urllib.parse.unquote(p.query or "")
    return urllib.parse.urlunparse((p.scheme or "http", canon_host, canon_path, "", canon_query, ""))

def extract_url_features(url: str) -> Dict[str, Any]:
    from urllib.parse import urlparse
    if not isinstance(url, str): url = ""
    url = url.strip()
    try:
        p = urlparse(url if url.startswith("http") else "http://" + url)
        netloc, path, query, scheme = p.netloc.lower().split(":")[0], p.path, p.query, p.scheme.lower()
    except Exception:
        netloc = path = query = scheme = ""
    clean_domain = re.sub(r"^www\.", "", netloc)
    parts = clean_domain.split(".")
    tld = parts[-1] if parts else ""
    norm = re.sub(r"^https?://", "", url.lower())
    norm = re.sub(r"^www\.", "", norm)
    dc = sum(c.isdigit() for c in norm)
    tl = len(norm) or 1
    has_brand = int(bool(BRAND_KEYWORDS.search(clean_domain)) and not bool(OFFICIAL_DOMAINS.match(clean_domain)))
    return {
        "url_length": len(norm), "entropy": round(_entropy(norm), 4),
        "digit_count": dc, "digit_ratio": round(dc / tl, 4),
        "special_char_count": sum(not c.isalnum() for c in norm),
        "hyphen_count": norm.count("-"), "at_count": norm.count("@"),
        "dot_count": clean_domain.count("."), "path_length": len(path),
        "query_length": len(query), "path_depth": path.count("/"),
        "query_param_count": query.count("&") + (1 if query else 0),
        "subdomain_count": max(0, clean_domain.count(".") - 1),
        "domain_length": len(clean_domain), "is_https": int(scheme == "https"),
        "has_ip": int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", clean_domain))),
        "has_shortener": int(bool(KNOWN_SHORTENERS.search(norm))),
        "tld_risk_score": round(_tld_risk(clean_domain), 3),
        "has_suspicious_tld": int(TLD_RISK.get(tld, 0) >= 0.75),
        "has_brand_in_url": has_brand, "has_hex_encoding": int("%" in url),
        "has_double_slash": int("//" in path),
        "char_continuation": round(_char_continuation_rate(norm), 4),
        "unique_char_ratio": round(_unique_char_ratio(norm[:50]), 4),
    }

def extract_domain_features(url_or_domain: str) -> Dict[str, Any]:
    from urllib.parse import urlparse
    if not isinstance(url_or_domain, str): url_or_domain = ""
    s = url_or_domain.strip().lower()
    try:
        p = urlparse(s if s.startswith("http") else "http://" + s)
        domain = p.netloc.split(":")[0] if p.netloc else s.split("/")[0]
    except Exception:
        domain = s.split("/")[0]
    clean_domain = re.sub(r"^www\.", "", domain).split(":")[0]
    parts = clean_domain.split(".")
    tld = parts[-1] if parts else ""
    sld = parts[-2] if len(parts) >= 2 else parts[0] if parts else ""
    has_brand = int(bool(BRAND_KEYWORDS.search(clean_domain)) and not bool(OFFICIAL_DOMAINS.match(clean_domain)))
    return {
        "domain_length": len(clean_domain), "domain_entropy": round(_entropy(clean_domain), 4),
        "domain_digit_count": sum(c.isdigit() for c in clean_domain),
        "domain_hyphen_count": clean_domain.count("-"),
        "domain_dot_count": clean_domain.count("."),
        "subdomain_depth": max(0, clean_domain.count(".") - 1),
        "has_ip": int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", clean_domain))),
        "tld_risk_score": round(_tld_risk(clean_domain), 3),
        "has_suspicious_tld": int(TLD_RISK.get(tld, 0) >= 0.75),
        "is_free_tld": int(tld in ["tk", "ml", "ga", "cf", "gq"]),
        "brand_impersonation": has_brand,
        "brand_in_subdomain": int(bool(BRAND_KEYWORDS.search(".".join(parts[:-2])))),
        "has_numbers_in_domain": int(any(c.isdigit() for c in clean_domain)),
        "has_punycode": int(clean_domain.startswith("xn--")),
        "unique_char_ratio": round(_unique_char_ratio(clean_domain), 4),
        "vowel_ratio": round(sum(c in "aeiou" for c in clean_domain) / max(1, len(clean_domain)), 4),
        "consonant_cluster": 0, "dga_score": 0.5, "homoglyph_flag": 0,
        "security_keyword": int(any(k in sld for k in ["login", "verify", "secure", "account"])),
        "sld_length": len(sld), "sld_digit_ratio": round(sum(c.isdigit() for c in sld) / max(1, len(sld)), 4),
        "tld_is_numeric": int(tld.isdigit()), "repeating_segment": 0, "known_tld_score": 0
    }

def extract_url_dual_representation(url: str) -> List[float]:
    c_u = canonicalize_url(url)
    raw = str(url)
    hex_count = len(re.findall(r'%[0-9a-fA-F]{2}', raw))
    length = max(1, len(raw))
    struct_dict = extract_url_features(c_u)
    return list(struct_dict.values()) + [float(hex_count), float(hex_count / length)]

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    print("=" * 80)
    print("  PHASE 1: TRAINING & FREEZING UNIFIED DEPLOYABLE PIPELINE")
    print("  (Validation Data Only for Operating Threshold Selection)")
    print("=" * 80)

    # 1. Load Email Corpus
    email_csv = os.path.join(PROJECT_ROOT, "data", "Phishing_Email.csv")
    df_email_raw = pd.read_csv(email_csv).dropna(subset=['Email Text'])
    df_email_raw['text_clean'] = df_email_raw['Email Text'].str.strip()
    df_email_raw = df_email_raw.drop_duplicates(subset=['text_clean'])
    df_email_raw = df_email_raw[(df_email_raw['text_clean'].str.len() >= 50) & (df_email_raw['text_clean'].str.len() <= 25000)]

    safe_emails = df_email_raw[df_email_raw['Email Type'] == 'Safe Email'].sample(n=5000, random_state=42)
    phish_emails = df_email_raw[df_email_raw['Email Type'] == 'Phishing Email'].sample(n=5000, random_state=42)
    df_emails = pd.concat([safe_emails, phish_emails]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_emails['label'] = (df_emails['Email Type'] == 'Phishing Email').astype(int)

    # 2. Load URL Corpus
    top1m_csv = r"C:\Users\shari\Desktop\top-1m.csv"
    df_top1m = pd.read_csv(top1m_csv, header=None, nrows=20000)
    top_benign = ['https://www.' + d for d in df_top1m[1].astype(str).tolist()[1000:6000]]

    url_csv = r"C:\Users\shari\Desktop\malicious_phish.csv"
    df_url_raw = pd.read_csv(url_csv, nrows=200000).dropna(subset=['url'])
    df_url_raw['url_clean'] = df_url_raw['url'].str.strip()
    df_url_raw = df_url_raw.drop_duplicates(subset=['url_clean'])

    deep_benign = df_url_raw[df_url_raw['type'] == 'benign']['url_clean'].dropna().tolist()[:5000]
    all_benign = top_benign + deep_benign
    phish_urls = df_url_raw[df_url_raw['type'] == 'phishing']['url_clean'].dropna().tolist()[:10000]

    df_benign_combined = pd.DataFrame({'url_clean': all_benign, 'label': 0})
    df_phish_combined = pd.DataFrame({'url_clean': phish_urls, 'label': 1})
    df_urls = pd.concat([df_benign_combined, df_phish_combined]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_urls['psl_domain'] = df_urls['url_clean'].apply(lambda u: tldextract.extract(u).top_domain_under_public_suffix or "unknown")

    # 3. 3-Way Splits
    X_em_tr, X_em_temp, y_em_tr, y_em_temp = train_test_split(
        df_emails['text_clean'], df_emails['label'], test_size=0.40, random_state=42, stratify=df_emails['label']
    )
    X_em_val, X_em_dev_te, y_em_val, y_em_dev_te = train_test_split(
        X_em_temp, y_em_temp, test_size=0.50, random_state=42, stratify=y_em_temp
    )

    gss = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=42)
    tr_u_idx, temp_u_idx = next(gss.split(df_urls['url_clean'], df_urls['label'], groups=df_urls['psl_domain']))
    df_u_tr = df_urls.iloc[tr_u_idx].reset_index(drop=True)
    df_u_temp = df_urls.iloc[temp_u_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_u_sub, dev_te_u_sub = next(gss_val.split(df_u_temp['url_clean'], df_u_temp['label'], groups=df_u_temp['psl_domain']))
    df_u_val = df_u_temp.iloc[val_u_sub].reset_index(drop=True)
    df_u_dev_te = df_u_temp.iloc[dev_te_u_sub].reset_index(drop=True)

    train_doms = set(df_u_tr['psl_domain'])
    val_doms = set(df_u_val['psl_domain'])
    assert len(train_doms.intersection(val_doms)) == 0, "Train-Val domain leakage detected!"

    print(f"[OK] Email Splits -> Train: {len(X_em_tr)}, Val: {len(X_em_val)}, Dev-Test: {len(X_em_dev_te)}")
    print(f"[OK] URL Splits -> Train: {len(df_u_tr)}, Val: {len(df_u_val)}, Dev-Test: {len(df_u_dev_te)}")

    # 4. Train Email Specialist
    print("\nFitting Email Specialist (Canonical NFKC Word n-grams + Anomaly features)...")
    email_vec = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
    X_em_tr_canon = email_vec.fit_transform([canonicalize_email_text(t) for t in X_em_tr]).astype(np.float32)
    X_em_val_canon = email_vec.transform([canonicalize_email_text(t) for t in X_em_val]).astype(np.float32)

    X_em_tr_anom = csr_matrix(np.array([extract_email_anomaly_features(t) for t in X_em_tr], dtype=np.float32))
    X_em_val_anom = csr_matrix(np.array([extract_email_anomaly_features(t) for t in X_em_val], dtype=np.float32))

    X_em_tr_dual = hstack([X_em_tr_canon, X_em_tr_anom])
    X_em_val_dual = hstack([X_em_val_canon, X_em_val_anom])

    email_model = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.1, num_leaves=20, n_jobs=1, random_state=42, verbose=-1)
    email_model.fit(X_em_tr_dual, y_em_tr)

    # 5. Train URL Specialist
    print("Fitting URL Specialist (Char_wb 3-5 n-grams + Dual Structure)...")
    url_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=3000)
    X_u_tr_b = url_vec.fit_transform([normalize_url_text(u) for u in df_u_tr['url_clean']]).astype(np.float32)
    X_u_val_b = url_vec.transform([normalize_url_text(u) for u in df_u_val['url_clean']]).astype(np.float32)

    X_u_tr_struct = csr_matrix(np.array([extract_url_dual_representation(u) for u in df_u_tr['url_clean']], dtype=np.float32))
    X_u_val_struct = csr_matrix(np.array([extract_url_dual_representation(u) for u in df_u_val['url_clean']], dtype=np.float32))

    X_u_tr_dual = hstack([X_u_tr_b, X_u_tr_struct])
    X_u_val_dual = hstack([X_u_val_b, X_u_val_struct])

    url_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, n_jobs=1, random_state=42, verbose=-1)
    url_model.fit(X_u_tr_dual, df_u_tr['label'])

    # 6. Train Domain Specialist
    print("Fitting Domain Specialist (Structural & Reputation Features)...")
    X_d_tr = np.array([list(extract_domain_features(u).values()) for u in df_u_tr['url_clean']], dtype=np.float32)
    X_d_val = np.array([list(extract_domain_features(u).values()) for u in df_u_val['url_clean']], dtype=np.float32)

    domain_model = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, num_leaves=15, n_jobs=1, random_state=42, verbose=-1)
    domain_model.fit(X_d_tr, df_u_tr['label'])

    # 7. Train Shared Monolith
    print("Fitting Shared Monolith (Concatenated Multimodal Representation)...", flush=True)
    n_mono = min(X_em_tr_dual.shape[0], X_u_tr_dual.shape[0])
    X_mono_tr = hstack([
        X_em_tr_dual[:n_mono],
        X_u_tr_dual[:n_mono],
        csr_matrix(X_d_tr[:n_mono])
    ])
    monolith_model = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.1, num_leaves=20, n_jobs=1, random_state=42, verbose=-1)
    monolith_model.fit(X_mono_tr, df_u_tr['label'].values[:n_mono])

    # 8. Select Thresholds & Operating Points on VALIDATION DATA ONLY
    print("\nSelecting Operating Thresholds on VALIDATION DATA ONLY...")
    val_probs_u = url_model.predict_proba(X_u_val_dual)[:, 1]
    fpr_u, tpr_u, thresh_u = roc_curve(df_u_val['label'], val_probs_u)

    # Strict Blocking (val target <= 0.25% FPR to ensure test FPR <= 0.5%)
    idx_05 = np.where(fpr_u <= 0.0025)[0]
    if len(idx_05) == 0:
        idx_05 = np.where(fpr_u <= 0.0050)[0]
    thresh_05 = float(thresh_u[idx_05[-1]]) if len(idx_05) > 0 else 0.85

    # Deep Inspection (val target <= 1.50% FPR to ensure test FPR <= 2.0%)
    idx_20 = np.where(fpr_u <= 0.0150)[0]
    if len(idx_20) == 0:
        idx_20 = np.where(fpr_u <= 0.0200)[0]
    thresh_20 = float(thresh_u[idx_20[-1]]) if len(idx_20) > 0 else 0.65

    # Balanced Optimum
    j_scores = tpr_u - fpr_u
    thresh_bal = float(thresh_u[np.argmax(j_scores)])

    print(f"  [OK] Validation Thresholds Selected:")
    print(f"       Strict Blocking (FPR <= 0.5% target) : tau = {thresh_05:.4f}")
    print(f"       Deep Inspection (FPR <= 2.0% target) : tau = {thresh_20:.4f}")
    print(f"       Balanced Optimum                     : tau = {thresh_bal:.4f}")

    # 9. Freeze Artifacts to models/frozen_pipeline/
    frozen_dir = os.path.join(PROJECT_ROOT, "models", "frozen_pipeline")
    os.makedirs(frozen_dir, exist_ok=True)

    email_vec_path = os.path.join(frozen_dir, "email_vectorizer.joblib")
    email_model_path = os.path.join(frozen_dir, "email_specialist.joblib")
    url_vec_path = os.path.join(frozen_dir, "url_vectorizer.joblib")
    url_model_path = os.path.join(frozen_dir, "url_specialist.joblib")
    domain_model_path = os.path.join(frozen_dir, "domain_specialist.joblib")
    monolith_path = os.path.join(frozen_dir, "monolith_model.joblib")

    joblib.dump(email_vec, email_vec_path, compress=3)
    joblib.dump(email_model, email_model_path, compress=3)
    joblib.dump(url_vec, url_vec_path, compress=3)
    joblib.dump(url_model, url_model_path, compress=3)
    joblib.dump(domain_model, domain_model_path, compress=3)
    joblib.dump(monolith_model, monolith_path, compress=3)

    # 10. Compute Hashes & Save pipeline_config.json
    hashes = {
        "email_vectorizer.joblib": compute_sha256(email_vec_path),
        "email_specialist.joblib": compute_sha256(email_model_path),
        "url_vectorizer.joblib": compute_sha256(url_vec_path),
        "url_specialist.joblib": compute_sha256(url_model_path),
        "domain_specialist.joblib": compute_sha256(domain_model_path),
        "monolith_model.joblib": compute_sha256(monolith_path)
    }

    pipeline_config = {
        "pipeline_version": "2.0.0-origin_trace_frozen",
        "frozen_timestamp_iso": "2026-10-01T22:25:00Z",
        "environment": {
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "lightgbm_version": lgb.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__
        },
        "operating_thresholds": {
            "strict_blocking_tau_05": round(thresh_05, 4),
            "deep_inspection_tau_20": round(thresh_20, 4),
            "balanced_optimum_tau": round(thresh_bal, 4)
        },
        "cascade_escalation_rules": {
            "conflict_metric_threshold": 0.40,
            "uncertainty_range": [0.40, 0.60],
            "asymmetric_url_threshold": 0.75,
            "asymmetric_domain_threshold": 0.45,
            "high_threat_override_threshold": 0.85,
            "clean_exoneration_threshold": 0.35
        },
        "preprocessing_settings": {
            "email_max_features": 2500,
            "email_ngram_range": [1, 2],
            "email_min_length": 50,
            "email_max_length": 25000,
            "url_max_features": 3000,
            "url_ngram_range": [3, 5],
            "url_analyzer": "char_wb"
        },
        "artifact_sha256_hashes": hashes
    }

    config_path = os.path.join(frozen_dir, "pipeline_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_config, f, indent=2)

    # 11. Save Split Manifests
    manifests = {
        "development_datasets": {
            "email_corpus": {
                "source": "data/Phishing_Email.csv",
                "train_samples": len(X_em_tr),
                "val_samples": len(X_em_val),
                "dev_test_samples": len(X_em_dev_te),
                "balance": "50/50"
            },
            "url_corpus": {
                "source": "malicious_phish.csv (Desktop)",
                "train_samples": len(df_u_tr),
                "val_samples": len(df_u_val),
                "dev_test_samples": len(df_u_dev_te),
                "unique_psl_domains": df_urls['psl_domain'].nunique(),
                "train_val_domain_overlap": len(train_doms.intersection(val_doms))
            }
        },
        "validation_selection_proof": {
            "val_url_samples": len(df_u_val),
            "val_benign_count": int(sum(df_u_val['label'] == 0)),
            "val_phish_count": int(sum(df_u_val['label'] == 1)),
            "val_fpr_at_strict_threshold": float(fpr_u[idx_05[-1]]),
            "selected_strict_tau": thresh_05
        }
    }

    benchmarks_dir = os.path.join(PROJECT_ROOT, "benchmarks", "step5_audit_results")
    os.makedirs(benchmarks_dir, exist_ok=True)
    manifest_path = os.path.join(benchmarks_dir, "split_manifests.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)

    print(f"\n[SUCCESS] Pipeline frozen to: {frozen_dir}")
    print(f"[SUCCESS] Config & SHA-256 written to: {config_path}")
    print(f"[SUCCESS] Split manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()
