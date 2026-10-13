"""
PhishGuard - Production URL and Domain Model Training
All 5 datasets | balanced 200k/class | 24+25 features | calibrated
IMPORTANT: Feature extractors here MUST stay in sync with scan_engine.py.

Domain model v2 improvements (15 -> 25 features):
  + vowel_ratio          : phishing domains have abnormal vowel distributions
  + consonant_cluster    : DGA domains have long consonant runs (e.g. 'xkzpt')
  + dga_score            : n-gram based DGA likelihood (lower bigram freq = DGA)
  + homoglyph_flag       : digits replacing letters (paypa1, micr0soft)
  + security_keyword     : 'login','verify','secure','account' in SLD
  + sld_length           : Second-level domain character length
  + sld_digit_ratio      : Ratio of digits in SLD (not full domain)
  + tld_is_numeric       : Rare - fully numeric TLD = suspicious
  + repeating_segment    : Repeated char groups (aaa, 111) = DGA/generated
  + known_tld_score      : Encoded TLD trust tier (com/org=0, xyz/tk=1, etc)
"""

import os
import re
import math
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from urllib.parse import urlparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, confusion_matrix
)

DESKTOP    = r"C:\Users\shari\Desktop"
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

TLD_RISK = {
    "tk": 1.0, "ml": 1.0, "ga": 1.0, "gq": 1.0, "cf": 1.0,
    "xyz": 0.9, "top": 0.9, "pw": 0.9, "cc": 0.85, "icu": 0.85,
    "club": 0.8, "online": 0.8, "site": 0.8, "click": 0.85,
    "link": 0.8, "download": 0.9, "zip": 0.9, "review": 0.8,
    "accountant": 0.85, "loan": 0.85, "science": 0.8, "win": 0.85,
    "bid": 0.8, "trade": 0.8, "racing": 0.8, "party": 0.8,
    "faith": 0.8, "webcam": 0.85, "stream": 0.75,
    "com": 0.1, "org": 0.05, "net": 0.1, "edu": 0.0, "gov": 0.0,
    "io": 0.1, "co": 0.1, "app": 0.1, "dev": 0.05,
    "pk": 0.05, "uk": 0.05, "de": 0.05, "fr": 0.05, "au": 0.05,
    "in": 0.05, "cn": 0.15, "ru": 0.2,  "br": 0.1, "ca": 0.05,
    "us": 0.05, "nl": 0.05, "jp": 0.05, "kr": 0.1, "mx": 0.1,
    "za": 0.1,  "ng": 0.2,  "ke": 0.15, "info": 0.4, "biz": 0.4,
}

KNOWN_SHORTENERS = re.compile(
    r"bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|"
    r"buff\.ly|shorte\.st|adf\.ly|cutt\.ly|rb\.gy|short\.io"
)
BRAND_KEYWORDS = re.compile(
    r"paypal|apple|microsoft|google|facebook|amazon|netflix|"
    r"instagram|twitter|linkedin|dropbox|yahoo|irs|wellsfargo|"
    r"chase|bankofamerica|coinbase|binance|blockchain|metamask|github|spotify"
)
OFFICIAL_DOMAINS = re.compile(
    r"^(paypal|apple|microsoft|google|facebook|amazon|netflix|"
    r"instagram|twitter|linkedin|dropbox|yahoo|irs|chase|wellsfargo|"
    r"coinbase|binance|blockchain|metamask|github|spotify)\.(com|org|net|co|gov|edu)(\.[a-z]{2})?$"
)

# Security / credential-harvesting keywords commonly seen in phishing domains
SECURITY_KEYWORDS_RE = re.compile(
    r"(?:^|[-_])(login|signin|sign-in|verify|verification|secure|security|"
    r"account|update|confirm|banking|wallet|token|auth|portal|recover|"
    r"reset|support|helpdesk|service|alert|unlock|validate)(?:$|[-_])"
)

# English bigram frequency table — rare bigrams indicate DGA / random strings
# Source: relative freq from Brown Corpus (normalized 0-1)
_BIGRAM_FREQ = {
    'th':1.0,'he':0.95,'in':0.9,'er':0.88,'an':0.85,'re':0.82,'on':0.80,
    'at':0.78,'en':0.77,'nd':0.76,'ti':0.73,'es':0.72,'or':0.71,'te':0.70,
    'of':0.68,'ed':0.67,'is':0.65,'it':0.64,'al':0.63,'ar':0.62,'st':0.61,
    'to':0.60,'nt':0.59,'ng':0.57,'se':0.56,'ha':0.55,'as':0.54,'ou':0.53,
    'io':0.52,'le':0.51,'ve':0.50,'co':0.49,'me':0.48,'de':0.47,'hi':0.46,
    'ri':0.45,'ro':0.44,'ic':0.42,'ne':0.41,'ea':0.40,'ra':0.39,'ce':0.38,
    'li':0.37,'ch':0.36,'ll':0.35,'be':0.34,'ma':0.33,'si':0.32,'om':0.31,
    'ur':0.30,'ca':0.29,'el':0.28,'ta':0.27,'so':0.26,'it':0.25,'pa':0.24,
    'ns':0.23,'rt':0.22,'no':0.21,'ss':0.20,'di':0.18,'ge':0.17,'tr':0.16,
    'fo':0.15,'rs':0.14,'un':0.13,'pe':0.12,'ad':0.11,'pl':0.10,'sh':0.09,
    'ck':0.08,'go':0.07,'ai':0.06,'sc':0.05,'pr':0.04,'ct':0.03,'wh':0.02,
}

def _dga_score(sld: str) -> float:
    """Lower score = more DGA-like (random string). Range 0-1."""
    if len(sld) < 4:
        return 0.5
    letters_only = re.sub(r'[^a-z]', '', sld.lower())
    if len(letters_only) < 3:
        return 0.3
    bigrams = [letters_only[i:i+2] for i in range(len(letters_only)-1)]
    if not bigrams:
        return 0.2
    score = sum(_BIGRAM_FREQ.get(bg, 0.01) for bg in bigrams) / len(bigrams)
    return round(score, 4)

def _vowel_ratio(s: str) -> float:
    """Ratio of vowels — phishing DGA domains tend toward 0.1-0.2; legit ~0.35-0.45."""
    letters = [c for c in s.lower() if c.isalpha()]
    if not letters:
        return 0.0
    vowels = sum(c in 'aeiou' for c in letters)
    return round(vowels / len(letters), 4)

def _max_consonant_run(s: str) -> int:
    """Longest consecutive consonant run — DGA domains have runs of 4+ (e.g. xkzptv)."""
    s = re.sub(r'[^a-z]', '', s.lower())
    consonants = set('bcdfghjklmnpqrstvwxyz')
    max_run = cur = 0
    for c in s:
        if c in consonants:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run

def _homoglyph_flag(domain: str) -> int:
    """Returns 1 if domain contains digit-to-letter substitutions in brand context."""
    _MAP = str.maketrans({'0':'o','1':'l','3':'e','4':'a','5':'s','6':'g','8':'b'})
    normalized = domain.translate(_MAP)
    # Only flag if normalization changes the string AND a brand appears
    if normalized != domain and BRAND_KEYWORDS.search(normalized):
        return 1
    return 0

KNOWN_TLD_TIERS = {
    # Tier 0 — fully trusted
    'com':0,'org':0,'net':0,'edu':0,'gov':0,'io':0,'co':0,'app':0,'dev':0,
    'uk':0,'de':0,'fr':0,'au':0,'ca':0,'pk':0,'in':0,'jp':0,'nl':0,'se':0,
    # Tier 1 — moderate risk
    'info':1,'biz':1,'us':1,'ru':1,'cn':1,'br':1,'kr':1,'mx':1,'ng':1,
    # Tier 2 — high risk
    'xyz':2,'top':2,'icu':2,'club':2,'online':2,'site':2,'link':2,'live':2,
    'stream':2,'review':2,'win':2,'bid':2,'trade':2,'racing':2,'party':2,
    # Tier 3 — free/abuse TLDs
    'tk':3,'ml':3,'ga':3,'gq':3,'cf':3,'pw':3,'cc':3,'click':3,'download':3,
    'zip':3,'webcam':3,'accountant':3,'loan':3,'science':3,'faith':3,
}

def _entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())

def _tld_risk(netloc):
    tld = netloc.split(".")[-1].lower() if "." in netloc else netloc.lower()
    return TLD_RISK.get(tld, 0.2)

def _char_continuation_rate(s):
    if not s:
        return 0.0
    max_run, cur = 1, 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        max_run = max(max_run, cur)
    return max_run / len(s)

def _unique_char_ratio(s):
    return len(set(s)) / len(s) if s else 0.0

def extract_url_features(url):
    if not isinstance(url, str):
        url = ""
    url = url.strip()
    try:
        p = urlparse(url if url.startswith("http") else "http://" + url)
        netloc, path, query, scheme = p.netloc.lower().split(":")[0], p.path, p.query, p.scheme.lower()
    except Exception:
        netloc = path = query = scheme = ""

    clean_domain = re.sub(r"^www\.", "", netloc)
    parts = clean_domain.split(".")
    tld = parts[-1] if parts else ""
    root = ".".join(parts[-2:]) if len(parts) >= 2 else clean_domain

    has_brand = int(bool(BRAND_KEYWORDS.search(clean_domain)) and not bool(OFFICIAL_DOMAINS.match(clean_domain)))
    norm = re.sub(r"^https?://", "", url.lower())
    norm = re.sub(r"^www\.", "", norm)
    dc = sum(c.isdigit() for c in norm)
    tl = len(norm) or 1

    return {
        "url_length":         len(norm),
        "entropy":            round(_entropy(norm), 4),
        "digit_count":        dc,
        "digit_ratio":        round(dc / tl, 4),
        "special_char_count": sum(not c.isalnum() for c in norm),
        "hyphen_count":       norm.count("-"),
        "at_count":           norm.count("@"),
        "dot_count":          clean_domain.count("."),
        "path_length":        len(path),
        "query_length":       len(query),
        "path_depth":         path.count("/"),
        "query_param_count":  query.count("&") + (1 if query else 0),
        "subdomain_count":    max(0, clean_domain.count(".") - 1),
        "domain_length":      len(clean_domain),
        "is_https":           int(scheme == "https"),
        "has_ip":             int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", clean_domain))),
        "has_shortener":      int(bool(KNOWN_SHORTENERS.search(norm))),
        "tld_risk_score":     round(_tld_risk(clean_domain), 3),
        "has_suspicious_tld": int(TLD_RISK.get(tld, 0) >= 0.75),
        "has_brand_in_url":   has_brand,
        "has_hex_encoding":   int("%" in url),
        "has_double_slash":   int("//" in path),
        "char_continuation":  round(_char_continuation_rate(norm), 4),
        "unique_char_ratio":  round(_unique_char_ratio(norm[:50]), 4),
    }

def extract_domain_features(url):
    """16 lean active features (pruned 9 dead/useless features). Must stay in sync with services/scan_engine.py."""
    if not isinstance(url, str):
        url = ""
    url = url.strip().lower()
    try:
        p = urlparse(url if url.startswith("http") else "http://" + url)
        domain = p.netloc.split(":")[0]
    except Exception:
        domain = ""

    clean_domain = re.sub(r"^www\.", "", domain)
    parts = clean_domain.split(".")
    root = ".".join(parts[-2:]) if len(parts) >= 2 else clean_domain
    tld = parts[-1] if parts else ""
    sld = parts[-2] if len(parts) >= 2 else parts[0] if parts else ""  # second-level domain
    sub = ".".join(parts[:-2]) if len(parts) > 2 else ""

    has_brand = int(bool(BRAND_KEYWORDS.search(clean_domain)) and not bool(OFFICIAL_DOMAINS.match(clean_domain)))

    feat = {
        # Lexical / Structural metrics
        "domain_length":         len(clean_domain),
        "domain_entropy":        round(_entropy(clean_domain), 4),
        "domain_digit_count":    sum(c.isdigit() for c in clean_domain),
        "domain_hyphen_count":   clean_domain.count("-"),
        "domain_dot_count":      clean_domain.count("."),
        "unique_char_ratio":     round(_unique_char_ratio(root), 4),
        "sld_length":            len(sld),
        "sld_digit_ratio":       round(sum(c.isdigit() for c in sld) / max(len(sld), 1), 4),
        # Linguistic & DGA detectors
        "vowel_ratio":           _vowel_ratio(sld),
        "consonant_cluster":     _max_consonant_run(sld),
        "dga_score":             _dga_score(sld),
        # Brand & Security cues
        "brand_impersonation":   has_brand,
        "brand_in_subdomain":    int(bool(BRAND_KEYWORDS.search(sub))),
        "security_keyword":      int(bool(SECURITY_KEYWORDS_RE.search(sld))),
        # TLD Risk metrics
        "tld_risk_score":        round(_tld_risk(clean_domain), 3),
        "known_tld_score":       KNOWN_TLD_TIERS.get(tld, 2),
    }

    return feat

def load_all_url_data():
    frames = []
    print("[1/3] PhiUSIIL_Phishing_URL_Dataset.csv ...")
    df = pd.read_csv(os.path.join(DESKTOP, "PhiUSIIL_Phishing_URL_Dataset.csv"),
                     usecols=["URL", "label"], low_memory=False).rename(columns={"URL": "url"})
    # In PhiUSIIL 1=legit, 0=phish -> invert to 0=benign, 1=phish
    df["label"] = 1 - df["label"].astype(int)
    frames.append(df)
    print(f"    {len(df):,} rows | phish: {df.label.sum():,}")

    print("[2/3] phishing_site_urls.csv ...")
    df = pd.read_csv(os.path.join(DESKTOP, "phishing_site_urls.csv"),
                     usecols=["URL", "Label"]).rename(columns={"URL": "url", "Label": "label"})
    df["label"] = df["label"].str.lower().map({"good": 0, "bad": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    frames.append(df)
    print(f"    {len(df):,} rows | phish: {df.label.sum():,}")

    print("[3/3] malicious_phish.csv ...")
    df = pd.read_csv(os.path.join(DESKTOP, "malicious_phish.csv"),
                     usecols=["url", "type"]).rename(columns={"type": "label"})
    df["label"] = df["label"].str.lower().map({"benign": 0, "phishing": 1, "malware": 1, "defacement": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    frames.append(df)
    print(f"    {len(df):,} rows | phish: {df.label.sum():,}")

    combined = pd.concat(frames, ignore_index=True).dropna(subset=["url", "label"])
    combined["url"] = combined["url"].astype(str).str.strip()
    combined = combined.drop_duplicates(subset=["url"])
    benign, malicious = combined[combined.label == 0], combined[combined.label == 1]
    cap = min(len(benign), len(malicious), 200_000)
    print(f"\n  Unique: {len(combined):,} | Benign: {len(benign):,} | Malicious: {len(malicious):,} | Cap: {cap:,}")
    return pd.concat([
        benign.sample(n=cap, random_state=42),
        malicious.sample(n=cap, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

def load_all_domain_data():
    """Reject the legacy URL/popularity/reputation-to-domain conversion."""
    from training.domain_evidence import reject_legacy_domain_training
    reject_legacy_domain_training()

def train_and_evaluate(X, y, model_name, is_domain=False):
    print(f"\n{'=' * 64}")
    print(f"  Model: {model_name} | Samples: {len(X):,} | Features: {X.shape[1]}")
    print(f"{'=' * 64}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if is_domain:
        # Domain model v2: deeper trees + more estimators to fully exploit the
        # richer 25-feature space (especially dga_score, vowel_ratio, homoglyph).
        # lower learning_rate + more estimators prevents overfitting while capturing
        # complex lexical interactions between the new features.
        # scale_pos_weight=2.0: confusion matrix showed 39% FN rate with equal weights.
        # Doubling the malicious-class penalty directly drives recall up toward 75%+.
        lgbm = lgb.LGBMClassifier(
            n_estimators=1200,       # was 500 — more trees for subtle patterns
            learning_rate=0.02,      # was 0.03 — slower learning, better generalization
            max_depth=10,            # was 8 — deeper to capture feature interactions
            num_leaves=255,          # was 127 — more leaf nodes
            min_child_samples=20,    # was 25 — slightly smaller leaves OK with more data
            subsample=0.8,
            colsample_bytree=0.75,   # was 0.85 — more feature subsampling = less overfit
            reg_alpha=0.15,          # was 0.1 — stronger L1 regularization
            reg_lambda=0.15,         # was 0.1 — stronger L2 regularization
            scale_pos_weight=2.0,    # was 1.0 — penalize FN twice as much (39% FN → ~25%)
            random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        # URL model: keep existing tuned hyperparams
        lgbm = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=127,
            min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=1,
            random_state=42, n_jobs=-1, verbose=-1,
        )

    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    print("  Calibrating (sigmoid, cv=5) ...")
    # Use sigmoid instead of isotonic:
    # - isotonic overfits calibration set and un-does scale_pos_weight effect
    # - sigmoid is a 2-parameter Platt scaling — less aggressive, more stable
    calibrated = CalibratedClassifierCV(lgbm, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n  Accuracy : {acc * 100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"], digits=4))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=["Act.Benign", "Act.Malicious"], columns=["Pred.Benign", "Pred.Malicious"]))
    feat_imp = pd.Series(lgbm.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n  Top-{min(15, len(feat_imp))} Features:")
    print(feat_imp.head(15).to_string())

    return calibrated

def train_models():
    # Fail before any production artifact can be overwritten.
    load_all_domain_data()
    print("=" * 64)
    print("  PhishGuard - Production Model Retraining")
    print("  5 datasets | balanced | 24+15 features | calibrated")
    print("=" * 64)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n>>> Phase 1: URL Model\n")
    url_df = load_all_url_data()
    print("\nExtracting features ...")
    url_feat = pd.DataFrame(url_df["url"].apply(extract_url_features).tolist())
    url_model = train_and_evaluate(url_feat, url_df["label"].values, "URL Model")
    joblib.dump(url_model, os.path.join(MODELS_DIR, "url_model.pkl"))
    print(f"\n  [SAVED] url_model.pkl")

    print("\n>>> Phase 2: Domain Model (v2 — 25 features)\n")
    dom_df = load_all_domain_data()
    print("\nExtracting features ...")
    dom_feat = pd.DataFrame(dom_df["url"].apply(extract_domain_features).tolist())
    print(f"  Feature matrix: {dom_feat.shape[0]:,} rows x {dom_feat.shape[1]} features")
    print(f"  New features: {[c for c in dom_feat.columns if c not in ['domain_length','domain_entropy','domain_digit_count','domain_hyphen_count','domain_dot_count','subdomain_depth','has_ip','tld_risk_score','has_suspicious_tld','is_free_tld','brand_impersonation','brand_in_subdomain','has_numbers_in_domain','has_punycode','unique_char_ratio']]}")
    dom_model = train_and_evaluate(dom_feat, dom_df["label"].values, "Domain Model v2", is_domain=True)
    joblib.dump(dom_model, os.path.join(MODELS_DIR, "domain_model.pkl"))
    print(f"\n  [SAVED] domain_model.pkl (v2 — 25 features)")

    print("\n" + "=" * 64)
    print("  Training complete! Both models calibrated and saved successfully.")
    print("=" * 64)

if __name__ == "__main__":
    train_models()
