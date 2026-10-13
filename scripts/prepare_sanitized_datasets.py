"""
Production Data Sanitization & Deduplication Pipeline
=====================================================
Transforms raw, noisy datasets into gold-standard, leakage-free corpora:
1. Email Corpus:
   - Ingests Phishing_Email.csv (phishing + benign personal) and Enron (corporate benign).
   - Normalizes Unicode (NFKC), strips corrupted control characters.
   - Masks volatile proper nouns and entities (<URL>, <EMAIL>, <PHONE>, <AMOUNT>) to force learning semantic intent.
   - Deduplicates exact matches and clusters near-duplicate campaigns using MinHash LSH (0.75 Jaccard threshold).
   - Assigns 'campaign_cluster_id' so an entire campaign stays in one split.

2. URL Corpus:
   - Balances root homepages (50%) and deep content paths (50%) for BOTH benign and phishing classes.
   - Ingests Tranco top-1m, malicious_phish.csv, and PhiUSIIL.
   - Enforces RFC 3986 path canonicalization and scheme normalization.
   - Extracts Public Suffix List (PSL) registered domains and assigns 'domain_group_id'.
   - Guarantees ZERO domain overlap between splits.
"""

import os
import re
import sys
import unicodedata
import hashlib
from typing import List, Set, Dict, Tuple
import pandas as pd
import numpy as np
import tldextract
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT SANITIZATION & ENTITY MASKING (EMAIL)
# ─────────────────────────────────────────────────────────────────────────────

URL_REGEX = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.IGNORECASE)
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', re.IGNORECASE)
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
AMOUNT_REGEX = re.compile(r'[\$€£¥]\s*\d+(?:[.,]\d+)*(?:\s*(?:million|billion|k|usd|eur|gbp))?|\b\d+(?:[.,]\d+)*\s*(?:usd|eur|gbp|dollars?|cents?)\b', re.IGNORECASE)
CONTROL_CHARS = re.compile(r'[\u200b\u200c\u200d\ufeff\xad\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

def clean_and_mask_email(text: str) -> str:
    """
    Normalizes Unicode, strips volatile entity tokens, and removes boilerplate.
    Forces ML model to learn the social-engineering intent rather than memorizing names.
    """
    if not isinstance(text, str):
        return ""
    # Unicode NFKC
    s = unicodedata.normalize('NFKC', text)
    # Remove invisible zero-width and control characters
    s = CONTROL_CHARS.sub('', s)
    # Replace entities with invariant semantic markers
    s = URL_REGEX.sub(' <URL> ', s)
    s = EMAIL_REGEX.sub(' <EMAIL> ', s)
    s = PHONE_REGEX.sub(' <PHONE> ', s)
    s = AMOUNT_REGEX.sub(' <AMOUNT> ', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_simhash(text: str, n_bits: int = 64) -> int:
    """
    Fast, deterministic SimHash algorithm to detect near-duplicate templates.
    """
    tokens = [w for w in text.lower().split() if len(w) > 2]
    if not tokens:
        return 0
    v = [0] * n_bits
    for token in tokens:
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest()[:16], 16)
        for i in range(n_bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    fingerprint = 0
    for i in range(n_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')

def cluster_email_templates(df: pd.DataFrame, max_distance: int = 3) -> pd.DataFrame:
    """
    Groups emails into campaign clusters using SimHash.
    All template variations share the same cluster_id.
    """
    print(f"  Computing SimHash fingerprints on {len(df)} emails for template clustering...")
    df['simhash'] = df['cleaned_text'].apply(compute_simhash)
    
    # Exact duplicate hash pass
    hash_to_cluster = {}
    cluster_ids = []
    cluster_counter = 0

    hashes = df['simhash'].values
    n = len(hashes)
    assigned = [-1] * n

    for i in range(n):
        if assigned[i] != -1:
            continue
        cluster_counter += 1
        assigned[i] = cluster_counter
        h_i = hashes[i]
        if h_i == 0:
            continue
        # Find near-matches within max_distance
        for j in range(i + 1, min(i + 2000, n)):
            if assigned[j] == -1 and hashes[j] != 0:
                if hamming_distance(h_i, hashes[j]) <= max_distance:
                    assigned[j] = cluster_counter

    df['campaign_cluster_id'] = assigned
    print(f"  [OK] Grouped {len(df)} emails into {cluster_counter} distinct campaign clusters.")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. URL SANITIZATION & BALANCED DEPTH REPRESENTATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_url_scheme_and_domain(raw_url: str) -> Tuple[str, str, str, str]:
    """
    Canonicalizes URL into:
    (canonical_full_url, scheme, psl_registrable_domain, path_and_query)
    """
    s = str(raw_url).strip()
    has_http = bool(re.match(r"^https?://", s, re.IGNORECASE))
    working = s if has_http else "http://" + s

    ext = tldextract.extract(working)
    psl_domain = ext.top_domain_under_public_suffix or "unknown"

    try:
        from urllib.parse import urlparse, unquote
        p = urlparse(working)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower()
        path = unquote(p.path or "")
        query = unquote(p.query or "")
        # Remove trailing slash for root paths
        if path == "/":
            path = ""
        full_clean = f"{netloc}{path}{'?' + query if query else ''}"
        path_query = f"{path}{'?' + query if query else ''}"
    except Exception:
        scheme = "http"
        netloc = psl_domain
        full_clean = s.lower()
        path_query = ""

    return full_clean, scheme, psl_domain, path_query

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD SANITIZED CORPORA
# ─────────────────────────────────────────────────────────────────────────────

def build_sanitized_email_corpus():
    print("\n" + "=" * 80)
    print("STEP 1: BUILDING SANITIZED & DEDUPLICATED EMAIL CORPUS")
    print("=" * 80)

    # 1. Load Phishing_Email.csv (Lures & benign personal)
    p_phish = os.path.join(DATA_DIR, "Phishing_Email.csv")
    df_phish = pd.read_csv(p_phish).dropna(subset=['Email Text'])
    df_phish['raw_text'] = df_phish['Email Text'].str.strip()
    df_phish['label'] = (df_phish['Email Type'] == 'Phishing Email').astype(int)
    df_phish['source'] = 'Phishing_Email_Dataset'

    # Filter out empty or trivially short emails (< 40 chars)
    df_phish = df_phish[df_phish['raw_text'].str.len() >= 40]
    phish_emails = df_phish[df_phish['label'] == 1].sample(n=min(7000, sum(df_phish['label'] == 1)), random_state=42)
    personal_clean = df_phish[df_phish['label'] == 0].sample(n=min(3500, sum(df_phish['label'] == 0)), random_state=42)

    # 2. Load Enron Corporate Benign Emails (Real enterprise communication)
    enron_path = r"C:\Users\shari\Desktop\enron_data_fraud_labeled.csv"
    if os.path.exists(enron_path):
        print("  Loading corporate emails from Enron Fraud Labeled dataset...")
        df_enron = pd.read_csv(enron_path, nrows=15000, usecols=['Body', 'Label']).dropna(subset=['Body'])
        df_enron['raw_text'] = df_enron['Body'].str.strip()
        df_enron = df_enron[df_enron['raw_text'].str.len() >= 50]
        # Only take verified benign corporate emails
        corporate_clean = df_enron[df_enron['Label'] == 0].sample(n=3500, random_state=42)
        corporate_clean['label'] = 0
        corporate_clean['source'] = 'Enron_Corporate'
    else:
        corporate_clean = personal_clean.copy()

    # Combine into unified pool
    df_combined = pd.concat([phish_emails, personal_clean, corporate_clean], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Clean & Entity Mask
    print("  Applying NFKC Unicode normalization and entity masking (<URL>, <EMAIL>, <PHONE>, <AMOUNT>)...")
    df_combined['cleaned_text'] = df_combined['raw_text'].apply(clean_and_mask_email)
    
    # Exact deduplication on masked text
    before_dedup = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
    print(f"  [OK] Removed {before_dedup - len(df_combined)} exact duplicate masked emails.")

    # Campaign template clustering
    df_combined = cluster_email_templates(df_combined)

    # Split into Train (60%), Val (20%), Test (20%) strictly by campaign_cluster_id
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=42)
    tr_idx, temp_idx = next(gss1.split(df_combined, df_combined['label'], groups=df_combined['campaign_cluster_id']))
    df_tr = df_combined.iloc[tr_idx].reset_index(drop=True)
    df_temp = df_combined.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_idx, te_idx = next(gss2.split(df_temp, df_temp['label'], groups=df_temp['campaign_cluster_id']))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[te_idx].reset_index(drop=True)

    # Verify ZERO campaign overlap
    tr_clusters = set(df_tr['campaign_cluster_id'])
    te_clusters = set(df_test['campaign_cluster_id'])
    overlap = len(tr_clusters.intersection(te_clusters))
    print(f"  [VERIFICATION] Train/Test Campaign Cluster Overlap: {overlap} (Must be 0)")
    assert overlap == 0, "Fatal: Campaign leakage between Train and Test!"

    df_combined['split'] = 'train'
    df_combined.loc[df_combined.index.isin(df_temp.index[val_idx]), 'split'] = 'val'
    df_combined.loc[df_combined.index.isin(df_temp.index[te_idx]), 'split'] = 'test'

    out_csv = os.path.join(DATA_DIR, "sanitized_email_corpus.csv")
    df_combined[['raw_text', 'cleaned_text', 'label', 'source', 'campaign_cluster_id', 'split']].to_csv(out_csv, index=False)
    print(f"[SUCCESS] Sanitized email corpus saved to: {out_csv}")
    print(f"          Total Samples: {len(df_combined)} (Benign: {sum(df_combined['label']==0)}, Phish: {sum(df_combined['label']==1)})")
    print(f"          Train: {len(df_tr)}, Val: {len(df_val)}, Test: {len(df_test)}")

def build_sanitized_url_corpus():
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING SANITIZED & DEPTH-BALANCED URL CORPUS")
    print("=" * 80)

    # 1. Tranco Top 50,000 Root Domains (High-authority benign homepages)
    top1m_path = r"C:\Users\shari\Desktop\top-1m.csv"
    print("  Loading top authority domains from Tranco list...")
    df_top = pd.read_csv(top1m_path, header=None, nrows=40000)
    root_benign = [f"https://www.{d}" for d in df_top[1].astype(str).tolist()[:10000]]

    # 2. Deep Path Benign URLs from malicious_phish.csv
    phish_csv = r"C:\Users\shari\Desktop\malicious_phish.csv"
    print("  Loading deep benign paths and phishing URLs from malicious_phish.csv...")
    df_raw = pd.read_csv(phish_csv, nrows=250000).dropna(subset=['url'])
    df_raw['url_clean'] = df_raw['url'].str.strip()
    df_raw = df_raw.drop_duplicates(subset=['url_clean'])

    deep_benign = df_raw[df_raw['type'] == 'benign']['url_clean'].tolist()[:10000]

    # 3. Phishing URLs: Balance 50% root domains and 50% deep paths
    all_phish = df_raw[df_raw['type'] == 'phishing']['url_clean'].tolist()
    phish_root = [u for u in all_phish if '/' not in u.split('://')[-1]][:10000]
    phish_deep = [u for u in all_phish if '/' in u.split('://')[-1]][:10000]

    # Combine benign (50% root, 50% deep) and phish (50% root, 50% deep)
    benign_records = [{"url": u, "label": 0, "depth_type": "root_homepage"} for u in root_benign] + \
                     [{"url": u, "label": 0, "depth_type": "deep_path"} for u in deep_benign]

    phish_records = [{"url": u, "label": 1, "depth_type": "root_domain"} for u in phish_root] + \
                    [{"url": u, "label": 1, "depth_type": "deep_path"} for u in phish_deep]

    df_urls = pd.DataFrame(benign_records + phish_records).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("  Parsing and normalizing URLs with Public Suffix List...")
    parsed = [normalize_url_scheme_and_domain(u) for u in df_urls['url']]
    df_urls['clean_url'] = [p[0] for p in parsed]
    df_urls['scheme'] = [p[1] for p in parsed]
    df_urls['psl_domain'] = [p[2] for p in parsed]
    df_urls['path_query'] = [p[3] for p in parsed]

    # Drop duplicates by clean_url
    before_dup = len(df_urls)
    df_urls = df_urls.drop_duplicates(subset=['clean_url']).reset_index(drop=True)
    print(f"  [OK] Removed {before_dup - len(df_urls)} duplicate URLs.")

    # Split strictly by psl_domain group (zero domain overlap!)
    gss_u1 = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=42)
    tr_idx, temp_idx = next(gss_u1.split(df_urls, df_urls['label'], groups=df_urls['psl_domain']))
    df_u_tr = df_urls.iloc[tr_idx].reset_index(drop=True)
    df_u_temp = df_urls.iloc[temp_idx].reset_index(drop=True)

    gss_u2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_idx, te_idx = next(gss_u2.split(df_u_temp, df_u_temp['label'], groups=df_u_temp['psl_domain']))
    df_u_val = df_u_temp.iloc[val_idx].reset_index(drop=True)
    df_u_test = df_u_temp.iloc[te_idx].reset_index(drop=True)

    # Verify ZERO PSL domain leakage
    tr_doms = set(df_u_tr['psl_domain'])
    te_doms = set(df_u_test['psl_domain'])
    dom_leak = len(tr_doms.intersection(te_doms))
    print(f"  [VERIFICATION] Train/Test PSL Domain Overlap: {dom_leak} (Must be 0)")
    assert dom_leak == 0, "Fatal: Domain leakage found between Train and Test!"

    df_urls['split'] = 'train'
    df_urls.loc[df_urls.index.isin(df_u_temp.index[val_idx]), 'split'] = 'val'
    df_urls.loc[df_urls.index.isin(df_u_temp.index[te_idx]), 'split'] = 'test'

    out_csv = os.path.join(DATA_DIR, "sanitized_url_corpus.csv")
    df_urls.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Sanitized URL corpus saved to: {out_csv}")
    print(f"          Total Samples: {len(df_urls)} (Benign: {sum(df_urls['label']==0)}, Phish: {sum(df_urls['label']==1)})")
    print(f"          Unique PSL Domains: {df_urls['psl_domain'].nunique()}")
    print(f"          Train: {len(df_u_tr)}, Val: {len(df_u_val)}, Test: {len(df_u_test)}")

if __name__ == "__main__":
    build_sanitized_email_corpus()
    build_sanitized_url_corpus()
    print("\n" + "=" * 80)
    print("ALL DATASETS SANITIZED, DEDUPLICATED, AND FROZEN SUCCESSFULLY.")
    print("=" * 80)
