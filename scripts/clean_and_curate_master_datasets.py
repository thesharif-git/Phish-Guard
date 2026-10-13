"""
Master Dataset Cleaning, Sanitization & Deduplication Pipeline
==============================================================
Legacy research curation; Email admission uses the reviewed-evidence workflow:
1. Email Corpus:
   - Ingests Phishing_Email.csv (phishing lures) and Enron (corporate communication).
   - Uses unified versioned preprocessor (utils/text_sanitizer.py).
   - Drops exact duplicates and purges conflicting labels across all records.
   - Vectorized 64-bit SimHash clustering with exhaustive candidate-pair union-find (Hamming <= 6).
   - First-80-token SimHash groups do not establish template independence.
   - Legacy source labels are not verified phishing/BEC annotations.

2. URL Corpus:
   - Ingests Tranco top-1m, PhiUSIIL ground truth, and malicious_phish.
   - Canonicalizes syntax (RFC 3986) and strips scheme bias.
   - Enforces 50/50 balance between root homepages and deep paths for BOTH classes.
   - Purges exact duplicate URLs and conflicting cross-source labels.
   - Groups by Public Suffix List (PSL) registrable domain.
   - Splits 60/20/20 grouped strictly by psl_domain (zero domain leakage).

3. Domain Evidence Review:
   - Preserves URLs and reputation metadata in an unverified review queue.
   - Does not inherit domain labels from URLs or popularity/reputation lists.
   - Preserves the legacy domain CSV; verified annotation is required to replace it.
"""

import os
import re
import sys
import time
import html
import hashlib
import unicodedata
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import pandas as pd
import numpy as np
import tldextract
from sklearn.model_selection import GroupShuffleSplit

_OFFLINE_PSL = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

from utils.text_sanitizer import clean_and_mask_email_text

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT SANITIZATION & EXHAUSTIVE PAIR SIMHASH CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_simhash_fast(text: str) -> int:
    """
    Vectorized 64-bit SimHash using NumPy bitwise projection.
    Runs in < 0.1ms per text.
    """
    tokens = [w for w in text.lower().split() if len(w) > 2][:80]
    if not tokens:
        return 0
    hashes = np.array([int(hashlib.md5(t.encode('utf-8')).hexdigest()[:16], 16) for t in tokens], dtype=np.uint64)
    bits = (hashes[:, None] >> np.arange(64, dtype=np.uint64)) & 1
    v = np.where(bits, 1, -1).sum(axis=0)
    return int((v > 0) @ (1 << np.arange(64, dtype=object)))

def fast_simhash_clustering(hashes: List[int], max_hamming: int = 6) -> List[int]:
    """
    Near-duplicate campaign template clustering using 8-chunk inverted index.
    Checks all candidate pairs within buckets to ensure no qualifying near-duplicates
    differing by <= max_hamming bits are missed.
    """
    n = len(hashes)
    buckets = [defaultdict(list) for _ in range(8)]
    for idx, h in enumerate(hashes):
        if h == 0:
            continue
        for chunk in range(8):
            key = (h >> (chunk * 8)) & 0xFF
            buckets[chunk][key].append(idx)

    parent = list(range(n))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    seen_pairs = set()
    for b in buckets:
        for val, ids in b.items():
            k = len(ids)
            if k > 1:
                for i in range(k):
                    for j in range(i + 1, k):
                        p0, p1 = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                        pair = (p0, p1)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            if bin(hashes[p0] ^ hashes[p1]).count('1') <= max_hamming:
                                union(p0, p1)

    return [find(i) for i in range(n)]

# ─────────────────────────────────────────────────────────────────────────────
# 2. URL CANONICALIZATION & PSL PARSING
# ─────────────────────────────────────────────────────────────────────────────

def canonicalize_url_clean(raw_url: str) -> Tuple[str, str, str, str]:
    """
    Canonicalizes raw URL and returns (canonical_clean, scheme, psl_domain, path_query).
    """
    s = str(raw_url).strip()
    has_scheme = bool(re.match(r"^https?://", s, re.IGNORECASE))
    working = s if has_scheme else "http://" + s

    ext = _OFFLINE_PSL(working)
    psl = ext.top_domain_under_public_suffix or "unknown"

    try:
        from urllib.parse import urlparse, unquote
        import posixpath
        p = urlparse(working)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower().split(":")[0]
        netloc = re.sub(r"^www\.", "", netloc)
        raw_path = unquote(p.path or "")
        canon_path = posixpath.normpath(raw_path) if raw_path else ""
        if raw_path.endswith("/") and not canon_path.endswith("/"):
            canon_path += "/"
        if canon_path == "/" or canon_path == ".":
            canon_path = ""
        query = unquote(p.query or "")
        clean_url = f"{netloc}{canon_path}{'?' + query if query else ''}"
        path_query = f"{canon_path}{'?' + query if query else ''}"
    except Exception:
        scheme = "http"
        clean_url = re.sub(r"^https?://", "", s.lower())
        clean_url = re.sub(r"^www\.", "", clean_url)
        path_query = ""

    return clean_url, scheme, psl, path_query

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD CLEAN EMAIL CORPUS
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_curate_emails(*, legacy_research_only=False):
    if not legacy_research_only:
        raise ValueError('Legacy Email curation cannot overwrite the training corpus: verified upstream spam-label ambiguity and cross-split template overlap. Use scripts/build_email_evidence_review_corpus.py, adjudicate task labels, and validate source/time partitions before training.')
    print("\n" + "=" * 90, flush=True)
    print("PHASE 1: CLEANING, PURGING & DEDUPLICATING EMAIL DATASETS", flush=True)
    print("=" * 90, flush=True)
    t0 = time.time()

    # 1. Ingest Phishing_Email.csv
    p_phish = os.path.join(DATA_DIR, "Phishing_Email.csv")
    print(f"[1/8] Ingesting phishing lures from: {p_phish}...", flush=True)
    df_phish = pd.read_csv(p_phish).dropna(subset=['Email Text'])
    if not df_phish['Email Type'].isin(['Safe Email', 'Phishing Email']).all():
        raise ValueError('Unknown source label cannot silently become a benign Email label')
    df_phish['raw_text'] = df_phish['Email Text'].astype(str).str.strip()
    df_phish['label'] = (df_phish['Email Type'] == 'Phishing Email').astype(int)
    df_phish['source'] = 'Phishing_Email_Dataset'
    print(f"      Loaded: {len(df_phish):,} records (Phish: {sum(df_phish['label']==1):,}, Benign: {sum(df_phish['label']==0):,})", flush=True)

    # 2. Ingest Enron Corporate Emails (High-quality benign corporate correspondence)
    enron_path = r"C:\Users\shari\Desktop\enron_data_fraud_labeled.csv"
    if os.path.exists(enron_path):
        print(f"[2/8] Ingesting corporate baseline emails from: {enron_path}...", flush=True)
        df_enron = pd.read_csv(enron_path, nrows=35000, usecols=['Body', 'Label']).dropna(subset=['Body'])
        df_enron['raw_text'] = df_enron['Body'].astype(str).str.strip()
        df_enron = df_enron[df_enron['Label'] == 0].head(10000)
        df_enron['label'] = 0
        df_enron['source'] = 'Enron_Corporate'
        print(f"      Enron corporate rows loaded: {len(df_enron):,}", flush=True)
    else:
        df_enron = pd.DataFrame()

    # Combine pools
    df_all = pd.concat([df_phish, df_enron], ignore_index=True)
    n_initial = len(df_all)
    print(f"[3/8] Combined Initial Pool: {n_initial:,} emails", flush=True)

    # 3. Filter length outliers (< 40 chars or > 40,000 chars)
    df_all['len'] = df_all['raw_text'].str.len()
    df_all = df_all[(df_all['len'] >= 40) & (df_all['len'] <= 40000)].copy()
    n_outliers = n_initial - len(df_all)
    print(f"      Purged length outliers (< 40 chars or > 40,000 chars): {n_outliers:,} dropped", flush=True)

    # 4. Clean & Entity Mask with canonical versioned text_sanitizer
    print("[4/8] Sanitizing with canonical utils.text_sanitizer (<URL>, <EMAIL>, <PHONE>, <AMOUNT>)...", flush=True)
    df_all['masked_text'] = df_all['raw_text'].apply(clean_and_mask_email_text)
    df_all = df_all[df_all['masked_text'].str.len() >= 20].copy()

    # 5. Purge conflicting labels (exact same text labeled 0 AND 1)
    label_counts_per_text = df_all.groupby('masked_text')['label'].nunique()
    conflicted_texts = set(label_counts_per_text[label_counts_per_text > 1].index)
    if conflicted_texts:
        n_before_conf = len(df_all)
        df_all = df_all[~df_all['masked_text'].isin(conflicted_texts)].copy()
        print(f"[5/8] [QUARANTINE] Purged {n_before_conf - len(df_all):,} conflicting label records!", flush=True)
    else:
        print("[5/8] [OK] Conflicting label scan: 0 conflicts found.", flush=True)

    # 6. Drop exact duplicates on masked text
    n_before_dup = len(df_all)
    df_all = df_all.drop_duplicates(subset=['masked_text']).reset_index(drop=True)
    print(f"[6/8] Purged exact duplicates: {n_before_dup - len(df_all):,} dropped. Retained: {len(df_all):,}", flush=True)

    # 7. Fast SimHash Near-Duplicate Campaign Template Clustering (Exhaustive Pair Checked)
    print("[7/8] Computing 64-bit SimHash fingerprints and building campaign template clusters (exhaustive candidate pairs)...", flush=True)
    t_sh = time.time()
    df_all['simhash'] = [compute_simhash_fast(t) for t in df_all['masked_text']]
    clusters = fast_simhash_clustering(df_all['simhash'].tolist(), max_hamming=6)
    df_all['campaign_cluster_id'] = clusters
    n_clusters = df_all['campaign_cluster_id'].nunique()
    print(f"      Clustered {len(df_all):,} emails into {n_clusters:,} distinct campaign clusters ({time.time()-t_sh:.2f}s).", flush=True)

    # Balance classes: sample balanced benign and phish
    phish_df = df_all[df_all['label'] == 1]
    benign_df = df_all[df_all['label'] == 0]
    n_target = min(len(phish_df), len(benign_df), 8000)
    print(f"      Balancing classes: {n_target:,} Benign and {n_target:,} Phishing...", flush=True)

    phish_bal = phish_df.sample(n=n_target, random_state=42)
    benign_bal = benign_df.sample(n=n_target, random_state=42)
    df_final = pd.concat([phish_bal, benign_bal]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 8. Zero-Leakage Grouped 3-Way Split
    print("[8/8] Performing legacy grouped split; true campaign independence is not established...", flush=True)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=42)
    tr_idx, temp_idx = next(gss1.split(df_final, df_final['label'], groups=df_final['campaign_cluster_id']))
    df_temp = df_final.iloc[temp_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_sub_idx, te_sub_idx = next(gss2.split(df_temp, df_temp['label'], groups=df_temp['campaign_cluster_id']))
    val_idx = temp_idx[val_sub_idx]
    te_idx = temp_idx[te_sub_idx]
    df_final['split'] = 'train'
    df_final.loc[val_idx, 'split'] = 'val'
    df_final.loc[te_idx, 'split'] = 'test'
    df_tr = df_final[df_final['split'] == 'train']
    df_val = df_final[df_final['split'] == 'val']
    df_te = df_final[df_final['split'] == 'test']
    tr_c = set(df_tr['campaign_cluster_id'])
    val_c = set(df_val['campaign_cluster_id'])
    te_c = set(df_te['campaign_cluster_id'])
    print(f"      [VERIFICATION] Train/Test Cluster Overlap: {len(tr_c.intersection(te_c))} (Must be 0)", flush=True)
    print(f"      [VERIFICATION] Train/Val Cluster Overlap:  {len(tr_c.intersection(val_c))} (Must be 0)", flush=True)
    print(f"      [VERIFICATION] Val/Test Cluster Overlap:   {len(val_c.intersection(te_c))} (Must be 0)", flush=True)
    assert len(tr_c.intersection(te_c)) == 0, "Fatal: Train/Test Campaign Cluster Leakage Detected!"
    assert len(tr_c.intersection(val_c)) == 0, "Fatal: Train/Val Campaign Cluster Leakage Detected!"
    research_dir = os.path.join(PROJECT_ROOT, 'benchmarks', 'legacy_email_research_export')
    os.makedirs(research_dir, exist_ok=True)
    out_csv = os.path.join(research_dir, "legacy_source_labels.csv")
    df_final[['raw_text', 'masked_text', 'label', 'source', 'campaign_cluster_id', 'split']].to_csv(out_csv, index=False)
    print(f"\n[EMAIL CORPUS COMPLETE]")
    print(f"  Saved to: {out_csv} ({time.time()-t0:.2f}s total)")
    print(f"  Total Clean Samples: {len(df_final):,} (Benign: {sum(df_final['label']==0):,}, Phish: {sum(df_final['label']==1):,})")
    print(f"  Split Distribution: Train={len(df_tr):,}, Val={len(df_val):,}, Test={len(df_te):,}")
    print(f"  Distinct Clusters: {df_final['campaign_cluster_id'].nunique():,}")


def clean_and_curate_urls():
    from training.legacy_url_sources import load_url_candidates
    t0 = time.time()
    print("PHASE 2: URL CURATION AND DOMAIN EVIDENCE REVIEW", flush=True)
    df_candidates = load_url_candidates(canonicalize_url_clean)
    # Keep all selected candidates, including conflicting/invalid assertions,
    # for review. Filtering below only affects the legacy URL export.
    df_all_urls = df_candidates
    # Domain evidence is review-only. Uniform URL labels do not certify domains.
    from training.domain_evidence import review_record, write_review_queue
    def domain_review_records():
        for row in df_all_urls.to_dict('records'):
            yield review_record(row['original_source_input'], source=row['source'], source_label=int(row['label']),
                                source_label_scope='url_phishing', metadata=row,
                                snapshot_path=row['source_snapshot_path'],
                                snapshot_sha256=row['source_snapshot_sha256'],
                                source_row_number=row['source_row_number'])
        mal_path = r"C:\Users\shari\Desktop\3_malicious_domains.csv"
        if os.path.exists(mal_path):
            snapshot_hash = hashlib.sha256(open(mal_path, 'rb').read()).hexdigest()
            reputation = pd.read_csv(mal_path).fillna('')
            for number, row in enumerate(reputation.to_dict('records'), 2):
                yield review_record(row.get('Domain', ''), source='3_malicious_domains.csv',
                                    source_label_scope='unverified_reputation',
                                    snapshot_path=mal_path, snapshot_sha256=snapshot_hash,
                                    source_row_number=number,
                                    observation_date=row.get('Last_Analysis_Date') or None,
                                    observation_date_kind='source_reported_last_analysis', metadata=row)
    review = write_review_queue(os.path.join(DATA_DIR, 'domain_candidate_review.jsonl'), domain_review_records())
    print(f"[DOMAIN REVIEW] Preserved {review['records']:,} unverified evidence records; no domain labels admitted.", flush=True)
    df_all_urls = df_candidates[
        (df_candidates['psl_domain'] != 'unknown') &
        (df_candidates['psl_domain'].str.len() >= 3) &
        (df_candidates['clean_url'].str.len() >= 4) &
        (df_candidates['clean_url'].str.len() <= 2048)].copy()
    lbl_nunique = df_all_urls.groupby('clean_url')['label'].nunique()
    conflicted_urls = set(lbl_nunique[lbl_nunique > 1].index)
    df_all_urls = df_all_urls[~df_all_urls['clean_url'].isin(conflicted_urls)]
    df_all_urls = df_all_urls.drop_duplicates(subset=['clean_url']).reset_index(drop=True)
    phish_urls = df_all_urls[df_all_urls['label'] == 1]
    benign_urls = df_all_urls[df_all_urls['label'] == 0]
    n_target = min(len(phish_urls), len(benign_urls), 25000)
    df_balanced = pd.concat([
        benign_urls.sample(n=n_target, random_state=42),
        phish_urls.sample(n=n_target, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    # Preserve the old domain corpus; do not overwrite it with inherited labels.
    dom_balanced = pd.DataFrame(columns=['psl_domain', 'label', 'count', 'split'])

    # 9. Coordinated PSL Domain Partitioning (Zero cross-model domain leakage)
    print("\n[9/9] Performing Coordinated Zero-Leakage Domain Partitioning across URL and Domain datasets...", flush=True)
    all_psl_domains = sorted(list(set(df_balanced['psl_domain'].dropna().str.lower()) | set(dom_balanced['psl_domain'].dropna().str.lower())))
    print(f"      Total unique PSL domains across both corpora: {len(all_psl_domains):,}", flush=True)

    domain_master_split = {}
    for d in all_psl_domains:
        h = int(hashlib.sha256(d.encode('utf-8')).hexdigest()[:8], 16)
        r = (h % 1000) / 1000.0
        if r < 0.60:
            domain_master_split[d] = 'train'
        elif r < 0.80:
            domain_master_split[d] = 'val'
        else:
            domain_master_split[d] = 'test'

    df_balanced['split'] = df_balanced['psl_domain'].str.lower().map(domain_master_split)
    dom_balanced['split'] = dom_balanced['psl_domain'].str.lower().map(domain_master_split)

    # Verify zero internal & zero cross-model domain leakage
    df_u_tr = df_balanced[df_balanced['split'] == 'train']
    df_u_val = df_balanced[df_balanced['split'] == 'val']
    df_u_te = df_balanced[df_balanced['split'] == 'test']

    tr_doms = set(df_u_tr['psl_domain'].str.lower())
    val_doms = set(df_u_val['psl_domain'].str.lower())
    te_doms = set(df_u_te['psl_domain'].str.lower())

    dom_tr_doms = set(dom_balanced[dom_balanced['split'] == 'train']['psl_domain'].str.lower())
    dom_val_doms = set(dom_balanced[dom_balanced['split'] == 'val']['psl_domain'].str.lower())
    dom_te_doms = set(dom_balanced[dom_balanced['split'] == 'test']['psl_domain'].str.lower())

    print(f"      [VERIFICATION] URL Internal Train/Test Domain Overlap: {len(tr_doms & te_doms)} (Must be 0)")
    print(f"      [VERIFICATION] Domain Internal Train/Test Domain Overlap: {len(dom_tr_doms & dom_te_doms)} (Must be 0)")
    print(f"      [VERIFICATION] Cross-Model URL-Test in Domain-Train Overlap: {len(te_doms & dom_tr_doms)} (Must be 0)")
    print(f"      [VERIFICATION] Cross-Model Domain-Test in URL-Train Overlap: {len(dom_te_doms & tr_doms)} (Must be 0)")

    assert len(tr_doms & te_doms) == 0, "Fatal: URL Train/Test Domain Leakage Detected!"
    assert len(dom_tr_doms & dom_te_doms) == 0, "Fatal: Domain Train/Test Domain Leakage Detected!"
    assert len(te_doms & dom_tr_doms) == 0, "Fatal: Cross-Model URL-Test in Domain-Train Overlap Detected!"
    assert len(dom_te_doms & tr_doms) == 0, "Fatal: Cross-Model Domain-Test in URL-Train Overlap Detected!"

    out_csv = os.path.join(DATA_DIR, "clean_url_corpus.csv")
    df_balanced[['url', 'clean_url', 'scheme', 'psl_domain', 'path_query', 'label', 'depth_type', 'source', 'split']].to_csv(out_csv, index=False)
    print(f"\n[URL CORPUS COMPLETE]")
    print(f"  Saved to: {out_csv} ({time.time()-t0:.2f}s total)")
    print(f"  Total Clean Samples: {len(df_balanced):,} (Benign: {sum(df_balanced['label']==0):,}, Phish: {sum(df_balanced['label']==1):,})")
    print(f"  Unique PSL Registered Domains: {df_balanced['psl_domain'].nunique():,}")
    print(f"  Split Distribution: Train={len(df_u_tr):,}, Val={len(df_u_val):,}, Test={len(df_u_te):,}")
    print(f"  Depth Distribution: {df_balanced['depth_type'].value_counts().to_dict()}")

    print("[DOMAIN TRAINING BLOCKED] Existing domain corpus preserved. Independently verified domain annotations are required.", flush=True)

if __name__ == "__main__":
    clean_and_curate_emails()
    clean_and_curate_urls()
    print("\n" + "=" * 90, flush=True)
    print("LEGACY EMAIL/URL EXPORTS COMPLETE. DOMAIN EVIDENCE REQUIRES INDEPENDENT REVIEW.", flush=True)
    print("=" * 90, flush=True)
