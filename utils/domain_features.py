"""
PhishGuard AI — Domain Structural, Lexical & Behavioral Feature Extractor (Phase 3)
Extracts 16 active features for standalone domain and URL-host reputation.
Optimized for pure-CPU sub-millisecond execution latency.
"""
import re
import math
from typing import Dict, Any, List, Union
import numpy as np

# Registrar abuse risk per TLD (synced with training pipeline)
TLD_RISK: Dict[str, float] = {
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

KNOWN_TLD_TIERS: Dict[str, int] = {
    # Tier 0 — established / trusted
    "com": 0, "org": 0, "net": 0, "edu": 0, "gov": 0, "io": 0, "co": 0, "app": 0, "dev": 0,
    "uk": 0, "de": 0, "fr": 0, "au": 0, "ca": 0, "pk": 0, "in": 0, "jp": 0, "nl": 0, "se": 0,
    # Tier 1 — moderate risk
    "info": 1, "biz": 1, "us": 1, "ru": 1, "cn": 1, "br": 1, "kr": 1, "mx": 1, "ng": 1,
    # Tier 2 — elevated abuse rate
    "xyz": 2, "top": 2, "icu": 2, "club": 2, "online": 2, "site": 2, "link": 2, "live": 2,
    "stream": 2, "review": 2, "win": 2, "bid": 2, "trade": 2, "racing": 2, "party": 2,
    # Tier 3 — free / high-abuse registrars
    "tk": 3, "ml": 3, "ga": 3, "gq": 3, "cf": 3, "pw": 3, "cc": 3, "click": 3, "download": 3,
    "zip": 3, "webcam": 3, "accountant": 3, "loan": 3, "science": 3, "faith": 3,
}

KNOWN_MULTI_CCTLDS = {
    "co.uk", "co.nz", "co.za", "co.in", "co.jp", "com.au", "net.au", "org.au",
    "com.br", "com.mx", "com.pk", "net.pk", "org.pk", "edu.pk", "gov.pk",
    "com.cn", "net.cn", "org.cn", "com.tw", "com.hk", "com.sg", "com.my",
    "ne.jp", "ac.uk", "gov.uk", "police.uk", "org.uk"
}

# Precompiled brand & security keywords
RE_BRAND_KEYWORDS = re.compile(
    r"\b(?:paypal|apple|microsoft|google|facebook|amazon|netflix|"
    r"instagram|twitter|linkedin|dropbox|yahoo|irs|wellsfargo|"
    r"chase|bankofamerica|coinbase|binance|blockchain|metamask|github|spotify)\b",
    re.I
)

RE_OFFICIAL_DOMAINS = re.compile(
    r"^(?:paypal|apple|microsoft|google|facebook|amazon|netflix|"
    r"instagram|twitter|linkedin|dropbox|yahoo|irs|chase|wellsfargo|"
    r"coinbase|binance|blockchain|metamask|github|spotify)\.(?:com|org|net|co|gov|edu)(?:\.[a-z]{2})?$",
    re.I
)

RE_SECURITY_KEYWORDS = re.compile(
    r"(?:^|[-_.])(?:login|signin|sign-in|verify|verification|secure|security|"
    r"account|update|confirm|banking|wallet|token|auth|portal|recover|"
    r"reset|support|helpdesk|service|alert|unlock|validate)(?:$|[-_.])",
    re.I
)

# English bigram frequency table — rare bigrams indicate DGA / random strings
# Normalized relative frequency from standard English corpora (0.01 - 1.0)
_BIGRAM_FREQ = {
    'th': 1.0, 'he': 0.95, 'in': 0.90, 'er': 0.88, 'an': 0.85, 're': 0.82, 'on': 0.80,
    'at': 0.78, 'en': 0.77, 'nd': 0.76, 'ti': 0.73, 'es': 0.72, 'or': 0.71, 'te': 0.70,
    'of': 0.68, 'ed': 0.67, 'is': 0.65, 'it': 0.64, 'al': 0.63, 'ar': 0.62, 'st': 0.61,
    'to': 0.60, 'nt': 0.59, 'ng': 0.57, 'se': 0.56, 'ha': 0.55, 'as': 0.54, 'ou': 0.53,
    'io': 0.52, 'le': 0.51, 've': 0.50, 'co': 0.49, 'me': 0.48, 'de': 0.47, 'hi': 0.46,
    'ri': 0.45, 'ro': 0.44, 'ic': 0.42, 'ne': 0.41, 'ea': 0.40, 'ra': 0.39, 'ce': 0.38,
    'li': 0.37, 'ch': 0.36, 'll': 0.35, 'be': 0.34, 'ma': 0.33, 'si': 0.32, 'om': 0.31,
    'ur': 0.30, 'ca': 0.29, 'el': 0.28, 'ta': 0.27, 'so': 0.26, 'it': 0.25, 'pa': 0.24,
    'ns': 0.23, 'rt': 0.22, 'no': 0.21, 'ss': 0.20, 'di': 0.18, 'ge': 0.17, 'tr': 0.16,
    'fo': 0.15, 'rs': 0.14, 'un': 0.13, 'pe': 0.12, 'ad': 0.11, 'pl': 0.10, 'sh': 0.09,
    'ck': 0.08, 'go': 0.07, 'ai': 0.06, 'sc': 0.05, 'pr': 0.04, 'ct': 0.03, 'wh': 0.02,
}

DOMAIN_FEATURE_NAMES = [
    "domain_length",
    "domain_entropy",
    "domain_digit_count",
    "domain_hyphen_count",
    "domain_dot_count",
    "unique_char_ratio",
    "sld_length",
    "sld_digit_ratio",
    "vowel_ratio",
    "consonant_cluster",
    "dga_score",
    "brand_impersonation",
    "brand_in_subdomain",
    "security_keyword",
    "tld_risk_score",
    "known_tld_score",
]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = float(len(s))
    return -sum((count / n) * math.log2(count / n) for count in freq.values())


def dga_score(sld: str) -> float:
    """Lower score = more DGA-like (random string). Range 0.0 - 1.0."""
    if len(sld) < 4:
        return 0.5
    letters_only = re.sub(r"[^a-z]", "", sld.lower())
    if len(letters_only) < 3:
        return 0.3
    bigrams = [letters_only[i:i + 2] for i in range(len(letters_only) - 1)]
    if not bigrams:
        return 0.2
    return round(sum(_BIGRAM_FREQ.get(bg, 0.01) for bg in bigrams) / len(bigrams), 4)


def vowel_ratio(s: str) -> float:
    """Ratio of vowels in alphabetic characters."""
    letters = [c for c in s.lower() if c.isalpha()]
    if not letters:
        return 0.0
    vowels = sum(c in "aeiou" for c in letters)
    return round(vowels / len(letters), 4)


def max_consonant_run(s: str) -> int:
    """Longest consecutive consonant run (e.g. DGA domains xkzptv)."""
    s = re.sub(r"[^a-z]", "", s.lower())
    consonants = set("bcdfghjklmnpqrstvwxyz")
    max_run = cur = 0
    for c in s:
        if c in consonants:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def extract_domain_features(domain_or_url: str) -> Dict[str, Any]:
    """
    Extracts the 16 active lexical, structural, and behavioral domain features.
    Accepts full URL or bare domain (e.g. 'https://learn.microsoft.com/path' or 'nyxlzy.com').
    Guaranteed backwards compatible with services/scan_engine.py.
    """
    if not isinstance(domain_or_url, str):
        domain_or_url = ""
    raw = domain_or_url.strip().lower()

    # Normalize protocol and path
    clean = re.sub(r"^https?://", "", raw)
    clean = re.sub(r"^www\.", "", clean)
    clean_domain = clean.split("/")[0].split(":")[0].split("?")[0].strip()

    parts = clean_domain.split(".")
    if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}" in KNOWN_MULTI_CCTLDS:
        tld = f"{parts[-2]}.{parts[-1]}"
        sld = parts[-3]
        sub = ".".join(parts[:-3])
    elif len(parts) >= 2:
        tld = parts[-1]
        sld = parts[-2]
        sub = ".".join(parts[:-2])
    else:
        tld = clean_domain
        sld = clean_domain
        sub = ""

    root = f"{sld}.{tld}" if sld and tld else clean_domain
    is_official = bool(RE_OFFICIAL_DOMAINS.match(root))

    has_brand = int(bool(RE_BRAND_KEYWORDS.search(clean_domain)) and not is_official)
    brand_in_sub = int(bool(RE_BRAND_KEYWORDS.search(sub)) and not is_official)
    sec_keyword = int(bool(RE_SECURITY_KEYWORDS.search(sld)))

    base_tld = tld.split(".")[-1] if "." in tld else tld
    tld_risk = TLD_RISK.get(tld, TLD_RISK.get(base_tld, 0.2))

    return {
        "domain_length": len(clean_domain),
        "domain_entropy": round(shannon_entropy(clean_domain), 4) if clean_domain else 0.0,
        "domain_digit_count": sum(c.isdigit() for c in clean_domain),
        "domain_hyphen_count": clean_domain.count("-"),
        "domain_dot_count": clean_domain.count("."),
        "unique_char_ratio": round(len(set(root)) / max(len(root), 1), 4),
        "sld_length": len(sld),
        "sld_digit_ratio": round(sum(c.isdigit() for c in sld) / max(len(sld), 1), 4),
        "vowel_ratio": vowel_ratio(sld),
        "consonant_cluster": max_consonant_run(sld),
        "dga_score": dga_score(sld),
        "brand_impersonation": has_brand,
        "brand_in_subdomain": brand_in_sub,
        "security_keyword": sec_keyword,
        "tld_risk_score": round(tld_risk, 3),
        "known_tld_score": KNOWN_TLD_TIERS.get(tld, KNOWN_TLD_TIERS.get(base_tld, 2)),
    }


def extract_domain_features_single(domain: str) -> np.ndarray:
    """High-speed single-domain NumPy extractor returning shape (1, 16) float32 array."""
    d = extract_domain_features(domain)
    arr = np.fromiter((d[k] for k in DOMAIN_FEATURE_NAMES), dtype=np.float32, count=16)
    return arr.reshape(1, 16)


def extract_domain_features_batch(domains: List[str]) -> np.ndarray:
    """High-speed batch extractor returning shape (N, 16) float32 array."""
    n = len(domains)
    matrix = np.empty((n, 16), dtype=np.float32)
    for i, d in enumerate(domains):
        feat = extract_domain_features(d)
        matrix[i] = [feat[k] for k in DOMAIN_FEATURE_NAMES]
    return matrix
