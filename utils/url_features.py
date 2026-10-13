"""
PhishGuard AI — URL Structural & Lexical Feature Extractor (Phase 2)
Extracts 33 dense behavioral, lexical, structural, and obfuscation features.
Designed for pure-CPU sub-millisecond extraction latency with precompiled regexes.
"""
import re
import math
from urllib.parse import urlparse
from typing import List, Union, Dict, Any
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

# Precompiled regexes for maximum throughput
RE_SHORTENERS = re.compile(
    r"\b(?:bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|"
    r"buff\.ly|shorte\.st|adf\.ly|cutt\.ly|rb\.gy|short\.io)\b",
    re.I
)

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

RE_SECURITY_WORDS = re.compile(
    r'(?:^|[._\-/])(login|signin|sign-in|verify|verification|secure|security|'
    r'account|update|confirm|banking|wallet|token|auth|portal|recover|reset|'
    r'support|admin|passcode|credential|webscr|ebayisapi|cmd=_login|ident|'
    r'log-in|sign_in|valida|validation|factura|billing|securite)(?:$|[._\-/])',
    re.I
)

RE_DDNS_DOMAINS = re.compile(
    r'\.(?:duckdns\.org|ngrok\.io|000webhostapp\.com|weebly\.com|wixsite\.com|'
    r'firebaseapp\.com|web\.app|workers\.dev|page\.link|glitch\.me|surge\.sh|'
    r'zapto\.org|servebeer\.com|bounceme\.net|ddns\.net|hopto\.org)\b',
    re.I
)

RE_IP_ADDR = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

KNOWN_MULTI_CCTLDS = {
    'co.uk', 'co.nz', 'co.za', 'co.in', 'co.jp', 'com.au', 'net.au', 'org.au',
    'com.br', 'com.mx', 'com.pk', 'net.pk', 'org.pk', 'edu.pk', 'gov.pk',
    'com.cn', 'net.cn', 'org.cn', 'com.tw', 'com.hk', 'com.sg', 'com.my',
    'ne.jp', 'ac.uk', 'gov.uk', 'police.uk', 'org.uk'
}

URL_FEATURE_NAMES = [
    "url_length",
    "entropy",
    "sld_entropy",
    "digit_count",
    "digit_ratio",
    "special_char_count",
    "hyphen_count",
    "domain_hyphen_count",
    "at_count",
    "dot_count",
    "path_length",
    "query_length",
    "path_depth",
    "query_param_count",
    "subdomain_count",
    "subdomain_length",
    "domain_length",
    "is_https",
    "has_ip",
    "has_shortener",
    "has_ddns",
    "tld_risk_score",
    "has_suspicious_tld",
    "has_brand_in_url",
    "has_brand_in_sub",
    "has_brand_in_path",
    "has_sec_in_path",
    "has_sec_in_sub",
    "has_sec_in_sld",
    "has_hex_encoding",
    "has_double_slash",
    "char_continuation",
    "unique_char_ratio",
]

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = float(len(s))
    return -sum((count / n) * math.log2(count / n) for count in freq.values())

def char_continuation_rate(s: str) -> float:
    if not s:
        return 0.0
    max_run, cur = 1, 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        max_run = max(max_run, cur)
    return max_run / len(s)

def unique_char_ratio(s: str) -> float:
    return len(set(s)) / len(s) if s else 0.0

def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extracts 33 lexical, structural, and obfuscation URL features as a named dictionary.
    Guaranteed backwards compatible with existing scan_engine / url_scanner APIs.
    """
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

    if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}" in KNOWN_MULTI_CCTLDS:
        tld = f"{parts[-2]}.{parts[-1]}"
        sld = parts[-3]
        subdomain = ".".join(parts[:-3])
    elif len(parts) >= 2:
        tld = parts[-1]
        sld = parts[-2]
        subdomain = ".".join(parts[:-2])
    else:
        tld = clean_domain
        sld = clean_domain
        subdomain = ""

    norm = re.sub(r"^https?://", "", url.lower())
    norm = re.sub(r"^www\.", "", norm)
    dc = sum(c.isdigit() for c in norm)
    tl = len(norm) or 1

    registered_domain = f"{sld}.{tld}" if sld and tld else clean_domain
    is_official = bool(RE_OFFICIAL_DOMAINS.match(registered_domain))
    has_brand_domain = int(bool(RE_BRAND_KEYWORDS.search(clean_domain)) and not is_official)
    has_brand_sub = int(bool(RE_BRAND_KEYWORDS.search(subdomain)) and not is_official)
    has_brand_path = int(bool(RE_BRAND_KEYWORDS.search(path)) and not is_official)

    has_sec_path = int(bool(RE_SECURITY_WORDS.search(path)))
    has_sec_sub = int(bool(RE_SECURITY_WORDS.search(subdomain)) and not is_official)
    has_sec_sld = int(bool(RE_SECURITY_WORDS.search(sld)))

    has_ddns = int(bool(RE_DDNS_DOMAINS.search(clean_domain)))

    base_tld = tld.split(".")[-1] if "." in tld else tld
    tld_risk = TLD_RISK.get(tld, TLD_RISK.get(base_tld, 0.2))

    # Authentic registered official domains with standard navigation paths (no security/harvesting words)
    # represent benign informational content, not phishing evasion payloads.
    path_len = len(path)
    path_dep = path.count("/")
    if is_official and not has_sec_path:
        path_len = 0
        path_dep = 0

    return {
        "url_length":         len(norm),
        "entropy":            round(shannon_entropy(norm), 4),
        "sld_entropy":        round(shannon_entropy(sld), 4),
        "digit_count":        dc,
        "digit_ratio":        round(dc / tl, 4),
        "special_char_count": sum(not c.isalnum() for c in norm),
        "hyphen_count":       norm.count("-"),
        "domain_hyphen_count": clean_domain.count("-"),
        "at_count":           norm.count("@"),
        "dot_count":          clean_domain.count("."),
        "path_length":        path_len,
        "query_length":       len(query),
        "path_depth":         path_dep,
        "query_param_count":  query.count("&") + (1 if query else 0),
        "subdomain_count":    subdomain.count(".") + 1 if subdomain else 0,
        "subdomain_length":   len(subdomain),
        "domain_length":      len(clean_domain),
        "is_https":           int(scheme == "https"),
        "has_ip":             int(bool(RE_IP_ADDR.search(clean_domain))),
        "has_shortener":      int(bool(RE_SHORTENERS.search(norm))),
        "has_ddns":           has_ddns,
        "tld_risk_score":     round(tld_risk, 3),
        "has_suspicious_tld": int(tld_risk >= 0.75),
        "has_brand_in_url":   has_brand_domain,
        "has_brand_in_sub":   has_brand_sub,
        "has_brand_in_path":  has_brand_path,
        "has_sec_in_path":    has_sec_path,
        "has_sec_in_sub":     has_sec_sub,
        "has_sec_in_sld":     has_sec_sld,
        "has_hex_encoding":   int("%" in url),
        "has_double_slash":   int("//" in path),
        "char_continuation":  round(char_continuation_rate(norm), 4),
        "unique_char_ratio":  round(unique_char_ratio(norm[:50]), 4),
    }

def extract_url_features_single(url: str) -> np.ndarray:
    """
    High-speed NumPy extractor returning shape (1, 33) float32 array.
    Avoids dictionary/DataFrame allocation overhead for sub-millisecond execution.
    """
    d = extract_url_features(url)
    arr = np.fromiter((d[k] for k in URL_FEATURE_NAMES), dtype=np.float32, count=33)
    return arr.reshape(1, 33)

def extract_url_features_batch(urls: List[str]) -> np.ndarray:
    """
    Extracts 33 dense features across a list of URL strings.
    Returns np.ndarray of shape (N, 33) with dtype float32.
    """
    n = len(urls)
    matrix = np.empty((n, 33), dtype=np.float32)
    for i, u in enumerate(urls):
        d = extract_url_features(u)
        matrix[i] = [d[k] for k in URL_FEATURE_NAMES]
    return matrix
