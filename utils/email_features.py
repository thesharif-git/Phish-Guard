"""
PhishGuard AI — Email Structural & Anomaly Feature Extractor (Phase 1)
Extracts 28 dense behavioral, structural, and obfuscation features.
Designed for pure-CPU sub-2ms extraction latency with precompiled regexes.
"""
import re
import math
import unicodedata
from collections import Counter
from typing import List, Union, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Precompiled regexes for maximum throughput
RE_URL = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.I)
RE_IP_URL = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', re.I)
RE_SUSPICIOUS_TLD = re.compile(r'\.(?:xyz|top|click|link|tk|pw|gq|ml|zip|bid|loan|racing|accountant)\b', re.I)
RE_SHORTENER = re.compile(r'\b(?:bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly|cutt\.ly|rb\.gy)\b', re.I)
RE_HTML_TAGS = re.compile(r'<(?:html|body|table|div|tbody|tr|td|span)\b', re.I)
RE_HTML_LINK = re.compile(r'<a\s+[^>]*href', re.I)
RE_BASE64_IMG = re.compile(r'data:image/[^;]+;base64', re.I)
RE_HIDDEN_CSS = re.compile(r'(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0)', re.I)
RE_ALL_CAPS = re.compile(r'\b[A-Z]{3,}\b')
RE_HEX_ENCODED = re.compile(r'%[0-9a-fA-F]{2}')
RE_ZERO_WIDTH = re.compile(r'[\u200b\u200c\u200d\ufeff\xad]')

# Cyber & Social Engineering Lexicons
RE_URGENT = re.compile(
    r'\b(?:urgent|urgently|verify|verification|suspended|suspension|immediate|immediately|'
    r'expires|expired|act\s+now|action\s+required|unauthorized|unusual\s+activity|'
    r'limited\s+time|security\s+alert|account\s+lock|locked\s+account|terminate|terminated)\b',
    re.I
)
RE_FINANCIAL = re.compile(
    r'\b(?:wire\s+transfer|direct\s+deposit|bank\s+account|routing\s+number|'
    r'vendor\s+invoice|overdue\s+invoice|remittance|gift\s+card|apple\s+gift\s+card|'
    r'payroll\s+update|payroll\s+change|cryptocurrency|wallet\s+phrase|seed\s+phrase)\b',
    re.I
)
RE_BRANDS = re.compile(
    r'\b(?:paypal|microsoft|office\s*365|m365|apple|icloud|amazon|google|gmail|'
    r'netflix|chase|wells\s*fargo|bank\s*of\s*america|citibank|docusign|dropbox|coinbase|binance)\b',
    re.I
)

HOMOGLYPH_CHARS = set('аеорсухіјѕАВЕКМНОРСХҮ')  # Cyrillic lookalikes

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = float(len(s))
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def extract_email_features_single(text: str) -> np.ndarray:
    """
    Extracts 28 numerical features from a single raw email string.
    Returns np.ndarray of shape (1, 28) with dtype float32.
    """
    if not isinstance(text, str):
        text = ""
    
    char_len = len(text)
    words = text.split()
    word_count = len(words)
    line_count = text.count('\n') + 1
    avg_word_len = char_len / max(word_count, 1)

    # Punctuation & Style
    excl_count = text.count('!')
    quest_count = text.count('?')
    dollar_count = text.count('$')
    comma_count = text.count(',')
    all_caps_count = len(RE_ALL_CAPS.findall(text))
    upper_chars = sum(1 for c in text if c.isupper())
    upper_ratio = upper_chars / max(char_len, 1)

    # URL Profile
    urls = RE_URL.findall(text)
    url_count = len(urls)
    has_ip = 1.0 if RE_IP_URL.search(text) else 0.0
    has_susp_tld = 1.0 if RE_SUSPICIOUS_TLD.search(text) else 0.0
    has_short = 1.0 if RE_SHORTENER.search(text) else 0.0
    html_links = len(RE_HTML_LINK.findall(text))
    url_chars_len = sum(len(u) for u in urls)
    link_to_text_ratio = url_chars_len / max(char_len, 1)

    # MIME & Formatting
    has_html_flag = 1.0 if RE_HTML_TAGS.search(text) else 0.0
    has_base64_flag = 1.0 if RE_BASE64_IMG.search(text) else 0.0
    hidden_css_flag = float(len(RE_HIDDEN_CSS.findall(text)))

    # Obfuscation & Anomaly
    zw_count = float(len(RE_ZERO_WIDTH.findall(text)))
    hg_count = float(sum(1 for c in text if c in HOMOGLYPH_CHARS))
    hex_count = float(len(RE_HEX_ENCODED.findall(text)))
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii_count / max(char_len, 1)
    entropy_val = shannon_entropy(text[:2000])

    # Semantic & Financial Keywords
    urgent_count = float(len(RE_URGENT.findall(text)))
    financial_count = float(len(RE_FINANCIAL.findall(text)))
    brand_count = float(len(RE_BRANDS.findall(text)))
    brand_action_flag = 1.0 if (brand_count > 0 and urgent_count > 0) else 0.0

    features = [
        float(char_len),
        float(word_count),
        float(line_count),
        float(avg_word_len),
        float(excl_count),
        float(quest_count),
        float(dollar_count),
        float(comma_count),
        float(all_caps_count),
        float(upper_ratio),
        float(url_count),
        has_ip,
        has_susp_tld,
        has_short,
        float(html_links),
        float(link_to_text_ratio),
        has_html_flag,
        has_base64_flag,
        hidden_css_flag,
        zw_count,
        hg_count,
        hex_count,
        float(non_ascii_ratio),
        float(entropy_val),
        urgent_count,
        financial_count,
        brand_count,
        brand_action_flag,
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)

def extract_email_features_batch(texts: List[str]) -> np.ndarray:
    """
    Extracts 28 numerical features from a list of email strings.
    Returns 2D np.ndarray of shape (N, 28).
    """
    n = len(texts)
    matrix = np.empty((n, 28), dtype=np.float32)
    for i, t in enumerate(texts):
        matrix[i] = extract_email_features_single(t)[0]
    return matrix

FEATURE_NAMES_28 = [
    "char_count", "word_count", "line_count", "avg_word_len",
    "exclamation_count", "question_count", "dollar_count", "comma_count",
    "all_caps_words", "uppercase_ratio",
    "url_count", "has_ip_url", "has_suspicious_tld", "has_shortener",
    "html_link_count", "link_to_text_ratio",
    "has_html", "has_base64_image", "hidden_css_elements",
    "zero_width_count", "homoglyph_count", "hex_encoding_count",
    "non_ascii_ratio", "text_entropy",
    "urgent_phrase_count", "financial_phrase_count", "brand_keyword_count",
    "brand_action_cooccurrence"
]
