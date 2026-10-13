"""
Canonical, Versioned Text Sanitization & Entity Masking Pipeline
===============================================================
Version: 2.1.0 (Strict Train/Inference Parity Guarantee)

This module MUST be used identically across:
1. Master dataset curation (clean_and_curate_master_datasets.py)
2. Production model training (train_production_models.py)
3. Live production inference (services/scan_engine.py)

Any difference in preprocessing between training and inference causes
severe distribution shift and invalidates model calibration.
"""

import re
import html
import unicodedata
from typing import Optional

PREPROCESSING_VERSION = "2.1.0"

URL_REGEX = re.compile(
    r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[a-zA-Z0-9.\-]+\.(?:com|org|net|xyz|top|info|biz|cc|tk|ml|ga|cf|pw|online|site|live|link|app|dev|io|co|uk|de|ru|cn|in|pk)/[^\s<>"\']*',
    re.IGNORECASE
)
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', re.IGNORECASE)
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
AMOUNT_REGEX = re.compile(
    r'[\$€£¥]\s*\d+(?:[.,]\d+)*(?:\s*(?:million|billion|k|usd|eur|gbp))?|\b\d+(?:[.,]\d+)*\s*(?:usd|eur|gbp|dollars?|cents?)\b',
    re.IGNORECASE
)
CONTROL_CHARS = re.compile(r'[\u200b\u200c\u200d\ufeff\xad\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
TAG_REGEX = re.compile(r'<(?!\/?(?:URL|EMAIL|PHONE|AMOUNT)\b)[^>]+>')

def clean_and_mask_email_text(text: Optional[str]) -> str:
    """
    Standardized, leakage-free text preparation:
    1. HTML entity unescaping (&nbsp;, &amp;)
    2. URL masking -> <URL> (capturing both plain text and href links)
    3. HTML tag stripping (preserving entity tokens)
    4. Unicode NFKC normalization
    5. Control character & zero-width character purging
    6. Email address masking -> <EMAIL>
    7. Phone number masking -> <PHONE>
    8. Monetary value masking -> <AMOUNT>
    9. Whitespace collapse and lowercasing for TF-IDF consistency
    """
    if not text or not isinstance(text, str):
        return ""

    s = html.unescape(text)
    # Mask URLs first so URLs embedded in hrefs or plain text become <URL>
    s = URL_REGEX.sub(' <URL> ', s)
    # Strip HTML tags except our custom tokens
    s = TAG_REGEX.sub(' ', s)
    s = unicodedata.normalize('NFKC', s)
    s = CONTROL_CHARS.sub('', s)
    s = EMAIL_REGEX.sub(' <EMAIL> ', s)
    s = PHONE_REGEX.sub(' <PHONE> ', s)
    s = AMOUNT_REGEX.sub(' <AMOUNT> ', s)
    # Normalize token spacing and lowercase
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s
