"""
PhishGuard Hybrid AI Scanning Engine
LightGBM + TF-IDF + Feature Engineering pipeline
"""
import re
import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional
from scipy.sparse import hstack, csr_matrix
import math
import ipaddress
from collections import Counter
from urllib.parse import urlparse
import requests
from threading import Lock

# ML Global State
ML_MODEL = None
TFIDF_VECTORIZER = None
URL_MODEL = None
DOMAIN_MODEL = None
MODEL_LOADED = False
EMAIL_INPUT_KIND = 'legacy_tfidf_structural'
MODEL_LOAD_ERRORS = {}
_MODEL_LOAD_LOCK = Lock()

from core.logger import logger
from services.security_pipeline import (
    security_pipeline_id, numeric_url_frame, domain_model_input,
    validate_structured_email_artifact,
)
from utils.email_input_contract import build_email_input, validate_email_input, contract_sha256

STRUCTURED_EMAIL_INPUT_KIND = 'structured_email_evidence_v1'

def load_ml_model():
    global ML_MODEL, TFIDF_VECTORIZER, URL_MODEL, DOMAIN_MODEL, MODEL_LOADED, EMAIL_INPUT_KIND
    with _MODEL_LOAD_LOCK:
        if MODEL_LOADED:
            return
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'models', 'phish_model.pkl')
        vec_path = os.path.join(project_root, 'models', 'vectorizer.pkl')
        url_model_path = os.path.join(project_root, 'models', 'url_model.pkl')
        domain_model_path = os.path.join(project_root, 'models', 'domain_model.pkl')
        
        MODEL_LOAD_ERRORS.clear()
        ML_MODEL = TFIDF_VECTORIZER = None
        EMAIL_INPUT_KIND = 'legacy_tfidf_structural'
        try:
            artifact = joblib.load(model_path)
            if isinstance(artifact, dict):
                if artifact.get('schema_version') == 3:
                    ML_MODEL = validate_structured_email_artifact(artifact)
                    EMAIL_INPUT_KIND = STRUCTURED_EMAIL_INPUT_KIND
                elif artifact.get('schema_version') == 2 and artifact.get('input_kind') == 'raw_email_model_text' and artifact.get('approved_for_deployment') is True:
                    ML_MODEL = artifact['model']
                    EMAIL_INPUT_KIND = 'raw_email_model_text'
                else:
                    raise ValueError('Unapproved or unsupported Email candidate artifact')
            else:
                ML_MODEL = artifact
                TFIDF_VECTORIZER = joblib.load(vec_path)
            # Small sparse Email requests otherwise oversubscribe CPU threads.
            # Keep the saved learned parameters intact; configure execution only.
            for calibrated in getattr(ML_MODEL, 'calibrated_classifiers_', []):
                estimator = calibrated.estimator
                if hasattr(estimator, 'named_steps'):
                    estimator.named_steps['classifier'].set_params(n_jobs=1)
                else:
                    estimator.set_params(n_jobs=1)
        except Exception as e:
            ML_MODEL = TFIDF_VECTORIZER = None
            MODEL_LOAD_ERRORS['email'] = str(e)
            logger.error(f"Email model load failed: {e}")
        URL_MODEL = None
        try:
            URL_MODEL = joblib.load(url_model_path)
        except Exception as e:
            MODEL_LOAD_ERRORS['url'] = str(e)
            logger.error(f"Embedded-link URL model load failed: {e}")
        DOMAIN_MODEL = None
        try:
            DOMAIN_MODEL = joblib.load(domain_model_path)
        except Exception as e:
            MODEL_LOAD_ERRORS['domain'] = str(e)
            logger.error(f"Embedded-link domain model load failed: {e}")
        MODEL_LOADED = True
        logger.info("Model loading completed; available components: email=%s, url=%s, domain=%s",
                    ML_MODEL is not None, URL_MODEL is not None, DOMAIN_MODEL is not None)

from utils.feature_extractor import clean_text as clean_text_for_ml, extract_features as extract_features_for_inference, extract_features_single
from utils.text_sanitizer import clean_and_mask_email_text
from utils.url_scanner import is_ugc_hosting_platform

PHISHING_KEYWORDS = [
    "verify your account", "suspended", "unusual activity", "click here", "expires today",
    "confirm your identity", "update your billing", "unauthorized access", "immediate action",
    "your account will be closed", "limited time", "act now", "urgent", "password expires",
    "verification code", "social security", "2-factor", "two-factor", "account termination",
]

SUSPICIOUS_TLDS = [".xyz", ".biz", ".info", ".click", ".link", ".top", ".tk", ".pw"]

MALICIOUS_DOMAIN_LIST = [
    "paypa1-alert.com", "microsoft-365verify.net", "bank0famerica.info",
    "corp-helpdesk.xyz", "secure-docusign.xyz", "dropbox-shared.net",
]


def _matches_local_domain_list(host: str) -> bool:
    """Match a hostname boundary, never a URL path, query, or embedded brand."""
    host = (host or '').lower().rstrip('.')
    return any(host == entry or host.endswith('.' + entry) for entry in MALICIOUS_DOMAIN_LIST)


def extract_urls(text: str) -> list[str]:
    pattern = r"https?://[^\s<>\"']+"
    return re.findall(pattern, text)


def extract_sender_domain(sender: str) -> Optional[str]:
    match = re.search(r"@([\w.\-]+)", sender)
    return match.group(1).lower() if match else None


def email_vocabulary_supported(texts, *, model=None, vectorizer=None):
    """No recognized token is insufficient statistical evidence, not a clean email."""
    cleaned = [clean_and_mask_email_text(text) for text in texts]
    if vectorizer is not None:
        return vectorizer.transform(cleaned).getnnz(axis=1) > 0
    estimators = getattr(model, 'calibrated_classifiers_', [])
    vectorizers = [c.estimator.named_steps['features'].vectorizer_ for c in estimators]
    if not vectorizers:
        raise ValueError('Email bundle lacks fold-owned vocabulary support checks')
    return np.logical_or.reduce([v.transform(cleaned).getnnz(axis=1) > 0 for v in vectorizers])


def predict_email_probabilities(texts):
    """Shared model input construction for live scoring and batch calibration."""
    if not MODEL_LOADED:
        load_ml_model()
    if ML_MODEL is None or (EMAIL_INPUT_KIND not in ('raw_email_model_text', STRUCTURED_EMAIL_INPUT_KIND) and TFIDF_VECTORIZER is None):
        raise RuntimeError("Email model unavailable")
    if EMAIL_INPUT_KIND == STRUCTURED_EMAIL_INPUT_KIND:
        features = [validate_email_input(record) for record in texts]
        support = np.asarray(ML_MODEL.named_steps['features'].supported(features), dtype=bool)
        if support.shape != (len(features),) or not support.all():
            raise ValueError('No recognized Email content or identity/link vocabulary; structured inference unavailable')
    else:
        # Deployed schema 1/2 inputs retain their exact original recipe.
        cleaned = [clean_and_mask_email_text(text) for text in texts]
        if any(len(text) <= 5 for text in cleaned):
            raise ValueError("Insufficient email text for model inference")
    if EMAIL_INPUT_KIND == 'raw_email_model_text':
        if not email_vocabulary_supported(texts, model=ML_MODEL).all():
            raise ValueError("No recognized email vocabulary; statistical inference unavailable")
        features = list(texts)  # Fold-owned transforms are inside the estimator.
    elif EMAIL_INPUT_KIND != STRUCTURED_EMAIL_INPUT_KIND:
        tfidf = TFIDF_VECTORIZER.transform(cleaned)
        from utils.email_features import extract_email_features_batch
        structural = csr_matrix(extract_email_features_batch(texts))
        if (tfidf.getnnz(axis=1) == 0).any():
            raise ValueError("No recognized email vocabulary; statistical inference unavailable")
        features = hstack([tfidf, structural]).tocsr()
    classes = np.asarray(ML_MODEL.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(classes) != 2 or len(positive) != 1 or not np.any(classes == 0):
        raise ValueError("Email model does not implement binary 0/1 class semantics")
    probabilities = np.asarray(ML_MODEL.predict_proba(features), dtype=float)
    if probabilities.shape != (len(texts), 2):
        raise ValueError("Email model returned an invalid probability matrix shape")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Invalid email model probabilities")
    values = probabilities[:, positive[0]]
    return values


def score_content(text: str, *, email_input=None) -> dict:
    text_lower = text.lower()
    keyword_hits = [kw for kw in PHISHING_KEYWORDS if kw in text_lower]
    content_score = None
    has_urgency = len(keyword_hits) > 0
    suspicious_phrases = keyword_hits[:5]

    # ── Multilingual Language Detection & Phishing Lexicons ───────────────────
    detected_lang = "und"
    lang_name = "Undetermined"
    is_english = None
    lang_res = {"method": "unavailable", "validated": False, "confidence": None}
    multilingual_hits = []
    try:
        from utils.language_detector import detect_language, find_multilingual_phishing_signals
        lang_res = detect_language(text)
        detected_lang = lang_res.get("lang", "und")
        lang_name = lang_res.get("name", "Undetermined")
        is_english = lang_res.get("is_english")

        if not is_english:
            multilingual_hits = find_multilingual_phishing_signals(text, detected_lang)
            if multilingual_hits:
                keyword_hits.extend(multilingual_hits)
                suspicious_phrases = keyword_hits[:5]
                has_urgency = True
                # Wording is an observation, not a substitute for a trained
                # probability. Legitimate notifications use the same phrases.
    except Exception as e:
        logger.debug(f"[ScanEngine] Multilingual detection skipped: {e}")
    
    ml_probability = 0.0
    ml_used = False
    ml_error = None
    
    # Lazy load on first use instead of at startup to prevent server hang
    if not MODEL_LOADED:
        load_ml_model()
    
    # Run ML prediction if model is loaded
    if MODEL_LOADED and ML_MODEL is not None and (TFIDF_VECTORIZER is not None or EMAIL_INPUT_KIND in ('raw_email_model_text', STRUCTURED_EMAIL_INPUT_KIND)):
        clean = clean_and_mask_email_text(text)
        structured = EMAIL_INPUT_KIND == STRUCTURED_EMAIL_INPUT_KIND
        if structured and email_input is None:
            ml_error = 'Structured Email inference requires the original evidence contract'
        elif structured or len(clean) > 5:
            try:
                values = predict_email_probabilities([email_input] if structured else [text])
                if len(values) == 1:
                    ml_probability = float(values[0])
                    ml_used = True

                    content_score = int(round(ml_probability * 100))
                else:
                    logger.warning("Email model returned unexpected output; content score unavailable.")
                    ml_error = 'Unexpected email prediction count'
            except Exception as e:
                logger.error(f"LightGBM inference error: {e}")
                # Missing/failed inference stays unavailable, never a phrase score.
                ml_error = str(e)
        else:
            ml_error = 'Insufficient email text for model inference'
    else:
        ml_error = MODEL_LOAD_ERRORS.get('email', 'Email model unavailable')

    # True brand impersonation requires brand name paired with urgency or account action language
    brand_action_patterns = [
        r"\b(paypal|microsoft|apple|amazon|google|netflix|chase|wells\s*fargo|bank\s*of\s*america)\b.*\b(verify|suspended|unauthorized|blocked|locked|security\s*alert|sign-in|billing\s*update|reset\s*password)\b",
        r"\b(verify|suspended|unauthorized|blocked|locked|security\s*alert|sign-in|billing\s*update|reset\s*password)\b.*\b(paypal|microsoft|apple|amazon|google|netflix|chase|wells\s*fargo|bank\s*of\s*america)\b"
    ]
    has_impersonation = any(re.search(pat, text_lower) for pat in brand_action_patterns)

    return {
        "score": content_score,
        "score_source": "email_model_probability" if ml_used else "unavailable",
        "keyword_hits": keyword_hits,
        "suspicious_phrases": suspicious_phrases,
        "has_urgency_language": has_urgency,
        "has_impersonation": has_impersonation,
        "readability": "Formatted" if "<html" in text_lower else "Plain text",
        "ml_powered": ml_used,
        "ml_probability": ml_probability if ml_used else None,
        "ml_error": ml_error,
        "language": lang_name,
        "lang_code": detected_lang,
        "is_english": is_english,
        "language_detection": {key: lang_res.get(key) for key in ("method", "validated", "confidence", "script")},
        "phrase_matches_are_phishing_proof": False,
    }


# TLD risk scores: 0.0=safest (edu/gov), 1.0=max abuse (free TLDs)
# Synced with training/train_url_and_domain_models.py
_TLD_RISK = {
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
_KNOWN_SHORTENERS_RE = re.compile(
    r"bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|"
    r"buff\.ly|shorte\.st|adf\.ly|cutt\.ly|rb\.gy|short\.io"
)
_BRAND_KEYWORDS_RE = re.compile(
    r"paypal|apple|microsoft|google|facebook|amazon|netflix|"
    r"instagram|twitter|linkedin|dropbox|yahoo|irs|wellsfargo|"
    r"chase|bankofamerica|coinbase|binance|blockchain|metamask|github|spotify"
)
_OFFICIAL_DOMAINS_RE = re.compile(
    r"^(paypal|apple|microsoft|google|facebook|amazon|netflix|"
    r"instagram|twitter|linkedin|dropbox|yahoo|irs|chase|wellsfargo|"
    r"coinbase|binance|blockchain|metamask|github|spotify)\.(com|org|net|co|gov|edu)(\.[a-z]{2})?$"
)

def entropy(s):
    p, lns = Counter(s), float(len(s))
    if lns == 0:
        return 0
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def _tld_risk_score(netloc: str) -> float:
    tld = netloc.split(".")[-1].lower() if "." in netloc else netloc.lower()
    return _TLD_RISK.get(tld, 0.2)

def _char_continuation_rate(s: str) -> float:
    if not s:
        return 0.0
    max_run, cur = 1, 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        max_run = max(max_run, cur)
    return max_run / len(s)

def _unique_char_ratio(s: str) -> float:
    return len(set(s)) / len(s) if s else 0.0

# ── Domain v2 helpers ─────────────────────────────────────────────────────────
_SECURITY_KEYWORDS_RE = re.compile(
    r"(?:^|[-_])(login|signin|sign-in|verify|verification|secure|security|"
    r"account|update|confirm|banking|wallet|token|auth|portal|recover|"
    r"reset|support|helpdesk|service|alert|unlock|validate)(?:$|[-_])"
)
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
_KNOWN_TLD_TIERS = {
    'com':0,'org':0,'net':0,'edu':0,'gov':0,'io':0,'co':0,'app':0,'dev':0,
    'uk':0,'de':0,'fr':0,'au':0,'ca':0,'pk':0,'in':0,'jp':0,'nl':0,'se':0,
    'info':1,'biz':1,'us':1,'ru':1,'cn':1,'br':1,'kr':1,'mx':1,'ng':1,
    'xyz':2,'top':2,'icu':2,'club':2,'online':2,'site':2,'link':2,'live':2,
    'stream':2,'review':2,'win':2,'bid':2,'trade':2,'racing':2,'party':2,
    'tk':3,'ml':3,'ga':3,'gq':3,'cf':3,'pw':3,'cc':3,'click':3,'download':3,
    'zip':3,'webcam':3,'accountant':3,'loan':3,'science':3,'faith':3,
}

def _dga_score(sld: str) -> float:
    if len(sld) < 4:
        return 0.5
    letters_only = re.sub(r'[^a-z]', '', sld.lower())
    if len(letters_only) < 3:
        return 0.3
    bigrams = [letters_only[i:i+2] for i in range(len(letters_only)-1)]
    if not bigrams:
        return 0.2
    return round(sum(_BIGRAM_FREQ.get(bg, 0.01) for bg in bigrams) / len(bigrams), 4)

def _vowel_ratio(s: str) -> float:
    letters = [c for c in s.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return round(sum(c in 'aeiou' for c in letters) / len(letters), 4)

def _max_consonant_run(s: str) -> int:
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

def _homoglyph_flag_domain(domain: str) -> int:
    _MAP = str.maketrans({'0':'o','1':'l','3':'e','4':'a','5':'s','6':'g','8':'b'})
    normalized = domain.translate(_MAP)
    if normalized != domain and _BRAND_KEYWORDS_RE.search(normalized):
        return 1
    return 0

def extract_url_features(url):
    from utils.url_features import extract_url_features as _extract_features
    return _extract_features(url)

def extract_domain_features(url):
    from utils.domain_features import extract_domain_features as _extract_features
    return _extract_features(domain_model_input(url))

_SHORT_URL_CACHE: dict[str, str] = {}
_SHORT_URL_LOCK = Lock()

def resolve_short_url(url: str) -> str:
    """Attempts to resolve a short URL to its final destination with thread-safe caching."""
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly", "bit.do", "mcaf.ee"]
    if any(s in url.lower() for s in shorteners):
        with _SHORT_URL_LOCK:
            if url in _SHORT_URL_CACHE:
                return _SHORT_URL_CACHE[url]
        try:
            r = requests.get(url, stream=True, timeout=2, allow_redirects=True)
            res = r.url
            with _SHORT_URL_LOCK:
                if len(_SHORT_URL_CACHE) < 1000:
                    _SHORT_URL_CACHE[url] = res
            return res
        except Exception:
            return url
    return url


def _triage_url_priority(url: str, is_safe_auth_fn, extract_dom_fn) -> float:
    """
    Fast zero-network structural triage priority for ordering URLs when an email
    exceeds processing capacity. Higher priority = evaluated first.
    Ensures potential attack vectors (non-authority domains, IP addresses, credentials,
    shorteners, brand keywords) are never pushed out by benign corporate footer links.
    """
    priority = 0.0
    u_lower = url.lower()

    # 1. IP address host or port specified
    if re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", u_lower) or re.search(r":\d{2,5}/", u_lower):
        priority += 100.0

    # 2. Known shortener
    if _KNOWN_SHORTENERS_RE.search(u_lower):
        priority += 80.0

    # 3. Security/action keywords
    if _SECURITY_KEYWORDS_RE.search(u_lower) or any(k in u_lower for k in ("verify", "signin", "login", "password", "account", "banking")):
        priority += 60.0

    # 4. Brand keywords
    if _BRAND_KEYWORDS_RE.search(u_lower):
        priority += 50.0

    # 5. Suspicious / low-reputation TLD
    parsed = urlparse(url)
    netloc = parsed.netloc.split(":")[0]
    if netloc:
        tld_risk = _tld_risk_score(netloc)
        priority += tld_risk * 40.0

    # 6. Deep path with query parameters
    if parsed.query or len(parsed.path.strip('/')) > 20:
        priority += 20.0

    # 7. Downgrade established authority domains with clean roots
    domain = extract_dom_fn(url)
    if is_safe_auth_fn(domain) and not is_ugc_hosting_platform(domain, parsed.path.strip('/')):
        priority -= 40.0

    return priority


def score_urls(text: str, *, additional_urls=None) -> dict:
    from utils.threat_intel import check_virustotal
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Safe-domain authority anchor — importing here avoids circular import at module load
    try:
        from utils.url_scanner import is_safe_authority_domain, extract_domain as _extract_domain
    except ImportError:
        is_safe_authority_domain = lambda d: False
        _extract_domain = lambda u: ""

    all_urls = list(dict.fromkeys(extract_urls(text) + list(additional_urls or [])))

    flagged = []
    inference_errors = []
    if len(all_urls) > 10:
        inference_errors.append("URL analysis incomplete: message exceeds the 10-destination limit")
        # Rank by risk triage so high-risk links are never dropped behind benign padding
        urls = sorted(all_urls, key=lambda u: _triage_url_priority(u, is_safe_authority_domain, _extract_domain), reverse=True)[:10]
    else:
        urls = all_urls

    total_score = 0

    if not MODEL_LOADED:
        load_ml_model()

    if urls and URL_MODEL is None:
        inference_errors.append("URL ML unavailable")

    # Resolve short URLs concurrently in parallel if shorteners are present
    shortener_markers = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly", "bit.do", "mcaf.ee")
    if any(any(s in u.lower() for s in shortener_markers) for u in urls):
        with ThreadPoolExecutor(max_workers=min(len(urls), 4)) as ex:
            resolved_map = dict(zip(urls, ex.map(resolve_short_url, urls)))
    else:
        resolved_map = {u: u for u in urls}

    # Pre-resolve and score heuristics fast (no network)
    url_work = []
    for raw_url in urls:
        url = resolved_map.get(raw_url, raw_url)
        url_score = 0
        flags = []

        domain = _extract_domain(url)
        is_safe_dom = is_safe_authority_domain(domain)

        # Authority anchor: Check if the URL is on a user-content platform or has suspicious path signals
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        query_lower = parsed.query.lower()
        is_ugc = is_ugc_hosting_platform(domain, path_lower)

        has_suspicious_path = bool(re.search(
            r'/(login|signin|sign-in|verify|verification|auth|authenticate|update|security|account|wallet|credential|password|recover|confirm|webscr|form|viewform|response)\b',
            path_lower
        ))
        has_cred_params = bool(re.search(
            r'(?:[?&]|^)(email|user|username|login|token|account|redirect|next|url|target|dest)=',
            query_lower
        ))

        # Host popularity does not establish safety of an individual URL.
        # All destinations proceed through URL inference and available threat feeds.

        if is_safe_dom and (is_ugc or has_suspicious_path or has_cred_params):
            if is_ugc:
                flags.append("User-content/form hosting observed; host popularity does not verify this content")
            if has_suspicious_path or has_cred_params:
                flags.append("Account-related path or query observed; this alone is not phishing evidence")

        if url != raw_url:
            flags.append(f"Redirects to: {url}")
        if _matches_local_domain_list(domain):
            url_score += 90
            flags.append("Hostname matches configured local domain list (not a live feed verdict)")
        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            url_score += 30
            flags.append("Suspicious TLD")
        try:
            ipaddress.ip_address(parsed.hostname or '')
            ip_hostname = True
        except ValueError:
            ip_hostname = False
        if ip_hostname:
            url_score += 40
            flags.append("Destination hostname is an IP address; not a verified malicious verdict")
        # Brand words in paths, prose, query strings or names such as Appleton
        # cannot establish impersonation. The Email identity pillar evaluates
        # actual sender assertions and bounded hostname relations separately.

        ml_url_prob = None
        if URL_MODEL:
            try:
                features = extract_url_features(url)
                feat_df = numeric_url_frame(URL_MODEL, [features])
                if hasattr(URL_MODEL, "_transform_urls"):
                    probs = URL_MODEL.predict_proba([url])
                else:
                    probs = URL_MODEL.predict_proba(feat_df)
                if len(probs) > 0 and len(probs[0]) > 1:
                    ml_url_prob = float(probs[0][1])
                    ml_url_score = int(ml_url_prob * 100)
                    if ml_url_score > 50:
                        url_score = max(url_score, ml_url_score)
                        flags.append(f"URL model estimates {ml_url_score}% phishing probability; reliability on unseen email links is unverified")
            except Exception as e:
                logger.error(f"URL ML error: {e}")
                inference_errors.append(f"URL ML: {e}")

        ml_domain_prob = None
        # If it's a verified safe authority domain, do not penalize domain reputation
        if DOMAIN_MODEL and not is_safe_dom:
            try:
                features = extract_domain_features(domain_model_input(url))
                feat_df = numeric_url_frame(DOMAIN_MODEL, [features])
                probs = DOMAIN_MODEL.predict_proba(feat_df)
                if len(probs) > 0 and len(probs[0]) > 1:
                    ml_domain_prob = float(probs[0][1])
                    # Legacy labels do not validate domain-wide reputation.
                    # Retain the output for diagnostics; it cannot alter risk or flags.
            except Exception as e:
                logger.error(f"Domain ML error: {e}")

        url_work.append({"raw_url": raw_url, "url": url, "score": url_score, "flags": flags,
                         "model_probability": ml_url_prob,
                         "domain_model_diagnostic": ml_domain_prob if DOMAIN_MODEL and not is_safe_dom else None})

    # Run VirusTotal checks for all URLs in parallel within the existing budget.
    def _vt_check(item):
        try:
            return item["url"], check_virustotal(item["url"])
        except Exception:
            return item["url"], {"available": False, "reason": "Check failed"}

    vt_results = {}
    if url_work:
        ex = ThreadPoolExecutor(max_workers=min(len(url_work), 4))
        try:
            futures = {ex.submit(_vt_check, item): item for item in url_work}
            try:
                for fut in as_completed(futures, timeout=3.5):
                    try:
                        resolved_url, vt_result = fut.result()
                        vt_results[resolved_url] = vt_result
                    except Exception:
                        pass
            except Exception:
                pass
        finally:
            ex.shutdown(wait=False, cancel_futures=True)


    # Assemble final results
    for item in url_work:
        url = item["url"]
        raw_url = item["raw_url"]
        url_score = item["score"]
        flags = item["flags"]
        vt_result = vt_results.get(url, {"available": False, "reason": "Check skipped"})

        # ── VirusTotal live check ───────────────────────────────────────────────
        try:
            if vt_result.get("available"):
                malicious = vt_result.get("malicious", 0)
                suspicious_vt = vt_result.get("suspicious", 0)
                if malicious >= 3:
                    url_score = max(url_score, 90)
                    flags.append(f"VirusTotal: {malicious}/{vt_result.get('total_engines', '?')} engines flagged as malicious")
                elif malicious > 0:
                    url_score = max(url_score, 65)
                    flags.append(f"VirusTotal: {malicious}/{vt_result.get('total_engines', '?')} engines flagged")
                elif suspicious_vt > 0:
                    url_score = max(url_score, 45)
                    flags.append(f"VirusTotal: {suspicious_vt} engines flagged as suspicious")
                else:
                    flags.append(f"VirusTotal: 0/{vt_result.get('total_engines', '?')} engines flagged; this does not establish legitimacy")
        except Exception as e:
            logger.warning(f"VT result processing skipped for {url[:60]}: {e}")

        url_score = min(url_score, 100)
        flagged.append({
            "url": raw_url,
            "resolved_url": url if url != raw_url else None,
            "score": url_score,
            "flags": flags,
            "risk": classify_risk(url_score),
            "virustotal": vt_result,
            "model_probability": item.get("model_probability"),
            "score_kind": "unvalidated_combined_review_score",
            "threat_probability_validated": False,
            "domain_model_diagnostic": item.get("domain_model_diagnostic"),
            "domain_model_role": "diagnostic_only_unverified_domain_labels",
        })
        total_score = max(total_score, url_score)

    suspicious_urls = [item for item in flagged if item["score"] >= 45]
    return {
        "inference_errors": inference_errors,
        "score": total_score,
        "urls_found": len(all_urls),       # true count, including any beyond the cap
        "urls_analyzed": len(urls),        # how many were actually checked
        "scan_complete": len(urls) == len(all_urls) and not inference_errors,
        "analyzed_urls": flagged,
        "flagged_urls": suspicious_urls,
        "has_redirect": any(u.get("resolved_url") for u in flagged),
        "redirect_evidence_kind": "observed_destination_change_only; absence does not establish no redirects",
        "has_shortener": any(_KNOWN_SHORTENERS_RE.fullmatch((urlparse(u["url"]).hostname or "").lower().removeprefix("www.")) for u in flagged),
    }



def score_headers(headers_text: str, spf: str = None, dkim: str = None, dmarc: str = None) -> dict:
    from utils.email_authentication import untrusted_authentication_report
    authentication = untrusted_authentication_report(headers_text, spf, dkim, dmarc)
    headers_lower = headers_text.lower() if headers_text else ""
    return {
        **authentication,
        "score": 0,
        "fail_count": 0,
        "has_suspicious_mailer": "anonymous" in headers_lower or "bulk" in headers_lower,
    }





def build_ai_paragraph(final_score: int | None, content_result: dict, url_result: dict, header_result: dict) -> str:
    if final_score is None:
        return "The Email model did not produce a prediction. No risk score is available; review the completed evidence checks."
    urls = url_result.get("flagged_urls", [])
    has_shortener = url_result.get("has_shortener", False)
    has_urgency = content_result.get("has_urgency_language", False)
    has_impersonation = content_result.get("has_impersonation", False)
    suspicious_phrases = content_result.get("suspicious_phrases", [])
    has_requests = any(w in " ".join(suspicious_phrases).lower() for w in ["password", "otp", "bank", "credit", "login", "verify", "account"])
    failed_checks = [name for name in ("spf", "dkim", "dmarc") if str(header_result.get(name, "UNKNOWN")).upper() == "FAIL"]
    header_failed = bool(failed_checks)
    auth_passed = all(str(header_result.get(name, "UNKNOWN")).upper() == "PASS" for name in ("spf", "dkim", "dmarc"))
    has_suspicious_mailer = header_result.get("has_suspicious_mailer", False)
    ml_high = content_result.get("ml_powered") and content_result.get("ml_probability", 0) > 0.6
    
    if final_score < 45:
        sentences = []
        sentences.append("The checks that completed did not identify high-risk indicators in this email.")
        if auth_passed:
            sentences.append("The supplied headers report SPF, DKIM and DMARC as passing.")
        elif header_failed:
            sentences.append(f"The supplied headers report a failed {', '.join(name.upper() for name in failed_checks)} check, so sender authentication needs review.")
        else:
            sentences.append("Sender authentication could not be fully assessed from the available headers.")
        if not has_urgency and not has_requests:
            sentences.append("No urgency or request for sensitive account information was identified in the message text.")
        return " ".join(sentences)

    sentences = ["The scan found warning signs that make this email risky."]
    if has_impersonation:
        sentences.append("The message mentions a brand alongside account-action language; this alone does not establish impersonation.")
    if has_urgency:
        sentences.append("The message uses urgency to push for immediate action.")
    if urls:
        sentences.append(f"{len(urls)} analyzed link{'s' if len(urls) != 1 else ''} showed suspicious signals.")
    elif has_shortener:
        sentences.append("A shortened link obscures its destination.")
    if header_failed:
        sentences.append(f"The supplied headers report failed {', '.join(name.upper() for name in failed_checks)} authentication.")
    if has_suspicious_mailer:
        sentences.append("The supplied headers name a mailer associated with bulk or anonymous sending.")
    if has_requests:
        sentences.append("The message requests sensitive account information.")
    if ml_high and not (has_urgency or urls or has_requests):
        sentences.append("The content model identified a phishing-like pattern.")
    return " ".join(sentences)

def build_explanation(final_score: int | None, content_result: dict, url_result: dict, header_result: dict) -> dict:
    """
    Build a human-readable explanation of WHY the email was flagged.
    Returns a dict with a verdict summary and a list of signal bullets.
    """
    signals = []
    positive_signals = []

    # --- Auth signals ---
    fail_count = header_result.get("fail_count", 0)
    spf = header_result.get("spf", "UNKNOWN")
    dkim = header_result.get("dkim", "UNKNOWN")
    dmarc = header_result.get("dmarc", "UNKNOWN")
    if fail_count > 0:
        failed_checks = [k for k, v in [("SPF", spf), ("DKIM", dkim), ("DMARC", dmarc)] if v == "FAIL"]
        signals.append(f"Supplied headers report failed authentication checks: {', '.join(failed_checks)}. This may indicate spoofing and needs review.")
    elif all(value == "PASS" for value in (spf, dkim, dmarc)):
        positive_signals.append("Supplied headers report SPF, DKIM and DMARC as passing.")

    if header_result.get("has_suspicious_mailer"):
        signals.append("A suspicious or anonymous mailer was detected in the email headers, commonly used by spam tools.")

    # --- Content signals ---
    phrases = content_result.get("suspicious_phrases", [])
    if phrases:
        signals.append(f"Configured account-action or urgency term matched: \"{phrases[0]}\". Legitimate security notifications can use the same wording; manipulation is not established.")
    if content_result.get("has_impersonation"):
        signals.append("Brand and account-action terms occur together. Verify the sender and destination; the text alone does not establish impersonation.")
    if content_result.get("ml_powered") and content_result.get("ml_probability", 0) > 0.5:
        prob_pct = int(content_result["ml_probability"] * 100)
        signals.append(f"The email content model estimates {prob_pct}% phishing probability. Reliability on unseen sources is not established.")
    elif content_result.get("ml_powered") and content_result.get("ml_probability", 1) < 0.3:
        positive_signals.append("The content model assigns a low phishing probability; this does not verify legitimacy.")

    # --- URL signals ---
    flagged_urls = url_result.get("flagged_urls", [])
    for u in flagged_urls[:2]:
        if u["flags"]:
            signals.append(f"Suspicious URL detected: {u['url'][:70]} — reason: {', '.join(u['flags'])}.")
    if url_result.get("has_shortener"):
        signals.append("A configured URL-shortener hostname is present. Its destination requires verification; shortening alone does not establish phishing.")
    if not flagged_urls and url_result.get("urls_found", 0) == 0:
        positive_signals.append("No URLs were found in the email body.")

    # --- Build verdict summary ---
    if final_score is None:
        summary = "The Email model prediction is unavailable. Review the available evidence; no numeric risk estimate was produced."
    elif final_score >= 75:
        summary = "The email content model assigns a high phishing probability. Review the available evidence and decision policy."
    elif final_score >= 45:
        summary = "This email has suspicious characteristics. It may not be safe — proceed with caution."
    else:
        summary = "No significant phishing indicators were found in the checks that completed."

    return {
        "summary": summary,
        "risk_signals": signals,
        "positive_signals": positive_signals if final_score is None or final_score < 75 else [],
        "ai_paragraph": build_ai_paragraph(final_score, content_result, url_result, header_result)
    }


def build_threat_intel(content_result: dict, url_result: dict, sender: str) -> dict:
    """Report configured list matches separately from unverified text heuristics."""
    matches = []
    sender_domain = extract_sender_domain(sender) if sender else None
    if _matches_local_domain_list(sender_domain):
        matches.append({
            "source": "Configured Local Domain List",
            "verdict": "LIST_MATCH",
            "detail": f"Sender hostname '{sender_domain}' matches a configured list entry. No live-feed verification or blocking was performed."
        })

    for u in url_result.get("flagged_urls", []):
        if _matches_local_domain_list(urlparse(u['url']).hostname):
            matches.append({
                "source": "Configured Local Domain List",
                "verdict": "LIST_MATCH",
                "detail": "URL hostname matches a configured list entry. This is not a live threat-feed verdict."
            })
            break

    return {
        "score": min(len(matches) * 40, 100),
        "matches": matches,
        "heuristic_observations": ["Brand/account-action and urgency terms co-occur; already reported as content evidence, not a threat-feed match."]
            if content_result.get('has_impersonation') and content_result.get('has_urgency_language') else [],
        "source_kind": "configured_local_list_not_live_feed",
    }


def classify_risk(score: int) -> str:
    if score >= 75:
        return "critical"
    elif score >= 45:
        return "medium"
    else:
        return "safe"


def get_status(score: int) -> str:
    if score >= 75:
        return "PHISHING"
    elif score >= 45:
        return "SUSPICIOUS"
    else:
        return "CLEAN"


from sqlmodel import Session

def evaluate_email_security(content: str, sender="", subject="", headers="", attachments=None, *, predictor=None, html_body="", parsing_info=None, supplementary_urls=None):
    """Shared production/calibration scorer; no database persistence or account policy."""
    # Preserve original request values before applying presentation defaults.
    # The same builder owns training and future structured model inputs.
    if supplementary_urls is not None and (not isinstance(supplementary_urls, list) or
            any(not isinstance(value, str) for value in supplementary_urls)):
        raise ValueError('Supplementary Email destinations must be a list of HTTP(S) or explicitly unsupported URL strings')
    email_input = build_email_input(content, sender=sender, subject=subject, headers=headers,
        html_body=html_body, supplementary_urls=supplementary_urls, parsing_info=parsing_info)
    input_diagnostics = {
        'contract_version': email_input['contract_version'],
        'contract_sha256': contract_sha256(email_input),
        'availability': dict(email_input['availability']),
        'parsing_completeness': email_input['parsing']['completeness'],
        'link_destination_count': len(email_input['link_destinations']),
        'unsupported_link_count': len(email_input['unsupported_urls']),
        'unsupported_link_reasons': dict(Counter(item['reason'] for item in email_input['unsupported_urls'])),
        'trusted_authentication_available': False,
    }
    # ── Feature D: OpenCV 5 Vision Scanner & Quishing Engine ──────────────────
    sensor_errors = []
    parsing_info = parsing_info or {}
    sensor_errors.extend('Email parsing: ' + str(e) for e in parsing_info.get('errors', []))
    if parsing_info.get('unscanned_parts'):
        sensor_errors.append('Email contains unscanned MIME parts; attachment evidence is incomplete')
    if email_input['unsupported_urls']:
        sensor_errors.append('URL evidence coverage incomplete: ' + str(len(email_input['unsupported_urls'])) +
                             ' unsupported or invalid URL occurrences were not scanned; this is not a phishing finding')
    vision_result = {
        "has_images": False,
        "images_analyzed": 0,
        "has_quishing": False,
        "qr_urls": [],
        "details": []
    }
    qr_urls_extracted = []
    if attachments:
        try:
            from services.vision_scanner import analyze_image_attachment
            for att in attachments:
                raw_data = att.get("data")
                filename = att.get("filename", "image.png")
                if raw_data and isinstance(raw_data, (bytes, bytearray)):
                    analysis = analyze_image_attachment(raw_data, filename=filename)
                    if analysis.get("is_valid_image"):
                        vision_result["has_images"] = True
                        vision_result["images_analyzed"] += 1
                        vision_result["details"].append(analysis)
                        if analysis.get('qr_scan_complete') is False:
                            sensor_errors.append('QR analysis incomplete: ' + str(analysis.get('qr_scan_error') or filename))
                        if analysis.get("has_qr"):
                            vision_result["has_quishing"] = True
                            for u in analysis.get("qr_urls", []):
                                if u not in qr_urls_extracted:
                                    qr_urls_extracted.append(u)
                    else:
                        sensor_errors.append('Image analysis unavailable or invalid image: ' + filename)
                else:
                    sensor_errors.append('Image payload unavailable: ' + filename)
            vision_result["qr_urls"] = qr_urls_extracted
        except Exception as e:
            logger.debug(f"[Vision Engine] Attachment analysis skipped: {e}")
            sensor_errors.append('[Vision Engine]' + ": " + str(e))

    # Retain the exact deployed body + space + subject call for schema 1/2.
    # The optional schema 3 model instead receives the original shared contract.
    was_structured = EMAIL_INPUT_KIND == STRUCTURED_EMAIL_INPUT_KIND
    content_result = (score_content(content + " " + subject, email_input=email_input)
                      if was_structured else score_content(content + " " + subject))
    if not was_structured and EMAIL_INPUT_KIND == STRUCTURED_EMAIL_INPUT_KIND:
        # A first lazy load can select schema 3 inside score_content. Its
        # contract-free preliminary result is never used as a scan decision.
        content_result = score_content(content + " " + subject, email_input=email_input)
    input_diagnostics['model_input_kind'] = EMAIL_INPUT_KIND

    # Decoder outputs remain a separate auxiliary channel: they were not part
    # of the original text/header/link training input and must not enter it.
    qr_input = build_email_input('', supplementary_urls=qr_urls_extracted)
    scan_url_content = ' '.join(qr_input['link_destinations'])
    input_diagnostics['attachment_qr'] = {
        'included_in_model_input': False,
        'link_destination_count': len(qr_input['link_destinations']),
        'unsupported_link_count': len(qr_input['unsupported_urls']),
    }
    if qr_input['unsupported_urls']:
        sensor_errors.append('QR URL evidence coverage incomplete: unsupported or invalid decoder destinations were not scanned')
    anchor_links_found, seen_anchors = [], set()
    for link in email_input['links']:
        if link['source'].endswith('_html_anchor'):
            key = (link['anchor_text'], link['destination'])
            if key not in seen_anchors:
                seen_anchors.add(key)
                anchor_links_found.append({'text': link['anchor_text'], 'href': link['destination']})
    # Only validated destinations reach remote checks; normalized Unicode links
    # are included, while script/style content and unsupported schemes are not.
    link_destinations = email_input['link_destinations']
    url_result = score_urls(scan_url_content, additional_urls=link_destinations) if link_destinations else score_urls(scan_url_content)
    url_result["anchor_links"] = anchor_links_found
    sensor_errors.extend(url_result.get("inference_errors", []))
    if not content_result.get("ml_powered"):
        sensor_errors.append("Content ML unavailable: " + (content_result.get('ml_error') or 'no valid model probability'))
    header_result = score_headers(headers)
    threat_result = build_threat_intel(content_result, url_result, sender)

    # ── Feature B: ChromaDB RAG Vector Store Query ────────────────────────────
    vector_result = {"has_match": False, "match_percentage": 0, "matches": []}
    try:
        from services.vector_store import search_similar_campaigns
        vector_result = search_similar_campaigns(content + " " + subject)
    except Exception as e:
        logger.debug(f"[VectorStore] RAG query skipped: {e}")
        sensor_errors.append('[VectorStore]' + ": " + str(e))

    # ── Feature C: DistilBERT Business Email Compromise (BEC) Intent Engine ───
    bec_result = {"is_bec": False, "bec_score": 0, "intent_category": "Clean", "signals": []}
    try:
        from services.bec_detector import predict_bec_intent
        urls_count = url_result.get("urls_found", len(extract_urls(content)))
        bec_result = predict_bec_intent(content + " " + subject, urls_found=urls_count)
        if bec_result.get('transformer_error'):
            sensor_errors.append('BEC inference failed: ' + str(bec_result['transformer_error']))
    except Exception as e:
        logger.debug(f"[BEC Engine] Intent evaluation skipped: {e}")
        sensor_errors.append('[BEC Engine]' + ": " + str(e))

    # ── OriginTrace 4-Pillars Security Architecture ───────────────────────────
    from services.origintrace_pillars import (
        ProvenanceMatrix,
        InfrastructureLineage,
        SemanticDeception,
        EvidentialFusion,
        CrossEntityContradictionEngine
    )

    extracted_scan_urls = list(dict.fromkeys(qr_input['link_destinations'] + link_destinations))

    # Cross-Entity Contradiction Matrix: 5-Layer Identity Alignment
    contra_result = CrossEntityContradictionEngine.analyze(
        sender_header=sender,
        raw_headers=headers,
        spf=header_result.get("spf", "UNKNOWN"),
        dkim=header_result.get("dkim", "UNKNOWN"),
        dmarc=header_result.get("dmarc", "UNKNOWN"),
        urls=extracted_scan_urls,
        anchor_links=anchor_links_found,
        destination_domain=url_result.get("domain", ""),
        destination_asn=url_result.get("asn", ""),
        destination_isp=url_result.get("isp", "")
    )
    if not contra_result.get('is_contradictory') and not header_result.get('verified'):
        contra_result['alignment_summary'] = 'No contradiction found in the available evidence; sender identity and authentication are not verified.'

    # Pillar 1: Provenance Matrix (Cryptographic Identity, Display Name Spoofing, Lookalike Domains)
    p1_result = ProvenanceMatrix.analyze(
        sender_header=sender,
        raw_headers=headers,
        spf=header_result.get("spf", "UNKNOWN"),
        dkim=header_result.get("dkim", "UNKNOWN"),
        dmarc=header_result.get("dmarc", "UNKNOWN"),
        urls=extracted_scan_urls
    )

    # Pillar 2: Infrastructure Lineage (Server Hops, Disposable VPS, Residential Relays)
    p2_result = InfrastructureLineage.analyze(
        headers_text=headers,
        urls=extracted_scan_urls,
        url_risk_score=url_result.get("score")
    )

    # Pillar 3: Semantic Deception (Zero-link BEC, CEO Wire Fraud, Gift Cards, Quishing)
    p3_result = SemanticDeception.analyze(
        bec_result=bec_result,
        content_result=content_result,
        vector_result=vector_result,
        vision_result=vision_result
    )

    # Model probability drives statistical decisions. Heuristic evidence can request
    # review but cannot masquerade as a calibrated probability or shrink model output.
    from services.conformal_evidential_engine import synthesize_email_security_opinion
    heuristic_evidence = {
        "url_risk_score": url_result.get("score", 0),
        "bec_risk_score": bec_result.get("bec_score", 0),
        "bec_detected": bool(bec_result.get("is_bec")),
        "contradiction_score": contra_result.get("contradiction_score", 0),
        "contradiction_detected": bool(contra_result.get("is_contradictory")),
        "qr_detected": bool(vision_result.get("has_quishing")),
        "threat_matches": threat_result.get("matches", []),
    }
    conformal_res = synthesize_email_security_opinion(
        predictor=predictor, pipeline_id=security_pipeline_id(),
        content_prob=content_result.get("ml_probability"),
        heuristic_evidence=heuristic_evidence,
    )

    if sensor_errors:
        conformal_res["evidence_incomplete"] = True
        if conformal_res["action"] != "AUTO_QUARANTINE":
            conformal_res.update(verdict="SUSPICIOUS", action="WARNING_BANNER", decision_basis="incomplete_evidence_review")
    if email_input['parsing']['completeness'] == 'INCOMPLETE':
        # Corrupted/partial content cannot acquire an automatic quarantine
        # decision from calibration on complete fixed-text inputs.
        conformal_res.update(verdict='SUSPICIOUS', action='WARNING_BANNER',
                             decision_basis='incomplete_message_text', calibration_valid=False)
    final_score = conformal_res["final_score"]
    verdict_state = conformal_res["verdict"]
    security_action = conformal_res["action"]
    pred_set = conformal_res["conformal_prediction_set"]
    uncertainty_val = conformal_res["epistemic_uncertainty"]

    # QR/BEC/URL auxiliary observations remain separate review evidence per the
    # calibrated single-estimator contract. They do not inflate the content model's
    # calibrated probability, but escalate action, status, and composite risk.
    url_score = url_result.get("score", 0)
    threat_score = threat_result.get("score", 0)
    max_payload = max(url_score, threat_score)
    composite_risk_score = max(final_score or 0, max_payload) if final_score is not None else max_payload

    # QR/BEC observations remain separate review evidence. They do not justify
    # inventing a higher aggregate score than the statistical scorer returned.

    return {
        'content_result': content_result,
        'url_result': url_result,
        'header_result': header_result,
        'threat_result': threat_result,
        'vector_result': vector_result,
        'bec_result': bec_result,
        'vision_result': vision_result,
        'qr_urls_extracted': qr_urls_extracted,
        'contra_result': contra_result,
        'p1_result': p1_result,
        'p2_result': p2_result,
        'p3_result': p3_result,
        'conformal_res': conformal_res,
        'final_score': final_score,
        'composite_risk_score': composite_risk_score,
        'verdict_state': verdict_state,
        'security_action': security_action,
        'pred_set': pred_set,
        'uncertainty_val': uncertainty_val,
        "sensor_errors": sensor_errors,
        "parsing_info": parsing_info,
        "email_input_diagnostics": input_diagnostics,
    }


def scan_email(content: str, session: Session, sender: str = "", subject: str = "", headers: str = "", gmail_id: str = "", attachments: list = None, user_id: int | None = None, *, html_body: str = "", parsing_info=None, supplementary_urls=None) -> dict:
    """
    Main scan function – runs all sub-engines and returns combined result.
    gmail_id: if provided, uses 'gm-{gmail_id}' as the scan_id for deduplication.
    attachments: optional list of extracted email attachment dicts (filename, data).
    """
    from data import store
    if any(not isinstance(value, str) for value in (content, sender, subject, headers, html_body)):
        raise TypeError('Email content and identity fields must be strings')
    
    # Check custom block rules first
    if sender and store.is_blocked(session, sender):
        logger.warning(f"Scan short-circuited: Sender '{sender}' is blocked by a custom rule.")
        final_score = 100
        blocked_result = {
            "final_score": final_score,
            "risk_level": "critical",
            "status": "PHISHING",
            # Retain the existing blocked-rule presentation, but never fabricate
            # model evidence from an administrative sender rule.
            "decision_basis": "manual_sender_block",
            "security_action": "BLOCK_BY_POLICY",
            "statistical_verdict": None,
            "score_kind": "administrative_block_priority",
            "action_execution": {"executed": False, "scope": "scan_recommendation_only"},
            "content_analysis": {"score": 0, "flags": [], "ml_probability": None, "ml_powered": False},
            "url_intelligence": {"score": 100, "flagged_urls": []},
            "header_authentication": score_headers(headers),
            "threat_intelligence": {
                "score": 100,
                "matches": [{
                    "source": "Custom Security Rule",
                    "verdict": "BLOCKED",
                    "detail": f"Sender {sender} matched a manual blocklist rule."
                }]
            },
            "explanation": "The sender matches a configured manual block rule. Blocking is recommended; this scan has not moved or blocked the message. This is an administrative decision, not a model classification.",
            "sender": sender,
            "subject": subject,
            "body_preview": content[:600] + ("..." if len(content) > 600 else ""),
            "source": "gmail" if gmail_id else "manual",
            "scan_id": f"gm-{gmail_id}" if gmail_id else None
        }
        store.add_scan(blocked_result, session, gmail_id=gmail_id, user_id=user_id)
        return blocked_result
    
    evaluated = evaluate_email_security(content, sender, subject, headers, attachments, html_body=html_body, parsing_info=parsing_info, supplementary_urls=supplementary_urls)
    content_result = evaluated['content_result']
    url_result = evaluated['url_result']
    header_result = evaluated['header_result']
    threat_result = evaluated['threat_result']
    vector_result = evaluated['vector_result']
    bec_result = evaluated['bec_result']
    vision_result = evaluated['vision_result']
    qr_urls_extracted = evaluated['qr_urls_extracted']
    contra_result = evaluated['contra_result']
    p1_result = evaluated['p1_result']
    p2_result = evaluated['p2_result']
    p3_result = evaluated['p3_result']
    conformal_res = evaluated['conformal_res']
    final_score = evaluated['final_score']
    verdict_state = evaluated['verdict_state']
    security_action = evaluated['security_action']
    pred_set = evaluated['pred_set']
    uncertainty_val = evaluated['uncertainty_val']
    explanation = build_explanation(final_score, content_result, url_result, header_result)

    # Inject Contradiction Matrix signals
    for sig in contra_result.get("signals", []):
        if sig not in explanation["risk_signals"]:
            explanation["risk_signals"].insert(0, f"[Contradiction Matrix] {sig}")

    # Inject Provenance Matrix signals
    for sig in p1_result.get("signals", []):
        if any(marker in sig for marker in ["Spoofing", "Display Name Domain Difference", "Lookalike", "Cryptographic Identity Failure"]):
            if sig not in explanation["risk_signals"]:
                explanation["risk_signals"].insert(0, f"[Pillar 1 Provenance] {sig}")

    # Inject Infrastructure Lineage signals
    for sig in p2_result.get("signals", []):
        if any(marker in sig for marker in ["Disposable VPS", "Residential Proxy", "Low-Reputation TLD"]):
            if sig not in explanation["risk_signals"]:
                explanation["risk_signals"].insert(0, f"[Pillar 2 Infrastructure] {sig}")

    # Add Quishing / QR code signal to explanation if detected
    if vision_result.get("has_quishing") and qr_urls_extracted:
        qr_msg = f"Quishing: QR destination extracted from an attached image: '{qr_urls_extracted[0]}'. Requires review; QR code presence alone does not establish phishing."
        if qr_msg not in explanation["risk_signals"]:
            explanation["risk_signals"].insert(0, qr_msg)

    # Add ChromaDB vector match signal to explanation if found
    if vector_result.get("has_match") and vector_result.get("matched_title"):
        vec_msg = (
            f"ChromaDB RAG Vector Match: {vector_result['match_percentage']}% semantic similarity "
            f"to campaign-memory entry '{vector_result['matched_title']}'. Similarity is not verified phishing evidence."
        )
        if vec_msg not in explanation["risk_signals"]:
            explanation["risk_signals"].insert(0, vec_msg)

    # Add BEC intent signal to explanation if detected
    if bec_result.get("is_bec") and bec_result.get("signals"):
        bec_msg = f"Possible Business Email Compromise (BEC) intent: {bec_result['intent_category']} (review score {bec_result['bec_score']}/100; not a calibrated probability)."
        if bec_msg not in explanation["risk_signals"]:
            explanation["risk_signals"].insert(0, bec_msg)

    # Rendering follows the actual policy decision, including abstention/errors,
    # rather than inferring a verdict from the display score.
    unique_signals, seen_signals = [], set()
    for signal in explanation['risk_signals']:
        evidence_text = re.sub(r'^\[[^]]+\]\s*', '', signal).strip().casefold()
        if evidence_text not in seen_signals:
            seen_signals.add(evidence_text)
            unique_signals.append(signal)
    explanation['risk_signals'] = unique_signals
    if security_action == 'WARNING_BANNER':
        basis = conformal_res.get('decision_basis', 'review_required')
        explanation['summary'] = 'Review required. Available evidence does not support an automatic safe-delivery or quarantine decision.'
        explanation['ai_paragraph'] = explanation['summary']
        if not explanation['risk_signals']:
            explanation['risk_signals'].append('Review required by the decision policy: ' + str(basis) + '.')
    elif security_action == 'AUTO_QUARANTINE':
        explanation['summary'] = 'The decision policy recommends quarantine; this scan has not blocked or moved the email.'
        explanation['ai_paragraph'] = explanation['summary']
    else:
        explanation['summary'] = 'The decision policy permits delivery. Completed checks do not establish that the email is legitimate.'
        explanation['ai_paragraph'] = explanation['summary']

    # Ground recommendation and action in the calibrated conformal prediction set
    if security_action == "AUTO_QUARANTINE":
        recommended_action = "Quarantine recommended by the email model. This scan has not moved or blocked the message. Avoid links and attachments until reviewed."
    elif security_action == "WARNING_BANNER":
        recommended_action = "Potential Risk Detected: Review sender carefully before interacting with links or replying."
    else:
        recommended_action = "The decision policy permits delivery; this is not proof of safety. Verify unexpected requests independently."

    ai_explanation = {
        "summary": explanation["ai_paragraph"],
        "key_risk": explanation["risk_signals"][0] if explanation["risk_signals"] and security_action != "SAFE_DELIVERY" else "No significant risk signal identified in completed checks.",
        "recommended_action": recommended_action,
        "model_used": "Email content model with separate evidence review; deployment generalization is unverified",
    }


    p4_result = {
        "pillar": "Pillar 4: Model Probability and Evidence Review",
        "final_score": final_score,
        "threat_probability": conformal_res["threat_probability"],
        "belief_phish": conformal_res["belief_phish"],
        "belief_clean": conformal_res["belief_clean"],
        "uncertainty": uncertainty_val,
        "aleatoric_uncertainty": conformal_res["aleatoric_uncertainty"],
        "conformal_prediction_set": pred_set,
        "security_action": security_action,
        "action_execution": {"executed": False, "scope": "scan_recommendation_only"},
        "modality_opinions": conformal_res["modality_opinions"],
        "heuristic_evidence": conformal_res["heuristic_evidence"],
        "decision_basis": conformal_res["decision_basis"],
        "modality_contributions": {
            "provenance_matrix": p1_result.get("score", 0),
            "infrastructure_lineage": p2_result.get("score", 0),
            "semantic_deception": p3_result.get("score", 0)
        }
    }

    result_payload = {
        "final_score": final_score,
        "composite_risk_score": evaluated.get('composite_risk_score', final_score),
        "risk_level": {"PHISHING": "critical", "SUSPICIOUS": "medium", "CLEAN": "safe"}[verdict_state],
        "status": verdict_state,
        "verdict_state": verdict_state,
        "security_action": security_action,
        "action_execution": {"executed": False, "scope": "scan_recommendation_only"},
        "conformal_prediction_set": pred_set,
        "sensor_errors": evaluated["sensor_errors"],
        "email_parsing": evaluated['parsing_info'],
        "email_input_diagnostics": evaluated['email_input_diagnostics'],
        "conformal_guarantee": {
            "calibration_valid": conformal_res["calibration_valid"],
            "calibration_rejection_reason": conformal_res.get("calibration_rejection_reason"),
            "assumptions": "Independent exchangeable calibration data and unchanged scorer; no FDR guarantee",
            "uncertainty_kind": "not_estimated",
            "mathematical_framework": "Split-conformal email-model decisions; heuristic review policies are separate"
        },
        "cascade_tier_executed": "TIER_2_DEEP_FORENSICS",
        "epistemic_uncertainty": uncertainty_val,
        "aleatoric_uncertainty": conformal_res["aleatoric_uncertainty"],
        "belief_phish": conformal_res["belief_phish"],
        "belief_clean": conformal_res["belief_clean"],
        "contradiction_analysis": contra_result,
        "content_analysis": content_result,
        "url_intelligence": url_result,
        "header_authentication": header_result,
        "threat_intelligence": threat_result,
        "vector_intelligence": vector_result,
        "bec_intelligence": bec_result,
        "vision_intelligence": vision_result,
        "explanation": explanation,
        "ai_explanation": ai_explanation,
        "pillars": {
            "pillar1_provenance": p1_result,
            "pillar2_infrastructure": p2_result,
            "pillar3_semantic": p3_result,
            "pillar4_fusion": p4_result
        },
        "language": content_result.get("language", "English"),
        "lang_code": content_result.get("lang_code", "en"),
        "is_english": content_result.get("is_english", True),
        "sender": sender,
        "subject": subject,
        "body_preview": content[:600] + ("..." if len(content) > 600 else ""),
        "source": "gmail" if gmail_id else "manual",
    }
    
    # Predictions are not independently confirmed campaign labels. Keep the
    # explicit indexing helper for reviewed campaigns, but do not self-label
    # scans into the known-campaign collection (or silently swallow write errors).
    result_payload["campaign_memory_write"] = {
        "executed": False, "reason": "Independent campaign confirmation required"}

    store.add_scan(result_payload, session, gmail_id=gmail_id, user_id=user_id)
    
    return result_payload

