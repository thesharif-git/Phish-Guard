"""
PhishGuard Hybrid AI Scanning Engine
LightGBM + TF-IDF + Feature Engineering pipeline
"""
import re
import os
import numpy as np
import joblib
from typing import Optional
from scipy.sparse import hstack, csr_matrix
import math
from collections import Counter
from urllib.parse import urlparse
import requests

# ML Global State
ML_MODEL = None
TFIDF_VECTORIZER = None
URL_MODEL = None
DOMAIN_MODEL = None
MODEL_LOADED = False

from core.logger import logger

def load_ml_model():
    global ML_MODEL, TFIDF_VECTORIZER, URL_MODEL, DOMAIN_MODEL, MODEL_LOADED
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'models', 'phish_model.pkl')
        vec_path = os.path.join(project_root, 'models', 'vectorizer.pkl')
        url_model_path = os.path.join(project_root, 'models', 'url_model.pkl')
        domain_model_path = os.path.join(project_root, 'models', 'domain_model.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vec_path):
            ML_MODEL = joblib.load(model_path)
            TFIDF_VECTORIZER = joblib.load(vec_path)
            
        if os.path.exists(url_model_path):
            URL_MODEL = joblib.load(url_model_path)
            
        if os.path.exists(domain_model_path):
            DOMAIN_MODEL = joblib.load(domain_model_path)
            
        MODEL_LOADED = True
        logger.info("ML Phishing Engine & URL/Domain Models Loaded Successfully")
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")

def clean_text_for_ml(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://[^\s<>\"\']+|www\.[^\s<>\"\']+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_features_for_inference(raw_text: str) -> np.ndarray:
    """
    Extracts the same structural/behavioral features used during LightGBM training.
    Must stay in sync with training/train_model.py → extract_features().
    """
    t = raw_text if isinstance(raw_text, str) else ""
    features = [
        # URL signals
        len(re.findall(r'https?://', t)),
        int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', t))),
        int(bool(re.search(r'\.(xyz|biz|info|click|link|top|tk|pw|gq|ml)\b', t))),
        int(bool(re.search(r'bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly', t))),
        # Urgency / manipulation signals
        len(re.findall(r'urgent|verify|suspended|click here|act now|expires|limited time|immediate|unauthorized|confirm|update your', t, re.I)),
        t.count('!'),
        t.count('$'),
        len(re.findall(r'\b[A-Z]{3,}\b', t)),
        # Brand impersonation signals
        len(re.findall(r'paypal|microsoft|apple|amazon|google|netflix|bank of america|docusign|dropbox', t, re.I)),
        # Structure signals
        int(bool(re.search(r'<html|<body|<table|<div', t, re.I))),
        len(re.findall(r'<a\s+href', t, re.I)),
        len(t),
        len(t.split()),
        len(t) / max(len(t.split()), 1),
        # Punctuation / style signals
        t.count('?'),
        t.count(','),
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)


PHISHING_KEYWORDS = [
    "verify your account", "suspended", "unusual activity", "click here", "expires today",
    "confirm your identity", "update your billing", "unauthorized access", "immediate action",
    "your account will be closed", "limited time", "act now", "urgent", "password expires",
]

SUSPICIOUS_TLDS = [".xyz", ".biz", ".info", ".click", ".link", ".top", ".tk", ".pw"]

MALICIOUS_DOMAIN_LIST = [
    "paypa1-alert.com", "microsoft-365verify.net", "bank0famerica.info",
    "corp-helpdesk.xyz", "secure-docusign.xyz", "dropbox-shared.net",
]


def extract_urls(text: str) -> list[str]:
    pattern = r"https?://[^\s<>\"']+"
    return re.findall(pattern, text)


def extract_sender_domain(sender: str) -> Optional[str]:
    match = re.search(r"@([\w.\-]+)", sender)
    return match.group(1).lower() if match else None


def score_content(text: str) -> dict:
    text_lower = text.lower()
    keyword_hits = [kw for kw in PHISHING_KEYWORDS if kw in text_lower]
    urgency_score = min(len(keyword_hits) * 12, 80)
    has_urgency = len(keyword_hits) > 0
    suspicious_phrases = keyword_hits[:5]
    
    ml_probability = 0.0
    ml_used = False
    
    # Lazy load on first use instead of at startup to prevent server hang
    if not MODEL_LOADED:
        load_ml_model()
    
    # Run ML prediction if model is loaded
    if MODEL_LOADED and ML_MODEL and TFIDF_VECTORIZER:
        clean = clean_text_for_ml(text)
        if len(clean) > 5:
            try:
                # Step 1: TF-IDF on cleaned text
                tfidf_vec = TFIDF_VECTORIZER.transform([clean])

                # Step 2: Structural/behavioral feature vector (must match training pipeline)
                struct_feat = csr_matrix(extract_features_for_inference(text))

                # Step 3: Combine both feature sets — same as training
                combined_vec = hstack([tfidf_vec, struct_feat])

                # Step 4: Predict with LightGBM
                probs = ML_MODEL.predict_proba(combined_vec)

                if len(probs) > 0 and len(probs[0]) > 1:
                    ml_probability = float(probs[0][1])
                    ml_used = True

                    # Blend: ML drives 80%, heuristic urgency 20%
                    ml_score_100 = int(ml_probability * 100)
                    urgency_score = int((ml_score_100 * 0.8) + (urgency_score * 0.2))
                else:
                    logger.warning("LightGBM returned unexpected output. Falling back to heuristics.")
            except Exception as e:
                logger.error(f"LightGBM inference error: {e}")
                # Graceful fallback — urgency_score stays heuristic-based

    return {
        "score": urgency_score,
        "keyword_hits": keyword_hits,
        "suspicious_phrases": suspicious_phrases,
        "has_urgency_language": has_urgency,
        "has_impersonation": any(brand in text_lower for brand in ["paypal", "microsoft", "apple", "amazon", "google", "bank of america", "netflix"]),
        "readability": "Formatted" if "<html" in text_lower else "Plain text",
        "ml_powered": ml_used,
        "ml_probability": round(ml_probability, 4) if ml_used else None
    }


def entropy(s):
    p, lns = Counter(s), float(len(s))
    if lns == 0:
        return 0
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def extract_url_features(url):
    if not isinstance(url, str):
        url = ""
    url = url.lower()
    
    # Extract parts before stripping
    try:
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        path = parsed.path
        query = parsed.query
        netloc = parsed.netloc
    except Exception:
        path = ""
        query = ""
        netloc = ""

    # Normalize url for structural features
    norm_url = re.sub(r"^https?://", "", url)
    norm_url = re.sub(r"^www\.", "", norm_url)
    
    features = {
        'length':             len(norm_url),
        'entropy':            entropy(norm_url) if len(norm_url) > 0 else 0,
        'digit_count':        sum(c.isdigit() for c in norm_url),
        'special_char_count': sum(not c.isalnum() for c in norm_url),
        'hyphen_count':       norm_url.count('-'),
        'at_count':           norm_url.count('@'),
        'dot_count':          norm_url.count('.'),
        'path_length':        len(path),
        'query_length':       len(query),
        'has_ip':             1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', norm_url) else 0,
        'has_shortener':      1 if re.search(r'bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly|is\.gd', norm_url) else 0,
        'has_suspicious_tld': 1 if re.search(r'\.(xyz|biz|info|tk|pw|gq|ml|top|cc|ru)\b', netloc) else 0,
        'subdomain_count':    netloc.count('.'),
        'path_depth':         path.count('/'),
    }
    return features

def extract_domain_features(url):
    if not isinstance(url, str):
        url = ""
    url = url.lower()
    try:
        if not url.startswith('http'):
            parsed = urlparse('http://' + url)
        else:
            parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0]
    except Exception:
        domain = ""
    
    features = {
        'domain_length':           len(domain),
        'domain_entropy':          entropy(domain) if len(domain) > 0 else 0,
        'domain_digit_count':      sum(c.isdigit() for c in domain),
        'domain_hyphen_count':     domain.count('-'),
        'domain_dot_count':        domain.count('.'),
        'has_ip':                  1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) else 0,
        'has_suspicious_tld':      1 if re.search(r'\.(xyz|biz|info|tk|pw|gq|ml|top|cc|ru)\b', domain) else 0,
        'brand_impersonation':     1 if re.search(r'paypal|apple|google|microsoft|facebook|amazon|netflix|yahoo|bank|chase|wellsfargo', domain)
                                   and not re.search(r'^(paypal|apple|google|microsoft|facebook|amazon|netflix|yahoo|chase|wellsfargo)\.(com|org|net|co\.uk)$', domain) else 0,
        'subdomain_depth':         domain.count('.'),
        'has_numbers_in_domain':   1 if re.search(r'\d', domain.split('.')[0] if domain else '') else 0,
    }
    return features

def resolve_short_url(url: str) -> str:
    """Attempts to resolve a short URL to its final destination."""
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly", "bit.do", "mcaf.ee"]
    if any(s in url.lower() for s in shorteners):
        try:
            r = requests.get(url, stream=True, timeout=2, allow_redirects=True)
            return r.url
        except Exception:
            return url
    return url
def score_urls(text: str) -> dict:
    all_urls = extract_urls(text)

    # Cap: analyze at most 10 URLs per email.
    # Phishing emails have 1–3 key URLs; processing 50+ links in a newsletter
    # wastes seconds and doesn't improve accuracy.
    urls = all_urls[:10]

    flagged = []
    total_score = 0

    from utils.threat_intel import check_virustotal
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Safe-domain set — importing here avoids circular import at module load
    try:
        from utils.url_scanner import SAFE_DOMAINS, extract_domain as _extract_domain
    except ImportError:
        SAFE_DOMAINS = set()
        _extract_domain = lambda u: ""

    if not MODEL_LOADED:
        load_ml_model()

    # Pre-resolve and score heuristics fast (no network)
    url_work = []
    for raw_url in urls:
        url = resolve_short_url(raw_url)
        url_score = 0
        flags = []

        if url != raw_url:
            flags.append(f"Redirects to: {url}")
        if any(dom in url for dom in MALICIOUS_DOMAIN_LIST):
            url_score += 90
            flags.append("Known malicious domain")
        if any(tld in url for tld in SUSPICIOUS_TLDS):
            url_score += 30
            flags.append("Suspicious TLD")
        if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
            url_score += 40
            flags.append("IP address in URL")
        if any(brand in url.lower() for brand in ["paypal", "microsoft", "apple", "amazon", "google"]):
            domain = re.search(r"https?://([^/]+)", url)
            if domain and not any(b in domain.group(1) for b in ["paypal.com", "microsoft.com", "apple.com", "amazon.com", "google.com"]):
                url_score += 50
                flags.append("Brand name in suspicious domain")

        ml_url_prob = 0.0
        if URL_MODEL:
            try:
                features = extract_url_features(url)
                feat_array = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
                probs = URL_MODEL.predict_proba(feat_array)
                if len(probs) > 0 and len(probs[0]) > 1:
                    ml_url_prob = float(probs[0][1])
                    ml_url_score = int(ml_url_prob * 100)
                    if ml_url_score > 50:
                        url_score = max(url_score, ml_url_score)
                        flags.append(f"AI URL Analysis: Suspicious structure ({ml_url_score}% risk)")
            except Exception as e:
                logger.error(f"URL ML error: {e}")

        ml_domain_prob = 0.0
        if DOMAIN_MODEL:
            try:
                features = extract_domain_features(url)
                feat_array = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
                probs = DOMAIN_MODEL.predict_proba(feat_array)
                if len(probs) > 0 and len(probs[0]) > 1:
                    ml_domain_prob = float(probs[0][1])
                    ml_domain_score = int(ml_domain_prob * 100)
                    if ml_domain_score > 50:
                        url_score = max(url_score, ml_domain_score)
                        flags.append(f"AI Domain Analysis: Suspicious domain reputation ({ml_domain_score}% risk)")
            except Exception as e:
                logger.error(f"Domain ML error: {e}")

        url_work.append({"raw_url": raw_url, "url": url, "score": url_score, "flags": flags})

    # Run VirusTotal checks for all URLs IN PARALLEL
    # Skip safe-domain URLs — they will never be flagged, and hitting the API
    # wastes a precious rate-limited call (VirusTotal free tier: 4 req/min).
    def _vt_check(item):
        domain = _extract_domain(item["url"])
        # Fast skip for known-good domains
        is_safe = any(domain == s or domain.endswith("." + s) for s in SAFE_DOMAINS)
        if is_safe:
            return item["url"], {"available": False, "reason": "Trusted domain — VT check skipped"}
        try:
            return item["url"], check_virustotal(item["url"])
        except Exception:
            return item["url"], {"available": False, "reason": "Check failed"}

    vt_results = {}
    if url_work:
        with ThreadPoolExecutor(max_workers=min(len(url_work), 5)) as ex:
            futures = {ex.submit(_vt_check, item): item for item in url_work}
            for fut in as_completed(futures, timeout=5):
                try:
                    resolved_url, vt_result = fut.result()
                    vt_results[resolved_url] = vt_result
                except Exception:
                    pass


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
                    flags.append(f"VirusTotal: Clean (0/{vt_result.get('total_engines', '?')} engines flagged)")
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
        })
        total_score = max(total_score, url_score)

    return {
        "score": total_score,
        "urls_found": len(all_urls),       # true count, including any beyond the cap
        "urls_analyzed": len(urls),        # how many were actually checked
        "flagged_urls": flagged,
        "has_redirect": any(u["score"] > 50 for u in flagged),
        "has_shortener": any("bit.ly" in u["url"] or "tinyurl" in u["url"] for u in flagged),
    }



def score_headers(headers_text: str, spf: str = None, dkim: str = None, dmarc: str = None) -> dict:
    headers_lower = headers_text.lower() if headers_text else ""

    # Auto-detect from header text
    if spf is None:
        spf = "FAIL" if "spf=fail" in headers_lower else ("PASS" if "spf=pass" in headers_lower else "UNKNOWN")
    if dkim is None:
        dkim = "FAIL" if "dkim=fail" in headers_lower else ("PASS" if "dkim=pass" in headers_lower else "UNKNOWN")
    if dmarc is None:
        dmarc = "FAIL" if "dmarc=fail" in headers_lower else ("PASS" if "dmarc=pass" in headers_lower else "UNKNOWN")

    fail_count = [spf, dkim, dmarc].count("FAIL")
    header_score = fail_count * 28

    return {
        "score": min(header_score, 85),
        "spf": spf,
        "dkim": dkim,
        "dmarc": dmarc,
        "all_passed": fail_count == 0,
        "fail_count": fail_count,
        "has_suspicious_mailer": "anonymous" in headers_lower or "bulk" in headers_lower,
    }





def build_ai_paragraph(final_score: int, content_result: dict, url_result: dict, header_result: dict) -> str:
    urls = url_result.get("flagged_urls", [])
    has_shortener = url_result.get("has_shortener", False)
    has_urgency = content_result.get("has_urgency_language", False)
    has_impersonation = content_result.get("has_impersonation", False)
    suspicious_phrases = content_result.get("suspicious_phrases", [])
    has_requests = any(w in " ".join(suspicious_phrases).lower() for w in ["password", "otp", "bank", "credit", "login", "verify", "account"])
    header_failed = header_result.get("score", 100) < 50
    has_suspicious_mailer = header_result.get("has_suspicious_mailer", False)
    ml_high = content_result.get("ml_powered") and content_result.get("ml_probability", 0) > 0.6
    
    if final_score < 45:
        # Legitimate
        sentences = []
        if not header_failed:
            sentences.append("This email appears legitimate because the sender address uses the official company domain and all authentication checks passed successfully.")
        else:
            sentences.append("This email appears mostly legitimate, though the sender domain authentication could not be fully verified.")
            
        if not urls and not has_shortener:
            sentences.append("The links inside the email redirect to trusted domains with no suspicious behavior detected.")
            
        if not has_urgency and not has_requests:
            sentences.append("The message tone is professional and does not use pressure tactics, threats, or fake urgency.")
            
        sentences.append("No spoofing indicators, harmful attachments, or credential harvesting attempts were identified during analysis.")
        
        return " ".join(sentences)
        
    else:
        # Phishing / Suspicious
        sentences = []
        
        if has_impersonation or header_failed:
            sentences.append("This email is likely phishing because the sender domain closely imitates a trusted organization but contains hidden spelling changes or fails authentication checks commonly used in phishing attacks.")
        else:
            sentences.append("This email is likely phishing due to multiple suspicious indicators detected in the message.")
            
        if has_urgency:
            sentences.append("The email creates fake urgency to scare the user and pushes for immediate action.")
            
        if urls or has_shortener:
            sentences.append("Link analysis detected redirects to suspicious or hidden external websites that do not belong to the claimed organization.")
            
        if header_failed or has_suspicious_mailer:
            sentences.append("SPF and DMARC verification failed or a suspicious mailer was used, which strongly indicates sender spoofing.")
            
        if has_requests:
            sentences.append("The message also requests sensitive account information, which is a hallmark phishing indicator.")
            
        if ml_high and not (has_urgency or urls or has_requests):
            sentences.append("The overall structure and behavioral patterns of this email strongly match known automated phishing attacks.")
            
        return " ".join(sentences)

def build_explanation(final_score: int, content_result: dict, url_result: dict, header_result: dict) -> dict:
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
        signals.append(f"Email authentication failed: {', '.join(failed_checks)} — the sender's domain could not be verified, indicating possible spoofing.")
    elif fail_count == 0 and spf != "UNKNOWN":
        positive_signals.append("All email authentication checks (SPF, DKIM, DMARC) passed — the sender domain is verified.")

    if header_result.get("has_suspicious_mailer"):
        signals.append("A suspicious or anonymous mailer was detected in the email headers, commonly used by spam tools.")

    # --- Content signals ---
    phrases = content_result.get("suspicious_phrases", [])
    if phrases:
        signals.append(f"Urgency and manipulation language detected: \"{phrases[0]}\" — this is a classic phishing pressure tactic.")
    if content_result.get("has_impersonation"):
        signals.append("The email impersonates a well-known brand (e.g. PayPal, Microsoft, Apple). Legitimate companies send from verified domains.")
    if content_result.get("ml_powered") and content_result.get("ml_probability", 0) > 0.5:
        prob_pct = int(content_result["ml_probability"] * 100)
        signals.append(f"The trained AI/ML model classified this email as phishing with {prob_pct}% confidence.")
    elif content_result.get("ml_powered") and content_result.get("ml_probability", 1) < 0.3:
        positive_signals.append(f"The AI/ML model found the email content consistent with legitimate correspondence.")

    # --- URL signals ---
    flagged_urls = url_result.get("flagged_urls", [])
    for u in flagged_urls[:2]:
        if u["flags"]:
            signals.append(f"Suspicious URL detected: {u['url'][:70]} — reason: {', '.join(u['flags'])}.")
    if url_result.get("has_shortener"):
        signals.append("A URL shortener was used, which hides the true destination and is a common phishing technique.")
    if not flagged_urls and url_result.get("urls_found", 0) == 0:
        positive_signals.append("No URLs were found in the email body.")

    # --- Build verdict summary ---
    if final_score >= 75:
        summary = "This email has multiple high-confidence phishing indicators and should be treated as malicious."
    elif final_score >= 45:
        summary = "This email has suspicious characteristics. It may not be safe — proceed with caution."
    else:
        summary = "This email appears to be legitimate. No significant phishing indicators were detected."
        if not signals:
            signals = positive_signals  # Show only positive for clean emails

    return {
        "summary": summary,
        "risk_signals": signals if signals else ["No significant risk signals detected."],
        "positive_signals": positive_signals if final_score < 75 else [],
        "ai_paragraph": build_ai_paragraph(final_score, content_result, url_result, header_result)
    }


def build_threat_intel(content_result: dict, url_result: dict, sender: str) -> dict:
    """Build a simplified threat intelligence section."""
    matches = []
    sender_domain = extract_sender_domain(sender) if sender else None
    if sender_domain and any(bad in sender_domain for bad in MALICIOUS_DOMAIN_LIST):
        matches.append({
            "source": "PhishGuard Blocklist",
            "verdict": "BLOCKED",
            "detail": f"Sender domain '{sender_domain}' is on the known phishing domain blocklist."
        })

    for u in url_result.get("flagged_urls", []):
        if any(dom in u["url"] for dom in MALICIOUS_DOMAIN_LIST):
            matches.append({
                "source": "URL Threat Feed",
                "verdict": "MALICIOUS",
                "detail": f"URL matches known malicious domain pattern."
            })
            break

    if content_result.get("has_impersonation") and content_result.get("has_urgency_language"):
        matches.append({
            "source": "Behavioral Analysis",
            "verdict": "HIGH RISK",
            "detail": "Brand impersonation combined with urgency language is the #1 phishing pattern."
        })

    return {
        "score": min(len(matches) * 40, 100),
        "matches": matches,
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

def scan_email(content: str, session: Session, sender: str = "", subject: str = "", headers: str = "", gmail_id: str = "") -> dict:
    """
    Main scan function – runs all sub-engines and returns combined result.
    gmail_id: if provided, uses 'gm-{gmail_id}' as the scan_id for deduplication.
    """
    from data import store
    
    # Check custom block rules first
    if sender and store.is_blocked(session, sender):
        logger.warning(f"Scan short-circuited: Sender '{sender}' is blocked by a custom rule.")
        final_score = 100
        return {
            "final_score": final_score,
            "risk_level": "critical",
            "status": "PHISHING",
            "content_analysis": {"score": 100, "flags": [], "ml_probability": 1.0},
            "url_intelligence": {"score": 100, "flagged_urls": []},
            "header_authentication": {"score": 100, "spf_pass": False, "dkim_pass": False, "dmarc_pass": False},
            "threat_intelligence": {
                "score": 100,
                "matches": [{
                    "source": "Custom Security Rule",
                    "verdict": "BLOCKED",
                    "detail": f"Sender {sender} matched a manual blocklist rule."
                }]
            },
            "explanation": "This email was automatically blocked because its sender matches a Custom Security Rule configured by the administrator.",
            "sender": sender,
            "subject": subject,
            "body_preview": content[:600] + ("..." if len(content) > 600 else ""),
            "source": "gmail" if gmail_id else "manual",
            "scan_id": f"gm-{gmail_id}" if gmail_id else None
        }
    
    content_result = score_content(content + " " + subject)
    url_result = score_urls(content)
    header_result = score_headers(headers)
    threat_result = build_threat_intel(content_result, url_result, sender)

    # Weighted combination (content 40%, URL 40%, headers 20%)
    combined = (
        content_result["score"] * 0.40 +
        url_result["score"] * 0.40 +
        header_result["score"] * 0.20
    )
    final_score = min(round(combined), 100)

    # Boost score if threat intel matched something dangerous
    if threat_result["matches"]:
        final_score = min(final_score + 15, 100)

    explanation = build_explanation(final_score, content_result, url_result, header_result)

    result_payload = {
        "final_score": final_score,
        "risk_level": classify_risk(final_score),
        "status": get_status(final_score),
        "content_analysis": content_result,
        "url_intelligence": url_result,
        "header_authentication": header_result,
        "threat_intelligence": threat_result,
        "explanation": explanation,
        "sender": sender,
        "subject": subject,
        "body_preview": content[:600] + ("..." if len(content) > 600 else ""),
        "source": "gmail" if gmail_id else "manual",
    }
    
    store.add_scan(result_payload, session, gmail_id=gmail_id)
    
    return result_payload

