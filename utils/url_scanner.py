"""
Dedicated URL scanning engine.
Runs heuristics + real threat intel (VirusTotal, PhishTank, URLhaus, WHOIS).
"""
import re
import socket
import requests as _req
import numpy as np
import threading
from core.logger import logger
from utils.threat_intel import full_threat_check
from utils.html_scanner import analyze_live_html
import services.scan_engine as scan_engine


def geolocate_ip(domain: str) -> dict:
    """Resolve domain to IP, then call ip-api.com for real geolocation data."""
    try:
        ip = socket.gethostbyname(domain)
    except Exception:
        ip = None

    fallback = {
        "ip": ip or "Unknown",
        "city": "Unknown",
        "region": "",
        "country": "Unknown",
        "countryCode": "",
        "isp": "Unknown",
        "lat": 37.7749,
        "lon": -122.4194,
    }

    if not ip:
        return fallback

    try:
        r = _req.get(f"http://ip-api.com/json/{ip}?fields=status,city,regionName,country,countryCode,isp,lat,lon", timeout=2)
        d = r.json()
        if d.get("status") == "success":
            return {
                "ip": ip,
                "city": d.get("city", "Unknown"),
                "region": d.get("regionName", ""),
                "country": d.get("country", "Unknown"),
                "countryCode": d.get("countryCode", ""),
                "isp": d.get("isp", "Unknown"),
                "lat": d.get("lat", 37.7749),
                "lon": d.get("lon", -122.4194),
            }
    except Exception as e:
        logger.warning(f"GeoIP lookup failed for {ip}: {e}")

    return fallback

SUSPICIOUS_TLDS = [".xyz", ".biz", ".info", ".click", ".link", ".top", ".tk", ".pw", ".ru", ".cn", ".cc"]
BRAND_NAMES = ["paypal", "microsoft", "apple", "amazon", "google", "netflix", "facebook", "instagram",
               "bankofamerica", "chase", "wellsfargo", "dhl", "fedex", "ups", "irs"]

# Well-known safe domains — never flag these regardless of ML score
SAFE_DOMAINS = {
    "google.com", "gmail.com", "google.co.uk", "google.com.pk", "googleapis.com",
    "github.com", "github.io", "githubusercontent.com",
    "microsoft.com", "office.com", "live.com", "outlook.com", "azure.com",
    "apple.com", "icloud.com",
    "amazon.com", "amazonaws.com", "aws.amazon.com",
    "facebook.com", "instagram.com", "meta.com",
    "twitter.com", "x.com", "t.co",
    "youtube.com", "youtu.be",
    "linkedin.com",
    "wikipedia.org", "wikimedia.org",
    "stackoverflow.com", "stackexchange.com",
    "cloudflare.com", "cdn.cloudflare.net",
    "npmjs.com", "pypi.org",
    "stripe.com", "paypal.com",
    "openai.com", "anthropic.com",
    "dropbox.com", "box.com",
    "zoom.us", "slack.com", "discord.com",
    "shopify.com", "wix.com",
    "wordpress.com", "wordpress.org",
    "reddit.com",
    "netflix.com",
}

MALICIOUS_DOMAIN_LIST = [
    "paypa1-alert.com", "microsoft-365verify.net", "bank0famerica.info",
    "corp-helpdesk.xyz", "secure-docusign.xyz", "dropbox-shared.net",
    "paypa1-secure.com", "paypa1-secure.ru",
]


def extract_domain(url: str) -> str:
    match = re.search(r"https?://([^/?\s]+)", url)
    if not match:
        return ""
    domain = match.group(1).lower().split(":")[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def run_url_heuristics(url: str) -> dict:
    """Fast local heuristics — no network calls."""
    url_lower = url.lower()
    domain = extract_domain(url)
    flags = []
    score = 0

    # Check blocklist
    if any(bad in url_lower for bad in MALICIOUS_DOMAIN_LIST):
        flags.append("Domain is on the PhishGuard known-malicious blocklist")
        score += 90

    # Suspicious TLD
    matching_tld = next((tld for tld in SUSPICIOUS_TLDS if domain.endswith(tld)), None)
    if matching_tld:
        flags.append(f"Suspicious TLD: '{matching_tld}' is commonly used in phishing")
        score += 25

    # IP address instead of domain
    if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
        flags.append("URL uses a raw IP address instead of a domain name")
        score += 40

    # Brand name in non-brand domain (typosquatting)
    for brand in BRAND_NAMES:
        if brand in url_lower:
            legit_domains = [f"{brand}.com", f"{brand}.net", f"{brand}.org"]
            if not any(domain == ld or domain.endswith(f".{ld}") for ld in legit_domains):
                flags.append(f"Brand name '{brand}' used in a non-official domain — possible impersonation")
                score += 50
                break

    # URL shortener
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "buff.ly", "is.gd"]
    if any(s in domain for s in shorteners):
        flags.append("URL shortener detected — hides the true destination")
        score += 20

    # Long subdomain chain (e.g. secure.login.verify.paypal.phish.com)
    subdomain_count = domain.count(".")
    if subdomain_count >= 4:
        flags.append(f"Unusually deep subdomain structure ({subdomain_count} levels) — common in phishing")
        score += 20

    # URL-encoded characters (obfuscation)
    if "%" in url and re.search(r"%[0-9a-fA-F]{2}", url):
        flags.append("URL contains percent-encoded characters, which can be used to obfuscate the destination")
        score += 15

    # HTTPS check
    is_https = url.lower().startswith("https://")

    return {
        "domain": domain,
        "flags": flags,
        "heuristic_score": min(score, 100),
        "is_https": is_https,
        "has_flags": len(flags) > 0,
    }

def is_safe_domain(url: str) -> bool:
    """Returns True if the domain is a known-safe popular site."""
    domain = extract_domain(url)
    # Check exact match or subdomain of a known safe domain
    for safe in SAFE_DOMAINS:
        if domain == safe or domain.endswith("." + safe):
            return True
    return False



def classify_risk(score: int) -> str:
    if score >= 75:
        return "critical"
    elif score >= 45:
        return "medium"
    return "safe"


def get_status(score: int) -> str:
    if score >= 75:
        return "PHISHING"
    elif score >= 45:
        return "SUSPICIOUS"
    return "CLEAN"


def _normalize_html_result(raw: dict) -> dict:
    """
    Ensure html_result always has a complete, consistent structure.
    When the HTML scanner timed out or failed, fields should clearly say
    'Could not retrieve' rather than returning None or crashing the template.
    """
    tech_raw = raw.get("tech") or {}
    return {
        "success":            raw.get("success", False),
        "score":              raw.get("score", 0),
        "flags":              raw.get("flags", []),
        "has_password_field": raw.get("has_password_field", False),
        "brand_mismatch":     raw.get("brand_mismatch", False),
        "obfuscated_js":      raw.get("obfuscated_js", False),
        # Server — only set if we actually got it from a real HTTP header
        "server":             raw.get("server") or "Could not retrieve",
        "error":              raw.get("error"),
        # SSL — always a dict, never None
        "ssl_info":           raw.get("ssl_info") or {
            "available": False,
            "valid": False,
            "issuer": "Could not retrieve",
            "error": "SSL check did not complete"
        },
        # Tech — always a dict with all keys, empty lists when not scanned
        "tech": {
            "cms":       tech_raw.get("cms", []),
            "analytics": tech_raw.get("analytics", []),
            "tracking":  tech_raw.get("tracking", []),
            "language":  tech_raw.get("language", "Could not retrieve"
                         if not raw.get("success") else "Unknown"),
        },
    }


# ── Main scan function ────────────────────────────────────────────────────────
def scan_url(raw_url: str) -> dict:
    """
    Main URL scan function.
    Combines local heuristics, Machine Learning models, and live threat intel APIs.
    """
    logger.info(f"Scanning URL: {raw_url[:100]}")

    if not scan_engine.MODEL_LOADED:
        scan_engine.load_ml_model()

    # 1. Unshorten URL
    url = scan_engine.resolve_short_url(raw_url)

    # 2. Extract domain
    domain_safe = extract_domain(url)
    
    # 3. Check custom block rules first
    from data.database import engine
    from sqlmodel import Session
    from data import store
    with Session(engine) as session:
        if store.is_blocked(session, domain_safe):
            logger.warning(f"Scan short-circuited: Domain '{domain_safe}' is blocked by a custom rule.")
            geo = geolocate_ip(domain_safe)
            return {
                "url": raw_url,
                "resolved_url": url if url != raw_url else None,
                "final_score": 100,
                "risk_level": "critical",
                "status": "PHISHING",
                "summary": "🚨 This URL was automatically blocked because its domain matches a Custom Security Rule configured by the administrator.",
                "signals": ["Custom Security Rule: Domain matches blocklist"],
                "heuristics": {"domain": domain_safe, "is_https": url.startswith("https")},
                "threat_intel": {"phish_tank": {"is_malicious": True, "available": True}, "urlhaus": {"is_malicious": True, "available": True}, "domain_age": {"available": False}},
                "html_analysis": {"success": False, "error": "Blocked by policy - scan aborted", "tech": {}, "ssl_info": {}},
                "domain": domain_safe,
                "is_https": url.startswith("https"),
                "has_redirect": url != raw_url,
                "has_shortener": url != raw_url,
                "geo": geo,
                "server": "Unknown",
            }

    # 4. Short-circuit: known-safe domains always return clean
    if is_safe_domain(url):
        geo = geolocate_ip(domain_safe)
        # Run real WHOIS for accurate domain info even on safe domains
        try:
            from utils.threat_intel import get_domain_age
            real_whois = get_domain_age(url)
        except Exception:
            real_whois = {"available": False, "reason": "WHOIS lookup skipped"}

        # Run HTML scan in background — safe domains might block bots, that's OK
        html_holder = {}
        def _safe_html():
            html_holder.update(analyze_live_html(url))
        ht = threading.Thread(target=_safe_html)
        ht.start(); ht.join(timeout=12)
        html_safe = _normalize_html_result(html_holder if html_holder else {})

        return {
            "url": raw_url,
            "resolved_url": url if url != raw_url else None,
            "final_score": 2,
            "risk_level": "safe",
            "status": "CLEAN",
            "summary": "✅ This URL belongs to a well-known, trusted domain. No threats detected.",
            "signals": ["Domain is on the PhishGuard trusted-site allowlist."],
            "heuristics": run_url_heuristics(url),
            "threat_intel": {
                "overall": "CLEAN",
                "overall_score": 0,
                "virustotal":  {"available": False, "reason": "Trusted domain — skipped"},
                "phishtank":   {"available": False, "reason": "Trusted domain — skipped"},
                "urlhaus":     {"available": False, "reason": "Trusted domain — skipped"},
                "domain_age":  real_whois,
            },
            "domain": domain_safe,
            "is_https": url.lower().startswith("https://"),
            "has_redirect": url != raw_url,
            "has_shortener": False,
            "geo": geo,
            "html_analysis": html_safe,
            "server": html_safe.get("server", "Unknown"),
        }

    # 3. Run fast local heuristics
    heuristics = run_url_heuristics(url)

    # 4. ML Predictions — capped to avoid overtrained model dominating
    ml_url_score = 0
    ml_domain_score = 0
    ai_flags = []
    
    if scan_engine.URL_MODEL:
        try:
            features = scan_engine.extract_url_features(url)
            feat_array = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
            probs = scan_engine.URL_MODEL.predict_proba(feat_array)
            if len(probs) > 0 and len(probs[0]) > 1:
                ml_url_score = int(float(probs[0][1]) * 100)
                # Only flag if model is very confident (≥70%)
                if ml_url_score >= 70:
                    ai_flags.append(f"AI URL Analysis: Suspicious structure ({ml_url_score}% risk)")
                else:
                    ml_url_score = 0  # ignore low-confidence ML signals
        except Exception as e:
            logger.error(f"URL ML error: {e}")

    if scan_engine.DOMAIN_MODEL:
        try:
            features = scan_engine.extract_domain_features(url)
            feat_array = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
            probs = scan_engine.DOMAIN_MODEL.predict_proba(feat_array)
            if len(probs) > 0 and len(probs[0]) > 1:
                ml_domain_score = int(float(probs[0][1]) * 100)
                if ml_domain_score >= 70:
                    ai_flags.append(f"AI Domain Analysis: Suspicious domain reputation ({ml_domain_score}% risk)")
                else:
                    ml_domain_score = 0
        except Exception as e:
            logger.error(f"Domain ML error: {e}")

    # 5. Run live threat intel, HTML analysis, and GeoIP all in parallel
    threat_intel_result = {}
    html_result_holder  = {}
    geo_result_holder   = {}

    def _run_threat_intel():
        threat_intel_result.update(full_threat_check(url))

    def _run_html_scan():
        html_result_holder.update(analyze_live_html(url))

    def _run_geo():
        geo_result_holder.update(geolocate_ip(heuristics["domain"]))

    t1 = threading.Thread(target=_run_threat_intel)
    t2 = threading.Thread(target=_run_html_scan)
    t3 = threading.Thread(target=_run_geo)
    t1.start(); t2.start(); t3.start()
    t1.join(timeout=12)   # threat intel: VT + PhishTank + URLhaus + WHOIS in parallel
    t2.join(timeout=12)   # HTML scan: SSL cert check (3s) + page fetch (3s) + parsing
    t3.join(timeout=4)    # GeoIP: fast UDP lookup

    threat_intel = threat_intel_result if threat_intel_result else {"overall": "CLEAN", "overall_score": 5}
    html_result  = _normalize_html_result(html_result_holder)
    geo          = geo_result_holder   if geo_result_holder   else geolocate_ip(heuristics["domain"])

    # 6. Combine scores: heuristics 25%, ML 15%, threat intel 40%, HTML live analysis 20%
    #    Threat intel (VirusTotal/PhishTank/URLhaus) is the most reliable signal.
    #    HTML analysis catches compromised domains that pass all other checks.
    heuristic_score = heuristics["heuristic_score"]
    ml_combined = max(ml_url_score, ml_domain_score)
    intel_score = threat_intel.get("overall_score", 5)
    html_score = html_result.get("score", 0)
    
    combined_score = min(round(
        heuristic_score * 0.25 +
        ml_combined   * 0.15 +
        intel_score   * 0.40 +
        html_score    * 0.20
    ), 100)
    
    # Hard overrides — any single definitive signal forces critical
    if intel_score >= 90:
        combined_score = max(combined_score, intel_score)
    if heuristic_score >= 75:
        # Blocklist hit or clear typosquatting — trust our own rules even if VT hasn't indexed it
        combined_score = max(combined_score, heuristic_score)
    if html_result.get("brand_mismatch"):          # e.g. title says PayPal, domain isn't paypal.com
        combined_score = max(combined_score, 92)
    if html_result.get("has_password_field") and not url.startswith("https://"):
        combined_score = max(combined_score, 88)   # password field over plain HTTP = very dangerous

    # If all real threat intel sources return clean AND heuristics are also low,
    # cap score at 44 so an overtrained ML model cannot push a clean URL into "critical"
    # But do NOT cap if heuristics or HTML are themselves already flagging something real
    if intel_score <= 10 and heuristic_score < 45 and html_score < 45:
        combined_score = min(combined_score, 44)

    final_score = combined_score
    risk_level = classify_risk(final_score)
    status = get_status(final_score)

    # Build explanation
    signals = []
    if url != raw_url:
        signals.append(f"Redirects to: {url}")
        
    signals.extend(ai_flags)
    signals.extend(heuristics["flags"])

    vt = threat_intel.get("virustotal", {})
    if vt.get("available") and vt.get("malicious", 0) > 0:
        signals.append(f"VirusTotal: {vt['malicious']}/{vt['total_engines']} security engines flagged this URL as malicious")
    elif vt.get("available"):
        signals.append(f"VirusTotal: 0/{vt.get('total_engines', '?')} engines flagged — no known detections")

    pt = threat_intel.get("phishtank", {})
    if pt.get("available") and pt.get("is_phishing"):
        signals.append("PhishTank: This URL is confirmed as a phishing site in the PhishTank community database")
    elif pt.get("available"):
        signals.append("PhishTank: URL not found in the phishing database")

    uh = threat_intel.get("urlhaus", {})
    if uh.get("available") and uh.get("is_malicious"):
        signals.append(f"URLhaus: Listed as malware distribution ({uh.get('threat', 'unknown')})")
    elif uh.get("available"):
        signals.append("URLhaus: Not found in the malware URL feed")

    da = threat_intel.get("domain_age", {})
    if da.get("available"):
        if da.get("is_suspicious"):
            signals.append(f"Domain Age: Only {da['age_days']} days old — very new domains are high risk")
        else:
            signals.append(f"Domain Age: {da['age_days']} days old — established domain")

    if not heuristics["is_https"]:
        signals.append("URL uses HTTP (not HTTPS) — no encryption, increased risk")

    # 7. HTML live analysis signals
    if html_result.get("success"):
        signals.extend(html_result.get("flags", []))
    elif html_result.get("error"):
        signals.append(f"Live HTML scan: {html_result['error']} — could not retrieve page content")

    # Verdict summary
    if final_score >= 75:
        summary = "🚨 This URL has multiple high-confidence malicious indicators. Do NOT visit it."
    elif final_score >= 45:
        summary = "⚠️ This URL has suspicious characteristics. Proceed with caution."
    else:
        summary = "✅ This URL appears to be safe. No significant threats detected."

    return {
        "url": raw_url,
        "resolved_url": url if url != raw_url else None,
        "final_score": final_score,
        "risk_level": risk_level,
        "status": status,
        "summary": summary,
        "signals": signals if signals else ["No suspicious signals detected."],
        "heuristics": heuristics,
        "threat_intel": threat_intel,
        "html_analysis": html_result,
        "domain": heuristics["domain"],
        "is_https": heuristics["is_https"],
        "has_redirect": url != raw_url,
        "has_shortener": url != raw_url,
        "geo": geo,
        "server": html_result.get("server", "Unknown"),
    }
