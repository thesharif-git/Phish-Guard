"""
Dedicated URL scanning engine.
Runs heuristics + real threat intel (VirusTotal, PhishTank, URLhaus, WHOIS).
"""
import re
import socket
from urllib.parse import urlparse
import requests as _req
import numpy as np
import pandas as pd
import time
import threading
from core.logger import logger
from utils.threat_intel import (
    full_threat_check,
    analyze_tls_cert_lineage,
    calculate_dns_ephemerality,
    check_asn_brand_incongruity
)
from utils.html_scanner import analyze_live_html
from services.origintrace_pillars import (
    ProvenanceMatrix,
    InfrastructureLineage,
    SemanticDeception,
    EvidentialFusion,
    CampaignAttributor
)


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

# Truly suspicious gTLDs/ccTLDs — almost exclusively used by phishing (not legitimate businesses)
SUSPICIOUS_TLDS = [".xyz", ".click", ".top", ".tk", ".pw", ".cc", ".link", ".gq", ".ml", ".ga", ".cf", ".work", ".live"]

# Legitimate national ccTLDs — official country domains used by real businesses worldwide
# These must NEVER be flagged as suspicious by TLD alone
TRUSTED_CCTLDS = {
    ".pk", ".in", ".uk", ".au", ".ca", ".de", ".fr", ".jp", ".br", ".mx", ".za", ".ng", ".ke",
    ".ae", ".sa", ".bd", ".ru", ".cn", ".tr", ".it", ".es", ".nl", ".se", ".no", ".dk", ".fi",
    ".nz", ".sg", ".my", ".ph", ".id", ".th", ".vn", ".eg", ".ma", ".tn", ".gh", ".tz",
    ".com.pk", ".net.pk", ".org.pk", ".edu.pk", ".gov.pk",
    ".co.uk", ".co.in", ".co.au", ".co.nz", ".co.za",
    ".biz", ".info",  # common gTLDs used by legit businesses too
}

# Explicit mapping of high-value target brands to their legitimate base domains
BRAND_LEGIT_DOMAINS = {
    "paypal": ["paypal.com", "paypal.me"],
    "microsoft": ["microsoft.com", "office.com", "live.com", "outlook.com", "azure.com", "office365.com", "msn.com"],
    "apple": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "amazonaws.com", "media-amazon.com"],
    "google": ["google.com", "gmail.com", "googleapis.com", "gstatic.com", "youtube.com", "youtu.be"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com", "messenger.com", "meta.com"],
    "instagram": ["instagram.com"],
    "bankofamerica": ["bankofamerica.com", "bofa.com"],
    "chase": ["chase.com", "jpmorgan.com", "jpmorganchase.com"],
    "wellsfargo": ["wellsfargo.com"],
    "dhl": ["dhl.com", "dhl.de"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "usps": ["usps.com"],
    "irs": ["irs.gov"],
    "coinbase": ["coinbase.com"],
    "binance": ["binance.com"],
    "metamask": ["metamask.io"],
    "whatsapp": ["whatsapp.com"],
    "telegram": ["telegram.org", "t.me"],
    "dropbox": ["dropbox.com"],
    "adobe": ["adobe.com"],
    "roblox": ["roblox.com"],
    "steam": ["steampowered.com", "steamcommunity.com"],
    "linkedin": ["linkedin.com"],
    "twitter": ["twitter.com", "x.com", "t.co"],
    "yahoo": ["yahoo.com"],
    "outlook": ["outlook.com", "office.com", "microsoft.com"],
    "office365": ["office365.com", "office.com", "microsoft.com"],
}

BRAND_NAMES = list(BRAND_LEGIT_DOMAINS.keys())

# Homoglyph / digit-substitution normalization map
# Attackers replace letters with look-alike digits to evade string matching:
#   paypa1.com  → paypal.com,  micr0soft.com → microsoft.com
#   g00gle.com  → google.com,  4mazon.com    → amazon.com
_HOMOGLYPH_MAP = str.maketrans({
    '0': 'o',   # zero → o
    '1': 'l',   # one  → l  (most common: paypa1, app1e)
    '3': 'e',   # 3    → e
    '4': 'a',   # 4    → a
    '5': 's',   # 5    → s
    '6': 'g',   # 6    → g  (6oogle)
    '8': 'b',   # 8    → b
    '@': 'a',   # @    → a
    '$': 's',   # $    → s
})

def _normalize_homoglyphs(text: str) -> str:
    """Normalize digit/symbol substitutions to letters for brand matching."""
    return text.translate(_HOMOGLYPH_MAP)

# Well-known safe domains — never flag these regardless of ML score
SAFE_DOMAINS = {
    "google.com", "gmail.com", "google.co.uk", "google.com.pk", "googleapis.com", "gstatic.com",
    "github.com", "github.io", "githubusercontent.com",
    "microsoft.com", "office.com", "live.com", "outlook.com", "azure.com", "office365.com", "msn.com",
    "apple.com", "icloud.com",
    "amazon.com", "amazonaws.com", "aws.amazon.com",
    "facebook.com", "instagram.com", "meta.com", "whatsapp.com", "messenger.com",
    "twitter.com", "x.com", "t.co",
    "youtube.com", "youtu.be",
    "linkedin.com",
    "wikipedia.org", "wikimedia.org",
    "stackoverflow.com", "stackexchange.com",
    "cloudflare.com", "cdn.cloudflare.net",
    "npmjs.com", "pypi.org",
    "stripe.com", "paypal.com", "paypal.me",
    "openai.com", "anthropic.com",
    "dropbox.com", "box.com",
    "zoom.us", "slack.com", "discord.com",
    "shopify.com", "wix.com",
    "wordpress.com", "wordpress.org",
    "reddit.com",
    "netflix.com",
    "bankofamerica.com", "bofa.com", "chase.com", "wellsfargo.com",
    "dhl.com", "fedex.com", "ups.com", "usps.com", "irs.gov",
    "coinbase.com", "binance.com", "metamask.io",
    "telegram.org", "adobe.com", "yahoo.com",
    "roblox.com", "steampowered.com", "steamcommunity.com"
}

def is_safe_authority_domain(domain: str) -> bool:
    """Return True if domain is in or a subdomain of SAFE_DOMAINS."""
    if not domain:
        return False
    d = domain.lower().strip().rstrip(".")
    if d.startswith("www."):
        d = d[4:]
    return any(d == safe or d.endswith("." + safe) for safe in SAFE_DOMAINS)


def extract_domain(url: str) -> str:
    if not url:
        return ""
    raw = url.strip()
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = "https://" + raw
    try:
        domain = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.strip().lower()


def run_url_heuristics(url: str) -> dict:
    """Fast local heuristics — no network calls."""
    domain = extract_domain(url)
    flags = []
    score = 0

    # Suspicious TLD — only flag genuinely high-risk gTLDs, never legitimate national ccTLDs
    # First check if domain uses a trusted/national ccTLD — skip TLD penalty entirely
    uses_trusted_cctld = any(domain.endswith(tld) for tld in TRUSTED_CCTLDS)
    matching_tld = None
    if not uses_trusted_cctld:
        matching_tld = next((tld for tld in SUSPICIOUS_TLDS if domain.endswith(tld)), None)
    if matching_tld:
        flags.append(f"Suspicious TLD: '{matching_tld}' is almost exclusively used by disposable phishing campaigns")
        score += 25

    # IP address instead of domain
    if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
        flags.append("URL uses a raw IP address instead of a domain name")
        score += 40

    # Brand impersonation check: protected brand in an official or homoglyph domain
    # Normalize homoglyphs (paypa1 → paypal, micr0soft → microsoft) before matching
    domain_normalized = _normalize_homoglyphs(domain)
    for brand, legit_domains in BRAND_LEGIT_DOMAINS.items():
        # Check both the raw domain (paypal.com) and the homoglyph-normalized version (paypa1 → paypal)
        brand_in_raw    = brand in domain
        brand_in_normed = brand in domain_normalized
        if brand_in_raw or brand_in_normed:
            is_legit = any(domain == ld or domain.endswith(f".{ld}") for ld in legit_domains)
            if not is_legit:
                if brand_in_normed and not brand_in_raw:
                    # Homoglyph attack detected (e.g. paypa1, micr0soft)
                    flags.append(
                        f"Homoglyph/Typosquatting attack: '{domain}' impersonates '{brand}' "
                        f"using digit/symbol substitution (normalized: '{domain_normalized}')"
                    )
                    score += 70  # Higher penalty — deliberate character obfuscation
                else:
                    flags.append(f"Brand impersonation alert: Protected brand '{brand}' appears in an unofficial domain")
                    score += 55
                # Check for credential harvesting or security cues anywhere in domain
                security_cue = re.search(
                    r"(?:^|[-_.])(secure|security|login|signin|sign-in|verify|verification|account|support|billing|update|auth|portal|recover|alert|banking|service|help|wallet|token)(?:$|[-_.])",
                    domain,
                )
                if security_cue:
                    flags.append(f"Credential harvesting indicator: Brand '{brand}' is paired with '{security_cue.group(1)}' in domain")
                    score += 40
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

_TRANCO_DOMAINS = None

def get_authority_domains() -> frozenset:
    global _TRANCO_DOMAINS
    if _TRANCO_DOMAINS is None:
        import os
        domains = set(SAFE_DOMAINS)
        tranco_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tranco_top50k.txt")
        if os.path.exists(tranco_path):
            try:
                with open(tranco_path, "r", encoding="utf-8") as f:
                    for line in f:
                        d = line.strip().lower()
                        if d:
                            domains.add(d)
            except Exception as e:
                logger.warning(f"Failed to load tranco_top50k.txt: {e}")
        _TRANCO_DOMAINS = frozenset(domains)
    return _TRANCO_DOMAINS

def is_safe_authority_domain(domain: str) -> bool:
    """
    Tier 0 Authority Anchor:
    Returns True if domain is an established global authority:
    - In Tranco Top 50,000 global ranking
    - Belongs to .edu, .gov, or .mil
    - In curated SAFE_DOMAINS
    """
    if not domain:
        return False
    d = domain.lower().strip()
    if d.startswith("www."):
        d = d[4:]
    if d.endswith(".edu") or d.endswith(".gov") or d.endswith(".mil"):
        return True
    auth = get_authority_domains()
    if d in auth:
        return True
    parts = d.split(".")
    if len(parts) > 2 and ".".join(parts[-2:]) in auth:
        return True
    if len(parts) > 3 and ".".join(parts[-3:]) in auth:
        return True
    return False

# Shared User-Generated Content (UGC), Cloud Storage, and Form Hosting Providers
# These domains have legitimate infrastructure, but host public user uploads frequently abused in phishing.
UGC_HOSTING_PLATFORMS = {
    "docs.google.com", "sites.google.com", "forms.gle", "drive.google.com", "storage.googleapis.com",
    "forms.office.com", "sway.office.com", "onedrive.live.com", "1drv.ms",
    "github.io", "pages.dev", "workers.dev",
    "firebaseapp.com", "web.app",
    "s3.amazonaws.com", "blob.core.windows.net",
    "notion.site", "canva.site", "weebly.com", "wixsite.com", "mystrikingly.com",
    "formstack.com", "typeform.com", "jotform.com", "surveymonkey.com"
}

def is_ugc_hosting_platform(domain: str, path: str = "") -> bool:
    """Detects whether a URL is hosted on a public user-generated content or form platform."""
    d = (domain or "").lower().strip()
    if d.startswith("www."):
        d = d[4:]
    if d in UGC_HOSTING_PLATFORMS:
        return True
    if any(d.endswith("." + p) for p in UGC_HOSTING_PLATFORMS):
        return True
    if "google.com" in d and any(seg in path.lower() for seg in ["/forms", "/spreadsheets", "/document"]):
        return True
    return False

def is_safe_domain(url: str) -> bool:
    """Returns True if the domain is a known-safe popular site or established global authority."""
    domain = extract_domain(url)
    return is_safe_authority_domain(domain)



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

    import services.scan_engine as scan_engine
    if not scan_engine.MODEL_LOADED:
        scan_engine.load_ml_model()

    raw_url = raw_url.strip() if raw_url else ""
    if not raw_url:
        return {
            "url": "",
            "final_score": 0,
            "risk_level": "safe",
            "status": "CLEAN",
            "summary": "No URL provided.",
            "signals": [],
            "heuristics": {"domain": "", "is_https": False},
            "threat_intel": {},
            "html_analysis": {},
            "domain": "",
            "is_https": False,
            "has_redirect": False,
            "has_shortener": False,
            "geo": {},
            "server": "Unknown",
        }

    # Normalize url so resolution and heuristics work smoothly with or without https://
    normalized_url = raw_url
    if not normalized_url.startswith(("http://", "https://")):
        normalized_url = "https://" + normalized_url

    if not scan_engine.MODEL_LOADED:
        scan_engine.load_ml_model()

    # 1. Unshorten URL
    url = scan_engine.resolve_short_url(normalized_url)

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

    # 3. Run fast local heuristics
    heuristics = run_url_heuristics(url)

    # 4. ML Predictions — differentiated for domain-only vs full URL with path
    ml_url_score = 0
    ml_domain_score = 0
    url_model_probability = None
    domain_model_probability = None
    ai_flags = []
    
    parsed_for_ml = urlparse(url)
    has_path_or_query = bool(parsed_for_ml.path.strip('/') or parsed_for_ml.query)

    # Root URLs are included in the URL training corpus and use the URL specialist.
    if scan_engine.URL_MODEL:
        try:
            features = scan_engine.extract_url_features(url)
            feat_df = scan_engine.numeric_url_frame(scan_engine.URL_MODEL, [features])
            if hasattr(scan_engine.URL_MODEL, "_transform_urls"):
                probs = scan_engine.URL_MODEL.predict_proba([url])
            else:
                probs = scan_engine.URL_MODEL.predict_proba(feat_df)
            if len(probs) > 0 and len(probs[0]) > 1:
                url_model_probability = float(probs[0][1])
                ml_url_score = int(url_model_probability * 100)
                if ml_url_score >= 70:
                    ai_flags.append(f"AI URL Structure Analysis: {ml_url_score}% phishing confidence")
                else:
                    ml_url_score = 0
        except Exception as e:
            logger.error(f"URL ML error: {e}")

    if scan_engine.DOMAIN_MODEL:
        try:
            features = scan_engine.extract_domain_features(scan_engine.domain_model_input(url))
            feat_df = scan_engine.numeric_url_frame(scan_engine.DOMAIN_MODEL, [features])
            probs = scan_engine.DOMAIN_MODEL.predict_proba(feat_df)
            if len(probs) > 0 and len(probs[0]) > 1:
                domain_model_probability = float(probs[0][1])
                # Diagnostics only: URL-derived training labels do not establish
                # domain-wide ownership/reputation. Never turn this into a threat flag.
        except Exception as e:
            logger.error(f"Domain ML error: {e}")

    # 5. Parallel Multi-Modal Intelligence Harvesting:
    #    - Live Threat Intel (VT, PhishTank, URLhaus, WHOIS domain age)
    #    - Live HTML DOM & Credential Analysis (DOM sinks, forms, SSL)
    #    - GeoIP (ip-api)
    #    - TLS Certificate Lineage Mismatch (X.509 SNI, Subject/Issuer org, SHA-256 fingerprint)
    #    - DNS Ephemerality & Fast-Flux Index (DoH TTL, MX, NS)
    #    - Autonomous System Brand Incongruity ABCI (BGP ASN vs Enterprise network)
    threat_intel_result = {}
    html_result_holder  = {}
    geo_result_holder   = {}
    tls_result_holder   = {}
    dns_result_holder   = {}
    asn_result_holder   = {}

    def _run_threat_intel():
        threat_intel_result.update(full_threat_check(url))

    def _run_html_scan():
        html_result_holder.update(analyze_live_html(url))

    def _run_geo():
        geo_result_holder.update(geolocate_ip(domain_safe))

    def _run_tls():
        tls_result_holder.update(analyze_tls_cert_lineage(domain_safe))

    def _run_dns():
        dns_result_holder.update(calculate_dns_ephemerality(domain_safe))

    def _run_asn():
        asn_result_holder.update(check_asn_brand_incongruity(domain_safe))

    threads = [
        threading.Thread(target=_run_threat_intel),
        threading.Thread(target=_run_html_scan),
        threading.Thread(target=_run_geo),
        threading.Thread(target=_run_tls),
        threading.Thread(target=_run_dns),
        threading.Thread(target=_run_asn),
    ]
    for t in threads:
        t.start()
    # Strictly bound the total wall-clock join time across all workers to 6.0 seconds
    deadline = time.time() + 6.0
    for t in threads:
        remaining = max(0.0, deadline - time.time())
        t.join(timeout=remaining)

    # Freeze the completed observations: timed-out workers must not mutate
    # evidence after the verdict has been computed.
    threat_intel = dict(threat_intel_result) if threat_intel_result else {"overall": "UNKNOWN", "overall_score": 0}
    html_result  = _normalize_html_result(html_result_holder)
    _GEO_FALLBACK = {"ip": "Unknown", "city": "Unknown", "region": "", "country": "Unknown", "countryCode": "", "isp": "Unknown", "lat": 37.7749, "lon": -122.4194}
    geo          = dict(geo_result_holder) if geo_result_holder else _GEO_FALLBACK
    tls_intel    = dict(tls_result_holder)
    dns_intel    = dict(dns_result_holder)
    asn_intel    = dict(asn_result_holder)

    # 6. OriginTrace 4-Pillars Web-Mode Execution
    # Pillar 1: Provenance Matrix (Cryptographic Identity & TLS Lineage Mismatch)
    p1_result = ProvenanceMatrix.analyze_web(
        domain=domain_safe,
        url=url,
        tls_intel=tls_intel
    )

    # Pillar 2: Infrastructure Lineage (Autonomous System Brand Incongruity & Fast-Flux DNS)
    p2_result = InfrastructureLineage.analyze_web(
        domain=domain_safe,
        dns_ephemerality=dns_intel,
        asn_intel=asn_intel,
        geo=geo
    )

    # Pillar 3: Semantic Deception (DOM Sinks, Credential Harvesting Forms, Evasion)
    p3_result = SemanticDeception.analyze_web(
        html_analysis=html_result,
        url=url,
        domain=domain_safe
    )

    # Pillar 4: Evidential Fusion (Mathematical Dempster-Shafer Synthesis)
    p4_result = EvidentialFusion.synthesize(
        provenance=p1_result,
        infrastructure=p2_result,
        semantic=p3_result,
        context="web"
    )

    # 7. Zero-Day Campaign Attribution Graph
    target_brand = p1_result.get("target_brand")
    campaign_attribution = CampaignAttributor.attribute(
        domain=domain_safe,
        brand=target_brand,
        asn=asn_intel.get("asn", ""),
        isp=asn_intel.get("isp", ""),
        tls_fingerprint=tls_intel.get("cert_fingerprint", ""),
        tls_issuer=tls_intel.get("issuer", ""),
        is_threat=p4_result["final_score"] >= 45
    )

    # 8. Unified Mathematical Scoring Integration
    combined_score = p4_result["final_score"]

    # Incorporate heuristic and ML signals
    heuristic_score = heuristics["heuristic_score"]
    
    # The URL specialist supplies ML risk for every URL, including roots; domain output is separate evidence.
    ml_combined = ml_url_score
    
    if heuristic_score >= 75:
        combined_score = max(combined_score, heuristic_score)
    elif heuristic_score >= 45:
        combined_score = max(combined_score, heuristic_score)

    # Corroboration Guard for ML Predictions:
    # A lexical ML model must NEVER single-handedly condemn an established, clean domain (>180 days old)
    # when all 4 OriginTrace pillars, heuristics, and threat feeds show ZERO threat indicators.
    da = threat_intel.get("domain_age", {})
    is_established_domain = da.get("available") and not da.get("is_suspicious") and da.get("age_days", 0) > 180

    if ml_combined >= 80:
        if is_established_domain and combined_score < 30 and heuristic_score < 30:
            logger.info(f"Suppressing uncorroborated ML score ({ml_combined}) on established domain '{domain_safe}' (Age: {da.get('age_days')} days)")
            ai_flags = [f for f in ai_flags if "phishing confidence" not in f]
        else:
            combined_score = max(combined_score, ml_combined)
    elif ml_combined >= 60:
        if not is_established_domain and (combined_score >= 25 or heuristic_score >= 25):
            combined_score = max(combined_score, ml_combined)

    # WHOIS new domain penalty (e.g. registered < 14 days ago)
    if da.get("available") and da.get("is_suspicious"):
        if heuristic_score >= 35 or ml_combined >= 50 or combined_score >= 40:
            combined_score = min(100, combined_score + 15)

    # External Threat Feeds Match (VirusTotal, PhishTank, URLhaus)
    vt = threat_intel.get("virustotal") or {}
    pt = threat_intel.get("phishtank") or {}
    uh = threat_intel.get("urlhaus") or {}
    if (vt.get("available") and vt.get("malicious", 0) > 0) or \
       (pt.get("available") and pt.get("is_phishing")) or \
       (uh.get("available") and uh.get("is_malicious")):
        combined_score = max(combined_score, 90)

    # Cross-Entity Contradiction Matrix: Evaluate Brand vs Domain vs ASN Mismatch
    from services.origintrace_pillars import CrossEntityContradictionEngine
    contra_analysis = CrossEntityContradictionEngine.analyze(
        urls=[url, raw_url] if url != raw_url else [url],
        destination_domain=domain_safe,
        destination_asn=asn_intel.get("asn", ""),
        destination_isp=asn_intel.get("isp", ""),
        tls_cert_org=tls_intel.get("subject_org", "")
    )

    if contra_analysis.get("is_contradictory"):
        combined_score = max(combined_score, int(contra_analysis["contradiction_score"]))

    # Build comprehensive explanation signals
    signals = []

    # Tier 0 Authority Domain Policy:
    # 1. Authority domains have ZERO domain reputation risk (domain is legitimate).
    # 2. However, user-generated content (UGC) hosting platforms or URLs with suspicious paths/threat intel
    #    must NEVER have their path, heuristic, ML, or threat intelligence scores wiped out.
    domain_lower = domain_safe.lower()
    parsed_path = urlparse(url).path.strip('/')
    is_auth_dom = is_safe_authority_domain(domain_lower)
    is_ugc = is_ugc_hosting_platform(domain_lower, parsed_path)

    if is_auth_dom:
        if is_ugc or parsed_path:
            # On UGC platforms or deep paths:
            # Domain penalty is zeroed, but heuristic/ML/threat intelligence scores on the path/payload stand
            if is_ugc:
                signals.append("[Cloud Hosting] Hosted on public user-generated content / form infrastructure")
                if heuristic_score >= 25 or ml_url_score >= 50:
                    combined_score = max(combined_score, heuristic_score, ml_url_score)
        else:
            # Pure root homepage of an established authority (e.g. https://google.com, https://github.com)
            # Only reset if threat feeds show zero malicious detections
            if combined_score < 45 and not ((vt.get("available") and vt.get("malicious", 0) > 0) or
                    (pt.get("available") and pt.get("is_phishing")) or
                    (uh.get("available") and uh.get("is_malicious")) or contra_analysis.get("is_contradictory")):
                combined_score = 0

    final_score = combined_score
    status = get_status(final_score)
    risk_level = classify_risk(final_score)

    if url != raw_url:
        signals.append(f"Redirects to: {url}")

    # Prioritize Contradiction Matrix Signals
    for sig in contra_analysis.get("signals", []):
        signals.append(f"[Contradiction Matrix] {sig}")

    # Prioritize OriginTrace Pillar Signals
    for sig in p1_result.get("signals", []):
        if any(marker in sig for marker in ["Lookalike", "Lineage Mismatch", "Self-Signed", "Expired"]):
            signals.append(f"[Pillar 1 Provenance] {sig}")

    for sig in p2_result.get("signals", []):
        if any(marker in sig for marker in ["Autonomous System", "Fast-Flux", "Zero Mail", "Disposable VPS", "Low-Reputation TLD"]):
            signals.append(f"[Pillar 2 Infrastructure] {sig}")

    for sig in p3_result.get("signals", []):
        if any(marker in sig for marker in ["DOM Brand", "Credential", "Obfuscated", "Deception"]):
            signals.append(f"[Pillar 3 Semantic] {sig}")

    # Campaign attribution signal
    if campaign_attribution.get("is_attributed") and campaign_attribution.get("campaign_id"):
        signals.append(
            f"[Campaign Graph] Linked to {campaign_attribution['campaign_name']} ({campaign_attribution['campaign_id']})"
        )

    signals.extend(ai_flags)
    signals.extend(heuristics["flags"])

    if vt.get("available") and vt.get("malicious", 0) > 0:
        signals.append(f"VirusTotal: {vt['malicious']}/{vt.get('total_engines', '?')} security engines flagged this URL as malicious")
    elif vt.get("available"):
        signals.append(f"VirusTotal: 0/{vt.get('total_engines', '?')} engines flagged — no known detections")

    if pt.get("available") and pt.get("is_phishing"):
        signals.append("PhishTank: This URL is confirmed as a phishing site in the PhishTank community database")

    if uh.get("available") and uh.get("is_malicious"):
        signals.append(f"URLhaus: Listed as malware distribution ({uh.get('threat', 'unknown')})")

    if da.get("available"):
        if da.get("is_suspicious"):
            signals.append(f"Domain Age: Only {da['age_days']} days old — very new domains are high risk")
        else:
            signals.append(f"Domain Age: {da['age_days']} days old — established domain")

    if not heuristics["is_https"]:
        signals.append("URL uses HTTP (not HTTPS) — no encryption, increased risk")

    if html_result.get("success"):
        for h_sig in html_result.get("flags", []):
            if h_sig not in signals:
                signals.append(h_sig)
    elif html_result.get("error"):
        signals.append(f"Live HTML scan: {html_result['error']} — could not retrieve page content")

    from services.web_evidence_policy import web_evidence_decision
    uncertainty_val = p4_result.get("uncertainty", 1.0)
    decision = web_evidence_decision(final_score, url_probability=url_model_probability,
        html=html_result, tls=tls_intel, dns=dns_intel, asn=asn_intel,
        intel=threat_intel, contradiction=contra_analysis, ignorance=uncertainty_val)

    return {
        "url": raw_url,
        "resolved_url": url if url != raw_url else None,
        "final_score": final_score,
        "combined_score": final_score,
        "ml_probability": url_model_probability,
        "ml_score": ml_url_score,
        **decision,
        "epistemic_uncertainty": None,
        "evidence_ignorance": uncertainty_val,
        "uncertainty_kind": "assigned_subjective_logic_ignorance",
        "contradiction_analysis": contra_analysis,
        "signals": signals if signals else ["No suspicious signals detected."],
        "heuristics": heuristics,
        "threat_intel": threat_intel,
        "html_analysis": html_result,
        "origintrace_pillars": {
            "provenance_matrix": p1_result,
            "infrastructure_lineage": p2_result,
            "semantic_deception": p3_result,
            "evidential_fusion": p4_result,
            "model_probabilities": {"url": url_model_probability, "domain": domain_model_probability, "domain_scope": scan_engine.domain_model_input(url), "domain_role": "diagnostic_only_unverified_domain_labels"},
        },
        "campaign_attribution": campaign_attribution,
        "tls_intel": tls_intel,
        "dns_ephemerality": dns_intel,
        "asn_intel": asn_intel,
        "domain": domain_safe,
        "is_https": heuristics["is_https"],
        "has_redirect": url != raw_url,
        "has_shortener": url != raw_url,
        "geo": geo,
        "server": html_result.get("server", "Unknown"),
    }
