import os
import re
import base64
import time
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from core.logger import logger
from functools import lru_cache
from typing import Optional, Dict, Any, List, Tuple

# ── Fix broken SSLKEYLOGFILE on this machine ─────────────────────────────────
_ssl_log = os.environ.get("SSLKEYLOGFILE", "")
if _ssl_log.startswith('"') or _ssl_log.startswith("'"):
    del os.environ["SSLKEYLOGFILE"]
    logger.info("Fixed malformed SSLKEYLOGFILE env var — HTTPS calls re-enabled.")

# Always reload .env so keys are picked up after hot-reload
load_dotenv(override=True)

# ── Config ───────────────────────────────────────────────────────────────────
def _vt_key() -> str:
    return os.getenv("VIRUSTOTAL_API_KEY", "").strip()

def _pt_key() -> str:
    return os.getenv("PHISHTANK_API_KEY", "").strip()

def _uh_key() -> str:
    return os.getenv("URLHAUS_AUTH_KEY", "").strip()

# Reduced timeouts — never let a single API call block the whole scan
REQUEST_TIMEOUT = 3  # seconds (fast fail for snappy scanning)

# ── Simple in-memory TTL cache ────────────────────────────────────────────────
# Avoids repeated external API calls for the same URL within a session.
# Format: { url: {"result": {...}, "expires_at": datetime} }
_cache: dict = {}
CACHE_TTL_MINUTES = 30

def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and entry["expires_at"] > datetime.now(timezone.utc).replace(tzinfo=None):
        return entry["result"]
    return None

def _cache_set(key: str, result: dict):
    _cache[key] = {
        "result": result,
        "expires_at": datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=CACHE_TTL_MINUTES)
    }
    # Evict old entries if cache grows large
    if len(_cache) > 2000:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        expired = [k for k, v in _cache.items() if v["expires_at"] <= now]
        for k in expired:
            del _cache[k]


# ── VirusTotal ────────────────────────────────────────────────────────────────
def check_virustotal(url: str) -> dict:
    """
    Submit a URL to VirusTotal and return a summary of detections.
    Requires VIRUSTOTAL_API_KEY in .env.
    Results are cached for 30 minutes to avoid redundant API calls.
    """
    cache_key = f"vt:{url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    VT_API_KEY = _vt_key()
    if not VT_API_KEY:
        return {"available": False, "reason": "No API key configured"}

    try:
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        headers = {"x-apikey": VT_API_KEY}

        resp = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        # 404 means URL not in VT cache yet — submit asynchronously in background
        # so we do NOT block the scan for another timeout cycle
        if resp.status_code == 404:
            import threading
            def _async_post_url():
                try:
                    requests.post(
                        "https://www.virustotal.com/api/v3/urls",
                        headers=headers,
                        data={"url": url},
                        timeout=REQUEST_TIMEOUT,
                    )
                except Exception:
                    pass
            threading.Thread(target=_async_post_url, daemon=True).start()

            result = {"available": False, "reason": "URL submitted to VT for analysis — check back soon"}
            _cache_set(cache_key, result)
            return result

        if resp.status_code != 200:
            return {"available": False, "reason": f"VT API error ({resp.status_code})"}

        data = resp.json()
        stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        total = sum(stats.values()) or 1

        result = {
            "available": True,
            "malicious": malicious,
            "suspicious": suspicious,
            "total_engines": total,
            "verdict": "MALICIOUS" if malicious > 0 else ("SUSPICIOUS" if suspicious > 0 else "CLEAN"),
            "display": f"{malicious}/{total} engines flagged" if malicious > 0 else f"0/{total} engines flagged",
            "link": f"https://www.virustotal.com/gui/url/{url_id}",
        }
        _cache_set(cache_key, result)
        return result

    except requests.exceptions.Timeout:
        logger.warning(f"VirusTotal timeout for URL: {url[:80]}")
        return {"available": False, "reason": "Request timed out"}
    except Exception as e:
        logger.error(f"VirusTotal error: {e}")
        return {"available": False, "reason": "Service error"}


# ── PhishTank ─────────────────────────────────────────────────────────────────
def check_phishtank(url: str) -> dict:
    """
    Check if a URL is in the PhishTank database.
    Results are cached for 30 minutes.
    """
    cache_key = f"pt:{url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    PT_API_KEY = _pt_key()
    try:
        payload = {"url": url, "format": "json"}
        if PT_API_KEY:
            payload["app_key"] = PT_API_KEY

        resp = requests.post(
            "https://checkurl.phishtank.com/checkurl/",
            data=payload,
            headers={"User-Agent": "phishguard-checker/1.0"},
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 403:
            result = {"available": False, "reason": "PhishTank requires a free API key. Add PHISHTANK_API_KEY to .env"}
            return result
        if resp.status_code != 200:
            return {"available": False, "reason": f"PhishTank error ({resp.status_code})"}

        data = resp.json()
        results = data.get("results", {})
        in_database = results.get("in_database", False)
        valid = results.get("valid", False)

        result = {
            "available": True,
            "in_database": in_database,
            "is_phishing": in_database and valid,
            "verdict": "PHISHING" if (in_database and valid) else ("LISTED (unverified)" if in_database else "NOT LISTED"),
            "display": "⚠ Known Phishing URL" if (in_database and valid) else "✓ Not in PhishTank database",
            "phish_id": results.get("phish_id"),
        }
        _cache_set(cache_key, result)
        return result

    except requests.exceptions.Timeout:
        logger.warning(f"PhishTank timeout for URL: {url[:80]}")
        return {"available": False, "reason": "Request timed out"}
    except Exception as e:
        logger.error(f"PhishTank error: {e}")
        return {"available": False, "reason": "Service error"}


# ── URLhaus (abuse.ch) ────────────────────────────────────────────────────────
def check_urlhaus(url: str) -> dict:
    """
    Check if a URL is in the URLhaus malware distribution feed.
    Results are cached for 30 minutes.
    """
    cache_key = f"uh:{url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    UH_KEY = _uh_key()
    if not UH_KEY:
        return {"available": False, "reason": "No Auth-Key configured. Get a free key at auth.abuse.ch"}

    try:
        resp = requests.post(
            "https://urlhaus-api.abuse.ch/v1/url/",
            data={"url": url},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Auth-Key": UH_KEY,
            },
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 401:
            return {"available": False, "reason": "Invalid URLhaus Auth-Key"}
        if resp.status_code != 200:
            return {"available": False, "reason": f"URLhaus error ({resp.status_code})"}

        data = resp.json()
        query_status = data.get("query_status", "")

        if query_status == "is_listed":
            threat = data.get("threat", "unknown")
            url_status = data.get("url_status", "unknown")
            result = {
                "available": True,
                "is_malicious": True,
                "threat": threat,
                "url_status": url_status,
                "verdict": "MALICIOUS",
                "display": f"⚠ Listed: {threat} ({url_status})",
                "link": data.get("urlhaus_reference", ""),
            }
        elif query_status == "no_results":
            result = {
                "available": True,
                "is_malicious": False,
                "verdict": "CLEAN",
                "display": "✓ Not in URLhaus database",
            }
        else:
            return {"available": False, "reason": f"Unexpected status: {query_status}"}

        _cache_set(cache_key, result)
        return result

    except requests.exceptions.Timeout:
        logger.warning(f"URLhaus timeout for URL: {url[:80]}")
        return {"available": False, "reason": "Request timed out"}
    except Exception as e:
        logger.error(f"URLhaus error: {e}")
        return {"available": False, "reason": "Service error"}


def _fallback_rdap(domain: str) -> dict:
    """Fallback to RDAP when standard WHOIS fails."""
    import urllib.request
    import json
    
    try:
        req = urllib.request.Request(f'https://rdap.org/domain/{domain}', headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
            
            # Extract registrar
            registrar = 'Privacy Protected / Not Disclosed'
            for ent in data.get('entities', []):
                if 'registrar' in ent.get('roles', []):
                    try:
                        vcard = ent.get('vcardArray', [[]])[1]
                        for prop in vcard:
                            if prop[0] == 'fn':
                                registrar = prop[3]
                                break
                    except Exception:
                        pass
            
            # Extract dates
            created_str = None
            expires_str = None
            for ev in data.get('events', []):
                if ev.get('eventAction') == 'registration':
                    created_str = ev.get('eventDate')
                if ev.get('eventAction') == 'expiration':
                    expires_str = ev.get('eventDate')
            
            if not created_str:
                return {"available": False, "reason": "RDAP returned no creation date"}
                
            from dateutil import parser
            created = parser.isoparse(created_str).replace(tzinfo=None)
            
            expires_formatted = "Unknown"
            if expires_str:
                expires = parser.isoparse(expires_str).replace(tzinfo=None)
                expires_formatted = expires.strftime("%Y-%m-%d")
                
            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - created).days
            
            # Nameservers
            ns_list = []
            for ns in data.get('nameservers', []):
                if 'ldhName' in ns:
                    ns_list.append(ns['ldhName'].lower())
            if not ns_list:
                ns_list = ["No nameservers found"]
                
            return {
                "available": True,
                "domain": domain,
                "age_days": age_days,
                "created": created.strftime("%Y-%m-%d"),
                "expires": expires_formatted,
                "registrar": registrar,
                "nameservers": ns_list,
                "verdict": "NEW" if age_days < 30 else ("RECENT" if age_days < 180 else "ESTABLISHED"),
                "display": f"{age_days} days old (created {created.strftime('%Y-%m-%d')})",
                "is_suspicious": age_days < 30,
            }
    except Exception as e:
        logger.warning(f"RDAP fallback failed for {domain}: {e}")
        return {"available": False, "reason": "RDAP lookup failed"}


# ── Domain Age (WHOIS) ────────────────────────────────────────────────────────
def get_domain_age(url: str) -> dict:
    """
    Extract the domain from a URL and compute its age via WHOIS.
    Results are cached for 30 minutes to avoid repeated WHOIS lookups.
    """
    cache_key = f"whois:{url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        import whois  # python-whois package
        import socket
        socket.setdefaulttimeout(3.0)  # Never let raw WHOIS port 43 sockets hang

        match = re.search(r"https?://([^/?\s]+)", url)
        if not match:
            return {"available": False, "reason": "Could not parse domain from URL"}

        domain = match.group(1).lower().split(":")[0]
        if domain.startswith("www."):
            domain = domain[4:]

        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]

        if not creation:
            return {"available": False, "reason": "WHOIS creation date not found"}

        if hasattr(creation, "tzinfo") and creation.tzinfo:
            creation = creation.replace(tzinfo=None)

        age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - creation).days

        registrar = w.get("registrar")
        if isinstance(registrar, list):
            registrar = registrar[0] if registrar else None
        # "Unknown" means WHOIS succeeded but registrar info was withheld (WHOIS privacy)
        if not registrar or registrar.strip().lower() in ("unknown", "n/a", ""):
            registrar = "Privacy Protected / Not Disclosed"

        expiration = w.get("expiration_date")
        if isinstance(expiration, list):
            expiration = expiration[0] if expiration else None

        expires_str = "Unknown"
        if expiration:
            if hasattr(expiration, "tzinfo") and expiration.tzinfo:
                expiration = expiration.replace(tzinfo=None)
            expires_str = expiration.strftime("%Y-%m-%d")

        ns = w.get("name_servers")
        ns_list = []
        if isinstance(ns, list):
            ns_list = [str(n).lower() for n in ns if n]
        elif isinstance(ns, str):
            ns_list = [ns.lower()]

        if not ns_list:
            ns_list = ["No nameservers found"]

        result = {
            "available": True,
            "domain": domain,
            "age_days": age_days,
            "created": creation.strftime("%Y-%m-%d"),
            "expires": expires_str,
            "registrar": registrar,
            "nameservers": ns_list,
            "verdict": "NEW" if age_days < 30 else ("RECENT" if age_days < 180 else "ESTABLISHED"),
            "display": f"{age_days} days old (created {creation.strftime('%Y-%m-%d')})",
            "is_suspicious": age_days < 30,
        }
        _cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"WHOIS lookup failed for {url[:80]}: {e}. Trying RDAP fallback...")
        fallback = _fallback_rdap(domain)
        if fallback.get("available"):
            _cache_set(cache_key, fallback)
            return fallback
        return {"available": False, "reason": "WHOIS and RDAP lookups failed"}


# ── Aggregate: run all checks on a URL ───────────────────────────────────
def full_threat_check(url: str) -> dict:
    """
    Run all threat intel sources against a URL IN PARALLEL.

    Cache hierarchy (fastest → slowest):
      L1 — In-memory TTL cache (30 min, per-process, lost on restart)
      L2 — Persistent SQLite DB cache (24 h, survives restarts)
      L3 — Live external APIs (VirusTotal, PhishTank, URLhaus, WHOIS)

    This means:
      - Second call within same session  → instant (L1 hit)
      - First call after server restart  → DB lookup only (L2 hit, ~1 ms)
      - Brand new URL never seen before  → live API calls (L3, parallelised)

    All results degrade gracefully — nothing throws.
    """
    from concurrent.futures import ThreadPoolExecutor

    # ── L1: in-memory cache (per-process, 30 min) ────────────────────────
    mem_key = f"ftc:{url}"
    cached = _cache_get(mem_key)
    if cached is not None:
        logger.debug(f"[ThreatCache] L1 hit: {url[:60]}")
        return cached

    # ── L2: persistent SQLite cache (24 h, survives restarts) ────────────
    try:
        from data.store import get_threat_cache, set_threat_cache
        db_cached = get_threat_cache(url)
        if db_cached is not None:
            logger.debug(f"[ThreatCache] L2 hit (DB): {url[:60]}")
            _cache_set(mem_key, db_cached)   # promote to L1
            return db_cached
    except Exception as e:
        logger.warning(f"[ThreatCache] DB cache read error: {e}")

    # ── L3: live API calls (first time or cache expired) ─────────────────
    logger.info(f"[ThreatCache] L3 miss — calling APIs for: {url[:80]}")

    ex = ThreadPoolExecutor(max_workers=4)
    try:
        fut_vt  = ex.submit(check_virustotal, url)
        fut_pt  = ex.submit(check_phishtank, url)
        fut_uh  = ex.submit(check_urlhaus, url)
        fut_da  = ex.submit(get_domain_age, url)

        api_timeout   = REQUEST_TIMEOUT + 0.5   # ~3.5s — for fast REST API calls
        whois_timeout = 4.0                     # WHOIS capped at 4s (avoids 15s freeze)
        try:
            vt = fut_vt.result(timeout=api_timeout)
        except Exception:
            vt = {"available": False, "reason": "Timed out"}
        try:
            pt = fut_pt.result(timeout=api_timeout)
        except Exception:
            pt = {"available": False, "reason": "Timed out"}
        try:
            uh = fut_uh.result(timeout=api_timeout)
        except Exception:
            uh = {"available": False, "reason": "Timed out"}
        try:
            domain_age = fut_da.result(timeout=whois_timeout)
            if not domain_age.get("available") and domain_age.get("reason", "") in (
                "python-whois not installed", "WHOIS lookup failed"
            ):
                domain_age = {"available": False, "reason": "WHOIS unavailable"}
        except Exception:
            domain_age = {"available": False, "reason": "Timed out"}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    is_malicious = (
        (vt.get("available") and vt.get("malicious", 0) > 0) or
        (pt.get("available") and pt.get("is_phishing", False)) or
        (uh.get("available") and uh.get("is_malicious", False))
    )
    is_suspicious = (
        (vt.get("available") and vt.get("suspicious", 0) > 0) or
        (domain_age.get("available") and domain_age.get("is_suspicious", False))
    )

    if is_malicious:
        overall = "MALICIOUS"
        overall_score = 90
    elif is_suspicious:
        overall = "SUSPICIOUS"
        overall_score = 55
    elif any(source.get("available") for source in (vt, pt, uh)):
        overall = "CLEAN"
        overall_score = 5
    else:
        overall = "UNKNOWN"
        overall_score = 0

    result = {
        "url": url,
        "virustotal": vt,
        "phishtank": pt,
        "urlhaus": uh,
        "domain_age": domain_age,
        "overall": overall,
        "overall_score": overall_score,
        "is_malicious": is_malicious,
        "reputation_checked": any(source.get("available") for source in (vt, pt, uh)),
    }

    # Persist to both caches for next time
    _cache_set(mem_key, result)
    try:
        from data.store import set_threat_cache
        set_threat_cache(url, result)
    except Exception as e:
        logger.warning(f"[ThreatCache] DB cache write error: {e}")

    return result


import socket

@lru_cache(maxsize=1024)
def resolve_country(domain_or_email: str) -> str:
    """
    Resolve a domain or email address to a 2-letter country code.
    Uses TLD primarily, and falls back to ip-api for generic TLDs (.com, .net).
    lru_cache ensures repeated lookups for the same domain are instant.
    """
    if not domain_or_email:
        return ''

    domain = domain_or_email
    if '@' in domain:
        domain = domain.split('@')[-1]

    if '://' in domain:
        domain = domain.split('://')[-1].split('/')[0]

    domain = domain.lower().strip()
    parts = domain.split('.')
    if len(parts) >= 2:
        tld = parts[-1]
        if len(tld) == 2 and tld.isalpha() and tld not in ['co', 'go', 'ac', 'edu']:
            return tld.upper()

    try:
        ip = socket.gethostbyname(domain)
        resp = requests.get(f"http://ip-api.com/json/{ip}?fields=countryCode", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if 'countryCode' in data:
                return data['countryCode']
    except Exception:
        pass

    return ''


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED FACTUAL INTELLIGENCE FOR ORIGINTRACE 4-PILLAR SECURITY ENGINE
# 1. TLS Certificate Lineage Mismatch Analysis
# 2. DNS Ephemerality & Fast-Flux Index (DoH-based)
# 3. Autonomous System Brand Incongruity (ABCI) Scoring
# ══════════════════════════════════════════════════════════════════════════════

import ssl
import hashlib
from urllib.parse import urlparse

# Well-known brands and their official organization entities in TLS certs
BRAND_OFFICIAL_ORGS = {
    "paypal": ["paypal, inc.", "paypal", "paypal holdings"],
    "microsoft": ["microsoft corporation", "microsoft"],
    "apple": ["apple inc.", "apple computer, inc."],
    "google": ["google llc", "google inc"],
    "amazon": ["amazon.com, inc.", "amazon.com", "amazon corporate llc"],
    "netflix": ["netflix, inc.", "netflix"],
    "chase": ["jpmorgan chase & co.", "jpmorgan chase bank, national association", "jpmorgan chase"],
    "bankofamerica": ["bank of america corporation", "bank of america"],
    "wellsfargo": ["wells fargo & company", "wells fargo bank, n.a."],
    "citibank": ["citigroup inc.", "citibank, n.a."],
    "docusign": ["docusign, inc.", "docusign"],
    "dropbox": ["dropbox, inc.", "dropbox"],
    "adobe": ["adobe inc.", "adobe systems incorporated"],
    "facebook": ["meta platforms, inc.", "facebook, inc."],
    "instagram": ["meta platforms, inc.", "instagram, llc"],
    "linkedin": ["linkedin corporation"],
    "coinbase": ["coinbase, inc.", "coinbase"],
    "binance": ["binance"],
}

# Free / Disposable DV Certificate Authorities commonly leveraged in fast-flux phishing
FREE_DV_ISSUERS = [
    "let's encrypt", "zerossl", "cpanel, inc.", "certum free", "buypass go ssl", "trustprovider", "free ssl"
]

# Bulletproof and disposable unmanaged VPS / hosting providers
DISPOSABLE_HOSTING_PROVIDERS = [
    "contabo", "hetzner", "digitalocean", "ovh", "linode", "vultr",
    "choopa", "hostinger", "scaleway", "kamatera", "m247", "frantech",
    "buyvm", "as-infra", "colocrossing"
]


def _extract_cert_rdn(items, key_name: str) -> str:
    """Extract a specific RDN (e.g. organizationName or commonName) from a parsed TLS cert tuple."""
    if not items:
        return ""
    for rdn in items:
        for k, v in rdn:
            if k == key_name:
                return str(v)
    return ""


def analyze_tls_cert_lineage(domain_or_url: str) -> dict:
    """
    Evaluates TLS/SSL Certificate Lineage Mismatch:
    - Extracts live X.509 certificate via SNI on port 443 (timeout 2.5s).
    - Computes cryptographic SHA-256 fingerprint for infrastructure campaign attribution.
    - Inspects Subject Alternative Names (SANs) and certificate issuance lifespan.
    - Detects Combosquatting Lineage Mismatch: A domain mimicking an enterprise or financial brand
      (e.g., PayPal, Microsoft, Chase) secured with an ephemeral/free DV certificate
      without official enterprise organization validation.
    """
    domain = domain_or_url.strip().lower()
    if "://" in domain:
        domain = domain.split("://")[-1].split("/")[0].split(":")[0]
    if "@" in domain:
        domain = domain.split("@")[-1].split(":")[0]
    if domain.startswith("www."):
        domain = domain[4:]

    res = {
        "available": False,
        "domain": domain,
        "issuer": "Unknown",
        "issuer_org": "Unknown",
        "subject_cn": "Unknown",
        "subject_org": "Unknown",
        "valid_from": "",
        "valid_to": "",
        "cert_age_days": 0,
        "cert_validity_days": 0,
        "san_list": [],
        "san_count": 0,
        "cert_fingerprint": "",
        "serial_number": "",
        "is_self_signed": False,
        "is_expired": False,
        "is_free_dv": False,
        "lineage_mismatch": False,
        "mismatch_reason": "",
        "signals": []
    }

    if not domain or domain in ["localhost", "127.0.0.1"]:
        return res

    der_cert = None
    cert_dict = None
    ssl_error_msg = ""

    # Attempt 1: Standard verification (gets parsed dict if certificate is valid)
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=2.5) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                cert_dict = ssock.getpeercert()
                der_cert = ssock.getpeercert(binary_form=True)
                res["available"] = True
    except ssl.SSLCertVerificationError as e:
        ssl_error_msg = str(e)
        res["available"] = True
        if "self signed" in ssl_error_msg.lower() or "self-signed" in ssl_error_msg.lower():
            res["is_self_signed"] = True
            res["signals"].append("Self-Signed SSL Certificate: Untrusted authority, high phishing risk.")
        elif "expired" in ssl_error_msg.lower():
            res["is_expired"] = True
            res["signals"].append("Expired SSL Certificate: Certificate validity window has elapsed.")
        elif "hostname" in ssl_error_msg.lower() or "match" in ssl_error_msg.lower():
            res["signals"].append("TLS Hostname Mismatch: Certificate SAN does not match requested domain.")
    except Exception as e:
        ssl_error_msg = str(e)

    # Attempt 2: If standard failed, fetch raw DER via unverified context to get fingerprint
    if der_cert is None:
        try:
            unverified_ctx = ssl.create_default_context()
            unverified_ctx.check_hostname = False
            unverified_ctx.verify_mode = ssl.CERT_NONE
            with socket.create_connection((domain, 443), timeout=2.0) as sock:
                with unverified_ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                    der_cert = ssock.getpeercert(binary_form=True)
                    res["available"] = True
        except Exception:
            pass

    # Extract SHA-256 fingerprint from DER bytes (essential for campaign clustering)
    if der_cert:
        res["cert_fingerprint"] = hashlib.sha256(der_cert).hexdigest()

    # If we have parsed certificate dictionary from successful handshake
    if cert_dict:
        res["available"] = True
        issuer_items = cert_dict.get("issuer", ())
        subject_items = cert_dict.get("subject", ())

        issuer_org = _extract_cert_rdn(issuer_items, "organizationName")
        issuer_cn = _extract_cert_rdn(issuer_items, "commonName")
        res["issuer"] = issuer_cn or issuer_org or "Unknown"
        res["issuer_org"] = issuer_org or issuer_cn or "Unknown"

        subject_org = _extract_cert_rdn(subject_items, "organizationName")
        subject_cn = _extract_cert_rdn(subject_items, "commonName")
        res["subject_cn"] = subject_cn or "Unknown"
        res["subject_org"] = subject_org or "Unknown"

        res["serial_number"] = str(cert_dict.get("serialNumber", ""))

        # SANs (Subject Alternative Names)
        sans = []
        for san_type, san_val in cert_dict.get("subjectAltName", ()):
            if san_type == "DNS":
                sans.append(san_val.lower())
        res["san_list"] = sans[:20]
        res["san_count"] = len(sans)

        # Dates & Lifespan
        not_before = cert_dict.get("notBefore", "")
        not_after = cert_dict.get("notAfter", "")
        res["valid_from"] = not_before
        res["valid_to"] = not_after

        try:
            fmt = "%b %d %H:%M:%S %Y %Z"
            d_start = datetime.strptime(not_before, fmt)
            d_end = datetime.strptime(not_after, fmt)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            res["cert_age_days"] = max(0, (now - d_start).days)
            res["cert_validity_days"] = max(0, (d_end - d_start).days)

            if now > d_end:
                res["is_expired"] = True
                res["signals"].append("Expired SSL Certificate: Certificate is past its expiration date.")
        except Exception:
            pass

        # Check for Free / Disposable DV provider
        issuer_str = f"{res['issuer']} {res['issuer_org']}".lower()
        if any(free_dv in issuer_str for free_dv in FREE_DV_ISSUERS) or (0 < res["cert_validity_days"] <= 90):
            res["is_free_dv"] = True

        # Check for Brand Lineage Mismatch (Combosquatting / Typosquatting)
        for brand, official_orgs in BRAND_OFFICIAL_ORGS.items():
            if brand in domain:
                # Is it the authentic brand domain? (e.g. paypal.com, login.microsoft.com)
                is_legit = domain == f"{brand}.com" or domain.endswith(f".{brand}.com")
                if not is_legit:
                    # Impersonating domain with free DV cert
                    if res["is_free_dv"]:
                        res["lineage_mismatch"] = True
                        res["mismatch_reason"] = (
                            f"TLS Certificate Lineage Mismatch: Domain targets brand '{brand}', "
                            f"but operates on a disposable short-lived DV certificate issued by '{res['issuer']}' "
                            f"without Enterprise EV/OV organization validation."
                        )
                        res["signals"].append(res["mismatch_reason"])
                        break
                    # Or cert subject organization does not match the official company
                    elif subject_org:
                        sub_org_lower = subject_org.lower()
                        if not any(org in sub_org_lower for org in official_orgs):
                            res["lineage_mismatch"] = True
                            res["mismatch_reason"] = (
                                f"TLS Subject Incongruity: Domain references '{brand}', but certificate "
                                f"organization is registered to '{subject_org}' instead of the official entity."
                            )
                            res["signals"].append(res["mismatch_reason"])
                            break

    elif ssl_error_msg and res["available"]:
        res["lineage_mismatch"] = True
        res["mismatch_reason"] = f"Untrusted TLS Layer: Handshake failed ({ssl_error_msg[:120]})"

    return res


def calculate_dns_ephemerality(domain_or_url: str) -> dict:
    """
    Calculates DNS Ephemerality & Fast-Flux Index:
    - Queries authoritative DNS records via Cloudflare DoH (with Google DoH fallback).
    - Evaluates record TTL: Ephemeral fast-flux phishing domains deploy ultra-low TTLs (<= 120s or 300s)
      to rapidly cycle proxy IPs and evade IP blacklists.
    - Evaluates Mail Infrastructure: Phishing landing domains mimicking enterprises routinely lack MX records.
    - Returns factual ephemerality index (0.0 to 1.0) and security signals.
    """
    domain = domain_or_url.strip().lower()
    if "://" in domain:
        domain = domain.split("://")[-1].split("/")[0].split(":")[0]
    if "@" in domain:
        domain = domain.split("@")[-1].split(":")[0]
    if domain.startswith("www."):
        domain = domain[4:]

    res = {
        "available": False,
        "domain": domain,
        "min_ttl": None,
        "a_records": [],
        "mx_records": [],
        "ns_records": [],
        "has_mx": False,
        "ephemerality_index": 0.0,  # 0.0 (Durable/Permanent) to 1.0 (Highly Ephemeral / Fast-Flux)
        "signals": []
    }

    if not domain or domain in ["localhost", "127.0.0.1"]:
        return res

    def _query_doh(q_type: str) -> list:
        headers = {"Accept": "application/dns-json"}
        # Cloudflare DoH
        try:
            r = requests.get(
                f"https://cloudflare-dns.com/dns-query?name={domain}&type={q_type}",
                headers=headers,
                timeout=2.0
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("Answer", [])
        except Exception:
            pass

        # Google DoH Fallback
        try:
            r = requests.get(
                f"https://dns.google/resolve?name={domain}&type={q_type}",
                headers=headers,
                timeout=2.0
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("Answer", [])
        except Exception:
            pass
        return []

    # 1. Query A records
    a_answers = _query_doh("A")
    ttls = []
    if a_answers:
        res["available"] = True
        for ans in a_answers:
            if ans.get("type") == 1:  # Type A
                res["a_records"].append(ans.get("data", ""))
                ttl = ans.get("TTL")
                if ttl is not None:
                    ttls.append(int(ttl))

    # 2. Query MX records
    mx_answers = _query_doh("MX")
    if mx_answers:
        res["available"] = True
        for ans in mx_answers:
            if ans.get("type") == 15:  # Type MX
                res["mx_records"].append(ans.get("data", ""))
        res["has_mx"] = len(res["mx_records"]) > 0

    # 3. Query NS records
    ns_answers = _query_doh("NS")
    if ns_answers:
        res["available"] = True
        for ans in ns_answers:
            if ans.get("type") == 2:  # Type NS
                res["ns_records"].append(ans.get("data", ""))

    # Ephemerality Index Calculation
    index = 0.0
    if ttls:
        min_ttl = min(ttls)
        res["min_ttl"] = min_ttl

        # Only flag very low TTLs (<=60s) as true fast-flux — many CDNs/ISPs use 120-300s legitimately
        if min_ttl <= 60:
            index += 0.45
            res["signals"].append(
                f"Fast-Flux DNS Detected: Host TTL is ultra-low ({min_ttl}s), characteristic of active botnet proxy cycling."
            )
        elif min_ttl <= 180:
            index += 0.15
            res["signals"].append(
                f"Short DNS TTL ({min_ttl}s): Slightly ephemeral record configuration (low but not definitive)."
            )
        # 181-600s is normal for legitimate sites — no penalty

    # Check for missing mail infrastructure on brand-mimicking domain
    # Only flag this if the domain actually contains a protected brand name in its root —
    # most legitimate websites (portfolios, blogs, tech firms) simply don't use email.
    is_brand_mimic = any(b in domain for b in BRAND_OFFICIAL_ORGS.keys())
    if is_brand_mimic and not res["has_mx"] and res["available"]:
        index += 0.20   # reduced from 0.35 — no MX alone is not a strong signal
        res["signals"].append(
            "Zero Mail Infrastructure: Domain lacks MX records despite targeting an established enterprise brand."
        )

    # Ephemeral / disposable nameservers check
    for ns in res["ns_records"]:
        ns_lower = ns.lower()
        if any(kw in ns_lower for kw in ["afraid.org", "duckdns", "no-ip", "changeip", "freedns"]):
            index += 0.30
            res["signals"].append(f"Dynamic/Disposable DNS Provider: Nameserver '{ns}' allows unverified anonymous routing.")
            break

    res["ephemerality_index"] = round(min(index, 1.0), 3)
    return res


def check_asn_brand_incongruity(domain_or_url: str, ip: Optional[str] = None) -> dict:
    """
    Computes Autonomous System Brand Incongruity (ABCI):
    - Resolves target IP and queries BGP Autonomous System Number (ASN) and ISP organization.
    - Evaluates architectural incongruity: Legitimate enterprises (Microsoft, Apple, PayPal, Google, etc.)
      routinely host infrastructure on their own verified ASNs (e.g. AS8075 Microsoft, AS714 Apple)
      or enterprise CDNs (Cloudflare, Akamai).
    - When a domain mimics a protected enterprise brand but resides on a cheap/bulletproof
      disposable VPS host (Contabo, Hetzner, Vultr, Choopa, DigitalOcean, Hostinger), this constitutes
      a 100% deterministic, mathematically verifiable infrastructure mismatch.
    """
    domain = domain_or_url.strip().lower()
    if "://" in domain:
        domain = domain.split("://")[-1].split("/")[0].split(":")[0]
    if "@" in domain:
        domain = domain.split("@")[-1].split(":")[0]
    if domain.startswith("www."):
        domain = domain[4:]

    res = {
        "available": False,
        "domain": domain,
        "ip": ip or "",
        "asn": "",
        "as_name": "",
        "isp": "",
        "org": "",
        "is_hosting": False,
        "is_incongruent": False,
        "incongruity_score": 0,  # 0 to 100
        "signals": []
    }

    if not domain or domain in ["localhost", "127.0.0.1"]:
        return res

    # Resolve IP if not given
    if not res["ip"]:
        try:
            res["ip"] = socket.gethostbyname(domain)
        except Exception:
            return res

    try:
        r = requests.get(
            f"http://ip-api.com/json/{res['ip']}?fields=status,as,org,isp,hosting,query",
            timeout=2.0
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                res["available"] = True
                res["asn"] = data.get("as", "")
                res["isp"] = data.get("isp", "")
                res["org"] = data.get("org", "")
                res["is_hosting"] = bool(data.get("hosting", False))

                # Split ASN string (e.g. "AS51167 Contabo GmbH" -> asn="AS51167", as_name="Contabo GmbH")
                if " " in res["asn"]:
                    parts = res["asn"].split(" ", 1)
                    res["as_name"] = parts[1]
                else:
                    res["as_name"] = res["isp"]
    except Exception as e:
        logger.debug(f"[ABCI] ASN query failed: {e}")
        return res

    # Evaluate Brand Incongruity
    asn_lower = f"{res['asn']} {res['isp']} {res['org']}".lower()

    for brand in BRAND_OFFICIAL_ORGS.keys():
        if brand in domain:
            is_legit = domain == f"{brand}.com" or domain.endswith(f".{brand}.com")
            if not is_legit:
                # Check if hosted on disposable / unmanaged VPS provider
                for vps in DISPOSABLE_HOSTING_PROVIDERS:
                    if vps in asn_lower:
                        res["is_incongruent"] = True
                        res["incongruity_score"] = 85
                        res["signals"].append(
                            f"Autonomous System Brand Incongruity (ABCI): Domain targets '{brand}' "
                            f"but is hosted on disposable unmanaged infrastructure '{res['isp']}' ({res['asn']}) "
                            f"instead of the verified enterprise autonomous system."
                        )
                        return res

                # Check if it is hosted on generic VPS/hosting (not enterprise CDN or official ASN)
                enterprise_networks = ["microsoft", "apple", "google", "amazon", "paypal", "akamai", "fastly", "cloudflare"]
                if not any(net in asn_lower for net in enterprise_networks) and res["is_hosting"]:
                    res["is_incongruent"] = True
                    res["incongruity_score"] = 70
                    res["signals"].append(
                        f"Autonomous System Discrepancy: Impersonated brand '{brand}' is hosted on "
                        f"third-party cloud provider '{res['isp']}' ({res['asn']})."
                    )
                    return res

    if res["is_hosting"]:
        for vps in DISPOSABLE_HOSTING_PROVIDERS:
            if vps in asn_lower:
                res["signals"].append(f"Disposable Infrastructure: Hosted on low-reputation VPS provider '{res['isp']}' ({res['asn']}).")
                break

    return res
