import os
import re
import base64
import time
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from core.logger import logger
from functools import lru_cache

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
REQUEST_TIMEOUT = 4  # seconds (was 6)

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

        # 404 means URL not in VT cache yet — submit it and return "not yet indexed"
        # instead of doing a slow double-fetch with sleep
        if resp.status_code == 404:
            try:
                requests.post(
                    "https://www.virustotal.com/api/v3/urls",
                    headers=headers,
                    data={"url": url},
                    timeout=REQUEST_TIMEOUT,
                )
            except Exception:
                pass
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

    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_vt  = ex.submit(check_virustotal, url)
        fut_pt  = ex.submit(check_phishtank, url)
        fut_uh  = ex.submit(check_urlhaus, url)
        fut_da  = ex.submit(get_domain_age, url)

        api_timeout  = REQUEST_TIMEOUT + 1   # 5s — for fast REST API calls
        whois_timeout = 15                    # WHOIS can be slow for ccTLDs (.pk, .io etc.)
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
            # If WHOIS returned a stale failure (e.g. from before library was installed),
            # do NOT cache it — let it retry on the next call
            if not domain_age.get("available") and domain_age.get("reason", "") in (
                "python-whois not installed", "WHOIS lookup failed"
            ):
                domain_age = {"available": False, "reason": "WHOIS unavailable"}
        except Exception:
            domain_age = {"available": False, "reason": "Timed out"}

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
    else:
        overall = "CLEAN"
        overall_score = 5

    result = {
        "url": url,
        "virustotal": vt,
        "phishtank": pt,
        "urlhaus": uh,
        "domain_age": domain_age,
        "overall": overall,
        "overall_score": overall_score,
        "is_malicious": is_malicious,
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
        return 'US'

    domain = domain_or_email
    if '@' in domain:
        domain = domain.split('@')[-1]

    if '://' in domain:
        domain = domain.split('://')[-1].split('/')[0]

    domain = domain.lower().strip()

    parts = domain.split('.')
    if len(parts) >= 2:
        tld = parts[-1]
        if len(tld) == 2 and tld not in ['co', 'go', 'ac', 'edu']:
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

    return 'US'
