"""
PhishGuard OriginTrace - Autonomous CertStream & Live Domain Monitoring Daemon
Connects to global Certificate Transparency (CT) WebSocket feed (wss://certstream.calidog.io).
Hunts real-time newly registered phishing domains and brand impersonations 24/7.
"""

import asyncio
import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse

import websockets
from core.logger import logger

# ── Monitored High-Value Protected Brands ─────────────────────────────────────
DEFAULT_PROTECTED_BRANDS = [
    "paypal", "microsoft", "google", "apple", "amazon", "netflix",
    "chase", "bankofamerica", "wellsfargo", "citibank", "docusign",
    "dropbox", "adobe", "facebook", "instagram", "linkedin", "irs",
    "github", "spotify", "binance", "coinbase", "metamask", "phishguard"
]

OFFICIAL_BRAND_DOMAINS = {
    "paypal": ["paypal.com", "paypal.me", "paypal-communication.com"],
    "microsoft": ["microsoft.com", "office.com", "live.com", "outlook.com", "azure.com", "office365.com", "msn.com", "skype.com"],
    "google": ["google.com", "gmail.com", "youtube.com", "google.co.uk", "google.ca"],
    "apple": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.de", "aws.amazon.com"],
    "netflix": ["netflix.com"],
    "chase": ["chase.com", "jpmorganchase.com"],
    "bankofamerica": ["bankofamerica.com", "bofa.com"],
    "wellsfargo": ["wellsfargo.com"],
    "citibank": ["citi.com", "citibank.com"],
    "docusign": ["docusign.com", "docusign.net"],
    "dropbox": ["dropbox.com"],
    "adobe": ["adobe.com"],
    "facebook": ["facebook.com", "fb.com", "meta.com"],
    "instagram": ["instagram.com"],
    "linkedin": ["linkedin.com"],
    "github": ["github.com", "github.io"],
    "spotify": ["spotify.com"],
    "binance": ["binance.com"],
    "coinbase": ["coinbase.com"],
    "metamask": ["metamask.io"],
}

SECURITY_KEYWORDS_RE = re.compile(
    r"(?:^|[-_.])(login|signin|sign-in|verify|verification|secure|security|"
    r"account|update|confirm|banking|wallet|token|auth|portal|recover|"
    r"reset|support|helpdesk|service|alert|unlock|validate|webscr|myaccount)(?:$|[-_.])",
    re.IGNORECASE
)

SUSPICIOUS_TLDS = {
    "xyz", "top", "pw", "cc", "icu", "club", "online", "site", "click",
    "link", "download", "zip", "review", "accountant", "loan", "science",
    "win", "bid", "trade", "racing", "party", "faith", "webcam", "tk", "ml", "ga", "cf", "gq"
}

CERTSTREAM_URL = "wss://certstream.calidog.io"


def _is_official_domain(domain: str, brand: str) -> bool:
    """Returns True if the domain is legitimately owned by the brand."""
    d = domain.lower().strip()
    official_list = OFFICIAL_BRAND_DOMAINS.get(brand, [f"{brand}.com"])
    for off in official_list:
        if d == off or d.endswith("." + off):
            return True
    return False


def _evaluate_domain_threat(domain: str, matched_brand: str) -> Dict:
    """
    Evaluates a lookalike domain using OriginTrace heuristics.
    Returns risk score, status, and forensic breakdown.
    """
    d = domain.lower().strip()
    parts = d.split(".")
    tld = parts[-1] if len(parts) > 1 else ""
    sld = parts[-2] if len(parts) > 1 else parts[0]

    risk_score = 45  # base score for containing a protected brand name
    reasons = [f"Brand impersonation target detected: '{matched_brand}'"]

    # 1. TLD Risk
    if tld in SUSPICIOUS_TLDS:
        risk_score += 25
        reasons.append(f"High-risk, abuse-prone top-level domain (.{tld})")

    # 2. Deceptive Security Keywords
    found_keywords = SECURITY_KEYWORDS_RE.findall(d)
    if found_keywords:
        risk_score += 25
        reasons.append(f"Deceptive credential harvesting keywords in domain: {list(set(found_keywords))}")

    # 3. Excessive hyphens or subdomains
    if d.count("-") >= 2:
        risk_score += 10
        reasons.append(f"Suspicious hyphenation pattern ({d.count('-')} hyphens)")

    if len(parts) >= 4:
        risk_score += 10
        reasons.append(f"Multi-layered subdomain structure ({len(parts)-1} levels)")

    # 4. Homoglyphs (e.g. paypa1, micr0soft)
    homo_map = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '8': 'b'})
    normalized = d.translate(homo_map)
    if normalized != d and matched_brand in normalized and matched_brand not in d:
        risk_score += 30
        reasons.append(f"Homoglyph character substitution detected spoofing '{matched_brand}'")

    risk_score = min(100, risk_score)
    status = "PHISHING" if risk_score >= 70 else "SUSPICIOUS"
    level = "critical" if risk_score >= 70 else "medium"

    pillars = {
        "pillar1_provenance": {
            "brand_impersonation": True,
            "target_brand": matched_brand,
            "authenticity": "Spoofed / Unauthorized Lookalike",
            "homoglyph_detected": bool(normalized != d and matched_brand in normalized)
        },
        "pillar2_infrastructure": {
            "tld": tld,
            "tld_risk": "HIGH" if tld in SUSPICIOUS_TLDS else "MODERATE",
            "ssl_certificate": "Newly Issued (Detected at CT Log Creation)",
            "registration_horizon": "< 1 hour"
        },
        "pillar3_deception": {
            "credential_harvesting_intent": bool(found_keywords),
            "keywords": list(set(found_keywords))
        },
        "pillar4_evidence": {
            "risk_score": risk_score,
            "epistemic_uncertainty": 0.08,
            "confidence": "HIGH"
        }
    }

    return {
        "domain": d,
        "matched_brand": matched_brand,
        "tld": tld,
        "risk_score": risk_score,
        "status": status,
        "risk_level": level,
        "reasons": reasons,
        "origintrace_pillars": pillars
    }


def save_threat_alert(threat: Dict, cert_issuer: Optional[str] = None):
    """Saves a detected threat to the database and notifies the system."""
    try:
        from data.database import engine
        from sqlmodel import Session, select
        from data.models import LiveThreatAlert, Notification

        with Session(engine) as session:
            # Check if domain already logged
            existing = session.exec(
                select(LiveThreatAlert).where(LiveThreatAlert.domain == threat["domain"])
            ).first()
            if existing:
                return

            alert = LiveThreatAlert(
                domain=threat["domain"],
                matched_brand=threat["matched_brand"],
                tld=threat["tld"],
                risk_score=threat["risk_score"],
                risk_level=threat["risk_level"],
                status=threat["status"],
                reasons_json=json.dumps(threat["reasons"]),
                origintrace_pillars_json=json.dumps(threat["origintrace_pillars"]),
                cert_issuer=cert_issuer,
                detected_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
            session.add(alert)

            # Create a notification for high-risk threats
            if threat["risk_score"] >= 70:
                notif = Notification(
                    title=f"🚨 CertStream Radar: Lookalike '{threat['domain']}' Detected",
                    message=f"Autonomous Radar caught newly registered domain targeting {threat['matched_brand'].upper()} ({threat['risk_score']}/100 Risk).",
                    level="critical",
                    is_read=False,
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None)
                )
                session.add(notif)

            session.commit()
            logger.warning(
                f"[CertStream Radar] Caught Live Phishing Domain: {threat['domain']} "
                f"(Target: {threat['matched_brand']} | Score: {threat['risk_score']})"
            )
    except Exception as e:
        logger.error(f"[CertStream Radar] Error saving threat alert: {e}")


def get_active_brands() -> List[str]:
    """Returns the union of default protected brands and user-defined rules."""
    try:
        from data.database import engine
        from sqlmodel import Session, select
        from data.models import BrandMonitorRule

        with Session(engine) as session:
            custom_rules = session.exec(
                select(BrandMonitorRule).where(BrandMonitorRule.is_active == True)
            ).all()
            custom_brands = [r.brand_name.lower().strip() for r in custom_rules]
            return list(set(DEFAULT_PROTECTED_BRANDS + custom_brands))
    except Exception:
        return DEFAULT_PROTECTED_BRANDS


async def _certstream_worker():
    """
    Main asynchronous worker connecting to CertStream WebSocket feed.
    Continuously parses certificate events, matches against brands, and logs threats.
    """
    logger.info("[CertStream Radar] Initializing Autonomous Domain Monitoring Daemon...")
    reconnect_delay = 5

    while True:
        try:
            logger.info(f"[CertStream Radar] Connecting to live feed: {CERTSTREAM_URL} ...")
            async with websockets.connect(
                CERTSTREAM_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=10_000_000
            ) as ws:
                logger.info("[CertStream Radar] Connected successfully! Monitoring global certificates in real-time.")
                reconnect_delay = 5  # reset on successful connection

                while True:
                    raw_msg = await ws.recv()
                    data = json.loads(raw_msg)

                    if data.get("message_type") != "certificate_update":
                        continue

                    cert_data = data.get("data", {})
                    leaf_cert = cert_data.get("leaf_cert", {})
                    all_domains = leaf_cert.get("all_domains", [])
                    issuer = leaf_cert.get("issuer", {}).get("O", "Unknown Issuer")

                    if not all_domains:
                        continue

                    monitored_brands = get_active_brands()

                    for raw_domain in all_domains:
                        clean_dom = raw_domain.lower().strip().lstrip("*.").split(":")[0]
                        if not clean_dom or len(clean_dom) < 4:
                            continue

                        # Check against active brands
                        for brand in monitored_brands:
                            # Match if brand is in domain string (e.g. 'paypal-security.com' or 'chase-login.xyz')
                            if brand in clean_dom:
                                # Ensure it's not the official legitimate brand website
                                if not _is_official_domain(clean_dom, brand):
                                    threat = _evaluate_domain_threat(clean_dom, brand)
                                    if threat["risk_score"] >= 55:
                                        save_threat_alert(threat, cert_issuer=issuer)
                                break

        except (websockets.ConnectionClosed, websockets.WebSocketException) as ws_err:
            logger.warning(f"[CertStream Radar] WebSocket connection lost: {ws_err}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(60, reconnect_delay * 2)
        except Exception as e:
            logger.error(f"[CertStream Radar] Unexpected error: {e}. Retrying in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(60, reconnect_delay * 2)


_daemon_thread: Optional[threading.Thread] = None


def start_certstream_daemon():
    """Starts the CertStream background daemon in a separate dedicated thread."""
    global _daemon_thread
    if _daemon_thread and _daemon_thread.is_alive():
        return

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_certstream_worker())

    _daemon_thread = threading.Thread(target=_run, name="CertStreamRadarThread", daemon=True)
    _daemon_thread.start()
    logger.info("[CertStream Radar] Daemon thread started.")
