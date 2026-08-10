"""
In-memory data store for the AI Phishing Detection Platform.
Replaces the static mock_data by storing real scans over the course of the server session.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
import threading
from typing import Dict, Any, Callable
import uuid

from sqlmodel import Session, select
from data.database import engine
from data.models import ScanRecord, KPIRecord, User, Campaign, Notification, ThreatCacheRecord

logger = logging.getLogger("phishguard")

# ── Persistent Threat Intel Cache Helpers ────────────────────────────────────
import hashlib
import os

_THREAT_CACHE_TTL_HOURS = int(os.getenv("THREAT_CACHE_TTL_HOURS", "24"))


def _url_hash(url: str) -> str:
    """Return a stable SHA-256 hex digest for a URL (used as the DB cache key)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def get_threat_cache(url: str) -> dict | None:
    """
    Look up a cached threat intel result for a URL.
    Returns the cached dict if it exists and hasn't expired, else None.
    Uses its own short-lived DB session so it can be called from anywhere.
    """
    try:
        with Session(engine) as session:
            record = session.exec(
                select(ThreatCacheRecord).where(ThreatCacheRecord.url_hash == _url_hash(url))
            ).first()
            if record and record.is_valid:
                return record.result
    except Exception:
        pass
    return None


def set_threat_cache(url: str, result: dict) -> None:
    """
    Persist a threat intel result for a URL into SQLite.
    Upserts: if a record already exists for the URL hash, it is overwritten.
    Never caches transient failure states (e.g. missing library, lookup timeout)
    so that the next scan attempt will retry the live API.
    """
    import json

    # Do NOT cache results where WHOIS failed transiently —
    # they would survive server restarts and block real data permanently.
    da = result.get("domain_age", {})
    bad_reasons = {"python-whois not installed", "WHOIS lookup failed", "Timed out", "WHOIS unavailable"}
    if not da.get("available", False) and da.get("reason", "") in bad_reasons:
        logger.debug(f"[ThreatCache] Skipping DB cache for {url[:60]} — WHOIS not available yet")
        return

    try:
        with Session(engine) as session:
            h = _url_hash(url)
            record = session.exec(
                select(ThreatCacheRecord).where(ThreatCacheRecord.url_hash == h)
            ).first()
            expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=_THREAT_CACHE_TTL_HOURS)
            if record:
                record.result_json = json.dumps(result)
                record.expires_at = expiry
                record.created_at = datetime.now(timezone.utc).replace(tzinfo=None)
            else:
                record = ThreatCacheRecord(
                    url_hash=h,
                    url=url[:2000],  # guard against enormous URLs
                    result_json=json.dumps(result),
                    expires_at=expiry,
                )
            session.add(record)
            session.commit()
    except Exception as e:
        logger.warning(f"[ThreatCache] Failed to persist cache for {url[:60]}: {e}")


# Simple thread-safe in-memory cache
_METRIC_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 30  # Low TTL ensures data stays relatively fresh
_cache_lock = threading.Lock()

def cached_metric(func: Callable):
    """Decorator to cache slow-running database metric calculations."""
    def wrapper(*args, **kwargs):
        cache_key = func.__name__
        now = datetime.now()
        
        with _cache_lock:
            if cache_key in _METRIC_CACHE:
                entry = _METRIC_CACHE[cache_key]
                if (now - entry["timestamp"]).total_seconds() < CACHE_TTL_SECONDS:
                    return entry["data"]
        
        # Cache miss or expired
        result = func(*args, **kwargs)
        
        with _cache_lock:
            _METRIC_CACHE[cache_key] = {
                "timestamp": now,
                "data": result
            }
        return result
    return wrapper

# We don't need in-memory lists anymore!
# SCANS = []
# CAMPAIGNS = []

def init_db():
    from data.database import create_db_and_tables
    # Importing all models here ensures SQLModel.metadata has all table definitions
    from data.models import (  # noqa: F401
        User, ScanRecord, KPIRecord, Campaign, Notification,
        InboxConnection, ThreatCacheRecord
    )
    create_db_and_tables()

    
    # Create an initial Admin user if none exists
    with Session(engine) as session:
        user = session.exec(select(User)).first()
        if not user:
            import os
            from services.security import get_password_hash
            admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@phishguard.io")
            admin_pass  = os.getenv("DEFAULT_ADMIN_PASS", "ChangeMe@Immediately!")
            admin = User(
                uuid=str(uuid.uuid4()),
                name="Admin User",
                email=admin_email,
                hashed_password=get_password_hash(admin_pass),
                role="admin"
            )
            session.add(admin)
            session.commit()

def _get_today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def get_kpi(session: Session) -> dict:
    """Fetch KPI data. Falls back to default if no records exist."""
    today_str = _get_today_str()
    kpi = session.exec(select(KPIRecord).where(KPIRecord.date_str == today_str)).first()
    
    if kpi:
        return {
            "emails_scanned_today": kpi.emails_scanned,
            "emails_scanned_change": 0.0,
            "phishing_detected": kpi.phishing_detected,
            "phishing_change": 0.0,
            "high_risk_domains": kpi.high_risk_domains,
            "high_risk_change": 0.0,
            "active_campaigns": kpi.active_campaigns,
            "active_campaigns_change": 0.0,
        }
    return {
        "emails_scanned_today": 0, "emails_scanned_change": 0.0,
        "phishing_detected": 0, "phishing_change": 0.0,
        "high_risk_domains": 0, "high_risk_change": 0.0,
        "active_campaigns": 0, "active_campaigns_change": 0.0,
    }


@cached_metric
def get_risk_trend(session: Session) -> dict:
    """Generates a trend for the last 7 days from the DB."""
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    labels, phishing, suspicious, clean = [], [], [], []
    
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        labels.append(d.strftime("%b %d"))
        
        # Calculate for this specific day using SQL bounds
        start_of_day = d.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = d.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        scans_that_day = session.exec(
            select(ScanRecord.status).where(
                ScanRecord.timestamp >= start_of_day,
                ScanRecord.timestamp <= end_of_day
            )
        ).all()
        
        day_p = sum(1 for status in scans_that_day if status == 'PHISHING')
        day_s = sum(1 for status in scans_that_day if status == 'SUSPICIOUS')
        day_c = sum(1 for status in scans_that_day if status == 'CLEAN')
        
        phishing.append(day_p)
        suspicious.append(day_s)
        clean.append(day_c)
        
    return {"labels": labels, "phishing": phishing, "suspicious": suspicious, "clean": clean}


@cached_metric
def get_phishing_types(session: Session) -> dict:
    """Dynamically compile phishing types based on actual scan flags in DB."""
    scans = session.exec(
        select(ScanRecord).where(ScanRecord.risk_level.in_(['critical', 'medium']))
    ).all()
    
    if not scans:
        return {"labels": ["Awaiting Data"], "values": [1]}
        
    types_count = {}
    for scan in scans:
        full_result = scan.full_analysis
        flags = []
        
        # Quick heuristic parsing of the JSON dump
        dump = str(full_result)
        if 'Known malicious domain' in dump: flags.append('Malware Link')
        if 'Suspicious TLD' in dump: flags.append('Suspicious Domain')
        if 'has_urgency' in dump: flags.append('Urgency Lure')
        if 'has_impersonation' in dump: flags.append('Brand Impersonation')
        if 'spf=fail' in dump or 'dkim=fail' in dump: flags.append('Spoofing/BEC')
        
        if not flags:
            flags.append('General Phishing')
            
        for ft in flags:
            types_count[ft] = types_count.get(ft, 0) + 1
                
    if not types_count:
        return {"labels": ["Awaiting Data"], "values": [1]}

    return {
        "labels": list(types_count.keys()),
        "values": list(types_count.values())
    }


@cached_metric
def get_top_targets(session: Session) -> dict:
    """Returns the most frequently seen sender domains (attack sources) from DB."""
    scans = session.exec(select(ScanRecord.sender)).all()
    
    if not scans:
        return {"labels": [], "values": []}
        
    sender_count = {}
    for sender in scans:
        sender = sender.strip()
        if sender and sender != "Unknown Sender":
            sender_count[sender] = sender_count.get(sender, 0) + 1
            
    sorted_senders = sorted(sender_count.items(), key=lambda x: x[1], reverse=True)[:6]
    return {
        "labels": [s[0] for s in sorted_senders],
        "values": [s[1] for s in sorted_senders]
    }


def add_scan(scan_result: dict, session: Session, gmail_id: str = "", scan_type: str = "email"):
    """
    Called by scan_engine to persist a new scan and update metrics dynamically.
    gmail_id: if provided, uses 'gm-{gmail_id}' as the scan_id (prevents duplicates during inbox sync).
    scan_type: 'email', 'url', or 'domain'
    """
    prefix = "gm" if gmail_id else ("ur" if scan_type == "url" else ("dm" if scan_type == "domain" else "em"))
    scan_id = f"gm-{gmail_id}" if gmail_id else f"{prefix}-{str(uuid.uuid4())[:8]}"
    
    # 1. Create the ScanRecord
    record = ScanRecord(
        scan_id=scan_id,
        scan_type=scan_type,
        sender=scan_result.get("sender", "Unknown Sender"),
        subject=scan_result.get("subject", "No Subject"),
        risk_score=scan_result.get("final_score", 0),
        status=scan_result.get("status", "CLEAN"),
        risk_level=scan_result.get("risk_level", "safe"),
    )
    record.full_analysis = scan_result
    session.add(record)
    
    # 2. Update Daily KPI
    today_str = _get_today_str()
    kpi = session.exec(select(KPIRecord).where(KPIRecord.date_str == today_str)).first()
    if not kpi:
        kpi = KPIRecord(date_str=today_str)
        session.add(kpi)
        
    kpi.emails_scanned += 1
    if record.status == "PHISHING":
        kpi.phishing_detected += 1
    if record.risk_level == "critical":
        kpi.high_risk_domains += 1
        
    session.commit()
    
    # Auto-create notification for critical/phishing scans
    if record.status == "PHISHING":
        notif = Notification(
            title="🚨 Phishing Detected",
            message=f"Email from '{record.sender}' - Subject: '{record.subject[:50]}'",
            level="critical",
            scan_id=scan_id
        )
        session.add(notif)
        session.commit()
    
    return scan_id

def get_recent_scans(session: Session, limit: int = 10):
    scans = session.exec(
        select(ScanRecord).order_by(ScanRecord.timestamp.desc()).limit(limit)
    ).all()
    
    result = []
    for s in scans:
        result.append({
            "id": s.scan_id,
            "sender": s.sender,
            "subject": s.subject,
            "risk_score": s.risk_score,
            "status": s.status,
            "risk_level": s.risk_level,
            "time": s.timestamp.strftime("%Y-%m-%d %H:%M")
        })
    return result


def get_all_scans(session: Session):
    """Returns all scans for the history page (all scan types), plus summary counts."""
    scans = session.exec(
        select(ScanRecord).order_by(ScanRecord.timestamp.desc())
    ).all()

    rows = []
    for s in scans:
        rows.append({
            "scan_id": s.scan_id,
            "scan_type": getattr(s, 'scan_type', 'email'),
            "sender": s.sender,
            "subject": s.subject,
            "risk_score": s.risk_score,
            "status": s.status,
            "risk_level": s.risk_level,
            "timestamp": s.timestamp.strftime("%Y-%m-%d %H:%M"),
        })

    counts = {
        "total": len(rows),
        "clean": sum(1 for r in rows if r["status"] == "CLEAN"),
        "suspicious": sum(1 for r in rows if r["status"] == "SUSPICIOUS"),
        "phishing": sum(1 for r in rows if r["status"] == "PHISHING"),
    }
    return rows, counts

def get_email_detail(email_id: str, session: Session):
    scan = session.exec(select(ScanRecord).where(ScanRecord.scan_id == email_id)).first()
    if not scan:
        return None
        
    an = scan.full_analysis or {}
    return {
        "id": email_id,
        "sender": an.get("sender", "Unknown"),
        "sender_name": an.get("sender_name", "Scanner Result"),
        "sender_ip": an.get("sender_ip", "Unknown"),
        "sender_country": an.get("sender_country", "Unknown"),
        "sender_isp": an.get("sender_isp", "Unknown"),
        "subject": an.get("subject", "No Subject"),
        "received_time": scan.timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.timestamp else "Unknown",
        "risk_score": scan.risk_score,
        "status": scan.status,
        "risk_level": scan.risk_level,
        "body_preview": an.get("body_preview", ""),
        "spf": an.get("header_authentication", {}).get("spf", "UNKNOWN") if isinstance(an.get("header_authentication"), dict) else "UNKNOWN",
        "dkim": an.get("header_authentication", {}).get("dkim", "UNKNOWN") if isinstance(an.get("header_authentication"), dict) else "UNKNOWN",
        "dmarc": an.get("header_authentication", {}).get("dmarc", "UNKNOWN") if isinstance(an.get("header_authentication"), dict) else "UNKNOWN",
        "domain_age_days": an.get("url_intelligence", {}).get("domain_age_days", 0) if isinstance(an.get("url_intelligence"), dict) else 0,
        "urls": an.get("url_intelligence", {}).get("flagged_urls", []) if isinstance(an.get("url_intelligence"), dict) else [],
        "attachments": an.get("attachments", []),
        "routing": an.get("routing", []),
        "raw_headers": an.get("headers", "No headers provided."),
        "why_flagged": an.get("threat_indicators", []) if isinstance(an.get("threat_indicators"), list) else [],
        "threat_intel": an.get("threat_intel", []),
    }

def get_all_users(session: Session):
    users = session.exec(select(User)).all()
    return [{
        "id": u.uuid,
        "name": u.name,
        "email": u.email,
        "role": u.role,
        "status": "Active",
        "last_login": u.last_login.strftime("%Y-%m-%d %H:%M:%S") if u.last_login else "Never"
    } for u in users]


def get_campaigns(session: Session):
    """Fetch all campaigns, ordered by most recently seen."""
    campaigns = session.exec(
        select(Campaign).order_by(Campaign.last_seen.desc())
    ).all()

    if not campaigns:
        return []

    return [{
        "name": c.name,
        "type": c.type,
        "source": c.source,
        "severity": c.severity,
        "targets": c.targets,
        "first_seen": c.first_seen.strftime("%Y-%m-%d %H:%M"),
        "last_seen": c.last_seen.strftime("%Y-%m-%d %H:%M"),
    } for c in campaigns]


def auto_build_campaigns(session: Session):
    """
    Automatically groups PHISHING scans by sender domain
    into Campaign records. Runs after each scan.
    Idempotent: merges with existing campaigns.
    """
    phishing_scans = session.exec(
        select(ScanRecord).where(ScanRecord.status == "PHISHING")
    ).all()

    domain_groups: dict = {}
    for scan in phishing_scans:
        sender = scan.sender or "unknown"
        # Extract domain from email-like strings
        domain = sender.split("@")[-1].strip() if "@" in sender else sender
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(scan)

    for domain, scans in domain_groups.items():
        existing = session.exec(
            select(Campaign).where(Campaign.source == domain)
        ).first()

        timestamps = [s.timestamp for s in scans]
        first = min(timestamps)
        last = max(timestamps)
        count = len(scans)

        # Determine severity from risk levels
        has_critical = any(s.risk_level == "critical" for s in scans)
        severity = "critical" if has_critical else "medium"

        if existing:
            existing.targets = count
            existing.last_seen = last
            existing.severity = severity
        else:
            # Generate a readable campaign name
            latest_subject = scans[-1].subject or "Unknown Attack"
            name = latest_subject[:60] + ("..." if len(latest_subject) > 60 else "")
            c = Campaign(
                name=name,
                type="Phishing Campaign",
                source=domain,
                severity=severity,
                targets=count,
                first_seen=first,
                last_seen=last,
            )
            session.add(c)

    session.commit()


def get_notifications(session: Session, limit: int = 20):
    """Fetch recent notifications, newest first."""
    notifs = session.exec(
        select(Notification).order_by(Notification.created_at.desc()).limit(limit)
    ).all()
    return [{
        "id": n.id,
        "title": n.title,
        "message": n.message,
        "level": n.level,
        "is_read": n.is_read,
        "scan_id": n.scan_id,
        "created_at": n.created_at.strftime("%Y-%m-%d %H:%M"),
    } for n in notifs]


def mark_notifications_read(session: Session):
    """Mark all unread notifications as read."""
    unread = session.exec(
        select(Notification).where(Notification.is_read == False)
    ).all()
    for n in unread:
        n.is_read = True
    session.commit()


from utils.threat_intel import resolve_country
from collections import Counter

@cached_metric
def get_top_threat_sources(session: Session, limit: int = 5) -> dict:
    """Fetch top threat sources by country for the global map."""
    scans = session.exec(
        select(ScanRecord).where(ScanRecord.status != 'CLEAN')
    ).all()
    
    counts = Counter()
    for scan in scans:
        if scan.sender:
            # Basic domain extraction if it's an email or URL
            domain = scan.sender
            if '@' in domain:
                domain = domain.split('@')[-1]
            elif '://' in domain:
                domain = domain.split('://')[-1].split('/')[0]
                
            country = resolve_country(domain)
            counts[country] += 1
            
    # Default data if no threats found so the map isn't completely empty
    if not counts:
        counts = Counter({'US': 1})
            
    top_list = counts.most_common(limit)
    
    # Map country codes to readable names
    country_names = {
        'US': 'United States', 'RU': 'Russia', 'CN': 'China', 'IN': 'India',
        'BR': 'Brazil', 'NG': 'Nigeria', 'UA': 'Ukraine', 'ZA': 'South Africa',
        'GB': 'United Kingdom', 'DE': 'Germany', 'FR': 'France', 'JP': 'Japan'
    }
    
    return {
        "map_data": dict(counts),
        "top_list": [{"code": c[0], "name": country_names.get(c[0], c[0]), "count": c[1]} for c in top_list]
    }

# ── Block Rules ────────────────────────────────────────────────────────────

def add_block_rule(session: Session, target: str, rule_type: str = "domain") -> bool:
    from data.models import BlockRule
    target = target.strip().lower()
    existing = session.exec(select(BlockRule).where(BlockRule.target == target)).first()
    if existing:
        return False
    rule = BlockRule(target=target, rule_type=rule_type)
    session.add(rule)
    session.commit()
    return True

def get_block_rules(session: Session):
    from data.models import BlockRule
    rules = session.exec(select(BlockRule).order_by(BlockRule.created_at.desc())).all()
    return [{"id": r.id, "target": r.target, "rule_type": r.rule_type, "created_at": r.created_at.strftime("%Y-%m-%d %H:%M")} for r in rules]

def remove_block_rule(session: Session, rule_id: int) -> bool:
    from data.models import BlockRule
    rule = session.get(BlockRule, rule_id)
    if rule:
        session.delete(rule)
        session.commit()
        return True
    return False

def is_blocked(session: Session, target: str) -> bool:
    from data.models import BlockRule
    target = target.strip().lower()
    # Check exact match
    exact = session.exec(select(BlockRule).where(BlockRule.target == target)).first()
    if exact:
        return True
    # If target is an email, check if its domain is blocked
    if "@" in target:
        domain = target.split("@")[-1]
        domain_match = session.exec(select(BlockRule).where(BlockRule.target == domain)).first()
        if domain_match:
            return True
    return False
