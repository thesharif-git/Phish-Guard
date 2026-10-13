from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Response, BackgroundTasks
from sqlmodel import Session
from data.database import get_session
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
import re

from services.scan_engine import scan_email as run_scan
from services.security import get_current_user, get_optional_current_user, verify_password, get_password_hash
from data.models import User
from data import store
from utils.email_parser import parse_eml_content, parse_pasted_email, visible_html_text
from utils.url_scanner import scan_url as run_url_scan
from services.plan_limits import require_scan_allowance

class ScanUrlRequest(BaseModel):
    url: str


router = APIRouter(prefix="/api")


@router.post("/scan-email")
async def scan_email_endpoint(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None),
    sender: Optional[str] = Form(""),
    subject: Optional[str] = Form(""),
    headers: Optional[str] = Form(""),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    import asyncio

    attachments = []
    html_body = ""
    parsing_info = None
    if file:
        file_bytes = await file.read()
        if (file.filename or '').lower().endswith('.eml'):
            parsed_email = parse_eml_content(file_bytes)
            content = parsed_email["body"]
            sender = parsed_email["sender"]
            subject = parsed_email["subject"]
            headers = parsed_email["headers"]
            attachments = parsed_email.get("attachments", [])
            html_body = parsed_email.get("html_body", "")
            parsing_info = parsed_email.get("parsing")
        elif (file.filename or '').lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
            # Direct image attachment upload for visual/Quishing inspection
            attachments = [{
                "filename": file.filename,
                "content_type": file.content_type or "image/png",
                "data": file_bytes
            }]
            if not content:
                content = ""  # Filenames are not observed message content.
        else:
            raise HTTPException(status_code=400, detail="Only .eml email files and image files (.png, .jpg, .jpeg, .webp) are supported.")

    # The message field also accepts a complete pasted RFC email. Use its real
    # header block when present instead of silently discarding it.
    if not file and content and not headers:
        parsed_email = parse_pasted_email(content)
        if parsed_email:
            content = parsed_email["body"]
            headers = parsed_email["headers"]
            sender = sender or parsed_email["sender"]
            subject = subject or parsed_email["subject"]
            attachments = parsed_email.get("attachments", [])
            html_body = parsed_email.get("html_body", "")
            parsing_info = parsed_email.get("parsing")
        elif re.search(r'<(?:html|body|p|div|table|a)\b[^>]*>', content, re.I):
            html_body = content
            content = visible_html_text(content)
            parsing_info = {'source_kind': 'pasted_html', 'text_complete': True, 'errors': [], 'unscanned_parts': []}
        
    if not content and not file:
        raise HTTPException(status_code=400, detail="Must provide either text content or an .eml/image file.")

    require_scan_allowance(session, user)

    # Run the synchronous scan in a thread pool — keeps the event loop free
    result = await asyncio.to_thread(
        run_scan,
        content=content,
        sender=sender,
        subject=subject,
        headers=headers,
        session=session,
        attachments=attachments,
        html_body=html_body,
        parsing_info=parsing_info,
        user_id=user.id,
    )

    # scan_email persists once, with the authenticated owner. A second write
    # here created duplicate history entries and a first ownerless record.

    if result.get("status") == "PHISHING" and user.browser_alerts:
        from utils.email_alerts import send_phishing_alert
        background_tasks.add_task(send_phishing_alert, user.email, user.name, result)

    return JSONResponse(content=result)


@router.post("/scan-url")
async def scan_url_endpoint(
    req: ScanUrlRequest,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_optional_current_user)
):
    """Dedicated URL threat intelligence scanner."""
    import asyncio

    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    require_scan_allowance(session, user)

    # Run in thread pool — scan makes many blocking network calls
    result = await asyncio.to_thread(run_url_scan, url)
    
    # Save URL scan to unified history
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc or url
        scan_result_for_db = {
            "sender": domain,
            "subject": url,
            "final_score": result.get("final_score", 0),
            "status": result.get("status", "CLEAN").upper(),
            "risk_level": result.get("risk_level", "safe"),
            "geo": {"countryCode": (result.get("geo") or {}).get("countryCode", "")},
        }
        store.add_scan(scan_result_for_db, session, scan_type="url", user_id=user.id if user else None)
    except Exception:
        pass
    
    return JSONResponse(content=result)


class ScanDomainRequest(BaseModel):
    domain: str

@router.post("/scan-domain")
async def scan_domain_endpoint(
    req: ScanDomainRequest,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_optional_current_user)
):
    """Dedicated Domain reputation scanner."""
    import asyncio

    domain = req.domain.strip()
    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required.")
    from services.domain_scan_policy import domain_scan_input, scope_domain_result
    try:
        domain, url = domain_scan_input(domain)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    require_scan_allowance(session, user)

    # Run in thread pool — makes blocking WHOIS + API calls
    result = await asyncio.to_thread(run_url_scan, url)
    result = scope_domain_result(result, domain)
    
    # Save domain scan to unified history with scan_type='domain'
    try:
        scan_result_for_db = {
            "sender": domain,
            "subject": domain,
            "final_score": result.get("final_score", 0),
            "status": result.get("status", "CLEAN").upper(),
            "risk_level": result.get("risk_level", "safe"),
            "geo": {"countryCode": (result.get("geo") or {}).get("countryCode", "")},
        }
        store.add_scan(scan_result_for_db, session, scan_type="domain", user_id=user.id if user else None)
    except Exception:
        pass
    
    return JSONResponse(content=result)


@router.get("/dashboard")
async def dashboard_data(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    kpi = store.get_kpi(session)
    return {
        "kpi": kpi,
        "risk_trend": store.get_risk_trend(session, user.id),
        "phishing_types": store.get_phishing_types(session, user.id),
        "top_targets": store.get_top_targets(session),
        "recent_scans": store.get_recent_scans(session, limit=10, user_id=user.id),
        "security_score": max(0, 100 - (
            int((kpi["phishing_detected"] / kpi["emails_scanned_today"]) * 100)
            if kpi["emails_scanned_today"] > 0 else 0
        )),
    }


@router.get("/campaigns")
async def campaigns_data(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    store.auto_build_campaigns(session)
    return {"campaigns": store.get_campaigns(session)}


@router.get("/emails")
async def emails_data(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    return {"emails": store.get_recent_scans(session, limit=50, user_id=user.id)}


@router.get("/scan-history")
async def scan_history_data(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    """Return the signed-in user's scan history for the slide-over panel."""
    rows, counts = store.get_all_scans(session, user.id)
    return {"scans": rows[:50], "counts": counts}


# ── Profile ───────────────────────────────────────────────────────────────────
from pydantic import BaseModel as PydanticBase, Field, EmailStr

class ProfileUpdate(PydanticBase):
    name: str | None = Field(None, min_length=2, max_length=50)
    current_password: str | None = None
    new_password: str | None = Field(None, min_length=8, max_length=128)
    theme: str | None = None
    browser_alerts: bool | None = None
    sync_alerts: bool | None = None
    weekly_reports: bool | None = None

@router.patch("/profile")
async def update_profile(
    update: ProfileUpdate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    if update.name:
        user.name = update.name.strip()

    if update.new_password:
        if not update.current_password:
            raise HTTPException(status_code=400, detail="Current password is required to set a new password.")
        if not verify_password(update.current_password, user.hashed_password):
            raise HTTPException(status_code=403, detail="Current password is incorrect.")
        user.hashed_password = get_password_hash(update.new_password)

    if update.theme is not None:
        user.theme = update.theme
    if update.browser_alerts is not None:
        user.browser_alerts = update.browser_alerts
    if update.sync_alerts is not None:
        user.sync_alerts = update.sync_alerts
    if update.weekly_reports is not None:
        user.weekly_reports = update.weekly_reports

    session.add(user)
    session.commit()
    session.refresh(user)
    return {"msg": "Profile updated successfully", "name": user.name}

@router.delete("/profile")
async def delete_account(
    response: Response,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Securely deletes the user account and logs them out."""
    session.delete(user)
    session.commit()
    response.delete_cookie(key="access_token")
    return {"msg": "Account permanently deleted"}


# ── Notifications ─────────────────────────────────────────────────────────────
@router.get("/notifications")
async def get_notifications(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    return {"notifications": store.get_notifications(session)}

@router.patch("/notifications/read-all")
async def mark_all_read(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    store.mark_notifications_read(session)
    return {"msg": "All notifications marked as read"}


# ── Inbox Integration ──────────────────────────────────────────────────────────

@router.get("/inbox/status")
async def inbox_status(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    """Return connected inbox info for the settings page."""
    from data.models import InboxConnection
    from sqlmodel import select

    conn = session.exec(
        select(InboxConnection).where(
            InboxConnection.user_id == user.id,
            InboxConnection.provider == "gmail",
            InboxConnection.is_active == True,
        )
    ).first()

    if not conn:
        return {"connected": False, "provider": "gmail"}

    return {
        "connected": True,
        "provider": "gmail",
        "email": conn.email_address,
        "last_synced": conn.last_synced.isoformat() if conn.last_synced else None,
        "token_valid": conn.token_expiry and conn.token_expiry > __import__('datetime').datetime.now(__import__('datetime').timezone.utc).replace(tzinfo=None),
    }


@router.post("/inbox/sync")
async def trigger_inbox_sync(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    """Manually trigger a Gmail inbox sync for the logged-in user."""
    from data.models import InboxConnection
    from sqlmodel import select
    from services.inbox_connector import sync_gmail_inbox

    conn = session.exec(
        select(InboxConnection).where(
            InboxConnection.user_id == user.id,
            InboxConnection.provider == "gmail",
            InboxConnection.is_active == True,
        )
    ).first()

    if not conn:
        raise HTTPException(status_code=404, detail="No Gmail inbox connected. Go to Settings to connect.")

    result = await sync_gmail_inbox(conn, session)

    if not result.get("success"):
        raise HTTPException(status_code=503, detail=result.get("error", "Sync failed."))

    return result

# ── Block Rules ──────────────────────────────────────────────────────────

from pydantic import BaseModel

class BlockRuleRequest(BaseModel):
    target: str
    rule_type: str = "domain"

@router.post("/rules/block")
async def create_block_rule(req: BlockRuleRequest, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    rule_id = store.add_block_rule(session, req.target, req.rule_type)
    if rule_id:
        return {"success": True, "id": rule_id, "message": f"Rule created for {req.target}"}
    raise HTTPException(status_code=400, detail="Rule already exists")

@router.get("/rules/block")
async def list_block_rules(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    return {"rules": store.get_block_rules(session)}

@router.delete("/rules/block/{rule_id}")
async def delete_block_rule(rule_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    if store.remove_block_rule(session, rule_id):
        return {"success": True, "message": "Rule removed"}
    raise HTTPException(status_code=404, detail="Rule not found")


class ScanFeedbackRequest(BaseModel):
    scan_id: str
    feedback_type: str  # "false_positive" or "false_negative"
    notes: Optional[str] = ""

@router.post("/scan-feedback")
async def submit_scan_feedback(
    req: ScanFeedbackRequest,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_optional_current_user)
):
    """Stores user active learning feedback for misclassification reporting."""
    from data.models import ScanFeedback

    if req.feedback_type not in ("false_positive", "false_negative"):
        raise HTTPException(status_code=400, detail="Invalid feedback_type. Must be 'false_positive' or 'false_negative'.")

    feedback = ScanFeedback(
        scan_id=req.scan_id,
        user_id=user.id if user else None,
        feedback_type=req.feedback_type,
        notes=req.notes or ""
    )
    session.add(feedback)
    session.commit()

    return {
        "status": "success",
        "message": "Feedback submitted successfully! Thank you for helping train the PhishGuard AI engine."
    }


# ── Chrome/Firefox Extension Fast Intercept Endpoint ──────────────────────────

class ExtensionCheckRequest(BaseModel):
    url: str

_EXT_CACHE: dict = {}

@router.post("/extension/quick-check")
async def extension_quick_check(req: ExtensionCheckRequest):
    """
    Real-time Multi-Fidelity URL inspection endpoint for the PhishGuard Chrome/Firefox Extension.
    Evaluates links dynamically using live threat intel and returns OriginTrace 4-Pillars forensics.
    """
    from utils.url_scanner import scan_url, extract_domain
    raw_url = req.url.strip()
    if not raw_url:
        return {"status": "UNKNOWN", "verdict_state": "INSUFFICIENT_EVIDENCE",
                "risk_level": "unknown", "risk_score": 0,
                "epistemic_uncertainty": None, "reasons": ["URL input is empty"],
                "origintrace_pillars": {}}

    # Reuse the actual URL pipeline rather than host/DNS-only clean shortcuts.
    # Run blocking model/network work outside the async request loop.
    import asyncio
    scan_res = await asyncio.to_thread(scan_url, raw_url)
    return {
        **scan_res,
        "url": raw_url,
        "domain": scan_res.get("domain") or extract_domain(raw_url),
        "risk_score": int(scan_res.get("final_score", 0)),
        "cascade_tier_executed": "TIER_2_DEEP_FORENSICS",
        "reasons": scan_res.get("signals", []),
        "origintrace_pillars": scan_res.get("origintrace_pillars") or {},
    }


# ── CertStream Autonomous Radar Endpoints ─────────────────────────────────────

@router.get("/radar/live-threats")
async def get_radar_live_threats(session: Session = Depends(get_session)):
    """Return the most recent phishing domains caught by the CertStream daemon."""
    from data.models import LiveThreatAlert
    from sqlmodel import select

    threats = session.exec(
        select(LiveThreatAlert).order_by(LiveThreatAlert.detected_at.desc()).limit(50)
    ).all()

    return {
        "count": len(threats),
        "threats": [
            {
                "id": t.id,
                "domain": t.domain,
                "matched_brand": t.matched_brand,
                "tld": t.tld,
                "risk_score": t.risk_score,
                "risk_level": t.risk_level,
                "status": t.status,
                "reasons": t.reasons,
                "origintrace_pillars": t.origintrace_pillars,
                "cert_issuer": t.cert_issuer or "Let's Encrypt / ZeroSSL",
                "detected_at": t.detected_at.strftime("%Y-%m-%d %H:%M:%S") if t.detected_at else ""
            }
            for t in threats
        ]
    }


@router.get("/radar/stats")
async def get_radar_stats(session: Session = Depends(get_session)):
    """Summary metrics of the live threat hunting radar."""
    from data.models import LiveThreatAlert
    from sqlmodel import select, func
    from services.certstream_daemon import get_active_brands

    total = session.exec(select(func.count(LiveThreatAlert.id))).one() or 0
    brands = get_active_brands()

    return {
        "radar_status": "ACTIVE_STREAMING",
        "total_threats_caught": total,
        "monitored_brands_count": len(brands),
        "monitored_brands": brands[:12]
    }
