from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Response, BackgroundTasks
from sqlmodel import Session
from data.database import get_session
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

from services.scan_engine import scan_email as run_scan
from services.security import get_current_user, verify_password, get_password_hash
from data.models import User
from data import store
from utils.email_parser import parse_eml_content
from utils.url_scanner import scan_url as run_url_scan

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

    if file:
        if not file.filename.endswith('.eml'):
            raise HTTPException(status_code=400, detail="Only .eml files are supported for upload.")
        
        file_bytes = await file.read()
        parsed_email = parse_eml_content(file_bytes)
        
        content = parsed_email["body"]
        sender = parsed_email["sender"]
        subject = parsed_email["subject"]
        headers = parsed_email["headers"]
        
    if not content and not file:
        raise HTTPException(status_code=400, detail="Must provide either text content or an .eml file.")

    # Run the synchronous scan in a thread pool — keeps the event loop free
    result = await asyncio.to_thread(
        run_scan,
        content=content,
        sender=sender,
        subject=subject,
        headers=headers,
        session=session
    )

    # Save the scan to the database
    try:
        scan_result_for_db = {
            "sender": sender,
            "subject": subject,
            "final_score": result.get("final_score", 0),
            "status": result.get("status", "CLEAN").upper(),
            "risk_level": result.get("risk_level", "safe"),
        }
        store.add_scan(scan_result_for_db, session, scan_type="email")
    except Exception:
        pass

    if result.get("status") == "PHISHING" and user.browser_alerts:
        from utils.email_alerts import send_phishing_alert
        background_tasks.add_task(send_phishing_alert, user.email, user.name, result)

    return JSONResponse(content=result)


@router.post("/scan-url")
async def scan_url_endpoint(
    req: ScanUrlRequest,
    session: Session = Depends(get_session)
):
    """Dedicated URL threat intelligence scanner."""
    import asyncio

    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

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
        }
        store.add_scan(scan_result_for_db, session, scan_type="url")
    except Exception:
        pass
    
    return JSONResponse(content=result)


class ScanDomainRequest(BaseModel):
    domain: str

@router.post("/scan-domain")
async def scan_domain_endpoint(
    req: ScanDomainRequest,
    session: Session = Depends(get_session)
):
    """Dedicated Domain reputation scanner."""
    import asyncio

    domain = req.domain.strip()
    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required.")
    url = "https://" + domain if not domain.startswith(("http://", "https://")) else domain

    # Run in thread pool — makes blocking WHOIS + API calls
    result = await asyncio.to_thread(run_url_scan, url)
    
    # Save domain scan to unified history with scan_type='domain'
    try:
        scan_result_for_db = {
            "sender": domain,
            "subject": domain,
            "final_score": result.get("final_score", 0),
            "status": result.get("status", "CLEAN").upper(),
            "risk_level": result.get("risk_level", "safe"),
        }
        store.add_scan(scan_result_for_db, session, scan_type="domain")
    except Exception:
        pass
    
    return JSONResponse(content=result)


@router.get("/dashboard")
async def dashboard_data(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    kpi = store.get_kpi(session)
    return {
        "kpi": kpi,
        "risk_trend": store.get_risk_trend(session),
        "phishing_types": store.get_phishing_types(session),
        "top_targets": store.get_top_targets(session),
        "recent_scans": store.get_recent_scans(session, limit=10),
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
    return {"emails": store.get_recent_scans(session, limit=50)}


@router.get("/scan-history")
async def scan_history_data(session: Session = Depends(get_session)):
    """Returns unified scan history (all scan types) for the slide-over panel."""
    rows, counts = store.get_all_scans(session)
    return {"scans": rows[:50], "counts": counts}


# ── Profile ───────────────────────────────────────────────────────────────────
from pydantic import BaseModel as PydanticBase, Field, EmailStr

class ProfileUpdate(PydanticBase):
    name: str | None = Field(None, min_length=2, max_length=50)
    current_password: str | None = None
    new_password: str | None = Field(None, min_length=8, max_length=128)
    theme: str | None = None
    browser_alerts: bool | None = None
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
    if store.add_block_rule(session, req.target, req.rule_type):
        return {"success": True, "message": f"Rule created for {req.target}"}
    raise HTTPException(status_code=400, detail="Rule already exists")

@router.get("/rules/block")
async def list_block_rules(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    return {"rules": store.get_block_rules(session)}

@router.delete("/rules/block/{rule_id}")
async def delete_block_rule(rule_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    if store.remove_block_rule(session, rule_id):
        return {"success": True, "message": "Rule removed"}
    raise HTTPException(status_code=404, detail="Rule not found")
