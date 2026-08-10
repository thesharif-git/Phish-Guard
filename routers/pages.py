from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from data.database import get_session
from data.models import User
from services.security import get_current_user
from data import store

templates = Jinja2Templates(directory="templates")
router = APIRouter()

class NotAuthenticatedException(Exception):
    pass

def require_auth(request: Request, session: Session = Depends(get_session)):
    try:
        user = get_current_user(request, session)
        return user
    except HTTPException:
        raise NotAuthenticatedException()


@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request, session: Session = Depends(get_session)):
    from data.models import ScanRecord
    from sqlmodel import select
    scans = store.get_recent_scans(session, limit=5)
    latest_scan = None
    if scans:
        latest_scan_record = session.exec(select(ScanRecord).where(ScanRecord.scan_id == scans[0]["id"])).first()
        if latest_scan_record:
            latest_scan = {
                "risk_score": latest_scan_record.risk_score,
                "status": latest_scan_record.status,
                "risk_level": latest_scan_record.risk_level,
                "subject": latest_scan_record.subject,
                "full_analysis": latest_scan_record.full_analysis
            }

    return templates.TemplateResponse("landing.html", {
        "request": request,
        "kpi": store.get_kpi(session),
        "active_users": len(store.get_all_users(session)),
        "recent_scans": scans,
        "latest_scan": latest_scan
    })


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, session: Session = Depends(get_session), user: User = Depends(require_auth)):
    recent_scans = store.get_recent_scans(session, limit=5)
    kpi = store.get_kpi(session)
    trend = store.get_risk_trend(session)
    threat_sources = store.get_top_threat_sources(session)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "active_page": "dashboard", 
        "user": user,
        "recent_scans": recent_scans,
        "kpi": kpi,
        "trend": trend,
        "threat_map_data": threat_sources["map_data"],
        "top_threats_list": threat_sources["top_list"]
    })

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("auth/login.html", {"request": request})

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("auth/register.html", {"request": request})



@router.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request, session: Session = Depends(get_session)):
    """Public scan page."""
    user = None
    try:
        user = get_current_user(request, session)
    except Exception:
        pass
    kpi = store.get_kpi(session) if user else None
    return templates.TemplateResponse("scan.html", {"request": request, "user": user, "kpi": kpi, "active_page": "scan"})

@router.get("/url-scan", response_class=HTMLResponse)
async def url_scan_page(request: Request, session: Session = Depends(get_session)):
    """Public URL Link Detector page."""
    user = None
    try:
        user = get_current_user(request, session)
    except Exception:
        pass
    kpi = store.get_kpi(session) if user else None
    return templates.TemplateResponse("url_scan.html", {"request": request, "user": user, "kpi": kpi})

@router.get("/domain-reputation", response_class=HTMLResponse)
async def domain_reputation_page(request: Request, session: Session = Depends(get_session)):
    """Public Domain Reputation Checker page."""
    user = None
    try:
        user = get_current_user(request, session)
    except Exception:
        pass
    kpi = store.get_kpi(session) if user else None
    return templates.TemplateResponse("domain_reputation.html", {
        "request": request, 
        "user": user, 
        "kpi": kpi,
        "active_page": "domain_reputation"
    })


@router.get("/inbox", response_class=HTMLResponse)
async def inbox(request: Request, session: Session = Depends(get_session), user: User = Depends(require_auth)):
    scans, counts = store.get_all_scans(session)
    return templates.TemplateResponse("inbox.html", {
        "request": request,
        "scans": scans,
        "total": counts["total"],
        "clean": counts["clean"],
        "suspicious": counts["suspicious"],
        "phishing": counts["phishing"],
        "active_page": "inbox",
        "user": user,
        "kpi": store.get_kpi(session),
    })

@router.get("/email/{email_id}", response_class=HTMLResponse)
async def email_detail(request: Request, email_id: str, session: Session = Depends(get_session), user: User = Depends(require_auth)):
    email = store.get_email_detail(email_id, session)
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    return templates.TemplateResponse("email_detail.html", {
        "request": request,
        "email": email,
        "active_page": "analyze",
        "user": user,
    })

@router.get("/campaigns", response_class=HTMLResponse)
async def campaigns(request: Request, session: Session = Depends(get_session), user: User = Depends(require_auth)):
    # Auto-group any new phishing scans into campaigns first
    store.auto_build_campaigns(session)
    real_campaigns = store.get_campaigns(session)
    return templates.TemplateResponse("campaigns.html", {
        "request": request,
        "campaigns": real_campaigns,
        "active_page": "campaigns",
        "user": user,
        "kpi": store.get_kpi(session),
    })

@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request, session: Session = Depends(get_session), user: User = Depends(require_auth)):
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "kpi": store.get_kpi(session),
        "active_page": "settings",
        "user": user,
        "block_rules": store.get_block_rules(session)
    })

@router.get("/pricing", response_class=HTMLResponse)
async def pricing_page(request: Request, session: Session = Depends(get_session)):
    """Pricing page accessible to both logged-in and public users."""
    user = None
    try:
        user = get_current_user(request, session)
    except Exception:
        pass
    return templates.TemplateResponse("pricing.html", {
        "request": request,
        "user": user,
        "kpi": store.get_kpi(session) if user else None,
        "active_page": "pricing"
    })

@router.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request, session: Session = Depends(get_session), user: User = Depends(require_auth)):
    scans, counts = store.get_all_scans(session)
    trend = store.get_risk_trend(session)
    return templates.TemplateResponse("reports.html", {
        "request": request,
        "scans": scans,
        "total": counts["total"],
        "clean": counts["clean"],
        "suspicious": counts["suspicious"],
        "phishing": counts["phishing"],
        "trend": trend,
        "active_page": "reports",
        "user": user,
        "kpi": store.get_kpi(session),
    })


# ── Info / Static pages ─────────────────────────────────────────────────────

@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("info/about.html", {"request": request})


@router.get("/blog", response_class=HTMLResponse)
async def blog_page(request: Request):
    return templates.TemplateResponse("info/blog.html", {"request": request})


@router.get("/careers", response_class=HTMLResponse)
async def careers_page(request: Request):
    return templates.TemplateResponse("info/careers.html", {"request": request})


@router.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("info/contact.html", {"request": request})


@router.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    return templates.TemplateResponse("info/privacy.html", {"request": request})


@router.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    return templates.TemplateResponse("info/terms.html", {"request": request})


@router.get("/cookies", response_class=HTMLResponse)
async def cookies_page(request: Request):
    return templates.TemplateResponse("info/cookies.html", {"request": request})


@router.get("/security", response_class=HTMLResponse)
async def security_page(request: Request):
    return templates.TemplateResponse("info/security.html", {"request": request})


@router.get("/pricing", response_class=HTMLResponse)
async def pricing_page(request: Request, session: Session = Depends(get_session)):
    from core.config import settings
    user = None
    try:
        user = get_current_user(request, session)
    except HTTPException:
        pass
    return templates.TemplateResponse("pricing.html", {
        "request": request,
        "user": user,
        "stripe_publishable_key": settings.STRIPE_PUBLISHABLE_KEY,
    })

# Trigger reload

# Trigger reload 2

# Trigger reload 3

# Trigger reload 4

# Trigger reload 5

# Trigger reload 6

# Trigger reload 7

# Trigger reload 6

# Trigger reload 7

# Trigger reload 8

# Trigger reload 9

# Trigger reload 10

# Trigger reload 11

# Trigger reload 12

# Trigger reload 13

# Trigger reload 14

# Trigger reload 15

# Trigger reload 16

# Trigger reload 17

# Trigger reload 18

# Trigger reload 19

# Trigger reload 20

# Trigger reload 21

# Trigger reload 22

# Trigger reload 23

# Trigger reload 24

# Trigger reload 25

# Trigger reload 26

# Trigger reload 27

# Trigger reload 28

# Trigger reload 29

# Trigger reload 30

# Trigger reload 31

# Trigger reload 32

# Trigger reload 33

# Trigger reload 34

# Trigger reload 35
