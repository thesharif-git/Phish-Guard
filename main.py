from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from routers import pages, api, auth, billing_routes
from routers.pages import NotAuthenticatedException
from data.store import init_db
from core.logger import setup_logging, logger
from core.config import settings
from fastapi.middleware.cors import CORSMiddleware

# Initialize structured logging
setup_logging()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — runs startup code before yielding,
    then teardown code after the yield (on shutdown).
    """
    logger.info("Initializing database and models...")
    init_db()

    import os
    if os.getenv('PHISHGUARD_PRELOAD_EMAIL', '1') == '1':
        import threading
        from services.email_runtime import preload_email_components
        def _warmup():
            try:
                logger.info('[Email startup] Pre-warming heavy AI models in background...')
                app.state.email_runtime = preload_email_components()
                logger.info('[Email startup] Model pre-warming complete: %s', app.state.email_runtime.get('components_available'))
            except Exception as e:
                logger.warning('[Email startup] Pre-warming deferred: %s', e)
        threading.Thread(target=_warmup, daemon=True).start()
    else:
        app.state.email_runtime = {'preloaded': False, 'components_available': None}
    
    # ── Background Inbox Polling Thread ─────────────────────────────
    import threading
    def _inbox_poll_loop():
        import time
        import asyncio
        from data.database import engine
        from sqlmodel import Session, select
        from data.models import InboxConnection
        from services.inbox_connector import sync_gmail_inbox
        from services.notification_scheduler import send_due_weekly_digests
        
        logger.info("[Background] Started Gmail inbox polling worker...")
        while True:
            try:
                with Session(engine) as session:
                    connections = session.exec(
                        select(InboxConnection).where(InboxConnection.is_active == True)
                    ).all()
                    for conn in connections:
                        asyncio.run(sync_gmail_inbox(conn, session))
            except Exception as e:
                logger.error(f"[Background] Error in inbox polling worker: {e}")
            try:
                asyncio.run(send_due_weekly_digests())
            except Exception as e:
                logger.error(f"[Background] Error sending weekly digests: {e}")
            time.sleep(120)  # poll every 2 minutes

    poll_thread = threading.Thread(target=_inbox_poll_loop, daemon=True)
    poll_thread.start()

    # Autonomous CertStream Brand Radar Daemon
    try:
        from services.certstream_daemon import start_certstream_daemon
        start_certstream_daemon()
    except Exception as e:
        logger.error(f"[Startup] Failed to start CertStream daemon: {e}")
    
    yield
    logger.info("Shutting down PhishGuard backend application...")


app = FastAPI(
    title="PhishGuard Enterprise AI",
    description="Enterprise Phishing & Threat Detection Platform API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)

# ── Scan API Rate Limiting (in-memory) ───────────────────────────
_scan_rate: dict = defaultdict(lambda: {"count": 0, "window_start": datetime.now(timezone.utc).replace(tzinfo=None)})
SCAN_LIMIT = 30        # max 30 scans per window
SCAN_WINDOW = 60       # per 60 seconds

@app.middleware("http")
async def scan_rate_limiter(request: Request, call_next):
    if request.url.path in ("/api/scan-email", "/api/scan-url", "/api/scan-domain") and request.method == "POST":
        ip = request.client.host
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        entry = _scan_rate[ip]
        if (now - entry["window_start"]).total_seconds() > SCAN_WINDOW:
            entry["count"] = 0
            entry["window_start"] = now
        entry["count"] += 1
        if entry["count"] > SCAN_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {SCAN_LIMIT} scans per minute. Please wait before scanning again."}
            )
    return await call_next(request)

@app.exception_handler(NotAuthenticatedException)
async def auth_exception_handler(request: Request, exc: NotAuthenticatedException):
    from fastapi.responses import RedirectResponse
    import urllib.parse
    current_url = str(request.url)
    # Only attach 'next' if it's not already login/register
    if "/login" not in current_url and "/register" not in current_url:
        return RedirectResponse(url=f"/login?next={urllib.parse.quote_plus(current_url)}", status_code=303)
    return RedirectResponse(url="/login", status_code=303)

@app.exception_handler(404)
async def custom_404_handler(request: Request, __):
    from routers.pages import templates
    return templates.TemplateResponse("errors/404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def custom_500_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error on {request.url}: {exc}")
    from routers.pages import templates
    return templates.TemplateResponse("errors/500.html", {"request": request}, status_code=500)

# Initialize Database (Migrations should be handled via Alembic)
try:
    logger.info("Ensuring database tables exist (Alembic managed)...")
    init_db()
    logger.info("Database readiness check complete.")
except Exception as e:
    logger.error(f"Failed to verify database: {e}")


app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(pages.router)
app.include_router(api.router)
app.include_router(auth.router)
app.include_router(billing_routes.router)


# ── Auto-Sync Background Task (runs every 5 minutes) ────────────────────────
import asyncio
import threading

async def _run_inbox_autosync():
    """
    Background coroutine: syncs all active inbox connections every 5 minutes.
    Runs in its own asyncio event loop on a daemon thread.
    """
    from data.models import InboxConnection
    from data.database import engine
    from services.inbox_connector import sync_gmail_inbox
    from sqlmodel import Session, select

    SYNC_INTERVAL = 300  # 5 minutes

    logger.info("[Inbox] Auto-sync daemon started (interval: 5 min)")

    while True:
        await asyncio.sleep(SYNC_INTERVAL)
        try:
            with Session(engine) as session:
                connections = session.exec(
                    select(InboxConnection).where(InboxConnection.is_active == True)
                ).all()

            if connections:
                logger.info(f"[Inbox] Auto-sync: checking {len(connections)} inbox(es)...")
                for conn in connections:
                    with Session(engine) as session:
                        conn_fresh = session.get(InboxConnection, conn.id)
                        if conn_fresh:
                            result = await sync_gmail_inbox(conn_fresh, session)
                            if result.get("new_scans", 0) > 0:
                                logger.info(
                                    f"[Inbox] Auto-sync [{conn_fresh.provider}:{conn_fresh.email_address}]: "
                                    f"{result['new_scans']} new scans, "
                                    f"{result['phishing_found']} phishing classifications, "
                                    f"{result.get('review_required', 0)} require review, "
                                    f"{result.get('results_unavailable', 0)} unavailable, "
                                    f"{result.get('policy_matches', 0)} sender-policy matches"
                                )
        except Exception as e:
            logger.error(f"[Inbox] Auto-sync error: {e}")



def _start_autosync_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run_inbox_autosync())


_sync_thread = threading.Thread(target=_start_autosync_thread, daemon=True)
_sync_thread.start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
