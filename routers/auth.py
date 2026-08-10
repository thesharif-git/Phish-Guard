from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from datetime import timedelta, datetime, timezone
import uuid
import re
import httpx
from collections import defaultdict

from data.database import get_session
from data.models import User
from services.security import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from core.config import settings
from pydantic import BaseModel, validator

router = APIRouter(prefix="/auth", tags=["auth"])

# ── In-memory rate limiting ──────────────────────────────────────
_login_attempts: dict = defaultdict(lambda: {"attempts": 0, "window_start": datetime.now(timezone.utc).replace(tzinfo=None), "locked_until": None})
MAX_ATTEMPTS = 5
WINDOW_SECONDS = 300   # 5 minutes
LOCKOUT_SECONDS = 900  # 15 minutes


def check_rate_limit(ip: str):
    entry = _login_attempts[ip]
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if entry["locked_until"] and now < entry["locked_until"]:
        wait_seconds = int((entry["locked_until"] - now).total_seconds())
        wait_minutes = (wait_seconds // 60) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Security lockout active. Please try again in {wait_minutes} minute(s)."
        )
    if (now - entry["window_start"]).total_seconds() > WINDOW_SECONDS:
        if not entry["locked_until"] or now > entry["locked_until"]:
            entry["attempts"] = 0
            entry["window_start"] = now
            entry["locked_until"] = None


def record_failed_attempt(ip: str):
    entry = _login_attempts[ip]
    entry["attempts"] += 1
    if entry["attempts"] >= MAX_ATTEMPTS:
        entry["locked_until"] = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=LOCKOUT_SECONDS)


def reset_attempts(ip: str):
    _login_attempts[ip] = {"attempts": 0, "window_start": datetime.now(timezone.utc).replace(tzinfo=None), "locked_until": None}


# ── Helper: Create JWT cookie and redirect ───────────────────────
def _login_user_and_redirect(user: User, response_target: str = "/dashboard") -> RedirectResponse:
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role},
        expires_delta=access_token_expires
    )
    is_prod = settings.ENV == "production"
    redirect = RedirectResponse(url=response_target, status_code=302)
    redirect.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        samesite="lax",
        secure=is_prod,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return redirect


# ── Helper: Find or create user from OAuth ───────────────────────
def _get_or_create_oauth_user(email: str, name: str, provider: str, session: Session) -> User:
    """
    Find or create a user by email from an OAuth callback.
    Supports BOTH login and registration — if the email already exists
    (whether from email/password signup, Google, or GitHub), the account is
    linked and the user is logged in seamlessly. Both providers can be
    connected to the same account at the same time.
    """
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        # Brand new user — create account via OAuth
        user = User(
            uuid=str(uuid.uuid4()),
            name=name or email.split("@")[0],
            email=email,
            hashed_password=get_password_hash(str(uuid.uuid4())),  # random unusable password
            role="user",
            auth_provider=provider,
            google_linked=(provider == "google"),
            github_linked=(provider == "github"),
        )
        session.add(user)
    else:
        # Existing user — link/update the OAuth provider
        if not user.auth_provider:
            user.auth_provider = provider  # set primary provider if not already set
        if provider == "google":
            user.google_linked = True
        elif provider == "github":
            user.github_linked = True

    user.last_login = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


# ── Register ─────────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

    @validator("name")
    def name_must_be_valid(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters.")
        if len(v) > 60:
            raise ValueError("Name is too long.")
        return v

    @validator("email")
    def email_must_be_valid(cls, v):
        v = v.strip().lower()
        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(pattern, v):
            raise ValueError("Please enter a valid email address.")
        return v

    @validator("password")
    def password_must_be_strong(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r'[0-9]', v):
            raise ValueError("Password must contain at least one number.")
        return v


@router.post("/register")
async def register(user: UserCreate, session: Session = Depends(get_session)):
    existing_user = session.exec(select(User).where(User.email == user.email)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="This email is already registered. Please log in instead.")
    hashed_pwd = get_password_hash(user.password)
    new_user = User(
        uuid=str(uuid.uuid4()),
        name=user.name,
        email=user.email,
        hashed_password=hashed_pwd
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return {"msg": "Account created successfully. You can now log in."}


# ── Login ─────────────────────────────────────────────────────────
@router.post("/login")
async def login(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
):
    client_ip = request.client.host
    check_rate_limit(client_ip)

    user = session.exec(select(User).where(User.email == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        record_failed_attempt(client_ip)
        remaining = max(0, MAX_ATTEMPTS - _login_attempts[client_ip]["attempts"])
        detail = "Incorrect email or password."
        if 0 < remaining <= 2:
            detail += f" {remaining} attempt(s) remaining before lockout."
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

    reset_attempts(client_ip)
    user.last_login = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add(user)
    session.commit()

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role},
        expires_delta=access_token_expires
    )
    is_prod = settings.ENV == "production"
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        samesite="lax",
        secure=is_prod,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ── Logout ────────────────────────────────────────────────────────
@router.post("/logout")
async def logout(response: Response):
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(key="access_token")
    return resp

@router.get("/logout")
async def logout_get(response: Response):
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(key="access_token")
    return resp


# ══════════════════════════════════════════════════════════════════
#  GOOGLE OAUTH 2.0
# ══════════════════════════════════════════════════════════════════

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GOOGLE_SCOPES = "openid email profile"


@router.get("/google/login")
async def google_login():
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured.")
    params = (
        f"?client_id={settings.GOOGLE_CLIENT_ID}"
        f"&redirect_uri={settings.GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={GOOGLE_SCOPES.replace(' ', '%20')}"
        f"&access_type=offline"
        f"&prompt=select_account"
    )
    return RedirectResponse(url=GOOGLE_AUTH_URL + params)


@router.get("/google/callback")
async def google_callback(code: str, session: Session = Depends(get_session)):
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code missing.")

    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            }
        )

    if token_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to exchange Google authorization code.")

    token_data = token_response.json()
    access_token = token_data.get("access_token")

    async with httpx.AsyncClient() as client:
        userinfo_response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"}
        )

    if userinfo_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch user info from Google.")

    userinfo = userinfo_response.json()
    email = userinfo.get("email")
    name = userinfo.get("name", "Google User")

    if not email:
        raise HTTPException(status_code=400, detail="Google did not return an email address.")

    user = _get_or_create_oauth_user(email=email, name=name, provider="google", session=session)
    return _login_user_and_redirect(user)


# ══════════════════════════════════════════════════════════════════
#  GITHUB OAUTH 2.0
# ══════════════════════════════════════════════════════════════════

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USERINFO_URL = "https://api.github.com/user"
GITHUB_EMAIL_URL = "https://api.github.com/user/emails"


@router.get("/github/login")
async def github_login():
    if not settings.GITHUB_CLIENT_ID:
        raise HTTPException(status_code=503, detail="GitHub OAuth is not configured yet.")
    params = (
        f"?client_id={settings.GITHUB_CLIENT_ID}"
        f"&redirect_uri={settings.GITHUB_REDIRECT_URI}"
        f"&scope=user:email"
    )
    return RedirectResponse(url=GITHUB_AUTH_URL + params)


@router.get("/github/callback")
async def github_callback(code: str, session: Session = Depends(get_session)):
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code missing.")

    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": settings.GITHUB_REDIRECT_URI,
            },
            headers={"Accept": "application/json"}
        )

    if token_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to exchange GitHub authorization code.")

    access_token = token_response.json().get("access_token")

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        userinfo_response = await client.get(GITHUB_USERINFO_URL, headers=headers)
        email_response = await client.get(GITHUB_EMAIL_URL, headers=headers)

    userinfo = userinfo_response.json()
    name = userinfo.get("name") or userinfo.get("login", "GitHub User")

    # Handle private GitHub emails — always fetch from emails endpoint
    email = None
    if email_response.status_code == 200:
        emails = email_response.json()
        # Prefer primary + verified email
        primary = next((e["email"] for e in emails if e.get("primary") and e.get("verified")), None)
        # Fall back to any verified email, then any email
        verified = next((e["email"] for e in emails if e.get("verified")), None)
        email = primary or verified or (emails[0]["email"] if emails else None)

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Could not retrieve your email from GitHub. Please make at least one email verified in your GitHub account settings."
        )


# ══════════════════════════════════════════════════════════════════
#  GMAIL INBOX CONNECTION (separate from login — needs gmail.readonly)
# ══════════════════════════════════════════════════════════════════

@router.get("/gmail/connect")
async def gmail_connect(request: Request, session: Session = Depends(get_session)):
    """Start the Gmail inbox OAuth flow. Requires the user to be logged in."""
    from services.security import get_current_user
    try:
        get_current_user(request, session)  # verify logged in
    except HTTPException:
        return RedirectResponse(url="/login?next=/settings", status_code=302)

    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured.")

    from services.inbox_connector import build_gmail_connect_url
    url = build_gmail_connect_url()
    return RedirectResponse(url=url)


@router.get("/gmail/inbox-callback")
async def gmail_inbox_callback(
    code: str,
    request: Request,
    session: Session = Depends(get_session),
):
    """Handle the Gmail OAuth callback — exchange code for tokens and store."""
    from services.security import get_current_user
    from services.inbox_connector import exchange_gmail_code, encrypt_token
    from data.models import InboxConnection

    # Must be logged in
    try:
        user = get_current_user(request, session)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=302)

    if not code:
        raise HTTPException(status_code=400, detail="Authorization code missing.")

    try:
        token_data = await exchange_gmail_code(code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gmail authorization failed: {e}")

    if not token_data.get("refresh_token"):
        raise HTTPException(
            status_code=400,
            detail="Google did not return a refresh token. Please try connecting again."
        )

    from datetime import timedelta
    expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=token_data.get("expires_in", 3600))

    # Check if connection already exists — update it
    existing = session.exec(
        select(InboxConnection).where(
            InboxConnection.user_id == user.id,
            InboxConnection.provider == "gmail"
        )
    ).first()

    if existing:
        existing.access_token_enc = encrypt_token(token_data["access_token"])
        existing.refresh_token_enc = encrypt_token(token_data["refresh_token"])
        existing.token_expiry = expiry
        existing.email_address = token_data.get("email", existing.email_address)
        existing.is_active = True
        session.add(existing)
    else:
        conn = InboxConnection(
            user_id=user.id,
            provider="gmail",
            email_address=token_data.get("email", ""),
            access_token_enc=encrypt_token(token_data["access_token"]),
            refresh_token_enc=encrypt_token(token_data["refresh_token"]),
            token_expiry=expiry,
        )
        session.add(conn)

    session.commit()
    return RedirectResponse(url="/settings?inbox=connected", status_code=302)


@router.post("/gmail/disconnect")
async def gmail_disconnect(request: Request, session: Session = Depends(get_session)):
    """Disconnect and remove the stored Gmail inbox connection."""
    from services.security import get_current_user
    from data.models import InboxConnection

    user = get_current_user(request, session)
    conn = session.exec(
        select(InboxConnection).where(
            InboxConnection.user_id == user.id,
            InboxConnection.provider == "gmail"
        )
    ).first()
    if conn:
        session.delete(conn)
        session.commit()
    return RedirectResponse(url="/settings?inbox=disconnected", status_code=302)

