"""
PhishGuard Inbox Connector Service
Handles Gmail (and future Outlook) OAuth token management + email fetching.
Tokens are encrypted at rest using Fernet symmetric encryption.
"""
import base64
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx
from cryptography.fernet import Fernet, InvalidToken

from core.config import settings
from core.logger import logger


# ── Token Encryption ──────────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    """Return Fernet instance from config key. Raises if key not set."""
    key = settings.INBOX_ENCRYPTION_KEY
    if not key:
        raise ValueError("INBOX_ENCRYPTION_KEY not configured in .env")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_token(token: str) -> str:
    """Encrypt a token string for database storage."""
    f = _get_fernet()
    return f.encrypt(token.encode()).decode()


def decrypt_token(encrypted: str) -> str:
    """Decrypt a stored token. Returns empty string on failure."""
    try:
        f = _get_fernet()
        return f.decrypt(encrypted.encode()).decode()
    except (InvalidToken, Exception) as e:
        logger.error(f"Token decryption failed: {e}")
        return ""


# ── Gmail API Constants ───────────────────────────────────────────────────────

GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_MESSAGES_URL = "https://www.googleapis.com/gmail/v1/users/me/messages"
GMAIL_MESSAGE_URL = "https://www.googleapis.com/gmail/v1/users/me/messages/{msg_id}"
GMAIL_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Scopes needed for inbox reading (separate from login)
GMAIL_INBOX_SCOPES = " ".join([
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
])


# ── Gmail OAuth URL Builder ───────────────────────────────────────────────────

def build_gmail_connect_url(state: str = "") -> str:
    """Build the Google OAuth authorization URL for inbox access."""
    params = (
        f"?client_id={settings.GOOGLE_CLIENT_ID}"
        f"&redirect_uri={settings.GOOGLE_INBOX_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={GMAIL_INBOX_SCOPES.replace(' ', '%20')}"
        f"&access_type=offline"           # needed to get refresh_token
        f"&prompt=consent"                # force consent screen to always get refresh_token
        f"&include_granted_scopes=false"
    )
    if state:
        params += f"&state={state}"
    return "https://accounts.google.com/o/oauth2/v2/auth" + params


# ── Gmail Token Exchange ──────────────────────────────────────────────────────

async def exchange_gmail_code(code: str) -> dict:
    """
    Exchange an authorization code for access + refresh tokens.
    Returns dict with: access_token, refresh_token, expires_in, email
    """
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GMAIL_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_INBOX_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )

    if token_resp.status_code != 200:
        raise ValueError(f"Token exchange failed: {token_resp.text}")

    token_data = token_resp.json()

    # Fetch user email so we know which mailbox this is
    async with httpx.AsyncClient() as client:
        userinfo_resp = await client.get(
            GMAIL_USERINFO_URL,
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )

    email = ""
    if userinfo_resp.status_code == 200:
        email = userinfo_resp.json().get("email", "")

    return {
        "access_token": token_data.get("access_token", ""),
        "refresh_token": token_data.get("refresh_token", ""),
        "expires_in": token_data.get("expires_in", 3600),
        "email": email,
    }


# ── Gmail Token Refresh ───────────────────────────────────────────────────────

async def refresh_gmail_access_token(refresh_token: str) -> dict:
    """
    Use a refresh token to get a new short-lived access token.
    Returns dict with: access_token, expires_in
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GMAIL_TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "grant_type": "refresh_token",
            },
        )

    if resp.status_code != 200:
        raise ValueError(f"Token refresh failed: {resp.text}")

    data = resp.json()
    return {
        "access_token": data.get("access_token", ""),
        "expires_in": data.get("expires_in", 3600),
    }


# ── Gmail Email Fetcher ───────────────────────────────────────────────────────

def _decode_base64_url(data: str) -> str:
    """Decode base64url-encoded string (Gmail uses this for message bodies)."""
    data = data.replace("-", "+").replace("_", "/")
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    try:
        return base64.b64decode(data).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body_from_payload(payload: dict) -> str:
    """Recursively extract text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime_type == "text/plain" and body_data:
        return _decode_base64_url(body_data)
    if mime_type == "text/html" and body_data:
        # Strip HTML tags for analysis
        text = _decode_base64_url(body_data)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # Recurse into multipart
    for part in payload.get("parts", []):
        result = _extract_body_from_payload(part)
        if result:
            return result

    return ""


def _extract_header(headers: list, name: str) -> str:
    """Extract a specific header value from the Gmail headers list."""
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


async def fetch_gmail_emails(access_token: str, max_results: int = 20) -> List[dict]:
    """
    Fetch the latest emails from Gmail inbox.
    Returns a list of dicts with: subject, sender, body, headers_raw, date
    """
    emails = []
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=20) as client:
        # Step 1: List message IDs from INBOX
        list_resp = await client.get(
            GMAIL_MESSAGES_URL,
            headers=headers,
            params={"maxResults": max_results, "labelIds": "INBOX", "q": "is:unread OR newer_than:3d"},
        )

        if list_resp.status_code != 200:
            logger.error(f"Gmail list failed: {list_resp.status_code} {list_resp.text[:200]}")
            return []

        messages = list_resp.json().get("messages", [])
        logger.info(f"Gmail: found {len(messages)} messages to fetch")

        # Step 2: Fetch each message in full
        for msg in messages[:max_results]:
            msg_id = msg.get("id")
            try:
                msg_resp = await client.get(
                    GMAIL_MESSAGE_URL.format(msg_id=msg_id),
                    headers=headers,
                    params={"format": "full"},
                )
                if msg_resp.status_code != 200:
                    continue

                data = msg_resp.json()
                payload = data.get("payload", {})
                msg_headers = payload.get("headers", [])

                subject = _extract_header(msg_headers, "Subject") or "(No Subject)"
                sender = _extract_header(msg_headers, "From") or "unknown@unknown.com"
                date_str = _extract_header(msg_headers, "Date") or ""
                body = _extract_body_from_payload(payload)

                # Build raw headers text for SPF/DKIM analysis
                headers_raw = "\n".join(f"{h['name']}: {h['value']}" for h in msg_headers)

                emails.append({
                    "gmail_id": msg_id,
                    "subject": subject,
                    "sender": sender,
                    "body": body or "(Empty body)",
                    "headers_raw": headers_raw,
                    "date": date_str,
                })
            except Exception as e:
                logger.warning(f"Failed to fetch Gmail message {msg_id}: {e}")
                continue

    return emails


# ── Sync Inbox (fetch + scan + store) ────────────────────────────────────────

async def sync_gmail_inbox(connection, session) -> dict:
    """
    Full sync cycle for a Gmail inbox connection:
    1. Refresh access token if needed
    2. Fetch latest emails
    3. Run PhishGuard scan on each
    4. Store results in ScanRecord
    Returns summary dict with counts.
    """
    from data.models import InboxConnection, User
    from services.scan_engine import scan_email as run_scan
    from sqlmodel import select

    user = session.get(User, connection.user_id)

    # 1. Refresh token if expired or expiring soon
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    needs_refresh = (
        not connection.token_expiry or
        connection.token_expiry <= now + timedelta(minutes=5)
    )

    access_token = decrypt_token(connection.access_token_enc)
    refresh_token = decrypt_token(connection.refresh_token_enc)

    if needs_refresh and refresh_token:
        try:
            new_tokens = await refresh_gmail_access_token(refresh_token)
            access_token = new_tokens["access_token"]
            connection.access_token_enc = encrypt_token(access_token)
            connection.token_expiry = now + timedelta(seconds=new_tokens["expires_in"])
            session.add(connection)
            session.commit()
            logger.info(f"Gmail token refreshed for user_id={connection.user_id}")
        except Exception as e:
            logger.error(f"Token refresh failed for user_id={connection.user_id}: {e}")
            return {"success": False, "error": "Token refresh failed. Please reconnect your inbox."}

    # 2. Fetch emails
    try:
        emails = await fetch_gmail_emails(access_token, max_results=15)
    except Exception as e:
        logger.error(f"Gmail fetch failed: {e}")
        return {"success": False, "error": "Failed to fetch emails from Gmail."}

    if not emails:
        connection.last_synced = now
        session.add(connection)
        session.commit()
        return {"success": True, "fetched": 0, "new_scans": 0, "phishing_found": 0}

    # 3. Scan each email (skip duplicates by checking gmail_id in existing scan notes)
    new_scans = 0
    phishing_found = 0

    for email in emails:
        # Build a unique scan_id using the Gmail message ID
        scan_id_check = f"gm-{email['gmail_id']}"
        from data.models import ScanRecord
        existing = session.exec(
            select(ScanRecord).where(ScanRecord.scan_id == scan_id_check)
        ).first()
        if existing:
            continue  # Already scanned

        try:
            result = run_scan(
                content=email["body"],
                sender=email["sender"],
                subject=email["subject"],
                headers=email["headers_raw"],
                session=session,
                gmail_id=email["gmail_id"],  # passed to store for custom scan_id prefix
            )
            session.commit()  # Commit each email to avoid bulk rollback on IntegrityError
            
            new_scans += 1
            if result.get("status") == "PHISHING":
                phishing_found += 1
                if user and user.browser_alerts:
                    from utils.email_alerts import send_phishing_alert
                    import asyncio
                    asyncio.create_task(send_phishing_alert(user.email, user.name, result))
            elif result.get("status") == "SUSPICIOUS":
                phishing_found += 1
        except Exception as e:
            session.rollback() # Clear the failed transaction so the loop can continue
            logger.warning(f"Scan failed for Gmail msg {email['gmail_id']}: {e}")
            continue

    # 4. Update last_synced
    connection.last_synced = now
    session.add(connection)
    session.commit()

    logger.info(f"Gmail sync complete: {new_scans} new scans, {phishing_found} threats found")
    return {
        "success": True,
        "fetched": len(emails),
        "new_scans": new_scans,
        "phishing_found": phishing_found,
    }
