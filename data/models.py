from sqlmodel import SQLModel, Field
from typing import Optional, Dict
from datetime import datetime, timezone
import json
from sqlalchemy import Column, String

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    name: str
    email: str = Field(index=True, unique=True)
    hashed_password: str
    role: str = Field(default="user")
    # OAuth linking — a user can connect BOTH at the same time
    auth_provider: Optional[str] = Field(default=None)   # first/primary provider used to create account
    google_linked: bool = Field(default=False)           # True once user has ever authenticated via Google
    github_linked: bool = Field(default=False)           # True once user has ever authenticated via GitHub
    
    # Preferences
    theme: str = Field(default="dark")
    browser_alerts: bool = Field(default=True)
    sync_alerts: bool = Field(default=False)
    weekly_reports: bool = Field(default=False)
    weekly_digest_last_sent_week: Optional[str] = Field(default=None)

    # Subscription / Billing (Lemon Squeezy)
    plan: str = Field(default="free") # "free", "free_plus", "pro", "enterprise"
    ls_customer_id: Optional[str] = Field(default=None)
    ls_subscription_id: Optional[str] = Field(default=None)
    plan_expires_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=utc_now)
    last_login: Optional[datetime] = None

class ScanRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    scan_id: str = Field(index=True, unique=True) # e.g. em-1234abcd
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    scan_type: str = Field(default="email")  # "email", "url", "domain"
    sender: str = Field(index=True)
    subject: str
    risk_score: Optional[int] = Field(default=None)
    status: str # CLEAN, SUSPICIOUS, PHISHING
    risk_level: str # safe, medium, critical
    timestamp: datetime = Field(default_factory=utc_now)
    
    # Store full JSON analysis results from the AI
    full_analysis_json: str = Field(default="{}")

    @property
    def full_analysis(self) -> dict:
        """Helper to parse JSON string back to dict."""
        try:
            return json.loads(self.full_analysis_json or "{}")
        except (ValueError, TypeError):
            return {}

    @full_analysis.setter
    def full_analysis(self, value: dict):
        """Helper to dump dict to JSON string."""
        self.full_analysis_json = json.dumps(value or {})

class KPIRecord(SQLModel, table=True):
    """
    Since KPIs are aggregated, we can store a single record updated daily 
    or dynamically calculate them from ScanRecords. For simplicity and 
    performance, we will store a daily snapshot.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    date_str: str = Field(index=True, unique=True) # YYYY-MM-DD
    emails_scanned: int = Field(default=0)
    phishing_detected: int = Field(default=0)
    high_risk_domains: int = Field(default=0)
    active_campaigns: int = Field(default=0)

class Campaign(SQLModel, table=True):
    """
    Represents a group of related phishing scan attempts,
    automatically derived from scan data.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str  # e.g. 'Microsoft 365 Password Expiry'
    type: str  # e.g. 'Credential Harvesting'
    source: str = Field(default="Unknown")  # Common sender domain
    severity: str = Field(default="medium")  # critical, medium, safe
    targets: int = Field(default=1)  # Number of scan records linked
    first_seen: datetime = Field(default_factory=utc_now)
    last_seen: datetime = Field(default_factory=utc_now)

class Notification(SQLModel, table=True):
    """Represents a system notification/alert for the user."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    title: str
    message: str
    level: str = Field(default="info")  # info, warning, critical
    is_read: bool = Field(default=False)
    scan_id: Optional[str] = Field(default=None)  # Link to the scan that triggered it
    created_at: datetime = Field(default_factory=utc_now)


class InboxConnection(SQLModel, table=True):
    """
    Stores OAuth tokens for connected email inboxes (Gmail, Outlook).
    Tokens are encrypted at rest using Fernet symmetric encryption.
    One record per user per provider (gmail or outlook).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)             # Links to User.id
    provider: str                                  # "gmail" or "outlook"
    email_address: str                             # The connected mailbox address
    access_token_enc: str                          # Fernet-encrypted access token
    refresh_token_enc: str                         # Fernet-encrypted refresh token
    token_expiry: Optional[datetime] = None        # When the access token expires
    last_synced: Optional[datetime] = None         # Last successful sync timestamp
    is_active: bool = Field(default=True)          # Can be deactivated without deleting
    created_at: datetime = Field(default_factory=utc_now)


class ThreatCacheRecord(SQLModel, table=True):
    """
    Persistent SQLite cache for external threat intel API results.
    Stores combined VT + PhishTank + URLhaus results per URL so repeated
    scans of the same URL skip all external API calls — even across restarts.
    TTL: 24 hours (configurable via THREAT_CACHE_TTL_HOURS env var).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    url_hash: str = Field(index=True, unique=True)  # SHA-256 hex of the URL (safe DB key)
    url: str                                          # Original URL (for reference / debugging)
    result_json: str = Field(default="{}")            # JSON-serialised full_threat_check() result
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime                              # Cache expires at this UTC time

    @property
    def result(self) -> dict:
        try:
            return json.loads(self.result_json or "{}")
        except (ValueError, TypeError):
            return {}

    @property
    def is_valid(self) -> bool:
        """Returns True if the cache entry has not expired yet."""
        if self.expires_at.tzinfo is None:
            return datetime.now(timezone.utc).replace(tzinfo=None) < self.expires_at
        return datetime.now(timezone.utc) < self.expires_at

class BlockRule(SQLModel, table=True):
    """
    Stores user-defined block rules (e.g., blocking a specific domain).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    target: str = Field(index=True, unique=True)      # e.g. "evil-domain.com"
    rule_type: str = Field(default="domain")          # e.g. "domain" or "email"
    created_at: datetime = Field(default_factory=utc_now)


class ScanFeedback(SQLModel, table=True):
    """
    Stores user active learning feedback for misclassifications (false positives / negatives).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    scan_id: str = Field(index=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    feedback_type: str = Field(default="false_positive")  # "false_positive" or "false_negative"
    notes: Optional[str] = Field(default="")
    created_at: datetime = Field(default_factory=utc_now)


class BrandMonitorRule(SQLModel, table=True):
    """
    Brands and keywords monitored 24/7 by the CertStream Live Radar Daemon.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    brand_name: str = Field(index=True)              # e.g. "paypal", "microsoft", "chase"
    domain_match: Optional[str] = Field(default=None) # e.g. "paypal.com" (official)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utc_now)


class LiveThreatAlert(SQLModel, table=True):
    """
    Real-time phishing lookalikes caught automatically by CertStream / CT logs.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    domain: str = Field(index=True, unique=True)     # e.g. "chase-verify-login.xyz"
    matched_brand: str = Field(index=True)           # e.g. "chase"
    tld: str                                         # e.g. "xyz"
    risk_score: int                                  # 0 to 100
    risk_level: str = Field(default="critical")      # critical, medium, safe
    status: str = Field(default="PHISHING")          # PHISHING, SUSPICIOUS
    reasons_json: str = Field(default="[]")
    origintrace_pillars_json: str = Field(default="{}")
    cert_issuer: Optional[str] = Field(default=None)
    detected_at: datetime = Field(default_factory=utc_now)

    @property
    def reasons(self) -> list:
        try:
            return json.loads(self.reasons_json or "[]")
        except Exception:
            return []

    @property
    def origintrace_pillars(self) -> dict:
        try:
            return json.loads(self.origintrace_pillars_json or "{}")
        except Exception:
            return {}
