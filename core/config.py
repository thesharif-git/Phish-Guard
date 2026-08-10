import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Config:
    # App
    SECRET_KEY = os.getenv("SECRET_KEY", "phishguard_fallback_secret_key_change_me")
    ALGORITHM = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 10080))
    ENV = os.getenv("ENV", "development")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./phishguard.db")

    # Google OAuth 2.0
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

    # GitHub OAuth
    GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
    GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback")

    # Gmail Inbox Connection (separate from login OAuth — needs gmail.readonly scope)
    GOOGLE_INBOX_REDIRECT_URI = os.getenv("GOOGLE_INBOX_REDIRECT_URI", "http://localhost:8000/auth/gmail/inbox-callback")

    # Microsoft / Outlook Inbox (requires Azure App Registration)
    MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
    MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
    MICROSOFT_REDIRECT_URI = os.getenv("MICROSOFT_REDIRECT_URI", "http://localhost:8000/auth/outlook/inbox-callback")
    MICROSOFT_TENANT = os.getenv("MICROSOFT_TENANT", "common")

    # Inbox token encryption (auto-generated Fernet key stored here)
    INBOX_ENCRYPTION_KEY = os.getenv("INBOX_ENCRYPTION_KEY", "")

    # Brevo (Sendinblue) Email Alerts — 300 emails/day free
    BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
    BREVO_FROM_EMAIL = os.getenv("BREVO_FROM_EMAIL", "")
    BREVO_FROM_NAME = os.getenv("BREVO_FROM_NAME", "PhishGuard Alerts")

    # Stripe Billing (Test Mode)
    STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRICE_FREE_PLUS = os.getenv("STRIPE_PRICE_FREE_PLUS", "")
    STRIPE_PRICE_PRO = os.getenv("STRIPE_PRICE_PRO", "")

settings = Config()
