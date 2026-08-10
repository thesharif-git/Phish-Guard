import uuid
from datetime import datetime, timedelta, timezone
import random
from sqlmodel import Session
from data.database import engine
from data.models import ScanRecord, KPIRecord, Campaign

def seed_scans():
    subjects = [
        "Urgent: Account Verification Required",
        "Your PayPal Invoice is Ready",
        "Meeting Agenda for Next Monday",
        "Suspicious Login Detected from New Device",
        "Microsoft 365: Action Required for Your Account",
        "Weekly Newsletter - Security Best Practices",
        "Congratulations! You've Won a Gift Card",
        "IT Support: Password Reset Confirmation",
        "Delivery Status Update: Package #98213",
        "Internal Memo: New Remote Work Policy",
        "Netflix: Your Membership is on Hold",
        "Google Security Alert: Access Granted to Third-party App",
        "Quarterly Financial Report Q1",
        "RE: Regarding your recent inquiry",
        "HR: Benefits Enrollment Period Starting Soon"
    ]
    
    senders = [
        "support@paypal-security.net",
        "no-reply@microsoft.com",
        "hr@phishguard.io",
        "billing@netflix.com",
        "admin@bank-of-america-verify.org",
        "marketing@newsletter.com",
        "security-noreply@google.com",
        "shipping@fedex-express.biz",
        "finance@phishguard.io",
        "giftcards@amazon-rewards.top",
        "https://secure-login-portal.info/auth",
        "http://verify-identity.xyz/account",
        "https://google.com",
        "https://github.com/login"
    ]
    
    with Session(engine) as session:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        for i in range(100):
            subject = random.choice(subjects)
            sender = random.choice(senders)
            
            # Heuristic for status based on sender domain
            if ".net" in sender or ".org" in sender or ".biz" in sender or ".top" in sender or "info" in sender or "xyz" in sender:
                status = random.choice(["PHISHING", "SUSPICIOUS"])
                risk_score = random.randint(45, 98)
                risk_level = "critical" if risk_score > 75 else "medium"
            else:
                status = "CLEAN"
                risk_score = random.randint(2, 15)
                risk_level = "safe"
            
            # Distribute over the last 15 days
            ts = now - timedelta(days=random.randint(0, 15), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            
            scan = ScanRecord(
                scan_id=f"em-{str(uuid.uuid4())[:8]}",
                sender=sender,
                subject=subject,
                risk_score=risk_score,
                status=status,
                risk_level=risk_level,
                timestamp=ts,
                full_analysis={
                    "sender": sender,
                    "subject": subject,
                    "final_score": risk_score,
                    "status": status,
                    "risk_level": risk_level,
                    "threat_indicators": ["Newly registered domain", "Suspicious TLD"] if status != "CLEAN" else []
                }
            )
            session.add(scan)
            
            # Update KPI for that day
            date_str = ts.strftime("%Y-%m-%d")
            # Note: direct SQL access for simplicity in seeding
            from sqlmodel import select
            kpi = session.exec(select(KPIRecord).where(KPIRecord.date_str == date_str)).first()
            if not kpi:
                kpi = KPIRecord(date_str=date_str)
                session.add(kpi)
            
            kpi.emails_scanned += 1
            if status == "PHISHING":
                kpi.phishing_detected += 1
            if risk_level == "critical":
                kpi.high_risk_domains += 1

        session.commit()
        print(f"Successfully seeded 100 scan records.")

if __name__ == "__main__":
    seed_scans()
