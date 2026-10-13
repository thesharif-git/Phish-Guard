from datetime import datetime, timezone

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from data.models import ScanRecord
from data.store import get_campaigns


def test_campaign_groups_require_repeated_phishing_emails_from_same_account():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        session.add_all([
            ScanRecord(scan_id="em-1", user_id=1, scan_type="email", sender="a@example.test", subject="First", risk_score=85, status="PHISHING", risk_level="critical", timestamp=now),
            ScanRecord(scan_id="em-2", user_id=1, scan_type="email", sender="b@example.test", subject="Second", risk_score=75, status="PHISHING", risk_level="medium", timestamp=now),
            ScanRecord(scan_id="em-3", user_id=2, scan_type="email", sender="c@example.test", subject="Other account", risk_score=90, status="PHISHING", risk_level="critical", timestamp=now),
            ScanRecord(scan_id="em-4", user_id=1, scan_type="email", sender="a@elsewhere.test", subject="Single", risk_score=90, status="PHISHING", risk_level="critical", timestamp=now),
        ])
        session.commit()
        groups = get_campaigns(session, 1)
        assert len(groups) == 1
        assert groups[0]["source"] == "example.test"
        assert groups[0]["count"] == 2
        assert get_campaigns(session, 2) == []
