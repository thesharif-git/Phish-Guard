from datetime import datetime, timezone

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from data.models import ScanRecord
from data.store import get_all_scans, get_email_detail, get_risk_trend


def test_history_and_details_are_scoped_to_account():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    now = datetime.now(timezone.utc)

    with Session(engine) as session:
        session.add_all([
            ScanRecord(scan_id="em-mine", user_id=1, scan_type="email", sender="mine@example.test",
                       subject="My message", risk_score=85, status="PHISHING", risk_level="critical", timestamp=now),
            ScanRecord(scan_id="dm-mine", user_id=1, scan_type="domain", sender="example.test",
                       subject="example.test", risk_score=15, status="UNKNOWN", risk_level="unknown", timestamp=now),
            ScanRecord(scan_id="em-other", user_id=2, scan_type="email", sender="other@example.test",
                       subject="Private message", risk_score=10, status="CLEAN", risk_level="safe", timestamp=now),
        ])
        session.commit()

        rows, counts = get_all_scans(session, user_id=1)
        assert {row["scan_id"] for row in rows} == {"em-mine", "dm-mine"}
        assert counts == {"total": 2, "clean": 0, "suspicious": 0, "phishing": 1, "unknown": 1}
        assert get_email_detail("em-other", session, user_id=1) is None
        assert get_risk_trend(session, user_id=1)["clean"][-1] == 0
