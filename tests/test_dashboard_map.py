from datetime import datetime, timedelta, timezone

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from data.models import ScanRecord
from data.store import get_threat_infrastructure_countries


def test_map_counts_only_saved_geo_for_flagged_url_and_domain_scans():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    now = datetime.now(timezone.utc)

    def scan(scan_id, scan_type, status, country, user_id=1, timestamp=now):
        record = ScanRecord(
            scan_id=scan_id,
            user_id=user_id,
            scan_type=scan_type,
            sender="example.test",
            subject="Test scan",
            risk_score=80,
            status=status,
            risk_level="critical",
            timestamp=timestamp,
        )
        record.full_analysis = {"geo": {"countryCode": country}} if country else {}
        return record

    with Session(engine) as session:
        session.add_all([
            scan("url-us", "url", "PHISHING", "US"),
            scan("domain-ca", "domain", "SUSPICIOUS", "CA"),
            scan("url-unknown", "url", "PHISHING", ""),
            scan("email-ru", "email", "PHISHING", "RU"),
            scan("url-clean", "url", "CLEAN", "US"),
            scan("url-other-user", "url", "PHISHING", "FR", user_id=2),
            scan("url-old", "url", "PHISHING", "DE", timestamp=now - timedelta(days=40)),
        ])
        session.commit()

        recent = get_threat_infrastructure_countries(session, user_id=1, days=30)
        assert recent["map_data"] == {"US": 1, "CA": 1}
        assert recent["unknown"] == 1
        assert recent["total"] == 3

        all_time = get_threat_infrastructure_countries(session, user_id=1, days=None)
        assert all_time["map_data"] == {"US": 1, "CA": 1, "DE": 1}
        assert all_time["total"] == 4
