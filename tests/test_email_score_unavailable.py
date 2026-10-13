"""Missing Email probabilities remain unavailable through storage and views."""
import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from jinja2 import ChoiceLoader, DictLoader, Environment, FileSystemLoader
import sqlalchemy as sa
from sqlmodel import Session, SQLModel, create_engine, select

from data import store
from data.models import ScanRecord, User
from services.conformal_evidential_engine import synthesize_email_security_opinion

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_session():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(uuid="unavailable-score-owner", name="Fixture", email="fixture@example.invalid", hashed_password="unused")
        session.add(user)
        session.commit()
        session.refresh(user)
        yield session, user.id
    engine.dispose()


def test_missing_model_does_not_invent_an_aggregate_score():
    result = synthesize_email_security_opinion(content_prob=None,
        heuristic_evidence={"bec_detected": True, "qr_detected": True})
    assert result["final_score"] is None
    assert result["threat_probability"] is None
    assert result["action"] == "WARNING_BANNER"
    assert result["conformal_prediction_set"] == ["CLEAN", "PHISHING"]
    assert not result["calibration_valid"]


@pytest.mark.parametrize("include_score", [True, False])
def test_unavailable_email_score_is_saved_and_returned_as_null(isolated_session, include_score):
    session, user_id = isolated_session
    payload = {"sender": "fixture@example.invalid", "subject": "Missing model fixture",
               "status": "SUSPICIOUS", "risk_level": "medium", "security_action": "WARNING_BANNER",
               "content_analysis": {"ml_probability": None, "ml_powered": False}}
    if include_score:
        payload["final_score"] = None
    scan_id = store.add_scan(payload, session=session, user_id=user_id)
    record = session.exec(select(ScanRecord).where(ScanRecord.scan_id == scan_id)).one()
    assert record.risk_score is None
    assert record.full_analysis.get("final_score") is None
    assert store.get_recent_scans(session, user_id=user_id)[0]["risk_score"] is None
    rows, counts = store.get_all_scans(session, user_id=user_id)
    assert rows[0]["risk_score"] is None and counts["suspicious"] == 1
    assert store.get_email_detail(scan_id, session, user_id)["risk_score"] is None


@pytest.mark.parametrize("scan_type,score", [("email", 0), ("email", 82), ("url", 0), ("url", 71), ("domain", 34)])
def test_real_numeric_scores_are_preserved(isolated_session, scan_type, score):
    session, user_id = isolated_session
    payload = {"sender": "fixture@example.invalid", "subject": "Numeric score fixture",
               "final_score": score, "status": "SUSPICIOUS", "risk_level": "medium"}
    scan_id = store.add_scan(payload, session=session, user_id=user_id, scan_type=scan_type)
    record = session.exec(select(ScanRecord).where(ScanRecord.scan_id == scan_id)).one()
    assert record.risk_score == score and record.full_analysis["final_score"] == score


@pytest.mark.parametrize("template_name", ["dashboard.html", "reports.html", "inbox.html", "email_detail.html"])
@pytest.mark.parametrize("score", [None, 0, 82])
def test_saved_score_templates_render_unavailability_and_preserve_zero(template_name, score):
    environment = Environment(loader=ChoiceLoader([
        DictLoader({"base.html": "{% block content %}{% endblock %}{% block scripts %}{% endblock %}"}),
        FileSystemLoader(ROOT / "templates"),
    ]))
    scan = {"id": "em-fixture", "scan_id": "em-fixture", "scan_type": "email",
            "sender": "fixture@example.invalid", "subject": "Fixture", "risk_score": score,
            "status": "SUSPICIOUS", "risk_level": "medium", "time": "2026-10-04 10:00",
            "timestamp": "2026-10-04 10:00", "received_time": "2026-10-04 10:00",
            "urls": [], "attachments": [], "routing": [], "why_flagged": [],
            "body_preview": "Fixture", "raw_headers": "", "domain_age_days": None}
    html = environment.get_template(template_name).render(
        recent_scans=[scan], scans=[scan], email=scan, user=None,
        metrics={"total": 1, "phishing": 0, "suspicious": 1, "clean": 0},
        map_summary={"mapped": 0, "unknown": 0, "top_list": [], "map_data": {}},
        recent_detections=[], total=1, phishing=0, suspicious=1, clean=0, unknown=0,
        trend={"labels": [], "values": []}, countries=[],
    )
    assert "None/100" not in html and "width: None%" not in html
    if score is None:
        assert "Unavailable" in html
        if template_name == "dashboard.html":
            assert 'style="width:' not in html
    else:
        assert "Unavailable" not in html
        assert str(score) in html
        if template_name == "dashboard.html":
            assert f"width: {score}%" in html
        if template_name in ("reports.html", "email_detail.html"):
            assert "/100" in html


def load_score_migration():
    path = ROOT / "alembic/versions/c7e2b0a61f84_nullable_scan_risk_score.py"
    spec = importlib.util.spec_from_file_location("score_nullability_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nullable_migration_preserves_rows_constraints_and_indexes(tmp_path, monkeypatch):
    engine = sa.create_engine("sqlite:///" + (tmp_path / "legacy_copy.db").as_posix())
    migration = load_score_migration()
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE user (id INTEGER PRIMARY KEY)"))
        connection.execute(sa.text("INSERT INTO user VALUES (7)"))
        connection.execute(sa.text("CREATE TABLE scanrecord (id INTEGER PRIMARY KEY, scan_id VARCHAR NOT NULL UNIQUE, risk_score INTEGER NOT NULL, user_id INTEGER REFERENCES user(id), full_analysis_json VARCHAR NOT NULL)"))
        connection.execute(sa.text("CREATE INDEX ix_scanrecord_user_id ON scanrecord(user_id)"))
        connection.execute(sa.text("INSERT INTO scanrecord VALUES (1, 'em-existing', 82, 7, '{}')"))
        before = connection.execute(sa.text("SELECT * FROM scanrecord ORDER BY id")).all()
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
        assert connection.execute(sa.text("SELECT * FROM scanrecord ORDER BY id")).all() == before
        assert next(c for c in sa.inspect(connection).get_columns("scanrecord") if c["name"] == "risk_score")["nullable"]
        assert any(i["name"] == "ix_scanrecord_user_id" for i in sa.inspect(connection).get_indexes("scanrecord"))
        assert sa.inspect(connection).get_foreign_keys("scanrecord")[0]["referred_table"] == "user"
        with pytest.raises(sa.exc.IntegrityError):
            connection.execute(sa.text("INSERT INTO scanrecord VALUES (2, 'em-existing', 33, 7, '{}')"))
        connection.execute(sa.text("INSERT INTO scanrecord VALUES (2, 'em-unavailable', NULL, 7, '{}')"))
        assert connection.execute(sa.text("PRAGMA foreign_key_check")).all() == []
        with pytest.raises(RuntimeError, match="unavailable scan scores"):
            migration.downgrade()
    engine.dispose()


def test_nullable_migration_can_downgrade_without_unavailable_records(tmp_path, monkeypatch):
    engine = sa.create_engine("sqlite:///" + (tmp_path / "numeric_copy.db").as_posix())
    migration = load_score_migration()
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE scanrecord (id INTEGER PRIMARY KEY, risk_score INTEGER NOT NULL)"))
        connection.execute(sa.text("INSERT INTO scanrecord VALUES (1, 0), (2, 82)"))
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
        migration.downgrade()
        assert connection.execute(sa.text("SELECT * FROM scanrecord ORDER BY id")).all() == [(1, 0), (2, 82)]
        assert not next(c for c in sa.inspect(connection).get_columns("scanrecord") if c["name"] == "risk_score")["nullable"]
    engine.dispose()
