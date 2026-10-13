"""Embedded startup migration keeps application diagnostics observable."""
import logging

import sqlalchemy as sa
from sqlmodel import SQLModel, Session, create_engine

from core.config import settings
from data import database, store
from data.models import User


def test_startup_migration_preserves_application_logging(tmp_path, monkeypatch, caplog):
    url = 'sqlite:///' + (tmp_path / 'startup_copy.db').as_posix()
    engine = create_engine(url)
    SQLModel.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(sa.text('CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)'))
        connection.execute(sa.text("INSERT INTO alembic_version VALUES ('a4c9d0e1f2a3')"))
    with Session(engine) as session:
        session.add(User(uuid='logging-fixture', name='Fixture', email='fixture@example.invalid', hashed_password='unused'))
        session.commit()
    monkeypatch.setattr(database, 'engine', engine)
    monkeypatch.setattr(store, 'engine', engine)
    monkeypatch.setattr(settings, 'DATABASE_URL', url)
    app_logger = logging.getLogger('phishguard')
    monkeypatch.setattr(app_logger, 'disabled', False)
    caplog.set_level(logging.INFO)
    store.init_db()
    with engine.connect() as connection:
        assert connection.execute(sa.text('SELECT version_num FROM alembic_version')).scalar() == 'c7e2b0a61f84'
    app_logger.info('Email startup timing remains observable')
    assert not app_logger.disabled
    assert 'Email startup timing remains observable' in caplog.text
    engine.dispose()
