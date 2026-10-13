from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from data.database import get_session
from data.models import User
from main import app


def test_google_callback_links_verified_existing_account():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(User(uuid="existing-test-user", name="Existing", email="person@example.test", hashed_password="unused"))
        session.commit()

    def test_session():
        with Session(engine) as session:
            yield session

    def mock_google(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "test-access-token"})
        return httpx.Response(200, json={
            "email": "Person@Example.Test",
            "email_verified": True,
            "name": "Existing",
        })

    real_client = httpx.AsyncClient

    def fake_client(**kwargs):
        return real_client(transport=httpx.MockTransport(mock_google), **kwargs)

    app.dependency_overrides[get_session] = test_session
    try:
        with patch("routers.auth.httpx.AsyncClient", side_effect=fake_client):
            with TestClient(app) as client:
                response = client.get("/auth/google/callback?code=test-code", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/dashboard"
        assert "access_token=" in response.headers["set-cookie"]
        with Session(engine) as session:
            users = session.exec(select(User)).all()
            assert len(users) == 1
            assert users[0].google_linked is True
    finally:
        app.dependency_overrides.pop(get_session, None)


def test_google_callback_rejects_unverified_email():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)

    def test_session():
        with Session(engine) as session:
            yield session

    def mock_google(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "test-access-token"})
        return httpx.Response(200, json={"email": "person@example.test", "email_verified": False})

    real_client = httpx.AsyncClient
    def fake_client(**kwargs):
        return real_client(transport=httpx.MockTransport(mock_google), **kwargs)

    app.dependency_overrides[get_session] = test_session
    try:
        with patch("routers.auth.httpx.AsyncClient", side_effect=fake_client):
            with TestClient(app) as client:
                response = client.get("/auth/google/callback?code=test-code", follow_redirects=False)
        assert response.status_code == 400
        with Session(engine) as session:
            assert session.exec(select(User)).all() == []
    finally:
        app.dependency_overrides.pop(get_session, None)


def test_google_callback_uses_verified_id_token_when_profile_endpoint_is_unreachable():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)

    def test_session():
        with Session(engine) as session:
            yield session

    def mock_google(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "test-access-token", "id_token": "test-id-token"})
        if request.url.path.endswith("/userinfo"):
            raise httpx.ConnectError("profile endpoint unreachable", request=request)
        assert request.url.path == "/tokeninfo"
        return httpx.Response(200, json={
            "aud": "test-client-id", "iss": "https://accounts.google.com",
            "email": "person@example.test", "email_verified": "true", "name": "Person",
        })

    real_client = httpx.AsyncClient
    def fake_client(**kwargs):
        return real_client(transport=httpx.MockTransport(mock_google), **kwargs)

    app.dependency_overrides[get_session] = test_session
    try:
        with patch("routers.auth.settings.GOOGLE_CLIENT_ID", "test-client-id"), patch("routers.auth.httpx.AsyncClient", side_effect=fake_client):
            with TestClient(app) as client:
                response = client.get("/auth/google/callback?code=test-code", follow_redirects=False)
        assert response.status_code == 302
        with Session(engine) as session:
            assert session.exec(select(User).where(User.email == "person@example.test")).first() is not None
    finally:
        app.dependency_overrides.pop(get_session, None)
