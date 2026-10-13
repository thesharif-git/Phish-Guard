from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import pages


def test_public_pricing_renders_with_settings_route_present(monkeypatch):
    # The /settings function previously replaced the configuration import,
    # causing /pricing to fail for every visitor, including signed-out users.
    monkeypatch.setattr(pages.app_settings, 'STRIPE_PRICE_FREE_PLUS_YEARLY', 'test-pro-price')
    monkeypatch.setattr(pages.app_settings, 'STRIPE_PRICE_PRO_YEARLY', '')
    app = FastAPI()
    app.include_router(pages.router)
    with TestClient(app) as client:
        response = client.get('/pricing')
    assert response.status_code == 200
    assert response.context['yearly_pro_available'] is True
    assert response.context['yearly_team_available'] is False
    assert '25 scans each month' in response.text
    assert 'id="yearly"' in response.text
