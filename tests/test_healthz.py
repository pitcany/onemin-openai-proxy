import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import os

    os.environ.setdefault("ONE_MIN_AI_API_KEY", "test-api-key")

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_healthz_returns_ok(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_healthz_content_type(client):
    response = client.get("/healthz")

    assert "application/json" in response.headers["content-type"]
