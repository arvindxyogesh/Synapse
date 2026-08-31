import os
import tempfile

import fakeredis
import pytest

os.environ["MOCK_MODE"] = "true"
os.environ["ADMIN_KEY"] = "test-admin-key"
_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"

import app.cache as cache_module  # noqa: E402
from app.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    fake = fakeredis.FakeStrictRedis(decode_responses=True)
    monkeypatch.setattr(cache_module.redis, "from_url", lambda *a, **k: fake)
    cache_module._cache_singleton = None
    get_settings.cache_clear()
    yield
    cache_module._cache_singleton = None
    get_settings.cache_clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def admin_headers():
    return {"x-admin-key": os.environ["ADMIN_KEY"]}


@pytest.fixture
def api_key(client, admin_headers):
    resp = client.post("/v1/admin/keys", json={"name": "test-key"}, headers=admin_headers)
    assert resp.status_code == 200
    return resp.json()["api_key"]
