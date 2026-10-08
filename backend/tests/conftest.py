import os
import tempfile

import fakeredis
import pytest

os.environ["MOCK_MODE"] = "true"
os.environ["ADMIN_KEY"] = "test-admin-key"
# TEST_DATABASE_URL lets the same suite run against a real Postgres (CI does
# this too) -- SQLite-only SQL like strftime() passes on SQLite and breaks on
# Postgres, which is what docker-compose actually runs.
if os.environ.get("TEST_DATABASE_URL"):
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
else:
    _db_fd, _db_path = tempfile.mkstemp(suffix=".db")
    os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"

import app.cache as cache_module  # noqa: E402
import app.redis_client as redis_client_module  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.models import ApiKey, RequestLog  # noqa: E402,F401

# Tests build the schema straight from the SQLAlchemy models (no Alembic
# involved) -- production/dev use `alembic upgrade head` instead, see
# app/main.py and alembic/. drop_all first so a reused Postgres database
# starts clean.
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    fake = fakeredis.FakeStrictRedis(decode_responses=True)
    monkeypatch.setattr(redis_client_module.redis, "from_url", lambda *a, **k: fake)
    redis_client_module.get_redis.cache_clear()
    cache_module._cache_singleton = None
    get_settings.cache_clear()
    yield
    redis_client_module.get_redis.cache_clear()
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
