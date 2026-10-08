"""Stats endpoints the dashboard charts are built from. Run under both
SQLite (default) and Postgres (TEST_DATABASE_URL, as in CI) -- the hourly
bucketing SQL is exactly the kind of thing that differs between them."""

from datetime import datetime, timedelta, timezone

import pytest

from app.db import SessionLocal
from app.models import ApiKey, RequestLog


@pytest.fixture
def seeded_logs(api_key):
    """Two requests in one hour, one in the next, plus one error row."""
    now = datetime.now(timezone.utc)
    this_hour = now.replace(minute=30, second=0, microsecond=0)
    if this_hour > now:  # keep every row in the past, inside the 24h window
        this_hour -= timedelta(hours=1)
    last_hour = this_hour - timedelta(hours=1)

    db = SessionLocal()
    try:
        db.query(RequestLog).delete()
        key_id = db.query(ApiKey.id).first()[0]
        rows = [
            RequestLog(api_key_id=key_id, provider="vllm", model="m", cached=False,
                       cost_usd=0.01, latency_ms=200.0, created_at=last_hour),
            RequestLog(api_key_id=key_id, provider="vllm", model="m", cached=False,
                       cost_usd=0.02, latency_ms=400.0, created_at=this_hour),
            RequestLog(api_key_id=key_id, provider="cache", model="m", cached=True,
                       cost_usd=0.0, latency_ms=2.0, created_at=this_hour),
            RequestLog(api_key_id=key_id, provider="error", model="m", cached=False,
                       cost_usd=0.0, latency_ms=60000.0, status="error", created_at=this_hour),
        ]
        db.add_all(rows)
        db.commit()
    finally:
        db.close()
    return last_hour, this_hour


def test_timeseries_buckets_by_hour(client, seeded_logs):
    last_hour, this_hour = seeded_logs
    resp = client.get("/v1/stats/timeseries?hours=24")
    assert resp.status_code == 200
    points = resp.json()

    assert [p["bucket"] for p in points] == [
        last_hour.strftime("%Y-%m-%dT%H:00:00"),
        this_hour.strftime("%Y-%m-%dT%H:00:00"),
    ]
    first, second = points
    assert (first["requests"], first["cache_hits"]) == (1, 0)
    assert first["avg_latency_ms"] == 200.0
    # 3 rows this hour (incl. the error), but latency averages successful
    # responses only: (400 + 2) / 2.
    assert (second["requests"], second["cache_hits"]) == (3, 1)
    assert second["avg_latency_ms"] == 201.0
    assert second["cost_usd"] == pytest.approx(0.02)


def test_provider_breakdown(client, seeded_logs):
    resp = client.get("/v1/stats/providers?hours=24")
    assert resp.status_code == 200
    by_provider = {row["provider"]: row["requests"] for row in resp.json()}
    assert by_provider == {"vllm": 2, "cache": 1, "error": 1}
