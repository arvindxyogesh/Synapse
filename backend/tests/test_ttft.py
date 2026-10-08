"""Time to first token is logged for streamed responses -- the metric that
tells you how long a user stares at a blank screen, separate from how long
the whole answer takes."""

from app.db import SessionLocal
from app.models import RequestLog


def _latest_row(model: str) -> RequestLog:
    db = SessionLocal()
    try:
        return db.query(RequestLog).filter(RequestLog.model == model).order_by(RequestLog.created_at.desc()).first()
    finally:
        db.close()


def _chat(client, api_key, model, prompt, stream):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    return resp


def test_streamed_miss_logs_ttft_before_end_of_response(client, api_key):
    # The mock backend streams one word every 10 ms, so the first token
    # arrives well before the last one.
    _chat(client, api_key, "ttft-stream", "a prompt long enough to stream several words", stream=True)
    row = _latest_row("ttft-stream")
    assert row.ttft_ms is not None
    assert 0 < row.ttft_ms < row.latency_ms


def test_streamed_cache_hit_logs_ttft(client, api_key):
    _chat(client, api_key, "ttft-hit", "cache me then stream me", stream=True)
    resp = _chat(client, api_key, "ttft-hit", "cache me then stream me", stream=True)
    assert resp.headers["x-cache"] == "hit"
    row = _latest_row("ttft-hit")
    assert row.cached is True
    assert row.ttft_ms is not None and row.ttft_ms <= row.latency_ms


def test_non_streamed_request_has_no_ttft(client, api_key):
    _chat(client, api_key, "ttft-none", "no streaming here", stream=False)
    assert _latest_row("ttft-none").ttft_ms is None


def test_request_log_endpoint_exposes_ttft(client, api_key):
    _chat(client, api_key, "ttft-api", "expose ttft please", stream=True)
    rows = client.get("/v1/stats/requests?limit=500").json()
    row = next(r for r in rows if r["model"] == "ttft-api")
    assert row["ttft_ms"] is not None
