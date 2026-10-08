"""A failing backend must surface as an error, never as a fast, healthy-
looking mock response (unless MOCK_FALLBACK is explicitly turned on)."""

import httpx
import pytest

import app.providers as providers_module
from app.config import get_settings
from app.db import SessionLocal
from app.models import ApiKey, RequestLog
from app.providers import ProviderError

_RealAsyncClient = httpx.AsyncClient


def _unreachable_backend(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "provider", "vllm")
    monkeypatch.setattr(settings, "mock_mode", False)
    monkeypatch.setattr(settings, "mock_fallback", False)

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_run_completion_raises_when_backend_fails(monkeypatch):
    _unreachable_backend(monkeypatch)
    with pytest.raises(ProviderError, match="vllm backend failed"):
        await providers_module.run_completion("m", [{"role": "user", "content": "hi"}], 0.0)


@pytest.mark.asyncio
async def test_run_streaming_completion_raises_before_first_chunk(monkeypatch):
    _unreachable_backend(monkeypatch)
    with pytest.raises(ProviderError):
        await providers_module.run_streaming_completion("m", [{"role": "user", "content": "hi"}], 0.0)


def _error_rows(model: str) -> list[RequestLog]:
    db = SessionLocal()
    try:
        return db.query(RequestLog).filter(RequestLog.model == model, RequestLog.status == "error").all()
    finally:
        db.close()


@pytest.mark.parametrize("stream", [False, True])
def test_gateway_returns_502_and_logs_error_row(client, api_key, monkeypatch, stream):
    _unreachable_backend(monkeypatch)
    model = f"down-model-stream-{stream}"
    resp = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": stream},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 502
    assert "vllm backend failed" in resp.json()["detail"]

    rows = _error_rows(model)
    assert len(rows) == 1
    assert rows[0].provider == "error"
    assert rows[0].cost_usd == 0.0


def test_summary_latency_ignores_error_rows(api_key):
    from app.api.stats import summary

    db = SessionLocal()
    try:
        db.query(RequestLog).delete()
        key_id = db.query(ApiKey.id).first()[0]  # the `api_key` fixture created one
        db.add(RequestLog(api_key_id=key_id, provider="vllm", model="m", latency_ms=100.0, status="ok"))
        db.add(RequestLog(api_key_id=key_id, provider="error", model="m", latency_ms=60000.0, status="error"))
        db.commit()

        result = summary(hours=24, db=db)
    finally:
        db.close()

    assert result.total_requests == 2
    assert result.avg_latency_ms == 100.0
    assert result.p95_latency_ms == 100.0
