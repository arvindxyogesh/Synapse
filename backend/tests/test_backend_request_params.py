"""Request parameters that must actually reach the model backend.

max_tokens must arrive in each backend's own wire format -- without it,
output length is uncontrolled and tokens/s comparisons between models are
meaningless. The backend timeout must be configurable, so long generations
in a deep queue aren't recorded as errors."""

import json

import httpx
import pytest

import app.providers as providers_module
from app.config import get_settings
from app.providers import OllamaProvider, VLLMProvider

_RealAsyncClient = httpx.AsyncClient


def _capture_requests(monkeypatch, response_json: dict) -> list[dict]:
    """Route every httpx call through a MockTransport and record the JSON
    bodies that were sent."""
    sent: list[dict] = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response_json)

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", factory)
    return sent


_VLLM_RESPONSE = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
_OLLAMA_RESPONSE = {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}
_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_vllm_sends_max_tokens(monkeypatch):
    sent = _capture_requests(monkeypatch, _VLLM_RESPONSE)
    await VLLMProvider("http://vllm").complete("m", _MESSAGES, 0.0, max_tokens=256)
    assert sent[0]["max_tokens"] == 256


@pytest.mark.asyncio
async def test_vllm_omits_max_tokens_when_unset(monkeypatch):
    sent = _capture_requests(monkeypatch, _VLLM_RESPONSE)
    await VLLMProvider("http://vllm").complete("m", _MESSAGES, 0.0)
    assert "max_tokens" not in sent[0]


@pytest.mark.asyncio
async def test_ollama_sends_num_predict(monkeypatch):
    sent = _capture_requests(monkeypatch, _OLLAMA_RESPONSE)
    await OllamaProvider("http://ollama").complete("m", _MESSAGES, 0.0, max_tokens=64)
    assert sent[0]["options"] == {"temperature": 0.0, "num_predict": 64}


@pytest.mark.asyncio
async def test_ollama_omits_num_predict_when_unset(monkeypatch):
    sent = _capture_requests(monkeypatch, _OLLAMA_RESPONSE)
    await OllamaProvider("http://ollama").complete("m", _MESSAGES, 0.0)
    assert "num_predict" not in sent[0]["options"]


def test_gateway_forwards_max_tokens_to_backend(client, api_key, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "provider", "vllm")
    monkeypatch.setattr(settings, "mock_mode", False)
    sent = _capture_requests(monkeypatch, _VLLM_RESPONSE)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "max tokens check"}], "max_tokens": 32},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    assert resp.json()["provider"] == "vllm"
    assert sent[0]["max_tokens"] == 32


def test_gateway_rejects_non_positive_max_tokens(client, api_key):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("stream", [False, True])
def test_requests_with_max_tokens_skip_the_cache(client, api_key, stream):
    """A capped (possibly truncated) reply must not be stored, and a capped
    request must not be served a full-length cached reply."""
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = f"cache skip check stream={stream}"
    capped = {"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": 4, "stream": stream}
    uncapped = {"model": "m", "messages": [{"role": "user", "content": prompt}], "stream": stream}

    first = client.post("/v1/chat/completions", json=capped, headers=headers)
    assert first.headers["x-cache"] == "bypass"

    # The capped reply wasn't stored, so this is a miss, not a hit...
    second = client.post("/v1/chat/completions", json=uncapped, headers=headers)
    assert second.headers["x-cache"] == "miss"

    # ...and now that a full reply *is* cached, a capped request still skips it.
    third = client.post("/v1/chat/completions", json=capped, headers=headers)
    assert third.headers["x-cache"] == "bypass"


@pytest.mark.asyncio
async def test_backend_timeout_is_configurable(monkeypatch):
    monkeypatch.setattr(get_settings(), "backend_timeout_seconds", 300.0)
    seen_timeouts = []

    def factory(*args, **kwargs):
        seen_timeouts.append(kwargs["timeout"])
        kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200, json=_VLLM_RESPONSE))
        return _RealAsyncClient(*args, **kwargs)

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", factory)
    await VLLMProvider("http://vllm").complete("m", _MESSAGES, 0.0)
    assert seen_timeouts == [300.0]
