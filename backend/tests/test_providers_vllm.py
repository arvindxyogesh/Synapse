import json

import httpx
import pytest

import app.providers as providers_module
from app.config import get_settings
from app.providers import OllamaProvider, VLLMProvider

_RealAsyncClient = httpx.AsyncClient


def _client_factory(handler):
    """Builds a stand-in for httpx.AsyncClient that always routes through a
    MockTransport, so VLLMProvider's real (unmodified) HTTP calls can be
    tested without a real vLLM server. Uses the real class captured above --
    patching providers_module.httpx.AsyncClient mutates the httpx module
    itself (providers_module.httpx *is* httpx), so referencing httpx.AsyncClient
    here would recurse into the patched version instead of the real one."""

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    return factory


def test_real_provider_selects_ollama_by_default():
    assert isinstance(providers_module._real_provider(get_settings()), OllamaProvider)


def test_real_provider_selects_vllm_when_configured(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "provider", "vllm")
    monkeypatch.setattr(settings, "vllm_base_url", "http://example:9000")
    provider = providers_module._real_provider(settings)
    assert isinstance(provider, VLLMProvider)
    assert provider.base_url == "http://example:9000"


@pytest.mark.asyncio
async def test_complete_parses_openai_response(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "llama3"
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello there"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", _client_factory(handler))
    provider = VLLMProvider("http://vllm:8000")
    text, pt, ct = await provider.complete("llama3", [{"role": "user", "content": "hi"}], 0.7)
    assert text == "hello there"
    assert pt == 5
    assert ct == 3


@pytest.mark.asyncio
async def test_complete_estimates_tokens_when_usage_missing(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", _client_factory(handler))
    provider = VLLMProvider("http://vllm:8000")
    text, pt, ct = await provider.complete("llama3", [{"role": "user", "content": "hello world"}], 0.7)
    assert text == "hi"
    assert pt >= 1
    assert ct >= 1


@pytest.mark.asyncio
async def test_complete_sends_bearer_token_when_api_key_set(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", _client_factory(handler))
    provider = VLLMProvider("http://vllm:8000", api_key="secret")
    await provider.complete("llama3", [{"role": "user", "content": "hi"}], 0.0)
    assert seen["auth"] == "Bearer secret"


@pytest.mark.asyncio
async def test_complete_omits_auth_header_without_api_key(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", _client_factory(handler))
    provider = VLLMProvider("http://vllm:8000")
    await provider.complete("llama3", [{"role": "user", "content": "hi"}], 0.0)
    assert seen["auth"] is None


@pytest.mark.asyncio
async def test_stream_yields_chunks_then_final_usage_chunk(monkeypatch):
    # OpenAI/vLLM streaming shape with stream_options.include_usage: content
    # deltas, a finish_reason chunk with an empty delta, then a trailing
    # usage-only chunk with empty choices, then [DONE].
    sse_body = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request):
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", _client_factory(handler))
    provider = VLLMProvider("http://vllm:8000")
    chunks = [c async for c in provider.stream("llama3", [{"role": "user", "content": "hi"}], 0.0)]

    assert [c.text for c in chunks] == ["Hel", "lo", ""]
    assert [c.done for c in chunks] == [False, False, True]
    assert chunks[-1].prompt_tokens == 4
    assert chunks[-1].completion_tokens == 2


@pytest.mark.asyncio
async def test_run_completion_uses_vllm_when_provider_configured(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "provider", "vllm")
    monkeypatch.setattr(settings, "mock_mode", False)

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "real vllm response"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", _client_factory(handler))
    text, pt, ct, provider_name = await providers_module.run_completion(
        "llama3", [{"role": "user", "content": "hi"}], 0.0
    )
    assert provider_name == "vllm"
    assert text == "real vllm response"
    assert pt == 2
    assert ct == 3


@pytest.mark.asyncio
async def test_run_completion_falls_back_to_mock_when_vllm_unreachable(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "provider", "vllm")
    monkeypatch.setattr(settings, "mock_mode", False)

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", _client_factory(handler))
    text, pt, ct, provider_name = await providers_module.run_completion(
        "llama3", [{"role": "user", "content": "hi"}], 0.0
    )
    assert provider_name == "mock"
