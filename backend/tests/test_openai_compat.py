"""Proves /v1/chat/completions is drop-in compatible with the real `openai`
SDK -- not just shape-similar to it. Point the SDK at base_url=".../v1"
with a gateway API key as api_key and it works unmodified: same request/
response schema, same Authorization: Bearer auth, same SSE streaming
format. Here it talks to the FastAPI app in-process via httpx's
ASGITransport instead of a real socket, so it's a normal, fast test
(ASGITransport only implements the async transport interface, hence
AsyncOpenAI/AsyncClient rather than the sync SDK)."""

import httpx
import openai
import pytest


@pytest.fixture
def openai_client(api_key):
    from app.main import app

    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    return openai.AsyncOpenAI(api_key=api_key, base_url="http://testserver/v1", http_client=http_client)


@pytest.mark.asyncio
async def test_real_openai_sdk_non_streaming(openai_client):
    resp = await openai_client.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": "hello from the real openai sdk"}],
    )
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.role == "assistant"
    assert "hello from the real openai sdk" in resp.choices[0].message.content
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens


@pytest.mark.asyncio
async def test_real_openai_sdk_streaming(openai_client):
    stream = await openai_client.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": "stream via the real sdk please"}],
        stream=True,
    )
    chunks = [c async for c in stream]
    assert len(chunks) > 1
    assert all(c.object == "chat.completion.chunk" for c in chunks)

    full_text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert "stream via the real sdk please" in full_text
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_real_openai_sdk_second_call_is_a_cache_hit(openai_client):
    messages = [{"role": "user", "content": "identical prompt for cache hit check"}]
    await openai_client.chat.completions.create(model="llama3", messages=messages)
    second = await openai_client.chat.completions.create(model="llama3", messages=messages)
    # cached/provider are additive extensions the SDK's typed model doesn't
    # know about but still exposes via model_extra, same as any client that
    # tolerates unknown JSON fields.
    assert second.model_extra["cached"] is True


@pytest.mark.asyncio
async def test_real_openai_sdk_wrong_key_is_rejected():
    from app.main import app

    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    bad_client = openai.AsyncOpenAI(
        api_key="llmgw_not-a-real-key", base_url="http://testserver/v1", http_client=http_client
    )
    with pytest.raises(openai.AuthenticationError):
        await bad_client.chat.completions.create(model="llama3", messages=[{"role": "user", "content": "hi"}])
