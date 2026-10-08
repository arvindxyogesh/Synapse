"""Shared test fixtures: a fake OpenAI-style streaming backend that every
httpx.AsyncClient created by the harness talks to instead of the network."""

import asyncio
import json

import httpx
import pytest

import synapse_bench.loadgen as loadgen

_RealAsyncClient = httpx.AsyncClient


def _sse(event) -> bytes:
    return f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n".encode()


class FakeBackend:
    """Streams `n_tokens` one-word chunks: waits `prefill_s` before the first,
    `per_token_s` between the rest, then a usage chunk and [DONE]. Tracks
    how many requests are in flight at once."""

    def __init__(self, prefill_s=0.05, per_token_s=0.01, n_tokens=5, provider="vllm", x_cache="bypass",
                 send_done=True, send_usage=True, status=200, raise_timeout=False, cached_tokens=None):
        self.prefill_s, self.per_token_s, self.n_tokens = prefill_s, per_token_s, n_tokens
        self.provider, self.x_cache = provider, x_cache
        self.send_done, self.send_usage, self.status = send_done, send_usage, status
        self.raise_timeout = raise_timeout
        self.cached_tokens = cached_tokens
        self.in_flight = self.max_in_flight = 0
        self.prompts_seen: list[str] = []
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.prompts_seen.append(body["messages"][0]["content"])
        if self.raise_timeout:
            # MockTransport never enforces timeouts (only real network
            # transports do), so simulate what httpx raises on one.
            raise httpx.ReadTimeout("simulated", request=request)
        headers = {"x-cache": self.x_cache} if self.x_cache else {}
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "backend down"}, headers=headers)
        return httpx.Response(200, content=self._stream(), headers=headers)

    async def _stream(self):
        # A request counts as in flight until its last event is produced.
        # (Not via try/finally: the client stops reading at [DONE], so a
        # finally block would only run whenever the generator is collected.)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self.prefill_s)
        for i in range(self.n_tokens):
            if i:
                await asyncio.sleep(self.per_token_s)
            yield _sse({"provider": self.provider, "choices": [{"delta": {"content": f"w{i} "}}]})
        if self.send_usage:
            usage = {"prompt_tokens": 3, "completion_tokens": self.n_tokens}
            if self.cached_tokens is not None:
                usage["prompt_tokens_details"] = {"cached_tokens": self.cached_tokens}
            yield _sse({"provider": self.provider, "choices": [], "usage": usage})
        self.in_flight -= 1
        if self.send_done:
            yield _sse("[DONE]")


@pytest.fixture
def fake(monkeypatch):
    """Route the harness's async HTTP clients to a FakeBackend."""
    backend = FakeBackend()

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(backend.handler)
        return _RealAsyncClient(*args, **kwargs)

    monkeypatch.setattr(loadgen.httpx, "AsyncClient", factory)
    return backend


