"""Regression test for the GPU benchmark's concurrency-64 failure: a request
must not hold a pooled DB connection while the model generates."""

import asyncio

import httpx
import pytest

from app.auth import require_api_key
from app.db import SessionLocal, engine


def test_auth_lookup_releases_its_connection(api_key):
    db = SessionLocal()
    try:
        key = require_api_key(authorization=f"Bearer {api_key}", db=db)
        assert not db.in_transaction()  # connection returned to the pool
        assert key.id  # attributes still readable after detaching
    finally:
        db.close()


def _serve_in_thread(app):
    """A real uvicorn server on a free port, in this process (so the test can
    look at the same SQLAlchemy engine). httpx's in-process ASGITransport
    buffers whole responses, so it can't show what happens *during* a stream."""
    import socket
    import threading
    import time

    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_streams_hold_no_db_connection_while_generating(api_key):
    """Mid-generation -- every stream has its first token, none has finished --
    count the DB connections checked out. If each in-flight stream holds one,
    64 concurrent streams exhaust the default 15-connection pool, which is the
    failure seen at concurrency 64 in the GPU benchmark."""
    from app.main import app

    server, thread, base = _serve_in_thread(app)
    n = 20
    started = 0
    all_started = asyncio.Event()
    release = asyncio.Event()
    checked_out_mid_stream = None
    headers = {"Authorization": f"Bearer {api_key}", "x-synapse-cache": "bypass"}
    try:
        async with httpx.AsyncClient(base_url=base, timeout=30) as client:

            async def stream(i):
                nonlocal started
                body = {"model": "m", "stream": True,
                        "messages": [{"role": "user", "content": f"hold check {i} " + "word " * 40}]}
                async with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
                    assert resp.status_code == 200
                    lines = resp.aiter_lines()
                    async for line in lines:
                        if line.startswith("data: "):
                            break  # first event received; the stream is now mid-flight
                    started += 1
                    if started == n:
                        all_started.set()
                    await release.wait()
                    rest = [line async for line in lines]
                assert "data: [DONE]" in rest

            async def measure():
                nonlocal checked_out_mid_stream
                await all_started.wait()
                checked_out_mid_stream = engine.pool.checkedout()
                release.set()

            await asyncio.gather(measure(), *(stream(i) for i in range(n)))
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert checked_out_mid_stream == 0, f"{checked_out_mid_stream} DB connections held by {n} in-flight streams"


@pytest.mark.asyncio
async def test_no_db_connection_held_while_waiting_for_first_token(api_key, monkeypatch):
    """The gateway waits for the backend's first token before starting a
    streamed response (so a dead backend gets a real 502). Under load that
    wait can be seconds (vLLM queues requests). If each waiting request holds a
    pooled DB connection, the 15-connection pool empties; the next request's
    synchronous pool checkout then blocks the whole event loop for up to 30 s.
    That is the concurrency-64 failure from the GPU benchmark."""
    import app.providers as providers_module
    from app.config import get_settings
    from app.main import app

    class SlowBackend(providers_module.BaseProvider):
        """Stands in for vLLM on the real (non-mock) path, where the gateway
        awaits the first chunk before returning the StreamingResponse."""

        name = "vllm"

        async def complete(self, *args, **kwargs):
            raise NotImplementedError

        async def stream(self, model, messages, temperature, max_tokens=None, ignore_eos=False):
            await asyncio.sleep(0.6)  # vLLM queueing / prefill under load
            for word in ("slow ", "first ", "token"):
                yield providers_module.StreamChunk(text=word)
            yield providers_module.StreamChunk(text="", done=True, prompt_tokens=3, completion_tokens=3)

    monkeypatch.setattr(get_settings(), "mock_mode", False)
    monkeypatch.setattr(providers_module, "provider_for", lambda route: SlowBackend())
    server, thread, base = _serve_in_thread(app)
    n = 10
    headers = {"Authorization": f"Bearer {api_key}", "x-synapse-cache": "bypass"}
    try:
        async with httpx.AsyncClient(base_url=base, timeout=30) as client:

            async def one(i):
                body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": f"wait {i}"}]}
                resp = await client.post("/v1/chat/completions", json=body, headers=headers)
                assert resp.status_code == 200 and "data: [DONE]" in resp.text

            requests = asyncio.gather(*(one(i) for i in range(n)))
            await asyncio.sleep(0.3)  # all n requests are now waiting for their first token
            held_while_waiting = engine.pool.checkedout()
            await requests
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert held_while_waiting == 0, f"{held_while_waiting} of {n} requests held a DB connection while waiting"
