"""Micro-benchmark: does reusing one httpx.AsyncClient beat creating a new
client per request (what app/providers.py does)?

Result so far: INCONCLUSIVE -- see results_2026-10-08_macbook.txt and
DECISIONS.md D5. Kept so the measurement can be repeated, not as evidence.

Usage (two terminals, from this directory):
    uvicorn stub_backend:app --port 18011 --workers 4 --log-level warning
    KEEPALIVE=20 python httpx_client_reuse.py
"""

import asyncio
import os
import statistics
import time

import httpx

URL = "http://127.0.0.1:18011/v1/chat/completions"
BODY = {"model": "m", "messages": []}
N = 400  # requests per measurement
KEEPALIVE = int(os.environ.get("KEEPALIVE", "20"))  # httpx's default is 20


async def per_request(concurrency: int):
    async def one():
        async with httpx.AsyncClient(timeout=60) as client:
            t = time.perf_counter()
            await client.post(URL, json=BODY)
            return time.perf_counter() - t

    return await run(one, concurrency)


async def shared(concurrency: int):
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=KEEPALIVE)
    client = httpx.AsyncClient(timeout=60, limits=limits)

    async def one():
        t = time.perf_counter()
        await client.post(URL, json=BODY)
        return time.perf_counter() - t

    try:
        return await run(one, concurrency)
    finally:
        await client.aclose()


async def run(one, concurrency: int) -> tuple[float, float]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded():
        async with sem:
            return await one()

    latencies_ms = sorted(x * 1000 for x in await asyncio.gather(*[guarded() for _ in range(N)]))
    return statistics.median(latencies_ms), latencies_ms[int(len(latencies_ms) * 0.95)]


async def main():
    for concurrency in (1, 16, 64):
        for name, fn in (("per-request", per_request), ("shared", shared)):
            await fn(concurrency)  # warm-up, discarded
            p50, p95 = await fn(concurrency)
            print(f"c={concurrency:<3} {name:<12} p50={p50:6.2f} ms  p95={p95:6.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
