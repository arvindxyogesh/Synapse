"""Closed-loop load generator for OpenAI-compatible streaming endpoints.

"Closed loop" means a fixed number of concurrent users (workers): each one
sends a request, waits for the full streamed reply, then immediately sends
the next. Concurrency stays exactly N for the whole level, which is what
makes "latency at concurrency N" a well-defined number. (The alternative,
open loop, fires requests at a fixed rate regardless of replies -- closer to
real traffic, but then concurrency drifts with server speed and two variants
end up being measured at different loads.)

Works against the Synapse gateway and against a vLLM server directly, since
both speak the same streaming format. Comparing the two at the same
concurrency gives the gateway's own overhead.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx

from synapse_bench.records import RequestRecord

# Errors that mean the measurement itself is invalid (not that the server
# was slow or failed). A run containing any of these must be thrown away.
INTEGRITY_ERRORS = ("cache not bypassed", "answered by mock", "answered by cache", "no usage in stream")


@dataclass
class Target:
    """Where to send requests, and what a valid answer looks like."""

    url: str  # full URL of the chat completions endpoint
    model: str
    headers: dict[str, str] = field(default_factory=dict)
    # Extra request fields, e.g. {"ignore_eos": True} for fixed-length outputs.
    extra_body: dict = field(default_factory=dict)
    # Through the gateway, every response must confirm x-cache: bypass and
    # name a real backend; a vLLM server queried directly sends neither.
    via_gateway: bool = True


@dataclass
class LevelResult:
    concurrency: int
    records: list[RequestRecord]
    wall_time_s: float
    # CPU time this client process used / wall time. Near or above 1.0 means
    # the load generator was saturating a CPU core and may itself have been
    # the bottleneck (see DECISIONS.md D5) -- reported so that's never hidden.
    client_cpu_fraction: float


def build_body(target: Target, prompt: str, max_tokens: int) -> dict:
    return {
        "model": target.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **target.extra_body,
    }


async def send_one(
    client: httpx.AsyncClient,
    target: Target,
    prompt: str,
    max_tokens: int,
    index: int,
    level_start: float,
) -> RequestRecord:
    """Send one streaming request and time it. Never raises: every failure
    becomes a record with ok=False and a short error string."""
    sent = time.perf_counter()
    status_code = x_cache = provider = None
    ttft = e2e = None
    prompt_tokens = output_tokens = None
    error = None

    try:
        async with client.stream("POST", target.url, json=build_body(target, prompt, max_tokens),
                                 headers=target.headers) as resp:
            status_code = resp.status_code
            x_cache = resp.headers.get("x-cache")
            if resp.status_code != 200:
                await resp.aread()
                error = f"HTTP {resp.status_code}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        e2e = time.perf_counter() - sent
                        break
                    event = json.loads(payload)
                    provider = event.get("provider", provider)
                    choices = event.get("choices") or []
                    if ttft is None and choices and (choices[0].get("delta") or {}).get("content"):
                        ttft = time.perf_counter() - sent
                    if event.get("usage"):
                        prompt_tokens = event["usage"].get("prompt_tokens")
                        output_tokens = event["usage"].get("completion_tokens")
    except httpx.TimeoutException:
        error = "timeout"
    except httpx.HTTPError as exc:
        error = type(exc).__name__
    except json.JSONDecodeError:
        error = "malformed stream event"

    if error is None:
        error = _validate(target, x_cache, provider, e2e, output_tokens)

    ok = error is None
    return RequestRecord(
        index=index,
        ok=ok,
        status_code=status_code,
        error=error,
        start_s=sent - level_start,
        ttft_s=ttft if ok else None,
        e2e_s=e2e if ok else None,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens if ok else None,
        x_cache=x_cache,
        provider=provider,
    )


def _validate(target: Target, x_cache, provider, e2e, output_tokens) -> str | None:
    """Reject responses that completed but can't be trusted as a measurement."""
    if e2e is None:
        return "incomplete stream (no [DONE])"
    if target.via_gateway:
        if x_cache != "bypass":
            return "cache not bypassed"
        if provider in ("mock", "cache"):
            return f"answered by {provider}"
    if output_tokens is None:
        return "no usage in stream"
    return None


async def run_level(
    target: Target,
    prompts: list[str],
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    warmup_requests: int = 0,
    timeout_s: float = 300.0,
    prompt_offset: int = 0,
) -> LevelResult:
    """Run one concurrency level: optional warm-up requests (discarded), then
    n_requests measured ones, always `concurrency` in flight. Request i uses
    prompts[(prompt_offset + i) % len(prompts)], so the sequence is the same
    for every model variant."""
    if concurrency < 1 or n_requests < 1:
        raise ValueError("concurrency and n_requests must be >= 1")
    if not prompts:
        raise ValueError("no prompts")

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:

        async def drain(indices: list[int], level_start: float, out: list[RequestRecord] | None) -> None:
            async def worker() -> None:
                while indices:
                    i = indices.pop(0)
                    prompt = prompts[(prompt_offset + i) % len(prompts)]
                    record = await send_one(client, target, prompt, max_tokens, i, level_start)
                    if out is not None:
                        out.append(record)

            await asyncio.gather(*(worker() for _ in range(concurrency)))

        if warmup_requests:
            # Warm-up takes prompts counting backwards from just before the
            # measured range, so as long as warmup + n_requests <= len(prompts)
            # no measured request repeats a warm-up prompt.
            warm = [-(k + 1) for k in range(warmup_requests)]
            await drain(warm, time.perf_counter(), out=None)

        records: list[RequestRecord] = []
        cpu_start = time.process_time()
        level_start = time.perf_counter()
        await drain(list(range(n_requests)), level_start, records)
        wall = time.perf_counter() - level_start
        cpu = time.process_time() - cpu_start

    records.sort(key=lambda r: r.index)
    return LevelResult(concurrency=concurrency, records=records, wall_time_s=wall, client_cpu_fraction=cpu / wall)
