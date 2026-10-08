import json
import random
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.background import fire_and_forget
from app.cache import CacheEntry, get_cache, prompt_from_messages
from app.config import get_settings
from app.db import SessionLocal, get_db
from app.judge import judge_same_intent
from app.model_registry import get_registry
from app.models import ApiKey, RequestLog
from app.pricing import estimate_cost_usd
from app.providers import ProviderError, StreamChunk, run_completion, run_streaming_completion
from app.ratelimit import RateLimiter, get_rate_limiter
from app.schemas import ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Usage
from app.threshold_controller import get_threshold_controller

router = APIRouter(prefix="/v1", tags=["gateway"])


def _enforce_limits(api_key: ApiKey, limiter: RateLimiter) -> None:
    rate = limiter.check_rate_limit(api_key.id, api_key.rate_limit_per_minute)
    if not rate.allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(rate.retry_after_seconds)},
        )
    if not limiter.has_quota_remaining(api_key.id, api_key.monthly_quota_usd):
        raise HTTPException(status_code=429, detail="Monthly usage quota exceeded")


def _log_and_bill(
    db: Session,
    limiter: RateLimiter,
    api_key: ApiKey,
    provider: str,
    model: str,
    cached: bool,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_ms: float,
    status: str = "ok",
    ttft_ms: float | None = None,
) -> None:
    db.add(
        RequestLog(
            api_key_id=api_key.id,
            provider=provider,
            model=model,
            cached=cached,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            status=status,
        )
    )
    db.commit()
    limiter.record_spend(api_key.id, cost_usd)


async def _shadow_verify(model: str, source_prompt: str, new_prompt: str) -> None:
    """Runs after a cache hit has already been served: asks an independent
    LLM-judge whether the prompt that originally produced the cached
    response and the prompt that just hit it are really asking the same
    thing, and feeds the result into the per-model adaptive threshold
    controller. Never raises -- a broken judge call must never surface
    anywhere near a real request."""
    try:
        same_intent = await judge_same_intent(model, source_prompt, new_prompt)
        get_threshold_controller().record_verification(model, is_false_positive=not same_intent)
    except Exception:
        pass


def _backend_error(
    db: Session, limiter: RateLimiter, api_key: ApiKey, model: str, start: float, exc: ProviderError
) -> HTTPException:
    """Log a failed backend call as an error row (so error rates show up per
    model instead of disappearing) and build the 502 to return."""
    latency_ms = (time.perf_counter() - start) * 1000
    _log_and_bill(db, limiter, api_key, "error", model, False, 0, 0, 0.0, latency_ms, status="error")
    return HTTPException(status_code=502, detail=str(exc))


CACHE_BYPASS = "bypass"


def _should_use_cache(body: ChatCompletionRequest, cache_header: str | None) -> bool:
    """Whether this request may read from / write to the semantic cache.

    - `x-synapse-cache: bypass` skips it explicitly (used by the benchmark
      to measure the model rather than the cache). Any other value is
      rejected, so a typo can't silently leave the cache on.
    - The cache is keyed on (model, prompt) only. A reply generated under a
      max_tokens cap may be cut short, so caching it would hand a truncated
      answer to a later request that asked for no cap -- requests that set
      max_tokens skip the cache too (see DECISIONS.md D2)."""
    if cache_header is not None and cache_header.lower() != CACHE_BYPASS:
        raise HTTPException(status_code=400, detail=f"x-synapse-cache must be '{CACHE_BYPASS}' if set")
    if cache_header is not None:
        return False
    return body.max_tokens is None


def _cache_header(use_cache: bool, cached: bool) -> str:
    if not use_cache:
        return "bypass"
    return "hit" if cached else "miss"


def _maybe_shadow_verify(model: str, hit: CacheEntry, new_prompt: str) -> None:
    if not hit.source_prompt:
        return  # entry was cached before source_prompt existed -- nothing to compare against
    if random.random() < get_settings().shadow_verify_sample_rate:
        fire_and_forget(_shadow_verify(model, hit.source_prompt, new_prompt))


@router.get("/models")
def list_models(api_key: ApiKey = Depends(require_api_key)):
    """OpenAI-compatible model list (what `client.models.list()` calls).
    Lists the registry's models; without a registry, just DEFAULT_MODEL --
    any other name still works, it just isn't advertised."""
    names = sorted(get_registry()) or [get_settings().default_model]
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "created": 0, "owned_by": "synapse"} for name in names],
    }


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    api_key: ApiKey = Depends(require_api_key),
    db: Session = Depends(get_db),
    limiter: RateLimiter = Depends(get_rate_limiter),
    x_synapse_cache: str | None = Header(default=None),
):
    use_cache = _should_use_cache(body, x_synapse_cache)
    _enforce_limits(api_key, limiter)

    start = time.perf_counter()
    messages = [m.model_dump() for m in body.messages]
    prompt = prompt_from_messages(messages)
    cache = get_cache()
    hit = None
    if use_cache:
        threshold = get_threshold_controller().get_threshold(body.model)
        hit = cache.lookup(body.model, messages, threshold=threshold)

    if body.stream:
        stream = provider = None
        if hit is None:
            # Start the backend stream *before* sending any response, so a
            # backend that's down can still get a real error status.
            try:
                stream, provider = await run_streaming_completion(
                    body.model, messages, body.temperature, body.max_tokens
                )
            except ProviderError as exc:
                raise _backend_error(db, limiter, api_key, body.model, start, exc) from exc
        return _stream_chat_completion(
            body, messages, prompt, use_cache, hit, stream, provider, api_key, limiter, start
        )

    if hit:
        text, prompt_tokens, completion_tokens, provider, cached = (
            hit.response_text,
            hit.prompt_tokens,
            hit.completion_tokens,
            "cache",
            True,
        )
        _maybe_shadow_verify(body.model, hit, prompt)
    else:
        try:
            text, prompt_tokens, completion_tokens, provider = await run_completion(
                body.model, messages, body.temperature, body.max_tokens
            )
        except ProviderError as exc:
            raise _backend_error(db, limiter, api_key, body.model, start, exc) from exc
        cached = False
        if use_cache:
            cache.store(body.model, messages, CacheEntry(text, prompt_tokens, completion_tokens))

    latency_ms = (time.perf_counter() - start) * 1000
    cost_usd = 0.0 if cached else estimate_cost_usd(body.model, prompt_tokens, completion_tokens)

    _log_and_bill(
        db, limiter, api_key, provider, body.model, cached, prompt_tokens, completion_tokens, cost_usd, latency_ms
    )

    response.headers["x-cache"] = _cache_header(use_cache, cached)
    response.headers["x-provider"] = provider

    return ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=body.model,
        provider=provider,
        cached=cached,
        choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=text))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _stream_chat_completion(
    body: ChatCompletionRequest,
    messages: list[dict],
    prompt: str,
    use_cache: bool,
    hit: CacheEntry | None,
    stream: AsyncIterator[StreamChunk] | None,
    provider: str | None,
    api_key: ApiKey,
    limiter: RateLimiter,
    start: float,
) -> StreamingResponse:
    completion_id = str(uuid.uuid4())
    cache = get_cache()
    # Time to first token, as the gateway sees it: from `start` (after auth
    # and rate limiting) to the first chunk with actual text being handed to
    # the response stream. Clients measure their own TTFT too, which adds
    # network time; this one is what gets logged per model.
    ttft_ms: float | None = None

    def _chunk_event(delta: str, provider: str, cached: bool, finish_reason: str | None = None) -> str:
        # provider/cached are echoed on every chunk (not just in response
        # headers) so browser clients get them even when a CORS config or
        # intermediary proxy doesn't expose custom headers to JS.
        return _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.model,
                "provider": provider,
                "cached": cached,
                "choices": [{"index": 0, "delta": {"content": delta} if delta else {}, "finish_reason": finish_reason}],
            }
        )

    def _content_event(delta: str, provider: str, cached: bool) -> str:
        nonlocal ttft_ms
        if ttft_ms is None and delta:
            ttft_ms = (time.perf_counter() - start) * 1000
        return _chunk_event(delta, provider, cached)

    async def _generate() -> AsyncIterator[str]:
        # A fresh DB session, because this generator outlives the request's
        # own `db` dependency once headers have already been sent.
        db = SessionLocal()
        try:
            if hit is not None:
                words = hit.response_text.split(" ")
                for i, word in enumerate(words):
                    piece = word if i == len(words) - 1 else word + " "
                    yield _content_event(piece, "cache", True)
                yield _chunk_event("", "cache", True, finish_reason="stop")
                yield "data: [DONE]\n\n"

                _maybe_shadow_verify(body.model, hit, prompt)
                latency_ms = (time.perf_counter() - start) * 1000
                _log_and_bill(
                    db, limiter, api_key, "cache", body.model, True,
                    hit.prompt_tokens, hit.completion_tokens, 0.0, latency_ms, ttft_ms=ttft_ms,
                )
                return

            full_text = ""
            prompt_tokens = completion_tokens = 0
            async for piece in stream:
                full_text += piece.text
                if piece.done:
                    prompt_tokens = piece.prompt_tokens or 0
                    completion_tokens = piece.completion_tokens or 0
                    if piece.text:
                        yield _content_event(piece.text, provider, False)
                    yield _chunk_event("", provider, False, finish_reason="stop")
                else:
                    yield _content_event(piece.text, provider, False)
            yield "data: [DONE]\n\n"

            if use_cache:
                cache.store(body.model, messages, CacheEntry(full_text, prompt_tokens, completion_tokens))
            latency_ms = (time.perf_counter() - start) * 1000
            cost_usd = estimate_cost_usd(body.model, prompt_tokens, completion_tokens)
            _log_and_bill(
                db, limiter, api_key, provider, body.model, False,
                prompt_tokens, completion_tokens, cost_usd, latency_ms, ttft_ms=ttft_ms,
            )
        finally:
            db.close()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "x-cache": _cache_header(use_cache, hit is not None),
            "x-provider": "cache" if hit is not None else provider,
            "Cache-Control": "no-cache",
        },
    )
