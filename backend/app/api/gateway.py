import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.cache import CacheEntry, get_cache
from app.db import SessionLocal, get_db
from app.models import ApiKey, RequestLog
from app.pricing import estimate_cost_usd
from app.providers import run_completion, run_streaming_completion
from app.ratelimit import RateLimiter, get_rate_limiter
from app.schemas import ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Usage

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
            status="ok",
        )
    )
    db.commit()
    limiter.record_spend(api_key.id, cost_usd)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    api_key: ApiKey = Depends(require_api_key),
    db: Session = Depends(get_db),
    limiter: RateLimiter = Depends(get_rate_limiter),
):
    _enforce_limits(api_key, limiter)

    start = time.perf_counter()
    messages = [m.model_dump() for m in body.messages]
    cache = get_cache()
    hit = cache.lookup(body.model, messages)

    if body.stream:
        return _stream_chat_completion(body, messages, hit, api_key, limiter, start)

    if hit:
        text, prompt_tokens, completion_tokens, provider, cached = (
            hit.response_text,
            hit.prompt_tokens,
            hit.completion_tokens,
            "cache",
            True,
        )
    else:
        text, prompt_tokens, completion_tokens, provider = await run_completion(
            body.model, messages, body.temperature
        )
        cached = False
        cache.store(body.model, messages, CacheEntry(text, prompt_tokens, completion_tokens))

    latency_ms = (time.perf_counter() - start) * 1000
    cost_usd = 0.0 if cached else estimate_cost_usd(body.model, prompt_tokens, completion_tokens)

    _log_and_bill(
        db, limiter, api_key, provider, body.model, cached, prompt_tokens, completion_tokens, cost_usd, latency_ms
    )

    response.headers["x-cache"] = "hit" if cached else "miss"
    response.headers["x-provider"] = provider

    return ChatCompletionResponse(
        id=str(uuid.uuid4()),
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
    hit: CacheEntry | None,
    api_key: ApiKey,
    limiter: RateLimiter,
    start: float,
) -> StreamingResponse:
    completion_id = str(uuid.uuid4())
    cache = get_cache()

    def _chunk_event(delta: str, provider: str, cached: bool, finish_reason: str | None = None) -> str:
        # provider/cached are echoed on every chunk (not just in response
        # headers) so browser clients get them even when a CORS config or
        # intermediary proxy doesn't expose custom headers to JS.
        return _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": body.model,
                "provider": provider,
                "cached": cached,
                "choices": [{"index": 0, "delta": {"content": delta} if delta else {}, "finish_reason": finish_reason}],
            }
        )

    async def _generate() -> AsyncIterator[str]:
        # A fresh DB session, because this generator outlives the request's
        # own `db` dependency once headers have already been sent.
        db = SessionLocal()
        try:
            if hit is not None:
                words = hit.response_text.split(" ")
                for i, word in enumerate(words):
                    piece = word if i == len(words) - 1 else word + " "
                    yield _chunk_event(piece, "cache", True)
                yield _chunk_event("", "cache", True, finish_reason="stop")
                yield "data: [DONE]\n\n"

                latency_ms = (time.perf_counter() - start) * 1000
                _log_and_bill(
                    db, limiter, api_key, "cache", body.model, True,
                    hit.prompt_tokens, hit.completion_tokens, 0.0, latency_ms,
                )
                return

            stream, provider = await run_streaming_completion(body.model, messages, body.temperature)
            full_text = ""
            prompt_tokens = completion_tokens = 0
            async for piece in stream:
                full_text += piece.text
                if piece.done:
                    prompt_tokens = piece.prompt_tokens or 0
                    completion_tokens = piece.completion_tokens or 0
                    if piece.text:
                        yield _chunk_event(piece.text, provider, False)
                    yield _chunk_event("", provider, False, finish_reason="stop")
                else:
                    yield _chunk_event(piece.text, provider, False)
            yield "data: [DONE]\n\n"

            cache.store(body.model, messages, CacheEntry(full_text, prompt_tokens, completion_tokens))
            latency_ms = (time.perf_counter() - start) * 1000
            cost_usd = estimate_cost_usd(body.model, prompt_tokens, completion_tokens)
            _log_and_bill(
                db, limiter, api_key, provider, body.model, False,
                prompt_tokens, completion_tokens, cost_usd, latency_ms,
            )
        finally:
            db.close()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "x-cache": "hit" if hit is not None else "miss",
            "x-provider": "cache" if hit is not None else "pending",
            "Cache-Control": "no-cache",
        },
    )
