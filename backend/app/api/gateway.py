import time
import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.cache import CacheEntry, get_cache
from app.db import get_db
from app.models import ApiKey, RequestLog
from app.pricing import estimate_cost_usd
from app.providers import run_completion
from app.schemas import ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Usage

router = APIRouter(prefix="/v1", tags=["gateway"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    api_key: ApiKey = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    start = time.perf_counter()
    messages = [m.model_dump() for m in body.messages]
    cache = get_cache()

    hit = cache.lookup(body.model, messages)
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

    log = RequestLog(
        api_key_id=api_key.id,
        provider=provider,
        model=body.model,
        cached=cached,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        status="ok",
    )
    db.add(log)
    db.commit()

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
