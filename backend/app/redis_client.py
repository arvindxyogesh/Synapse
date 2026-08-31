"""Shared Redis connection, used by the semantic cache and the rate limiter
so both share one connection pool instead of opening their own."""

from functools import lru_cache

import redis

from app.config import get_settings


@lru_cache
def get_redis() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)
