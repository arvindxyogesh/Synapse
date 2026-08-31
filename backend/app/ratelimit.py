"""Per-API-key rate limiting (requests/minute) and usage quotas (USD/month),
enforced with simple Redis counters.

Rate limiting uses a fixed 60-second window keyed by wall-clock minute --
cheap and good enough at this scale (a rolling window would be smoother
right at window edges, but isn't worth the extra Redis round trips here).

Quotas track cumulative estimated cost per calendar month per key. Spend is
only recorded for non-cached requests (cache hits cost $0), so a key that's
hitting cache heavily doesn't burn its quota.
"""

import time
from dataclasses import dataclass

import redis

from app.redis_client import get_redis

QUOTA_KEY_TTL_SECONDS = 60 * 60 * 24 * 40  # ~40 days, well past any month


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_seconds: int = 0


class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    def check_rate_limit(self, key_id: str, limit_per_minute: int | None) -> RateLimitResult:
        """Increments the current-minute counter for this key and reports
        whether the request is within its per-minute limit. Always
        increments (even when over limit) so bursts don't get free retries
        within the same window."""
        if not limit_per_minute:
            return RateLimitResult(allowed=True)

        now = time.time()
        window = int(now // 60)
        redis_key = f"ratelimit:{key_id}:{window}"
        count = self._redis.incr(redis_key)
        if count == 1:
            self._redis.expire(redis_key, 60)

        if count > limit_per_minute:
            retry_after = 60 - int(now % 60)
            return RateLimitResult(allowed=False, retry_after_seconds=max(retry_after, 1))
        return RateLimitResult(allowed=True)

    def _quota_key(self, key_id: str) -> str:
        period = time.strftime("%Y-%m", time.gmtime())
        return f"quota:{key_id}:{period}"

    def has_quota_remaining(self, key_id: str, monthly_quota_usd: float | None) -> bool:
        if not monthly_quota_usd:
            return True
        spent = float(self._redis.get(self._quota_key(key_id)) or 0.0)
        return spent < monthly_quota_usd

    def record_spend(self, key_id: str, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        redis_key = self._quota_key(key_id)
        self._redis.incrbyfloat(redis_key, cost_usd)
        self._redis.expire(redis_key, QUOTA_KEY_TTL_SECONDS)

    def current_spend(self, key_id: str) -> float:
        return float(self._redis.get(self._quota_key(key_id)) or 0.0)


def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_redis())
