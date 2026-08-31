"""Per-model adaptive similarity threshold for the semantic cache.

Rather than one fixed cache_similarity_threshold for every model, each
model's threshold is tuned online by a simple closed-loop controller: a
sample of cache hits is shadow-verified (app/judge.py) against an
independent LLM-judge signal, and the observed false-positive rate nudges
the threshold up (stricter -- fewer wrong hits) or down (looser -- more
hits) toward a target false-positive rate. Small bounded steps + an
exponential moving average keep it from oscillating on noisy samples.

State lives in Redis (JSON blobs, one per model) so it survives restarts
and is shared across gateway workers.
"""

import json
import time
from dataclasses import asdict, dataclass

import redis

from app.config import get_settings
from app.redis_client import get_redis

STATE_KEY_PREFIX = "threshold:state"


@dataclass
class ThresholdState:
    model: str
    threshold: float
    fp_rate_ewma: float = 0.0
    verified_count: int = 0
    samples_since_adjustment: int = 0
    last_adjusted_at: float | None = None
    last_direction: str | None = None  # "up" | "down" | None


class ThresholdController:
    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    def _key(self, model: str) -> str:
        return f"{STATE_KEY_PREFIX}:{model}"

    def _load(self, model: str) -> ThresholdState:
        raw = self._redis.get(self._key(model))
        if raw:
            return ThresholdState(**json.loads(raw))
        return ThresholdState(model=model, threshold=get_settings().cache_similarity_threshold)

    def _save(self, state: ThresholdState) -> None:
        self._redis.set(self._key(state.model), json.dumps(asdict(state)))

    def get_threshold(self, model: str) -> float:
        if not get_settings().adaptive_threshold_enabled:
            return get_settings().cache_similarity_threshold
        return self._load(model).threshold

    def get_state(self, model: str) -> ThresholdState:
        return self._load(model)

    def all_models(self) -> list[str]:
        prefix = f"{STATE_KEY_PREFIX}:"
        return [key[len(prefix) :] for key in self._redis.scan_iter(f"{prefix}*")]

    def record_verification(self, model: str, is_false_positive: bool) -> ThresholdState:
        settings = get_settings()
        state = self._load(model)

        observation = 1.0 if is_false_positive else 0.0
        alpha = settings.threshold_fp_rate_ewma_alpha
        state.fp_rate_ewma = observation if state.verified_count == 0 else (
            alpha * observation + (1 - alpha) * state.fp_rate_ewma
        )
        state.verified_count += 1
        state.samples_since_adjustment += 1

        ready = (
            state.verified_count >= settings.threshold_min_samples_before_adjust
            and state.samples_since_adjustment >= settings.threshold_adjustment_cooldown_samples
        )
        if ready:
            target = settings.target_false_positive_rate
            step = settings.cache_threshold_step
            if state.fp_rate_ewma > target:
                # Too many wrong hits -- tighten so fewer near-miss prompts
                # qualify as a hit at all.
                state.threshold = min(settings.cache_threshold_max, state.threshold + step)
                state.last_direction, state.last_adjusted_at = "up", time.time()
                state.samples_since_adjustment = 0
            elif state.fp_rate_ewma < target * 0.5:
                # Comfortably under target -- ease back down so the cache
                # doesn't stay needlessly strict (and under-utilized)
                # forever once it's earned some slack.
                state.threshold = max(settings.cache_threshold_min, state.threshold - step)
                state.last_direction, state.last_adjusted_at = "down", time.time()
                state.samples_since_adjustment = 0
            # else: fp rate is in the dead zone between target and
            # target/2 -- leave it alone and keep accumulating samples.

        self._save(state)
        return state


def get_threshold_controller() -> ThresholdController:
    return ThresholdController(get_redis())
