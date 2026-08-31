"""Semantic response cache backed by Redis.

Exact-hash matches are checked first (cheap). If no exact match, we compare
the query's embedding against a bounded set of recent cached embeddings per
model with cosine similarity, and reuse the response above a threshold.

This is a linear scan over a per-model candidate set, which is fine at demo
scale. A production swap-in would use pgvector or a Redis vector index
(RediSearch) for approximate nearest-neighbor search instead.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import redis

from app.config import get_settings
from app.embeddings import cosine_similarity, get_embedder

MAX_CANDIDATES_PER_MODEL = 200


@dataclass
class CacheEntry:
    response_text: str
    prompt_tokens: int
    completion_tokens: int


class SemanticCache:
    def __init__(self, redis_url: str, ttl_seconds: int, similarity_threshold: float):
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold
        self._embedder = get_embedder()

    def _prompt_key(self, messages: list[dict]) -> str:
        return " ".join(m["content"] for m in messages if m["role"] == "user")

    def _exact_key(self, model: str, prompt: str) -> str:
        digest = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
        return f"cache:exact:{model}:{digest}"

    def _candidates_key(self, model: str) -> str:
        return f"cache:candidates:{model}"

    def lookup(self, model: str, messages: list[dict]) -> CacheEntry | None:
        prompt = self._prompt_key(messages)

        exact = self._redis.get(self._exact_key(model, prompt))
        if exact:
            data = json.loads(exact)
            return CacheEntry(**data)

        query_vec = self._embedder.embed(prompt)
        candidate_ids = self._redis.lrange(self._candidates_key(model), 0, MAX_CANDIDATES_PER_MODEL - 1)
        best_score, best_entry = 0.0, None
        for entry_id in candidate_ids:
            raw = self._redis.get(f"cache:entry:{entry_id}")
            if not raw:
                continue
            record = json.loads(raw)
            score = cosine_similarity(query_vec, _decode_vec(record["embedding"]))
            if score > best_score:
                best_score, best_entry = score, record

        if best_entry and best_score >= self._threshold:
            return CacheEntry(
                response_text=best_entry["response_text"],
                prompt_tokens=best_entry["prompt_tokens"],
                completion_tokens=best_entry["completion_tokens"],
            )
        return None

    def store(self, model: str, messages: list[dict], entry: CacheEntry) -> None:
        prompt = self._prompt_key(messages)
        exact_payload = json.dumps(
            {
                "response_text": entry.response_text,
                "prompt_tokens": entry.prompt_tokens,
                "completion_tokens": entry.completion_tokens,
            }
        )
        self._redis.set(self._exact_key(model, prompt), exact_payload, ex=self._ttl)

        entry_id = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        vec = self._embedder.embed(prompt)
        full_payload = json.dumps(
            {
                "response_text": entry.response_text,
                "prompt_tokens": entry.prompt_tokens,
                "completion_tokens": entry.completion_tokens,
                "embedding": _encode_vec(vec),
            }
        )
        self._redis.set(f"cache:entry:{entry_id}", full_payload, ex=self._ttl)
        self._redis.lrem(self._candidates_key(model), 0, entry_id)
        self._redis.lpush(self._candidates_key(model), entry_id)
        self._redis.ltrim(self._candidates_key(model), 0, MAX_CANDIDATES_PER_MODEL - 1)
        self._redis.expire(self._candidates_key(model), self._ttl)


def _encode_vec(vec) -> list[float]:
    return [round(float(x), 6) for x in vec]


def _decode_vec(values: list[float]) -> np.ndarray:
    return np.array(values, dtype="float32")


_cache_singleton: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache_singleton
    if _cache_singleton is None:
        settings = get_settings()
        _cache_singleton = SemanticCache(
            redis_url=settings.redis_url,
            ttl_seconds=settings.cache_ttl_seconds,
            similarity_threshold=settings.cache_similarity_threshold,
        )
    return _cache_singleton
