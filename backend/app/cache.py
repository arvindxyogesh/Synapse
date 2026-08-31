"""Semantic response cache backed by Redis.

Exact-hash matches are checked first (cheap). Beyond that, prompts are
compared by embedding similarity to reuse responses for near-duplicate
queries. Two implementations back that similarity search:

- ANN (preferred): Redis's vector search (the "Redis Query Engine" /
  RediSearch module, bundled in the `redis/redis-stack-server` image) with
  an HNSW index -- sub-linear lookup, one index per (model, embedding
  dimension).
- Linear scan (fallback): a bounded per-model candidate list compared by
  brute-force cosine similarity in Python. Used automatically whenever the
  vector search module isn't loaded -- e.g. plain `redis:7-alpine`, or
  fakeredis in tests -- so the cache (and the whole gateway) keeps working
  with zero extra setup.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.config import get_settings
from app.embeddings import cosine_similarity, get_embedder
from app.redis_client import get_redis

MAX_CANDIDATES_PER_MODEL = 200
ANN_CANDIDATE_KEY_PREFIX = "cache:vec"


@dataclass
class CacheEntry:
    response_text: str
    prompt_tokens: int
    completion_tokens: int


class SemanticCache:
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int, similarity_threshold: float):
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold
        self._embedder = get_embedder()
        self._known_indexes: set[str] = set()
        self._ann_supported = self._probe_ann_support()

    def _probe_ann_support(self) -> bool:
        try:
            self._redis.execute_command("FT._LIST")
            return True
        except Exception:
            return False

    def _prompt_key(self, messages: list[dict]) -> str:
        return " ".join(m["content"] for m in messages if m["role"] == "user")

    def _exact_key(self, model: str, prompt: str) -> str:
        digest = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
        return f"cache:exact:{model}:{digest}"

    def _candidates_key(self, model: str) -> str:
        return f"cache:candidates:{model}"

    # -- Vector index (ANN) -------------------------------------------------

    def _index_name(self, model: str, dim: int) -> str:
        return f"idx:cache:{model}:{dim}"

    def _key_prefix(self, model: str) -> str:
        return f"{ANN_CANDIDATE_KEY_PREFIX}:{model}:"

    def _ensure_index(self, model: str, dim: int) -> bool:
        """Best-effort: returns True if a vector index is ready to use for
        this (model, embedding-dimension) pair, creating it on first use.
        Falls back (permanently, for this process) to linear scan the
        moment any FT.* command fails -- covers both "module not loaded"
        and any other unexpected server-side rejection."""
        if not self._ann_supported:
            return False

        name = self._index_name(model, dim)
        if name in self._known_indexes:
            return True

        try:
            self._redis.ft(name).info()
            self._known_indexes.add(name)
            return True
        except ResponseError:
            pass  # doesn't exist yet -- fall through and create it
        except Exception:
            self._ann_supported = False
            return False

        try:
            schema = (
                VectorField(
                    "embedding",
                    "HNSW",
                    {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                ),
            )
            self._redis.ft(name).create_index(
                schema,
                definition=IndexDefinition(prefix=[self._key_prefix(model)], index_type=IndexType.HASH),
            )
            self._known_indexes.add(name)
            return True
        except Exception:
            self._ann_supported = False
            return False

    def _ann_lookup(self, model: str, query_vec: np.ndarray) -> CacheEntry | None:
        dim = query_vec.shape[0]
        query = (
            Query("*=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("score", "response_text", "prompt_tokens", "completion_tokens")
            .dialect(2)
        )
        params = {"vec": query_vec.astype("float32").tobytes()}
        try:
            result = self._redis.ft(self._index_name(model, dim)).search(query, query_params=params)
        except Exception:
            self._ann_supported = False
            return None

        if not result.docs:
            return None

        doc = result.docs[0]
        # HNSW + COSINE distance metric: score is cosine *distance*, so
        # similarity = 1 - distance.
        similarity = 1.0 - float(doc.score)
        if similarity < self._threshold:
            return None
        return CacheEntry(
            response_text=doc.response_text,
            prompt_tokens=int(doc.prompt_tokens),
            completion_tokens=int(doc.completion_tokens),
        )

    def _ann_store(self, model: str, prompt: str, query_vec: np.ndarray, entry: CacheEntry) -> None:
        entry_id = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        doc_key = f"{self._key_prefix(model)}{entry_id}"
        self._redis.hset(
            doc_key,
            mapping={
                "embedding": query_vec.astype("float32").tobytes(),
                "response_text": entry.response_text,
                "prompt_tokens": entry.prompt_tokens,
                "completion_tokens": entry.completion_tokens,
            },
        )
        self._redis.expire(doc_key, self._ttl)

    # -- Linear scan (fallback) ----------------------------------------------

    def _linear_scan_lookup(self, model: str, query_vec: np.ndarray) -> CacheEntry | None:
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

    def _linear_scan_store(self, model: str, prompt: str, query_vec: np.ndarray, entry: CacheEntry) -> None:
        entry_id = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        full_payload = json.dumps(
            {
                "response_text": entry.response_text,
                "prompt_tokens": entry.prompt_tokens,
                "completion_tokens": entry.completion_tokens,
                "embedding": _encode_vec(query_vec),
            }
        )
        self._redis.set(f"cache:entry:{entry_id}", full_payload, ex=self._ttl)
        self._redis.lrem(self._candidates_key(model), 0, entry_id)
        self._redis.lpush(self._candidates_key(model), entry_id)
        self._redis.ltrim(self._candidates_key(model), 0, MAX_CANDIDATES_PER_MODEL - 1)
        self._redis.expire(self._candidates_key(model), self._ttl)

    # -- Public API -----------------------------------------------------------

    def lookup(self, model: str, messages: list[dict]) -> CacheEntry | None:
        prompt = self._prompt_key(messages)

        exact = self._redis.get(self._exact_key(model, prompt))
        if exact:
            data = json.loads(exact)
            return CacheEntry(**data)

        query_vec = self._embedder.embed(prompt)
        if self._ensure_index(model, query_vec.shape[0]):
            return self._ann_lookup(model, query_vec)
        return self._linear_scan_lookup(model, query_vec)

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

        query_vec = self._embedder.embed(prompt)
        if self._ensure_index(model, query_vec.shape[0]):
            self._ann_store(model, prompt, query_vec, entry)
        else:
            self._linear_scan_store(model, prompt, query_vec, entry)


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
            redis_client=get_redis(),
            ttl_seconds=settings.cache_ttl_seconds,
            similarity_threshold=settings.cache_similarity_threshold,
        )
    return _cache_singleton
