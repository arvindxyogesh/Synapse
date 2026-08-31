"""Pluggable embedder for the semantic cache.

Primary path: sentence-transformers (open-source, MIT-licensed, runs
locally -- no API key, no per-call cost). Falls back to a deterministic
hashing embedding when the model can't be loaded (e.g. no network to
download weights, as in a minimal CI container), so the cache -- and the
whole gateway -- still works without it.
"""

import hashlib
from functools import lru_cache

import numpy as np

from app.config import get_settings

_HASH_DIM = 256


def _hash_embed(text: str) -> np.ndarray:
    vec = np.zeros(_HASH_DIM, dtype=np.float32)
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vec[h % _HASH_DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Embedder:
    def __init__(self):
        self._model = None
        self._tried_load = False

    def _load(self):
        if self._tried_load:
            return
        self._tried_load = True
        try:
            from sentence_transformers import SentenceTransformer

            settings = get_settings()
            self._model = SentenceTransformer(settings.embedding_model_name)
        except Exception:
            self._model = None

    def embed(self, text: str) -> np.ndarray:
        self._load()
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        return _hash_embed(text)


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
