"""Pluggable embedder for the semantic cache.

Primary path: sentence-transformers (open-source, MIT-licensed, runs
locally -- no API key, no per-call cost). Falls back to a deterministic
hashing embedding when the model can't be loaded (e.g. no network to
download weights, as in a minimal CI container), so the cache -- and the
whole gateway -- still works without it.
"""

import hashlib
import logging
from functools import lru_cache

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

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
            # This used to fail silently -- degrading the entire semantic
            # cache to a much cruder bag-of-words fallback with zero
            # operational visibility that it happened (found the hard way:
            # a whole benchmark run was silently using the fallback because
            # the real model's weights couldn't be downloaded, and nothing
            # said so). A cache that can't recognize paraphrases is a
            # correctness regression, not just a performance one -- log it
            # loudly enough that it shows up in normal server logs.
            logger.warning(
                "Falling back to the hashing embedder -- could not load '%s' "
                "(no network access to download weights, or another load "
                "failure). Semantic cache hit rate on genuine paraphrases "
                "will be substantially worse than with the real model.",
                get_settings().embedding_model_name,
                exc_info=True,
            )

    def embed(self, text: str) -> np.ndarray:
        self._load()
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        return _hash_embed(text)

    @property
    def backend(self) -> str:
        """'sentence-transformers' or 'hash-fallback', surfaced on
        GET /health specifically so silently running in degraded mode
        (see the warning in _load()) is visible without reading logs.
        Deliberately does *not* trigger loading -- a liveness endpoint
        shouldn't eagerly do a first-time model download/load; it reports
        'not-yet-initialized' until the first real embed() call happens."""
        if not self._tried_load:
            return "not-yet-initialized"
        return "sentence-transformers" if self._model is not None else "hash-fallback"


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
