from app.embeddings import Embedder, cosine_similarity


def test_backend_reports_not_yet_initialized_before_first_embed():
    e = Embedder()
    assert e.backend == "not-yet-initialized"


def test_backend_reports_hash_fallback_when_real_model_unavailable():
    # requirements-dev.txt deliberately excludes sentence-transformers (see
    # its comment) so this exercises the same fallback path CI runs under.
    e = Embedder()
    e.embed("hello")
    assert e.backend == "hash-fallback"


def test_hash_fallback_embeddings_are_deterministic_and_normalized():
    e = Embedder()
    v1 = e.embed("How do I reset my password?")
    v2 = e.embed("How do I reset my password?")
    assert (v1 == v2).all()
    assert abs(cosine_similarity(v1, v1) - 1.0) < 1e-5


def test_cosine_similarity_mismatched_shapes_returns_zero():
    import numpy as np

    assert cosine_similarity(np.zeros(4), np.zeros(8)) == 0.0
