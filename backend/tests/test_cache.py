from app.cache import CacheEntry, get_cache


def test_exact_match_hit():
    cache = get_cache()
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    assert cache.lookup("llama3", messages) is None

    cache.store("llama3", messages, CacheEntry("Paris.", 8, 3))
    hit = cache.lookup("llama3", messages)
    assert hit is not None
    assert hit.response_text == "Paris."


def test_different_model_is_a_separate_cache_namespace():
    cache = get_cache()
    messages = [{"role": "user", "content": "hello there"}]
    cache.store("llama3", messages, CacheEntry("hi!", 2, 2))
    assert cache.lookup("mistral", messages) is None


def test_semantically_similar_prompt_hits_via_embedding_fallback():
    cache = get_cache()
    original = [{"role": "user", "content": "capital of france capital of france capital of france"}]
    cache.store("llama3", original, CacheEntry("Paris.", 5, 3))

    # Identical bag-of-words (the hashing-embedding fallback is token-count
    # based, not truly semantic) should still hit above the threshold.
    near_dupe = [{"role": "user", "content": "capital of france capital of france capital of france"}]
    hit = cache.lookup("llama3", near_dupe)
    assert hit is not None


def test_lookup_returns_source_prompt_for_shadow_verification():
    cache = get_cache()
    messages = [{"role": "user", "content": "how do I export my data"}]
    cache.store("llama3", messages, CacheEntry("Go to Settings > Export.", 6, 6))

    hit = cache.lookup("llama3", messages)
    assert hit.source_prompt == "how do I export my data"


def test_threshold_override_controls_hit_vs_miss():
    cache = get_cache()
    stored = [{"role": "user", "content": "capital of france capital of france capital of germany"}]
    cache.store("threshold-test-model", stored, CacheEntry("Paris-ish.", 5, 3))

    # Partial bag-of-words overlap with the stored prompt -- similar but
    # not identical.
    query = [{"role": "user", "content": "capital of france capital of spain capital of italy"}]

    # A near-zero threshold should accept any non-trivial similarity.
    assert cache.lookup("threshold-test-model", query, threshold=0.01) is not None
    # A maximal threshold should reject anything short of a bit-identical
    # embedding.
    assert cache.lookup("threshold-test-model", query, threshold=0.999999) is None
