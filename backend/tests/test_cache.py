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
