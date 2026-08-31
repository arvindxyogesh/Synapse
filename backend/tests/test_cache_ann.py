"""Unit tests for the ANN (RediSearch/vector-search) branch of SemanticCache
using a mocked Redis client -- fakeredis (used elsewhere in the test suite)
doesn't implement FT.* commands, so it always exercises the linear-scan
fallback instead. These tests cover the ANN code path's own logic:
index creation, the KNN query/params shape, and threshold handling."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from redis.exceptions import ResponseError

from app.cache import CacheEntry, SemanticCache


def _make_redis_with_ann():
    redis_mock = MagicMock()
    redis_mock.execute_command.return_value = []  # FT._LIST succeeds -> module present
    redis_mock.get.return_value = None  # no exact-match hit

    ft_mock = MagicMock()
    ft_mock.info.side_effect = ResponseError("Unknown index name")  # index doesn't exist yet
    redis_mock.ft.return_value = ft_mock
    return redis_mock, ft_mock


def test_store_uses_ann_hset_when_vector_search_available():
    redis_mock, ft_mock = _make_redis_with_ann()
    cache = SemanticCache(redis_client=redis_mock, ttl_seconds=3600, similarity_threshold=0.9)

    cache.store("llama3", [{"role": "user", "content": "hi"}], CacheEntry("hello!", 2, 2))

    ft_mock.create_index.assert_called_once()
    assert redis_mock.hset.called
    doc_key, kwargs = redis_mock.hset.call_args.args[0], redis_mock.hset.call_args.kwargs
    assert doc_key.startswith("cache:vec:llama3:")
    assert set(kwargs["mapping"]) == {
        "embedding",
        "response_text",
        "prompt_tokens",
        "completion_tokens",
        "source_prompt",
    }
    # the linear-scan fallback path should not have been used
    redis_mock.lpush.assert_not_called()


def test_lookup_uses_ann_search_and_hits_above_threshold():
    redis_mock, ft_mock = _make_redis_with_ann()
    cache = SemanticCache(redis_client=redis_mock, ttl_seconds=3600, similarity_threshold=0.9)

    doc = SimpleNamespace(score="0.02", response_text="Paris.", prompt_tokens="3", completion_tokens="2")
    ft_mock.search.return_value = SimpleNamespace(docs=[doc])

    hit = cache.lookup("llama3", [{"role": "user", "content": "capital of france"}])
    assert hit is not None
    assert hit.response_text == "Paris."
    assert hit.prompt_tokens == 3

    _, kwargs = ft_mock.search.call_args
    assert "vec" in kwargs["query_params"]
    ft_mock.create_index.assert_called_once()  # index was created on first use


def test_lookup_returns_none_below_similarity_threshold():
    redis_mock, ft_mock = _make_redis_with_ann()
    cache = SemanticCache(redis_client=redis_mock, ttl_seconds=3600, similarity_threshold=0.9)

    # distance 0.5 -> similarity 0.5, below the 0.9 threshold
    doc = SimpleNamespace(score="0.5", response_text="irrelevant", prompt_tokens="1", completion_tokens="1")
    ft_mock.search.return_value = SimpleNamespace(docs=[doc])

    assert cache.lookup("llama3", [{"role": "user", "content": "capital of france"}]) is None


def test_lookup_returns_none_with_no_docs():
    redis_mock, ft_mock = _make_redis_with_ann()
    cache = SemanticCache(redis_client=redis_mock, ttl_seconds=3600, similarity_threshold=0.9)
    ft_mock.search.return_value = SimpleNamespace(docs=[])

    assert cache.lookup("llama3", [{"role": "user", "content": "anything"}]) is None


def test_ann_module_missing_falls_back_to_linear_scan():
    redis_mock = MagicMock()
    redis_mock.execute_command.side_effect = ResponseError("unknown command 'FT._LIST'")
    redis_mock.get.return_value = None

    cache = SemanticCache(redis_client=redis_mock, ttl_seconds=3600, similarity_threshold=0.9)
    cache.store("llama3", [{"role": "user", "content": "hi"}], CacheEntry("hello!", 2, 2))

    redis_mock.ft.assert_not_called()
    assert redis_mock.lpush.called


def test_index_reused_across_calls_for_same_model_and_dimension():
    redis_mock, ft_mock = _make_redis_with_ann()
    cache = SemanticCache(redis_client=redis_mock, ttl_seconds=3600, similarity_threshold=0.9)

    cache.store("llama3", [{"role": "user", "content": "one"}], CacheEntry("a", 1, 1))
    cache.store("llama3", [{"role": "user", "content": "two"}], CacheEntry("b", 1, 1))

    # info() is only consulted once per (model, dim); after that the index
    # name is cached in-process and create_index isn't called again.
    assert ft_mock.create_index.call_count == 1
