"""x-synapse-cache: bypass must skip the cache completely -- no lookup (so
no hit), no store (so nothing is left behind for later requests), and no
embedding call (so cache-off measurements don't pay for one)."""

import pytest

import app.cache as cache_module


def _post(client, api_key, prompt, stream=False, cache_header=None):
    headers = {"Authorization": f"Bearer {api_key}"}
    if cache_header is not None:
        headers["x-synapse-cache"] = cache_header
    return client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": prompt}], "stream": stream},
        headers=headers,
    )


@pytest.mark.parametrize("stream", [False, True])
def test_bypass_skips_lookup_and_store(client, api_key, stream):
    prompt = f"bypass check stream={stream}"

    first = _post(client, api_key, prompt, stream, cache_header="bypass")
    assert first.status_code == 200
    assert first.headers["x-cache"] == "bypass"

    # Nothing was stored by the bypassed request, so a normal one misses...
    assert _post(client, api_key, prompt, stream).headers["x-cache"] == "miss"
    # ...and once something *is* cached, a bypassed request still ignores it.
    assert _post(client, api_key, prompt, stream).headers["x-cache"] == "hit"
    assert _post(client, api_key, prompt, stream, cache_header="bypass").headers["x-cache"] == "bypass"


def test_bypass_never_computes_an_embedding(client, api_key, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("cache was touched on a bypassed request")

    cache = cache_module.get_cache()
    monkeypatch.setattr(cache, "lookup", fail)
    monkeypatch.setattr(cache, "store", fail)

    assert _post(client, api_key, "no embedding please", cache_header="bypass").status_code == 200


def test_bypass_header_is_case_insensitive(client, api_key):
    assert _post(client, api_key, "case check", cache_header="BYPASS").headers["x-cache"] == "bypass"


def test_unknown_cache_header_value_is_rejected(client, api_key):
    resp = _post(client, api_key, "typo check", cache_header="bypas")
    assert resp.status_code == 400
