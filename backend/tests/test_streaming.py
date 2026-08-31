import json


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or not block.startswith("data: "):
            continue
        payload = block[len("data: ") :]
        if payload == "[DONE]":
            events.append({"done": True})
        else:
            events.append(json.loads(payload))
    return events


def test_streaming_miss_yields_chunks_and_caches(client, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": "llama3", "messages": [{"role": "user", "content": "stream this"}], "stream": True}

    with client.stream("POST", "/v1/chat/completions", json=payload, headers=headers) as resp:
        assert resp.status_code == 200
        assert resp.headers["x-cache"] == "miss"
        body = "".join(resp.iter_text())

    events = _parse_sse(body)
    assert events[-1] == {"done": True}
    assert any(e.get("choices", [{}])[0].get("finish_reason") == "stop" for e in events if "choices" in e)

    full_text = "".join(
        e["choices"][0]["delta"].get("content", "") for e in events if "choices" in e
    )
    assert "stream this" in full_text

    # Cache should now hold this exact prompt.
    follow_up = client.post(
        "/v1/chat/completions",
        json={"model": "llama3", "messages": [{"role": "user", "content": "stream this"}]},
        headers=headers,
    )
    assert follow_up.json()["cached"] is True


def test_streaming_hit_serves_from_cache(client, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": "llama3", "messages": [{"role": "user", "content": "prime the cache"}]}
    client.post("/v1/chat/completions", json=payload, headers=headers)

    stream_payload = {**payload, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=stream_payload, headers=headers) as resp:
        assert resp.headers["x-cache"] == "hit"
        body = "".join(resp.iter_text())

    events = _parse_sse(body)
    full_text = "".join(
        e["choices"][0]["delta"].get("content", "") for e in events if "choices" in e
    )
    assert "prime the cache" in full_text


def test_streaming_requires_auth(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 401
