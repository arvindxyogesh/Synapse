def test_chat_completion_requires_auth(client):
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 401


def test_chat_completion_mock_mode_roundtrip(client, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "llama3", "messages": [{"role": "user", "content": "hello"}]},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "mock"
    assert body["cached"] is False
    assert "hello" in body["choices"][0]["message"]["content"]
    assert resp.headers["x-cache"] == "miss"


def test_second_identical_request_is_cached(client, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": "llama3", "messages": [{"role": "user", "content": "what time is it"}]}

    first = client.post("/v1/chat/completions", json=payload, headers=headers)
    second = client.post("/v1/chat/completions", json=payload, headers=headers)

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.headers["x-cache"] == "hit"
    assert second.json()["cost_usd"] == 0.0


def test_revoked_key_is_rejected(client, admin_headers):
    create = client.post("/v1/admin/keys", json={"name": "will-revoke"}, headers=admin_headers)
    key_id, raw_key = create.json()["id"], create.json()["api_key"]
    client.post(f"/v1/admin/keys/{key_id}/revoke", headers=admin_headers)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 401


def test_stats_summary_reflects_requests(client, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    client.post(
        "/v1/chat/completions",
        json={"model": "llama3", "messages": [{"role": "user", "content": "stats check"}]},
        headers=headers,
    )
    resp = client.get("/v1/stats/summary")
    assert resp.status_code == 200
    assert resp.json()["total_requests"] >= 1


def test_cache_threshold_endpoint_lists_verified_models(client):
    from app.threshold_controller import get_threshold_controller

    get_threshold_controller().record_verification("stats-endpoint-model", is_false_positive=False)

    resp = client.get("/v1/stats/cache-threshold")
    assert resp.status_code == 200
    models = {row["model"] for row in resp.json()}
    assert "stats-endpoint-model" in models


def test_cache_threshold_endpoint_empty_when_nothing_verified(client):
    resp = client.get("/v1/stats/cache-threshold")
    assert resp.status_code == 200
    assert resp.json() == []
