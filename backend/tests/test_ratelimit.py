def test_rate_limit_blocks_after_threshold(client, admin_headers):
    created = client.post(
        "/v1/admin/keys",
        json={"name": "rate-limited", "rate_limit_per_minute": 2},
        headers=admin_headers,
    )
    assert created.status_code == 200
    headers = {"Authorization": f"Bearer {created.json()['api_key']}"}

    payload = {"model": "llama3", "messages": [{"role": "user", "content": "one"}]}
    assert client.post("/v1/chat/completions", json=payload, headers=headers).status_code == 200

    payload2 = {"model": "llama3", "messages": [{"role": "user", "content": "two"}]}
    assert client.post("/v1/chat/completions", json=payload2, headers=headers).status_code == 200

    payload3 = {"model": "llama3", "messages": [{"role": "user", "content": "three"}]}
    third = client.post("/v1/chat/completions", json=payload3, headers=headers)
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_unlimited_key_is_not_rate_limited(client, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    for i in range(5):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llama3", "messages": [{"role": "user", "content": f"msg {i}"}]},
            headers=headers,
        )
        assert resp.status_code == 200


def test_monthly_quota_blocks_once_exceeded(client, admin_headers):
    created = client.post(
        "/v1/admin/keys",
        json={"name": "quota-limited", "monthly_quota_usd": 0.0000001},
        headers=admin_headers,
    )
    headers = {"Authorization": f"Bearer {created.json()['api_key']}"}

    first = client.post(
        "/v1/chat/completions",
        json={"model": "llama3", "messages": [{"role": "user", "content": "spend some quota"}]},
        headers=headers,
    )
    assert first.status_code == 200
    assert first.json()["cost_usd"] > 0

    second = client.post(
        "/v1/chat/completions",
        json={"model": "llama3", "messages": [{"role": "user", "content": "should be blocked now"}]},
        headers=headers,
    )
    assert second.status_code == 429


def test_admin_can_update_and_clear_limits(client, admin_headers):
    created = client.post("/v1/admin/keys", json={"name": "adjustable"}, headers=admin_headers)
    key_id = created.json()["id"]
    assert created.json()["rate_limit_per_minute"] is None

    updated = client.patch(
        f"/v1/admin/keys/{key_id}",
        json={"rate_limit_per_minute": 10, "monthly_quota_usd": 5.0},
        headers=admin_headers,
    )
    assert updated.status_code == 200
    assert updated.json()["rate_limit_per_minute"] == 10
    assert updated.json()["monthly_quota_usd"] == 5.0

    cleared = client.patch(
        f"/v1/admin/keys/{key_id}",
        json={"clear_rate_limit": True, "clear_monthly_quota": True},
        headers=admin_headers,
    )
    assert cleared.status_code == 200
    assert cleared.json()["rate_limit_per_minute"] is None
    assert cleared.json()["monthly_quota_usd"] is None
