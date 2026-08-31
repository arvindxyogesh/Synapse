from app.pricing import estimate_cost_usd


def test_known_model_rate():
    cost = estimate_cost_usd("llama3", 1000, 1000)
    assert cost == 0.0002 + 0.0002


def test_unknown_model_falls_back_to_default_rate():
    cost = estimate_cost_usd("some-unlisted-model", 1000, 0)
    assert cost == 0.0002


def test_zero_tokens_is_zero_cost():
    assert estimate_cost_usd("llama3", 0, 0) == 0.0
