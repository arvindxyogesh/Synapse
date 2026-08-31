"""Integration coverage for the gateway's shadow-verification wiring: a
cache hit should (probabilistically) trigger a background LLM-judge check
that feeds the adaptive threshold controller, without adding latency or
ever surfacing an error to the caller.
"""

import asyncio

import pytest

from app.cache import CacheEntry
from app.config import get_settings
from app.threshold_controller import get_threshold_controller


async def _drain_background_tasks():
    """Shadow verification runs as a fire-and-forget asyncio task (see
    app/background.py) -- give the event loop a couple of ticks so it
    actually finishes before we assert on its effects."""
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cache_hit_with_full_sampling_updates_threshold_controller(client, api_key, monkeypatch):
    monkeypatch.setattr(get_settings(), "shadow_verify_sample_rate", 1.0)
    headers = {"Authorization": f"Bearer {api_key}"}

    payload = {"model": "llama3", "messages": [{"role": "user", "content": "how do I export my data"}]}
    client.post("/v1/chat/completions", json=payload, headers=headers)  # populates the cache
    resp = client.post("/v1/chat/completions", json=payload, headers=headers)  # exact-match hit
    assert resp.json()["cached"] is True

    await _drain_background_tasks()

    state = get_threshold_controller().get_state("llama3")
    assert state.verified_count == 1
    # Identical prompt -> judge_same_intent short-circuits to True -> not a
    # false positive.
    assert state.fp_rate_ewma == 0.0


@pytest.mark.asyncio
async def test_zero_sample_rate_never_triggers_verification(client, api_key, monkeypatch):
    monkeypatch.setattr(get_settings(), "shadow_verify_sample_rate", 0.0)
    headers = {"Authorization": f"Bearer {api_key}"}

    payload = {"model": "mistral", "messages": [{"role": "user", "content": "how do I add a team member"}]}
    client.post("/v1/chat/completions", json=payload, headers=headers)
    client.post("/v1/chat/completions", json=payload, headers=headers)

    await _drain_background_tasks()

    assert "mistral" not in get_threshold_controller().all_models()


@pytest.mark.asyncio
async def test_entry_without_source_prompt_skips_verification(monkeypatch):
    """Defensive: an entry stored before source_prompt existed (or by an
    older client) shouldn't crash verification -- it should just be
    skipped. SemanticCache.store() always fills source_prompt itself now,
    so this exercises the gateway's own guard directly rather than trying
    to contrive a cache miss that skips it."""
    from app.api.gateway import _maybe_shadow_verify

    monkeypatch.setattr(get_settings(), "shadow_verify_sample_rate", 1.0)
    legacy_hit = CacheEntry("cached answer", 3, 3, source_prompt="")

    _maybe_shadow_verify("legacy-model", legacy_hit, "some new prompt")
    await _drain_background_tasks()

    assert "legacy-model" not in get_threshold_controller().all_models()


def test_gateway_passes_controller_threshold_into_cache_lookup(client, api_key, monkeypatch):
    """Wiring check: the gateway must fetch the per-model adaptive
    threshold and hand it to SemanticCache.lookup rather than letting the
    cache fall back to its own static default (test_cache.py covers what
    the cache itself does with that value)."""
    controller = get_threshold_controller()
    for _ in range(10):  # threshold_min_samples_before_adjust
        controller.record_verification("wired-model", is_false_positive=True)
    expected_threshold = controller.get_threshold("wired-model")
    assert expected_threshold != get_settings().cache_similarity_threshold

    from app.cache import SemanticCache

    seen_thresholds = []
    original_lookup = SemanticCache.lookup

    def spying_lookup(self, model, messages, threshold=None):
        seen_thresholds.append(threshold)
        return original_lookup(self, model, messages, threshold=threshold)

    monkeypatch.setattr(SemanticCache, "lookup", spying_lookup)

    headers = {"Authorization": f"Bearer {api_key}"}
    client.post(
        "/v1/chat/completions",
        json={"model": "wired-model", "messages": [{"role": "user", "content": "anything"}]},
        headers=headers,
    )

    assert seen_thresholds == [expected_threshold]
