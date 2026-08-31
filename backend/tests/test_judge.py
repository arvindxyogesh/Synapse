import pytest

import app.judge as judge_module
from app.judge import judge_same_intent


@pytest.mark.asyncio
async def test_identical_prompts_are_always_same_intent():
    assert await judge_same_intent("llama3", "how do I reset my password", "how do I reset my password") is True


@pytest.mark.asyncio
async def test_identical_prompts_case_and_whitespace_insensitive():
    assert await judge_same_intent("llama3", "  Reset Password?  ", "reset password?") is True


@pytest.mark.asyncio
async def test_mock_mode_uses_heuristic_for_close_paraphrase():
    # High token overlap -- the heuristic fallback (used because MOCK_MODE
    # is set in tests, see conftest.py) should call this the same intent.
    a = "How do I reset my forgotten password"
    b = "How can I reset my forgotten password"
    assert await judge_same_intent("llama3", a, b) is True


@pytest.mark.asyncio
async def test_mock_mode_heuristic_rejects_unrelated_prompts():
    a = "How do I reset my password"
    b = "What payment methods do you accept"
    assert await judge_same_intent("llama3", a, b) is False


@pytest.mark.asyncio
async def test_mock_mode_heuristic_rejects_near_miss_confuser():
    # Real-world false-positive shape: topically adjacent, different intent.
    a = "How do I cancel my subscription?"
    b = "How do I pause my subscription instead of cancelling?"
    assert await judge_same_intent("llama3", a, b) is False


@pytest.mark.asyncio
async def test_real_provider_yes_response_is_parsed(monkeypatch):
    async def fake_run_completion(model, messages, temperature):
        return "YES, these ask for the same thing.", 20, 8, "ollama"

    monkeypatch.setattr(judge_module, "run_completion", fake_run_completion)
    assert await judge_same_intent("llama3", "prompt a", "prompt b") is True


@pytest.mark.asyncio
async def test_real_provider_no_response_is_parsed(monkeypatch):
    async def fake_run_completion(model, messages, temperature):
        return "No, these are different questions.", 20, 9, "ollama"

    monkeypatch.setattr(judge_module, "run_completion", fake_run_completion)
    assert await judge_same_intent("llama3", "prompt a", "prompt b") is False


@pytest.mark.asyncio
async def test_provider_error_falls_back_to_heuristic(monkeypatch):
    async def failing_run_completion(model, messages, temperature):
        raise RuntimeError("boom")

    monkeypatch.setattr(judge_module, "run_completion", failing_run_completion)
    # Same heuristic-favorable pair as the paraphrase test above.
    a = "How do I reset my forgotten password"
    b = "How can I reset my forgotten password"
    assert await judge_same_intent("llama3", a, b) is True
