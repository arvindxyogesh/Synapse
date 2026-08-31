from abc import ABC, abstractmethod

import httpx

from app.config import get_settings


class ProviderError(Exception):
    pass


class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def complete(self, model: str, messages: list[dict], temperature: float) -> tuple[str, int, int]:
        """Return (response_text, prompt_tokens, completion_tokens)."""


def _estimate_tokens(text: str) -> int:
    # Rough, provider-agnostic estimate (~4 chars/token) used when a backend
    # doesn't report exact counts. Good enough for cost/latency dashboards.
    return max(1, len(text) // 4)


class OllamaProvider(BaseProvider):
    """Talks to a local Ollama server (https://ollama.com) running open-weight
    models such as llama3 or mistral. Free, runs on your own hardware."""

    name = "ollama"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def complete(self, model: str, messages: list[dict], temperature: float) -> tuple[str, int, int]:
        payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count") or _estimate_tokens(" ".join(m["content"] for m in messages))
        completion_tokens = data.get("eval_count") or _estimate_tokens(text)
        return text, prompt_tokens, completion_tokens


class MockProvider(BaseProvider):
    """Deterministic canned responses -- no external dependency at all.
    Used automatically when Ollama is unreachable, and in tests/CI, so the
    whole gateway + dashboard is runnable and demoable with zero local
    model setup."""

    name = "mock"

    async def complete(self, model: str, messages: list[dict], temperature: float) -> tuple[str, int, int]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = f"[mock:{model}] This is a canned response to: {last_user[:120]}"
        prompt_tokens = _estimate_tokens(" ".join(m["content"] for m in messages))
        completion_tokens = _estimate_tokens(text)
        return text, prompt_tokens, completion_tokens


async def run_completion(model: str, messages: list[dict], temperature: float) -> tuple[str, int, int, str]:
    """Route to Ollama, falling back to the mock provider if Ollama is
    unreachable or MOCK_MODE is set. Returns (text, prompt_tokens,
    completion_tokens, provider_name)."""
    settings = get_settings()
    if not settings.mock_mode:
        provider = OllamaProvider(settings.ollama_base_url)
        try:
            text, pt, ct = await provider.complete(model, messages, temperature)
            return text, pt, ct, provider.name
        except (httpx.HTTPError, ProviderError):
            pass  # fall through to mock

    provider = MockProvider()
    text, pt, ct = await provider.complete(model, messages, temperature)
    return text, pt, ct, provider.name
