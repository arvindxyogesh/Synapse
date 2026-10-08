import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    pass


@dataclass
class StreamChunk:
    text: str
    done: bool = False
    # Only populated on the final chunk (done=True).
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def complete(
        self, model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
    ) -> tuple[str, int, int]:
        """Return (response_text, prompt_tokens, completion_tokens)."""

    @abstractmethod
    def stream(
        self, model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Yield StreamChunk pieces as they become available; the final
        chunk has done=True and carries the token counts."""


def _timeout() -> float:
    return get_settings().backend_timeout_seconds


def _estimate_tokens(text: str) -> int:
    # Rough, provider-agnostic estimate (~4 chars/token) used when a backend
    # doesn't report exact counts. Good enough for cost/latency dashboards.
    return max(1, len(text) // 4)


def _ollama_options(temperature: float, max_tokens: int | None) -> dict:
    options = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens  # Ollama's name for max_tokens
    return options


class OllamaProvider(BaseProvider):
    """Talks to a local Ollama server (https://ollama.com) running open-weight
    models such as llama3 or mistral. Free, runs on your own hardware."""

    name = "ollama"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def complete(
        self, model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
    ) -> tuple[str, int, int]:
        options = _ollama_options(temperature, max_tokens)
        payload = {"model": model, "messages": messages, "stream": False, "options": options}
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count") or _estimate_tokens(" ".join(m["content"] for m in messages))
        completion_tokens = data.get("eval_count") or _estimate_tokens(text)
        return text, prompt_tokens, completion_tokens

    async def stream(
        self, model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
    ) -> AsyncIterator[StreamChunk]:
        options = _ollama_options(temperature, max_tokens)
        payload = {"model": model, "messages": messages, "stream": True, "options": options}
        prompt_fallback = _estimate_tokens(" ".join(m["content"] for m in messages))
        text_so_far = ""
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    text_so_far += piece
                    if data.get("done"):
                        yield StreamChunk(
                            text=piece,
                            done=True,
                            prompt_tokens=data.get("prompt_eval_count") or prompt_fallback,
                            completion_tokens=data.get("eval_count") or _estimate_tokens(text_so_far),
                        )
                    else:
                        yield StreamChunk(text=piece)


class VLLMProvider(BaseProvider):
    """Talks to a vLLM OpenAI-compatible server (`vllm serve <model>`, or the
    `vllm/vllm-openai` Docker image -- see docker-compose.yml's optional
    `vllm` service, profile `vllm`). Needs a CUDA GPU to be worth running;
    unlike Ollama it speaks the OpenAI chat-completions wire format
    natively, so this provider is mostly just passthrough."""

    name = "vllm"

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def complete(
        self, model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
    ) -> tuple[str, int, int]:
        payload = {"model": model, "messages": messages, "stream": False, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions", json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or _estimate_tokens(" ".join(m["content"] for m in messages))
        completion_tokens = usage.get("completion_tokens") or _estimate_tokens(text)
        return text, prompt_tokens, completion_tokens

    async def stream(
        self, model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
    ) -> AsyncIterator[StreamChunk]:
        # stream_options.include_usage asks vLLM (an OpenAI-spec extension)
        # for a final chunk carrying exact token counts, instead of falling
        # back to the char-count estimate used when a backend doesn't
        # report them.
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        prompt_fallback = _estimate_tokens(" ".join(m["content"] for m in messages))
        text_so_far = ""
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: ") :]
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    choices = data.get("choices") or []
                    piece = choices[0]["delta"].get("content", "") if choices else ""
                    usage = data.get("usage")
                    text_so_far += piece
                    if usage:
                        # The usage-carrying chunk has empty choices (no new
                        # text of its own) per the OpenAI streaming spec.
                        yield StreamChunk(
                            text=piece,
                            done=True,
                            prompt_tokens=usage.get("prompt_tokens") or prompt_fallback,
                            completion_tokens=usage.get("completion_tokens") or _estimate_tokens(text_so_far),
                        )
                    elif piece:
                        yield StreamChunk(text=piece)


class MockProvider(BaseProvider):
    """Deterministic canned responses -- no external dependency at all.
    Used when MOCK_MODE is set (tests/CI), or when the real backend fails
    and MOCK_FALLBACK is set, so the whole gateway + dashboard is runnable
    and demoable with zero local model setup."""

    name = "mock"

    async def complete(
        self, model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
    ) -> tuple[str, int, int]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = f"[mock:{model}] This is a canned response to: {last_user[:120]}"
        prompt_tokens = _estimate_tokens(" ".join(m["content"] for m in messages))
        completion_tokens = _estimate_tokens(text)
        return text, prompt_tokens, completion_tokens

    async def stream(
        self, model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
    ) -> AsyncIterator[StreamChunk]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = f"[mock:{model}] This is a canned response to: {last_user[:120]}"
        prompt_tokens = _estimate_tokens(" ".join(m["content"] for m in messages))
        words = text.split(" ")
        for i, word in enumerate(words):
            piece = word if i == len(words) - 1 else word + " "
            await asyncio.sleep(0.01)
            yield StreamChunk(text=piece)
        yield StreamChunk(text="", done=True, prompt_tokens=prompt_tokens, completion_tokens=_estimate_tokens(text))


def _real_provider(settings) -> BaseProvider:
    """The non-mock backend selected by PROVIDER (default "ollama")."""
    if settings.provider == "vllm":
        return VLLMProvider(settings.vllm_base_url, settings.vllm_api_key)
    return OllamaProvider(settings.ollama_base_url)


def _backend_failed(provider: BaseProvider, exc: Exception) -> ProviderError:
    return ProviderError(f"{provider.name} backend failed: {exc!r}")


async def run_completion(
    model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
) -> tuple[str, int, int, str]:
    """Route to the configured provider (see PROVIDER). Returns (text,
    prompt_tokens, completion_tokens, provider_name). Raises ProviderError
    if the backend fails, unless MOCK_FALLBACK is set, in which case the
    mock provider answers instead (and a warning is logged)."""
    settings = get_settings()
    if not settings.mock_mode:
        provider = _real_provider(settings)
        try:
            text, pt, ct = await provider.complete(model, messages, temperature, max_tokens)
            return text, pt, ct, provider.name
        except (httpx.HTTPError, ProviderError) as exc:
            if not settings.mock_fallback:
                raise _backend_failed(provider, exc) from exc
            logger.warning("%s backend failed (%r); serving a mock response (MOCK_FALLBACK=true)", provider.name, exc)

    provider = MockProvider()
    text, pt, ct = await provider.complete(model, messages, temperature, max_tokens)
    return text, pt, ct, provider.name


async def run_streaming_completion(
    model: str, messages: list[dict], temperature: float, max_tokens: int | None = None
) -> tuple[AsyncIterator[StreamChunk], str]:
    """Same routing/failure behavior as run_completion, but streamed. Waits
    for the first chunk before returning, so a backend that fails up front
    raises ProviderError here -- before any response has been sent, while
    the caller can still return a proper error status. A failure *after*
    the first chunk ends the stream early (no [DONE] marker)."""
    settings = get_settings()
    if not settings.mock_mode:
        provider = _real_provider(settings)
        agen = provider.stream(model, messages, temperature, max_tokens)
        try:
            first_chunk = await agen.__anext__()
        except StopAsyncIteration as exc:
            if not settings.mock_fallback:
                raise ProviderError(f"{provider.name} backend returned an empty stream") from exc
            logger.warning("%s backend returned an empty stream; serving a mock response", provider.name)
        except (httpx.HTTPError, ProviderError) as exc:
            if not settings.mock_fallback:
                raise _backend_failed(provider, exc) from exc
            logger.warning("%s backend failed (%r); serving a mock response (MOCK_FALLBACK=true)", provider.name, exc)
        else:
            async def _prefixed() -> AsyncIterator[StreamChunk]:
                yield first_chunk
                async for chunk in agen:
                    yield chunk

            return _prefixed(), provider.name

    provider = MockProvider()
    return provider.stream(model, messages, temperature, max_tokens), provider.name
