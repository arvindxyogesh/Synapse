"""One row per benchmark request -- the raw data every summary is computed
from. Saved as-is (JSONL), so any number in a report can be recomputed."""

from dataclasses import dataclass


@dataclass
class RequestRecord:
    index: int  # position in the run's request sequence
    ok: bool  # got a complete response (HTTP 200 and a [DONE] marker)
    status_code: int | None  # None if the request never got an HTTP response
    error: str | None  # short description when ok is False
    start_s: float  # seconds since the start of the level (perf_counter based)
    ttft_s: float | None  # send -> first chunk with text
    e2e_s: float | None  # send -> [DONE]
    prompt_tokens: int | None
    # Prompt tokens the backend served from its own prefix/prompt cache, when
    # it reports them (usage.prompt_tokens_details.cached_tokens). Anything
    # above 0 means TTFT was measured on partly pre-processed prompts.
    cached_prompt_tokens: int | None
    output_tokens: int | None
    x_cache: str | None  # gateway's cache verdict ("bypass"/"hit"/"miss"), None when talking to vLLM directly
    provider: str | None  # who answered, as reported in the stream ("vllm", "cache", "mock", ...)
