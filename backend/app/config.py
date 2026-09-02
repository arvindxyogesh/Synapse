import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./gateway.db"
    redis_url: str = "redis://localhost:6379/0"

    # Open-weight model serving backend, selected by PROVIDER:
    #   "ollama" (default) -- https://ollama.com, runs on CPU, zero GPU
    #       setup, the default so the whole stack demos with `docker
    #       compose up` alone.
    #   "vllm" -- an OpenAI-compatible vLLM server (`vllm serve <model>`,
    #       or the optional `vllm` service in docker-compose.yml, profile
    #       "vllm"). Needs a CUDA GPU; in exchange gets vLLM's throughput
    #       (continuous batching, PagedAttention) and features like
    #       multi-LoRA serving that Ollama doesn't offer.
    # Either way it's free/self-hosted -- no per-token API cost.
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8001"
    vllm_api_key: str | None = None
    default_model: str = "llama3"

    # Master key used to create/revoke gateway API keys via /v1/admin/*.
    # Individual gateway API keys (created through that endpoint) are what
    # callers of /v1/chat/completions authenticate with.
    admin_key: str = "change-me-admin-key"

    # Semantic cache
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 60 * 60 * 24 * 7  # 7 days
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Adaptive cache threshold: cache_similarity_threshold above is just the
    # starting point / fallback. When enabled, each model's threshold is
    # instead tuned online by a closed-loop controller (see
    # app/threshold_controller.py), driven by LLM-judge shadow verification
    # of a sample of cache hits (see app/judge.py) -- the same "hold an
    # operating metric near a target via small bounded adjustments" shape
    # as an SLO-adaptive controller, applied here to cache correctness.
    adaptive_threshold_enabled: bool = True
    cache_threshold_min: float = 0.80
    cache_threshold_max: float = 0.99
    cache_threshold_step: float = 0.01
    shadow_verify_sample_rate: float = 0.2
    target_false_positive_rate: float = 0.05
    # An EWMA's effective memory is roughly 1/alpha samples. At a modest,
    # spread-out false-positive rate (e.g. ~10-15%, not clustered in a
    # burst), too short a memory means a couple of consecutive *correct*
    # verifications decay the estimate back toward zero before the true
    # rate is ever reflected -- alpha=0.3 (~3-sample memory) measurably
    # undercounted a real 14% false-positive rate down to ~0% in practice.
    # 0.1 gives roughly a 10-sample memory, matched to the sample sizes
    # below.
    threshold_fp_rate_ewma_alpha: float = 0.1
    threshold_min_samples_before_adjust: int = 10
    # At most one adjustment per this many verified samples -- without a
    # cooldown, a single EWMA lags behind a sudden change in the true
    # false-positive rate (e.g. right after a run of bad luck), so it keeps
    # adjusting in the *old* direction for several more samples even after
    # the underlying rate has already flipped, overshooting badly. The
    # cooldown gives the EWMA time to catch up between adjustments.
    threshold_adjustment_cooldown_samples: int = 10

    # If true (or if Ollama is unreachable), the gateway serves canned
    # responses instead of calling a model -- lets the whole stack run and
    # be demoed with zero local setup.
    mock_mode: bool = os.getenv("MOCK_MODE", "false").lower() == "true"

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
