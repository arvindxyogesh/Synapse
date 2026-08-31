import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./gateway.db"
    redis_url: str = "redis://localhost:6379/0"

    # Open-weight model serving. Ollama (https://ollama.com) runs models like
    # llama3/mistral locally for free -- no API key, no per-token cost.
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3"

    # Master key used to create/revoke gateway API keys via /v1/admin/*.
    # Individual gateway API keys (created through that endpoint) are what
    # callers of /v1/chat/completions authenticate with.
    admin_key: str = "change-me-admin-key"

    # Semantic cache
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 60 * 60 * 24 * 7  # 7 days
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # If true (or if Ollama is unreachable), the gateway serves canned
    # responses instead of calling a model -- lets the whole stack run and
    # be demoed with zero local setup.
    mock_mode: bool = os.getenv("MOCK_MODE", "false").lower() == "true"

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
