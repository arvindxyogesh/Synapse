"""Model registry: which backend serves which model name.

Without a registry, every request goes to the single backend picked by
PROVIDER, and the `model` string is passed through unchanged -- the
original behavior, still the default.

With MODEL_REGISTRY_PATH pointing at a TOML file, each registered gateway
model name gets its own backend. That's what lets several variants of one
model (e.g. 16-bit and 4-bit) sit behind one gateway at the same time, each
on its own vLLM server, with per-variant metrics in the request log:

    [models."qwen2.5-7b-awq"]
    provider = "vllm"                                # "vllm" or "ollama"
    base_url = "http://localhost:8002"
    upstream_model = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # the backend's own name

`upstream_model` is optional and defaults to the gateway name. Model names
that aren't in the file still fall back to PROVIDER, so adding a registry
never breaks a client that was already working. See models.example.toml.
"""

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings

SUPPORTED_PROVIDERS = ("ollama", "vllm")
_ALLOWED_KEYS = {"provider", "base_url", "upstream_model"}


class RegistryError(ValueError):
    """The registry file is missing or malformed. Raised at startup, so a typo
    stops the gateway instead of silently routing to the wrong backend."""


@dataclass(frozen=True)
class ModelRoute:
    name: str  # what clients send as "model"
    provider: str  # "ollama" or "vllm"
    base_url: str
    upstream_model: str  # what the backend itself calls the model
    registered: bool  # False = the PROVIDER default route


def load_registry(path: str | Path) -> dict[str, ModelRoute]:
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as exc:
        raise RegistryError(f"model registry not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RegistryError(f"model registry {path} is not valid TOML: {exc}") from exc

    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise RegistryError(f"model registry {path} has no [models.\"<name>\"] entries")

    routes = {}
    for name, entry in models.items():
        where = f"model registry {path}, model '{name}'"
        if not isinstance(entry, dict):
            raise RegistryError(f"{where}: expected a table of settings")
        unknown = set(entry) - _ALLOWED_KEYS
        if unknown:
            raise RegistryError(f"{where}: unknown setting(s) {sorted(unknown)}")
        provider = entry.get("provider")
        if provider not in SUPPORTED_PROVIDERS:
            raise RegistryError(f"{where}: provider must be one of {SUPPORTED_PROVIDERS}, got {provider!r}")
        base_url = entry.get("base_url")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise RegistryError(f"{where}: base_url must be an http(s) URL, got {base_url!r}")
        upstream = entry.get("upstream_model", name)
        if not isinstance(upstream, str) or not upstream:
            raise RegistryError(f"{where}: upstream_model must be a non-empty string")

        routes[name] = ModelRoute(
            name=name,
            provider=provider,
            base_url=base_url.rstrip("/"),
            upstream_model=upstream,
            registered=True,
        )
    return routes


@lru_cache
def _cached_registry(path: str) -> dict[str, ModelRoute]:
    return load_registry(path)


def get_registry(settings: Settings | None = None) -> dict[str, ModelRoute]:
    """Registered routes, or {} if MODEL_REGISTRY_PATH isn't set."""
    settings = settings or get_settings()
    if not settings.model_registry_path:
        return {}
    return _cached_registry(settings.model_registry_path)


def resolve_model(model: str, settings: Settings | None = None) -> ModelRoute:
    """The backend that should serve `model`: its registry entry if it has
    one, otherwise the PROVIDER default with the name passed through."""
    settings = settings or get_settings()
    route = get_registry(settings).get(model)
    if route is not None:
        return route
    # Same rule as before the registry existed: anything but "vllm" means Ollama.
    provider = "vllm" if settings.provider == "vllm" else "ollama"
    base_url = settings.vllm_base_url if provider == "vllm" else settings.ollama_base_url
    return ModelRoute(
        name=model,
        provider=provider,
        base_url=base_url.rstrip("/"),
        upstream_model=model,
        registered=False,
    )
