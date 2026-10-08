"""Model registry: per-model backends, so several variants of one model can
sit behind one gateway and be measured separately."""

import json
import re
from pathlib import Path

import httpx
import pytest

import app.model_registry as registry_module
import app.providers as providers_module
from app.config import get_settings
from app.db import SessionLocal
from app.model_registry import RegistryError, load_registry, resolve_model
from app.models import RequestLog

REPO_ROOT = Path(__file__).resolve().parents[2]
_RealAsyncClient = httpx.AsyncClient

TWO_VARIANTS = """
[models."qwen-bf16"]
provider = "vllm"
base_url = "http://gpu-host:8001/"
upstream_model = "Qwen/Qwen2.5-7B-Instruct"

[models."qwen-awq"]
provider = "vllm"
base_url = "http://gpu-host:8002"
upstream_model = "Qwen/Qwen2.5-7B-Instruct-AWQ"

[models."tiny"]
provider = "ollama"
base_url = "http://cpu-host:11434"
"""


@pytest.fixture(autouse=True)
def _fresh_registry_cache():
    registry_module._cached_registry.cache_clear()
    yield
    registry_module._cached_registry.cache_clear()


@pytest.fixture
def registry_file(tmp_path, monkeypatch):
    path = tmp_path / "models.toml"
    path.write_text(TWO_VARIANTS)
    monkeypatch.setattr(get_settings(), "model_registry_path", str(path))
    return path


def _write(tmp_path, text: str) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(text)
    return path


# -- Loading and validation -------------------------------------------------


def test_example_registry_in_repo_is_valid():
    routes = load_registry(REPO_ROOT / "models.example.toml")
    assert {"qwen2.5-7b-bf16", "qwen2.5-7b-awq", "qwen2.5-7b-gptq-int8"} <= set(routes)
    assert routes["qwen2.5-7b-awq"].upstream_model == "Qwen/Qwen2.5-7B-Instruct-AWQ"


def test_load_registry_parses_entries(tmp_path):
    routes = load_registry(_write(tmp_path, TWO_VARIANTS))
    bf16 = routes["qwen-bf16"]
    assert (bf16.provider, bf16.base_url, bf16.upstream_model) == (
        "vllm",
        "http://gpu-host:8001",  # trailing slash stripped
        "Qwen/Qwen2.5-7B-Instruct",
    )
    assert bf16.registered is True
    assert routes["tiny"].upstream_model == "tiny"  # defaults to the gateway name


@pytest.mark.parametrize(
    "text, message",
    [
        ('[models."m"]\nprovider = "openai"\nbase_url = "http://x"', "provider must be one of"),
        ('[models."m"]\nprovider = "vllm"\nbase_url = "localhost:8001"', "base_url must be an http"),
        ('[models."m"]\nprovider = "vllm"\nbase_url = "http://x"\nbase_ur = "typo"', "unknown setting"),
        ('[models."m"]\nprovider = "vllm"\nbase_url = "http://x"\nupstream_model = ""', "upstream_model"),
        ("title = 'no models'", "no [models"),
        ("not = valid = toml", "not valid TOML"),
    ],
)
def test_load_registry_rejects_bad_files(tmp_path, text, message):
    with pytest.raises(RegistryError, match=re.escape(message)):
        load_registry(_write(tmp_path, text))


def test_load_registry_missing_file(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path / "nope.toml")


# -- Resolution ---------------------------------------------------------------


def test_registered_model_resolves_to_its_own_backend(registry_file):
    route = resolve_model("qwen-awq")
    assert route.registered is True
    assert route.base_url == "http://gpu-host:8002"


def test_unregistered_model_falls_back_to_provider(registry_file, monkeypatch):
    monkeypatch.setattr(get_settings(), "provider", "vllm")
    route = resolve_model("something-else")
    assert route.registered is False
    assert (route.provider, route.upstream_model) == ("vllm", "something-else")


def test_no_registry_configured_means_provider_for_everything():
    assert get_settings().model_registry_path is None
    assert resolve_model("qwen-awq").registered is False


# -- End to end through the gateway -----------------------------------------


def test_gateway_routes_each_variant_to_its_own_server(client, api_key, registry_file, monkeypatch):
    monkeypatch.setattr(get_settings(), "mock_mode", False)
    calls = []

    def handler(request):
        calls.append((request.url.host, request.url.port, json.loads(request.content)["model"]))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        )

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    monkeypatch.setattr(providers_module.httpx, "AsyncClient", factory)
    headers = {"Authorization": f"Bearer {api_key}", "x-synapse-cache": "bypass"}
    for model in ("qwen-bf16", "qwen-awq"):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "route me"}]},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == model  # clients see the gateway name

    assert calls == [
        ("gpu-host", 8001, "Qwen/Qwen2.5-7B-Instruct"),
        ("gpu-host", 8002, "Qwen/Qwen2.5-7B-Instruct-AWQ"),
    ]

    # The request log records the gateway name, so metrics are per variant.
    db = SessionLocal()
    try:
        logged = {r.model for r in db.query(RequestLog).filter(RequestLog.model.in_(["qwen-bf16", "qwen-awq"]))}
    finally:
        db.close()
    assert logged == {"qwen-bf16", "qwen-awq"}


def test_models_endpoint_lists_registry(client, api_key, registry_file):
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    assert [m["id"] for m in resp.json()["data"]] == ["qwen-awq", "qwen-bf16", "tiny"]


def test_models_endpoint_without_registry_lists_default_model(client, api_key):
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    assert [m["id"] for m in resp.json()["data"]] == [get_settings().default_model]


def test_models_endpoint_requires_auth(client):
    assert client.get("/v1/models").status_code == 401


def test_broken_registry_raises_instead_of_routing_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "model_registry_path", str(_write(tmp_path, "[models.m]\nprovider = 1")))
    with pytest.raises(RegistryError):
        registry_module.get_registry()
