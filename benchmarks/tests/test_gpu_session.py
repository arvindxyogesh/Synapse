"""Pure helpers in gpu_session.py (the rest needs a GPU)."""

from pathlib import Path

import gpu_session


def test_pip_cuda_home_finds_toolkit_next_to_vllm(tmp_path):
    env = tmp_path / "envs" / "vllm"
    nvcc = env / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("")
    (env / "bin").mkdir()
    assert gpu_session.pip_cuda_home(str(env / "bin" / "vllm")) == nvcc.parent.parent


def test_pip_cuda_home_none_without_toolkit_or_path(tmp_path):
    (tmp_path / "bin").mkdir()
    assert gpu_session.pip_cuda_home(str(tmp_path / "bin" / "vllm")) is None
    assert gpu_session.pip_cuda_home("vllm") is None


def test_vllm_env_prefers_existing_cuda_home(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-from-module")
    cfg = gpu_session.SessionConfig(mode="smoke", results_dir=tmp_path, gateway_python="python",
                                    gpu_label="test", gpu_price_per_hour=None, vllm_bin=str(tmp_path / "bin" / "vllm"))
    env = gpu_session._vllm_env(cfg)
    assert env["CUDA_HOME"] == "/opt/cuda-from-module"
    assert env["PATH"].split(":")[0] == str(Path(cfg.vllm_bin).parent)


def test_vllm_env_turns_off_flashinfer_sampler_and_usage_stats(tmp_path):
    cfg = gpu_session.SessionConfig(mode="smoke", results_dir=tmp_path, gateway_python="python",
                                    gpu_label="test", gpu_price_per_hour=None)
    env = gpu_session._vllm_env(cfg)
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["VLLM_NO_USAGE_STATS"] == "1" and env["DO_NOT_TRACK"] == "1"
