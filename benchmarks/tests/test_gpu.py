"""GPU memory helpers. The log lines below are written in the formats vLLM
is known to use; they are examples for the parser, not captured output from
a real run -- milestone 4 checks the parser against a real server log."""

import time

from synapse_bench.gpu import GpuMemorySampler, parse_memory_used, parse_vllm_log

SAMPLE_LOG = """\
INFO 10-20 14:02:11 [api_server.py:1090] vLLM API server version 0.9.2
INFO 10-20 14:02:30 [awq_marlin.py:117] The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.
INFO 10-20 14:02:41 [gpu_model_runner.py:1595] Model loading took 5.4337 GiB and 9.81 seconds
INFO 10-20 14:02:55 [gpu_worker.py:227] Available KV cache memory: 32.17 GiB
INFO 10-20 14:02:55 [kv_cache_utils.py:715] GPU KV cache size: 602,416 tokens
INFO 10-20 14:02:55 [kv_cache_utils.py:719] Maximum concurrency for 4,096 tokens per request: 147.07x
"""


def test_parse_vllm_log_newer_format():
    parsed = parse_vllm_log(SAMPLE_LOG)
    assert parsed["vllm_version"] == "0.9.2"
    assert parsed["weights_gib"] == 5.4337
    assert parsed["kv_cache_memory_gib"] == 32.17
    assert parsed["kv_cache_tokens"] == 602416
    assert parsed["max_concurrency"] == {"tokens_per_request": 4096, "requests": 147.07}
    assert "awq_marlin" in parsed["quant_kernel_line"]
    assert parsed["gpu_blocks"] is None  # not in this format -> None, not a guess


def test_parse_vllm_log_older_format():
    parsed = parse_vllm_log("INFO Loading model weights took 14.2487 GB\nINFO # GPU blocks: 2,048, # CPU blocks: 512\n")
    assert parsed["weights_gib"] == 14.2487
    assert parsed["gpu_blocks"] == 2048


def test_parse_vllm_log_unrelated_text_gives_all_none():
    assert all(v is None for v in parse_vllm_log("nothing useful here").values())


def test_parse_vllm_log_last_match_wins():
    # vLLM can log a value more than once (e.g. after a profiling pass).
    parsed = parse_vllm_log("Available KV cache memory: 1.0 GiB\nAvailable KV cache memory: 2.5 GiB\n")
    assert parsed["kv_cache_memory_gib"] == 2.5


def test_parse_memory_used_multi_gpu():
    assert parse_memory_used("0, 40123\n1, 512\n") == {0: 40123, 1: 512}


def _wait_until(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return predicate()


def test_sampler_keeps_peak_per_gpu():
    readings = iter(["0, 100\n", "0, 900\n", "0, 300\n"] + ["0, 200\n"] * 1000)
    with GpuMemorySampler(interval_s=0.01, query=lambda: next(readings)) as sampler:
        assert _wait_until(lambda: sampler.samples >= 3)
    result = sampler.result()
    assert result["peak_mib"] == {0: 900}
    assert result["samples"] >= 4  # includes the final reading on exit


def test_sampler_survives_a_failed_poll():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("nvidia-smi hiccup")
        return "0, 42\n"

    with GpuMemorySampler(interval_s=0.01, query=flaky) as sampler:
        assert _wait_until(lambda: sampler.samples >= 1)
    assert sampler.result()["peak_mib"] == {0: 42}


def test_sampler_without_nvidia_smi_does_nothing(monkeypatch):
    monkeypatch.setattr("synapse_bench.gpu.shutil.which", lambda name: None)
    with GpuMemorySampler() as sampler:
        pass
    assert sampler.available is False and sampler.result() is None
