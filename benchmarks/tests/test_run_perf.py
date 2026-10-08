"""run_perf end to end against the fake backend: the files it writes, the
secrets it must not write, and the exit code when results are invalid."""

import json

import pytest

import synapse_bench.metadata as metadata
from synapse_bench import run_perf

API_KEY = "llmgw_secret_value_that_must_not_leak"


@pytest.fixture(autouse=True)
def _no_network_probes(monkeypatch):
    monkeypatch.setattr(metadata, "http_json", lambda url, headers=None: {"stub": url})
    monkeypatch.setenv("SYNAPSE_API_KEY", API_KEY)


def _run(tmp_path, *extra):
    out = tmp_path / "result"
    argv = ["--model", "m", "--concurrency", "1,2", "--requests", "4", "--warmup", "1", "--out", str(out), *extra]
    return run_perf.main(argv), out


def test_writes_metadata_raw_records_and_summary(fake, tmp_path):
    code, out = _run(tmp_path, "--gpu-price-per-hour", "1.5")
    assert code == 0

    records = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    assert len(records) == 2 * 4  # 2 levels x 4 measured requests (warm-up not recorded)
    assert {r["concurrency"] for r in records} == {1, 2}
    assert all(r["ok"] and r["target"] == "gateway" for r in records)

    summary = json.loads((out / "summary.json").read_text())
    assert [(row["target"], row["concurrency"]) for row in summary] == [("gateway", 1), ("gateway", 2)]
    assert all(row["cost_per_1k_output_tokens_usd"] is not None for row in summary)

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["args"]["model"] == "m" and meta["args"]["gpu_price_per_hour"] == 1.5
    assert len(meta["workload"]["sha256"]) == 64


def test_api_key_never_written_anywhere(fake, tmp_path):
    _, out = _run(tmp_path)
    for path in out.iterdir():
        assert API_KEY not in path.read_text(), path.name


def test_fixed_length_by_default_and_optional(fake, tmp_path):
    _run(tmp_path)
    assert all(body["ignore_eos"] is True for body in fake.bodies)
    fake.bodies.clear()
    _run(tmp_path / "second", "--no-fixed-length")
    assert all("ignore_eos" not in body for body in fake.bodies)


def test_direct_target_runs_too(fake, tmp_path):
    code, out = _run(tmp_path, "--direct-url", "http://vllm:8002", "--direct-model", "Qwen/X")
    assert code == 0
    targets = [row["target"] for row in json.loads((out / "summary.json").read_text())]
    assert targets == ["gateway", "gateway", "direct", "direct"]
    assert {b["model"] for b in fake.bodies} == {"m", "Qwen/X"}


def test_integrity_failure_saves_results_but_exits_nonzero(fake, tmp_path):
    fake.x_cache = "miss"  # the gateway didn't bypass its cache
    code, out = _run(tmp_path)
    assert code == 2
    records = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    assert all(r["error"] == "cache not bypassed" for r in records)


def test_backend_errors_are_counted_not_fatal(fake, tmp_path):
    fake.status = 502
    code, out = _run(tmp_path)
    assert code == 0  # a failing server is a (bad) result, not an invalid measurement
    summary = json.loads((out / "summary.json").read_text())
    assert all(row["n_errors"] == 4 and row["error_types"] == {"HTTP 502": 4} for row in summary)


def test_refuses_to_overwrite_results(fake, tmp_path):
    _run(tmp_path)
    with pytest.raises(FileExistsError):
        _run(tmp_path)


def test_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("SYNAPSE_API_KEY")
    assert _run(tmp_path)[0] == 1


def test_direct_url_requires_direct_model(tmp_path):
    with pytest.raises(SystemExit):
        run_perf.parse_args(["--model", "m", "--out", str(tmp_path), "--direct-url", "http://x"])
