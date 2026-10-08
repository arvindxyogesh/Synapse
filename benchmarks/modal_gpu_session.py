"""Run the quantization benchmark on one Modal L40S GPU.

    modal run benchmarks/modal_gpu_session.py::check            # CPU only: verify vLLM version + flags
    modal run benchmarks/modal_gpu_session.py --mode smoke      # ~15 min on GPU: one variant, small runs
    modal run benchmarks/modal_gpu_session.py --mode full       # all variants, full runs
    modal volume get synapse-bench-results <run_id> benchmarks/results/<run_id>

Everything runs inside ONE container on ONE GPU, one variant at a time:
vLLM serving the variant, the Synapse gateway in front of it (with Redis),
and the benchmark client. So:
- every variant is measured on the same physical GPU (separate containers
  could each land on a different L40S, adding card-to-card variation);
- gateway <-> vLLM <-> client traffic is all localhost, like a real
  deployment where the gateway sits next to the model server, so no
  internet latency ends up inside TTFT;
- only one variant holds the GPU at a time, so variants never compete for
  memory.

Results are written to a Modal Volume and committed after each variant, so
a failure halfway through keeps everything finished so far.
"""

import json
import os
import secrets
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

VLLM_VERSION = "0.31.0"
GPU = "L40S"
# Modal on-demand L40S price, USD/hour, from modal.com/pricing on 2026-10-08
# ($0.000542/s). Only used to turn measured throughput into cost per 1K
# tokens; recorded in every result's metadata.
GPU_PRICE_PER_HOUR = 1.95

# The same model, Qwen2.5-7B-Instruct, at three weight precisions -- all
# first-party checkpoints from the Qwen team, pinned to exact revisions so a
# re-upload can't silently change what's being measured.
VARIANTS = {
    "qwen2.5-7b-bf16": {"repo": "Qwen/Qwen2.5-7B-Instruct",
                        "revision": "a09a35458c702b33eeacc393d103063234e8bc28"},
    "qwen2.5-7b-awq": {"repo": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                       "revision": "b25037543e9394b818fdfca67ab2a00ecc7dd641"},
    "qwen2.5-7b-gptq-int8": {"repo": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
                             "revision": "6711f2b6fc95545efe6018c469eca6832e28cefd"},
}

# Identical for every variant. No --quantization flag: vLLM reads the
# method (AWQ / GPTQ) from each checkpoint's config.json and picks the
# kernel itself; the chosen kernel is recorded from its log (DECISIONS D11).
VLLM_ARGS = [
    "--gpu-memory-utilization", "0.90",   # same memory budget for all variants (D11)
    "--max-model-len", "4096",            # same max sequence length -> comparable KV-cache capacity
    "--max-num-seqs", "128",              # scheduler cap on concurrent sequences, above our max concurrency (64)
    "--no-enable-prefix-caching",         # every request pays its full prompt cost (D12)
    "--enable-prompt-tokens-details",     # ...and reports cached tokens, so the harness can verify that
    "--seed", "0",
]

RUN_PROFILES = {
    # Shakes out problems for ~15 GPU-minutes before spending on a full run.
    "smoke": {"variants": ["qwen2.5-7b-awq"], "concurrency": "1,16", "requests": 20, "warmup": 3,
              "repeat_concurrency": None, "eval_limit": 50},
    # The real thing (docs/PLAN.md section 3).
    "full": {"variants": list(VARIANTS), "concurrency": "1,4,16,64", "requests": 100, "warmup": 10,
             "repeat_concurrency": "1,16", "eval_limit": None},
}

HF_CACHE = "/hf-cache"
RESULTS = "/results"
GATEWAY_PY = "/opt/gateway/bin/python"

app = modal.App("synapse-quant-benchmark")
hf_volume = modal.Volume.from_name("synapse-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("synapse-bench-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("redis-server")
    .pip_install(f"vllm=={VLLM_VERSION}")
    # The gateway and the benchmark client get their own virtualenv with the
    # repo's pinned dev requirements, so their dependency pins never fight
    # vLLM's. (The gateway runs with its semantic cache bypassed, so it
    # doesn't need sentence-transformers.)
    .add_local_file(REPO / "backend" / "requirements-dev.txt", "/tmp/gateway-requirements.txt", copy=True)
    .run_commands("python -m venv /opt/gateway", f"{GATEWAY_PY} -m pip install -q -r /tmp/gateway-requirements.txt")
    .env({"HF_HOME": HF_CACHE})
    .add_local_dir(REPO / "backend", "/repo/backend", ignore=["**/__pycache__", "**/.venv", "*.db"])
    .add_local_dir(REPO / "benchmarks", "/repo/benchmarks",
                   ignore=["**/__pycache__", "**/.venv", "results", "modal_gpu_session.py"])
)


# -- helpers (run inside the container) ---------------------------------------


def _wait_http(url: str, timeout_s: float, proc: subprocess.Popen | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process exited with code {proc.returncode} while waiting for {url}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(3)
    raise TimeoutError(f"{url} not up after {timeout_s:.0f}s")


def _post_json(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _run_logged(cmd: list[str], log_path: Path, **kwargs) -> int:
    with open(log_path, "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, **kwargs).returncode


def _serve_variant(name: str, out: Path, profile: dict, git_env: dict) -> dict:
    spec = VARIANTS[name]
    out.mkdir(parents=True, exist_ok=False)
    status = {"variant": name, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    vllm_log = open(out / "vllm.log", "w")
    vllm = subprocess.Popen(
        ["vllm", "serve", spec["repo"], "--revision", spec["revision"], "--port", "8001", *VLLM_ARGS],
        stdout=vllm_log, stderr=subprocess.STDOUT,
    )
    redis = gateway = None
    try:
        t0 = time.monotonic()
        _wait_http("http://127.0.0.1:8001/health", timeout_s=1800, proc=vllm)
        status["vllm_ready_after_s"] = round(time.monotonic() - t0, 1)

        redis = subprocess.Popen(["redis-server", "--port", "6379", "--save", "", "--appendonly", "no"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        registry = out / "models.toml"
        registry.write_text(
            f'[models."{name}"]\nprovider = "vllm"\nbase_url = "http://127.0.0.1:8001"\n'
            f'upstream_model = "{spec["repo"]}"\n'
        )
        admin_key = secrets.token_urlsafe(16)
        gw_env = {**os.environ, "DATABASE_URL": f"sqlite:///{out}/gateway.db", "REDIS_URL": "redis://127.0.0.1:6379/0",
                  "MODEL_REGISTRY_PATH": str(registry), "MOCK_FALLBACK": "false", "ADMIN_KEY": admin_key,
                  "BACKEND_TIMEOUT_SECONDS": "900"}
        subprocess.run([GATEWAY_PY, "-m", "alembic", "upgrade", "head"], cwd="/repo/backend", env=gw_env, check=True,
                       capture_output=True)
        gateway_log = open(out / "gateway.log", "w")
        gateway = subprocess.Popen(
            [GATEWAY_PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000",
             "--log-level", "warning"],
            cwd="/repo/backend", env=gw_env, stdout=gateway_log, stderr=subprocess.STDOUT,
        )
        _wait_http("http://127.0.0.1:8000/health", timeout_s=120, proc=gateway)
        api_key = _post_json("http://127.0.0.1:8000/v1/admin/keys", {"name": "benchmark"},
                             {"x-admin-key": admin_key})["api_key"]

        bench_env = {**os.environ, **git_env, "SYNAPSE_API_KEY": api_key, "PYTHONPATH": "/repo/benchmarks"}
        common = ["--model", name, "--gateway-url", "http://127.0.0.1:8000"]

        perf = [GATEWAY_PY, "-m", "synapse_bench.run_perf", *common,
                "--direct-url", "http://127.0.0.1:8001", "--direct-model", spec["repo"],
                "--concurrency", profile["concurrency"], "--requests", str(profile["requests"]),
                "--warmup", str(profile["warmup"]), "--max-tokens", "256",
                "--gpu-price-per-hour", str(GPU_PRICE_PER_HOUR), "--gpu-sample",
                "--vllm-log", str(out / "vllm.log"), "--timeout", "900", "--out", str(out / "perf")]
        status["perf_exit"] = _run_logged(perf, out / "perf.log", cwd="/repo/benchmarks", env=bench_env)

        if profile["repeat_concurrency"]:
            # Extra repeats at two levels, gateway only, to show run-to-run spread.
            repeats = [GATEWAY_PY, "-m", "synapse_bench.run_perf", *common,
                       "--concurrency", profile["repeat_concurrency"], "--repeat", "2",
                       "--requests", str(profile["requests"]), "--warmup", str(profile["warmup"]),
                       "--max-tokens", "256", "--gpu-price-per-hour", str(GPU_PRICE_PER_HOUR),
                       "--timeout", "900", "--out", str(out / "perf_repeats")]
            status["perf_repeats_exit"] = _run_logged(repeats, out / "perf_repeats.log", cwd="/repo/benchmarks",
                                                      env=bench_env)

        evaluate = [GATEWAY_PY, "-m", "synapse_bench.run_eval", *common, "--concurrency", "32",
                    "--timeout", "900", "--out", str(out / "gsm8k")]
        if profile["eval_limit"]:
            evaluate += ["--limit", str(profile["eval_limit"])]
        status["eval_exit"] = _run_logged(evaluate, out / "eval.log", cwd="/repo/benchmarks", env=bench_env)
    finally:
        for proc in (gateway, redis, vllm):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
        vllm_log.close()
        status["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


# -- Modal functions --------------------------------------------------------------


@app.function(image=image, timeout=15 * 60)
def check() -> dict:
    """CPU only, costs cents: confirm the vLLM version and that every flag in
    VLLM_ARGS exists in this version's `vllm serve` before paying for a GPU."""
    import vllm

    help_text = subprocess.run(["vllm", "serve", "--help=all"], capture_output=True, text=True).stdout
    if "--max-model-len" not in help_text:  # older/newer CLIs that don't support --help=all
        help_text = subprocess.run(["vllm", "serve", "--help"], capture_output=True, text=True).stdout
    flags = [a for a in VLLM_ARGS if a.startswith("--")]
    return {"vllm_version": vllm.__version__, "missing_flags": [f for f in flags if f not in help_text],
            "help_chars": len(help_text)}


@app.function(image=image, gpu=GPU, cpu=8.0, memory=32768, timeout=6 * 3600,
              volumes={HF_CACHE: hf_volume, RESULTS: results_volume})
def run_session(mode: str, run_id: str, git_commit: str, git_dirty: bool) -> list[dict]:
    profile = RUN_PROFILES[mode]
    git_env = {"SYNAPSE_GIT_COMMIT": git_commit, "SYNAPSE_GIT_DIRTY": "true" if git_dirty else "false"}
    session_dir = Path(RESULTS) / run_id
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "session.json").write_text(json.dumps({
        "mode": mode, "profile": profile, "gpu": GPU, "gpu_price_per_hour": GPU_PRICE_PER_HOUR,
        "vllm_version": VLLM_VERSION, "vllm_args": VLLM_ARGS, "variants": {v: VARIANTS[v] for v in profile["variants"]},
        "git_commit": git_commit, "git_dirty": git_dirty,
        "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout,
    }, indent=2))
    (session_dir / "pip-freeze.txt").write_text(
        subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout)

    statuses = []
    for name in profile["variants"]:
        try:
            statuses.append(_serve_variant(name, session_dir / name, profile, git_env))
        except Exception as exc:  # keep going: a failed variant shouldn't lose the others
            statuses.append({"variant": name, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            results_volume.commit()
            hf_volume.commit()
    (session_dir / "statuses.json").write_text(json.dumps(statuses, indent=2))
    results_volume.commit()
    return statuses


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    if mode not in RUN_PROFILES:
        raise SystemExit(f"--mode must be one of {list(RUN_PROFILES)}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout)
    if dirty and mode == "full":
        raise SystemExit("commit your changes first: full-run results must map to an exact commit")
    run_id = f"{datetime.now(timezone.utc):%Y-%m-%dT%H%M}_{GPU.lower()}_{mode}"
    print(f"run_id={run_id} commit={commit[:10]} dirty={dirty}")
    for status in run_session.remote(mode, run_id, commit, dirty):
        print(json.dumps(status))
    print(f"\nfetch results: modal volume get synapse-bench-results {run_id} benchmarks/results/{run_id}")
