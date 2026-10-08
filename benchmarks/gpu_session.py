#!/usr/bin/env python3
"""Run the quantization benchmark session on a Linux machine with an NVIDIA GPU.

For each model variant, one after another on the same GPU:
  1. start vLLM serving that variant (pinned revision, identical flags),
  2. start Redis + the Synapse gateway in front of it, all on localhost,
  3. run the speed benchmark (through the gateway AND directly against vLLM),
     extra repeats at two concurrency levels, and the GSM8K eval,
  4. stop everything, so the next variant gets the whole GPU.

Same code on a lab machine, a rented box, or inside Modal
(modal_gpu_session.py is a thin wrapper around run_session()).

    python benchmarks/gpu_session.py --mode smoke \\
        --results-dir /data/me/results --gateway-python /data/me/envs/gateway/bin/python \\
        --cuda-device 2 --gpu-memory-utilization 0.30 --gpu-label "H200 (shared lab node)"

Everything the run writes goes under --results-dir; model downloads go to
$HF_HOME (set it to keep weights out of your home directory).
"""

import argparse
import json
import os
import secrets
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VLLM_VERSION = "0.31.0"

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

# Identical for every variant within a session. No --quantization flag: vLLM
# reads the method (AWQ / GPTQ) from each checkpoint's config.json and picks
# the kernel itself; the chosen kernel is recorded from its log (DECISIONS D11).
# --gpu-memory-utilization is added per session (SessionConfig), because the
# right value depends on the machine -- but it's the same for every variant.
VLLM_ARGS = [
    "--max-model-len", "4096",            # same max sequence length -> comparable KV-cache capacity
    "--max-num-seqs", "128",              # scheduler cap on concurrent sequences, above our max concurrency (64)
    "--no-enable-prefix-caching",         # every request pays its full prompt cost (D12)
    "--enable-prompt-tokens-details",     # ...and reports cached tokens, so the harness can verify that
    "--seed", "0",
]

RUN_PROFILES = {
    # Shakes out problems quickly before a full run.
    "smoke": {"variants": ["qwen2.5-7b-awq"], "concurrency": "1,16", "requests": 20, "warmup": 3,
              "repeat_concurrency": None, "eval_limit": 50},
    # The real thing (docs/PLAN.md section 3).
    "full": {"variants": list(VARIANTS), "concurrency": "1,4,16,64", "requests": 100, "warmup": 10,
             "repeat_concurrency": "1,16", "eval_limit": None},
}

VLLM_PORT, GATEWAY_PORT, REDIS_PORT = 8001, 8000, 6379


@dataclass
class SessionConfig:
    mode: str
    results_dir: Path  # this session's folder is created inside it
    gateway_python: str  # python of a venv with backend/requirements-dev.txt installed
    gpu_label: str  # human description recorded in results, e.g. "L40S (Modal)"
    gpu_price_per_hour: float | None  # USD/h, for cost per 1K tokens; None = don't compute cost
    gpu_memory_utilization: float = 0.90
    cuda_device: str | None = None  # e.g. "2" -> CUDA_VISIBLE_DEVICES=2 for vLLM only
    vllm_bin: str = "vllm"
    redis_bin: str = "redis-server"
    port_offset: int = 0  # shift all ports, to avoid clashing with other users on a shared machine
    git_commit: str | None = None
    git_dirty: bool | None = None
    on_variant_done: object = None  # optional callback (Modal uses it to commit its volume)

    @property
    def ports(self) -> tuple[int, int, int]:
        return VLLM_PORT + self.port_offset, GATEWAY_PORT + self.port_offset, REDIS_PORT + self.port_offset


# -- helpers --------------------------------------------------------------------------


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


def _stop(procs) -> None:
    for proc in procs:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def _vllm_env(cfg: SessionConfig) -> dict:
    env = dict(os.environ)
    if "/" in cfg.vllm_bin:
        # Same effect as activating vLLM's environment: its bin/ goes first on
        # PATH. Needed because FlashInfer compiles a sampling kernel on first
        # start and looks for the `ninja` build tool on PATH (found the hard
        # way: the first lab smoke run died with "No such file: 'ninja'").
        env["PATH"] = f"{Path(cfg.vllm_bin).parent}{os.pathsep}{env.get('PATH', '')}"
    if cfg.cuda_device is not None:
        # CUDA's default numbering is "fastest first", which isn't guaranteed to
        # match nvidia-smi's. PCI_BUS_ID order makes --cuda-device N the same
        # physical GPU as nvidia-smi's index N -- on a shared machine, the
        # difference is landing on someone else's GPU.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_device
    return env


def preflight(cfg: SessionConfig) -> dict:
    """Confirm the vLLM version and that every flag exists before any model is
    downloaded or loaded. Needs the GPU machine: vLLM 0.31.0 can't build its
    argument parser without a GPU ("Failed to infer device type")."""
    version = subprocess.run([cfg.vllm_bin, "--version"], capture_output=True, text=True, env=_vllm_env(cfg))
    proc = subprocess.run([cfg.vllm_bin, "serve", "--help=all"], capture_output=True, text=True, env=_vllm_env(cfg))
    help_text = proc.stdout + proc.stderr
    flags = [a for a in VLLM_ARGS if a.startswith("--")] + ["--gpu-memory-utilization"]
    result = {"vllm_version": version.stdout.strip().splitlines()[-1] if version.stdout.strip() else None,
              "help_exit": proc.returncode, "missing_flags": [f for f in flags if f not in help_text]}
    if result["vllm_version"] != VLLM_VERSION or result["missing_flags"] or proc.returncode != 0:
        raise RuntimeError(f"preflight failed, not loading any model: {result}")
    return result


def serve_and_measure(cfg: SessionConfig, name: str, out: Path) -> dict:
    spec = VARIANTS[name]
    profile = RUN_PROFILES[cfg.mode]
    vllm_port, gw_port, redis_port = cfg.ports
    out.mkdir(parents=True, exist_ok=False)
    status = {"variant": name, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    vllm_log = open(out / "vllm.log", "w")
    vllm = subprocess.Popen(
        [cfg.vllm_bin, "serve", spec["repo"], "--revision", spec["revision"], "--host", "127.0.0.1",
         "--port", str(vllm_port), "--gpu-memory-utilization", str(cfg.gpu_memory_utilization), *VLLM_ARGS],
        stdout=vllm_log, stderr=subprocess.STDOUT, env=_vllm_env(cfg),
    )
    redis = gateway = None
    try:
        t0 = time.monotonic()
        _wait_http(f"http://127.0.0.1:{vllm_port}/health", timeout_s=1800, proc=vllm)
        status["vllm_ready_after_s"] = round(time.monotonic() - t0, 1)

        redis = subprocess.Popen([cfg.redis_bin, "--port", str(redis_port), "--bind", "127.0.0.1",
                                  "--save", "", "--appendonly", "no"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        registry = out / "models.toml"
        registry.write_text(
            f'[models."{name}"]\nprovider = "vllm"\nbase_url = "http://127.0.0.1:{vllm_port}"\n'
            f'upstream_model = "{spec["repo"]}"\n'
        )
        admin_key = secrets.token_urlsafe(16)
        gw_env = {**os.environ, "DATABASE_URL": f"sqlite:///{out}/gateway.db",
                  "REDIS_URL": f"redis://127.0.0.1:{redis_port}/0", "MODEL_REGISTRY_PATH": str(registry),
                  "MOCK_FALLBACK": "false", "ADMIN_KEY": admin_key, "BACKEND_TIMEOUT_SECONDS": "900"}
        subprocess.run([cfg.gateway_python, "-m", "alembic", "upgrade", "head"], cwd=REPO / "backend", env=gw_env,
                       check=True, capture_output=True)
        gateway_log = open(out / "gateway.log", "w")
        gateway = subprocess.Popen(
            [cfg.gateway_python, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(gw_port),
             "--log-level", "warning"],
            cwd=REPO / "backend", env=gw_env, stdout=gateway_log, stderr=subprocess.STDOUT,
        )
        gw_url = f"http://127.0.0.1:{gw_port}"
        _wait_http(f"{gw_url}/health", timeout_s=120, proc=gateway)
        api_key = _post_json(f"{gw_url}/v1/admin/keys", {"name": "benchmark"}, {"x-admin-key": admin_key})["api_key"]

        bench_env = {**os.environ, "SYNAPSE_API_KEY": api_key, "PYTHONPATH": str(REPO / "benchmarks")}
        if cfg.git_commit:
            bench_env |= {"SYNAPSE_GIT_COMMIT": cfg.git_commit, "SYNAPSE_GIT_DIRTY": str(bool(cfg.git_dirty)).lower()}
        py = [cfg.gateway_python, "-m"]
        common = ["--model", name, "--gateway-url", gw_url, "--max-tokens", "256", "--timeout", "900"]
        price = ["--gpu-price-per-hour", str(cfg.gpu_price_per_hour)] if cfg.gpu_price_per_hour else []
        gpu_index = ["--gpu-index", cfg.cuda_device] if cfg.cuda_device is not None else []

        perf = [*py, "synapse_bench.run_perf", *common, *price, *gpu_index,
                "--direct-url", f"http://127.0.0.1:{vllm_port}", "--direct-model", spec["repo"],
                "--concurrency", profile["concurrency"], "--requests", str(profile["requests"]),
                "--warmup", str(profile["warmup"]), "--gpu-sample", "--vllm-log", str(out / "vllm.log"),
                "--out", str(out / "perf")]
        status["perf_exit"] = _run_logged(perf, out / "perf.log", cwd=REPO / "benchmarks", env=bench_env)

        if profile["repeat_concurrency"]:
            # Extra repeats at two levels, gateway only, to show run-to-run spread.
            repeats = [*py, "synapse_bench.run_perf", *common, *price,
                       "--concurrency", profile["repeat_concurrency"], "--repeat", "2",
                       "--requests", str(profile["requests"]), "--warmup", str(profile["warmup"]),
                       "--out", str(out / "perf_repeats")]
            status["perf_repeats_exit"] = _run_logged(repeats, out / "perf_repeats.log", cwd=REPO / "benchmarks",
                                                      env=bench_env)

        evaluate = [*py, "synapse_bench.run_eval", "--model", name, "--gateway-url", gw_url,
                    "--concurrency", "32", "--timeout", "900", "--out", str(out / "gsm8k")]
        if profile["eval_limit"]:
            evaluate += ["--limit", str(profile["eval_limit"])]
        status["eval_exit"] = _run_logged(evaluate, out / "eval.log", cwd=REPO / "benchmarks", env=bench_env)
    finally:
        _stop((gateway, redis, vllm))
        vllm_log.close()
        status["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


def run_session(cfg: SessionConfig, run_id: str) -> list[dict]:
    pre = preflight(cfg)
    profile = RUN_PROFILES[cfg.mode]
    session_dir = Path(cfg.results_dir) / run_id
    session_dir.mkdir(parents=True, exist_ok=False)
    smi_args = ["nvidia-smi"] + (["-i", cfg.cuda_device] if cfg.cuda_device is not None else [])
    (session_dir / "session.json").write_text(json.dumps({
        "run_id": run_id, "mode": cfg.mode, "profile": profile, "gpu_label": cfg.gpu_label,
        "gpu_price_per_hour": cfg.gpu_price_per_hour, "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "cuda_device": cfg.cuda_device, "vllm_version": VLLM_VERSION,
        "vllm_args": [*VLLM_ARGS, "--gpu-memory-utilization", str(cfg.gpu_memory_utilization)],
        "variants": {v: VARIANTS[v] for v in profile["variants"]},
        "git_commit": cfg.git_commit, "git_dirty": cfg.git_dirty, "preflight": pre,
        "nvidia_smi_at_start": subprocess.run(smi_args, capture_output=True, text=True).stdout,
    }, indent=2, default=str))
    # Exact package versions of the vLLM environment (the python next to the vllm executable).
    vllm_python = str(Path(cfg.vllm_bin).with_name("python")) if "/" in cfg.vllm_bin else "python3"
    (session_dir / "pip-freeze.txt").write_text(
        subprocess.run([vllm_python, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)

    statuses = []
    for name in profile["variants"]:
        try:
            statuses.append(serve_and_measure(cfg, name, session_dir / name))
        except Exception as exc:  # keep going: a failed variant shouldn't lose the others
            statuses.append({"variant": name, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if callable(cfg.on_variant_done):
                cfg.on_variant_done()
    (session_dir / "statuses.json").write_text(json.dumps(statuses, indent=2))
    return statuses


def git_state() -> tuple[str | None, bool | None]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
    if commit.returncode != 0:
        return None, None
    return commit.stdout.strip(), bool(status.stdout.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(RUN_PROFILES), default="smoke")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--gateway-python", required=True)
    p.add_argument("--gpu-label", required=True, help='recorded in results, e.g. "H200 (shared lab node)"')
    p.add_argument("--gpu-price-per-hour", type=float, help="USD/h for cost per 1K tokens (omit if not meaningful)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--cuda-device", help="GPU index to use (sets CUDA_VISIBLE_DEVICES for vLLM)")
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--redis-bin", default="redis-server")
    p.add_argument("--port-offset", type=int, default=0)
    args = p.parse_args()

    commit, dirty = git_state()
    if args.mode == "full" and dirty is not False:
        raise SystemExit("full-run results must map to an exact, clean commit -- commit (or check out) first")
    cfg = SessionConfig(mode=args.mode, results_dir=args.results_dir, gateway_python=args.gateway_python,
                        gpu_label=args.gpu_label, gpu_price_per_hour=args.gpu_price_per_hour,
                        gpu_memory_utilization=args.gpu_memory_utilization, cuda_device=args.cuda_device,
                        vllm_bin=args.vllm_bin, redis_bin=args.redis_bin, port_offset=args.port_offset,
                        git_commit=commit, git_dirty=dirty)
    run_id = f"{datetime.now(timezone.utc):%Y-%m-%dT%H%M}_{args.mode}"
    print(f"run_id={run_id} commit={(commit or '?')[:10]} dirty={dirty} config={asdict(cfg)}", flush=True)
    for status in run_session(cfg, run_id):
        print(json.dumps(status), flush=True)
    print(f"results: {args.results_dir / run_id}")


if __name__ == "__main__":
    main()
