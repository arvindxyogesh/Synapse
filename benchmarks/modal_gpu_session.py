"""Run the benchmark session (gpu_session.py) on one Modal L40S.

    modal run benchmarks/modal_gpu_session.py --mode smoke
    modal run benchmarks/modal_gpu_session.py --mode full
    modal volume get synapse-bench-results <run_id> benchmarks/results/<run_id>

A thin wrapper: the container image provides vLLM, Redis and the gateway's
virtualenv; gpu_session.run_session() does the actual work, exactly as it
does on any other Linux GPU machine. Results go to a Modal Volume that is
committed after each variant, so a preemption or failure halfway through
keeps everything finished so far.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
import gpu_session  # noqa: E402

GPU = "L40S"
# Modal on-demand L40S price, USD/hour, from modal.com/pricing on 2026-10-08
# ($0.000542/s). Only used to turn measured throughput into cost per 1K tokens.
GPU_PRICE_PER_HOUR = 1.95
HF_CACHE, RESULTS = "/hf-cache", "/results"
GATEWAY_PY = "/opt/gateway/bin/python"

app = modal.App("synapse-quant-benchmark")
hf_volume = modal.Volume.from_name("synapse-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("synapse-bench-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("redis-server")
    .pip_install(f"vllm=={gpu_session.VLLM_VERSION}")
    # The gateway and benchmark client get their own virtualenv with the
    # repo's pinned dev requirements, so their pins never fight vLLM's.
    .add_local_file(REPO / "backend" / "requirements-dev.txt", "/tmp/gateway-requirements.txt", copy=True)
    .run_commands("python -m venv /opt/gateway", f"{GATEWAY_PY} -m pip install -q -r /tmp/gateway-requirements.txt")
    .env({"HF_HOME": HF_CACHE})
    .add_local_dir(REPO / "backend", "/repo/backend", ignore=["**/__pycache__", "**/.venv", "*.db"])
    .add_local_dir(REPO / "benchmarks", "/repo/benchmarks", ignore=["**/__pycache__", "**/.venv", "results"])
)


@app.function(image=image, gpu=GPU, cpu=8.0, memory=32768, timeout=6 * 3600,
              volumes={HF_CACHE: hf_volume, RESULTS: results_volume})
def run_session(mode: str, run_id: str, git_commit: str, git_dirty: bool) -> list[dict]:
    sys.path.insert(0, "/repo/benchmarks")
    import gpu_session as session  # the copy inside the image, so its REPO resolves to /repo

    def commit_volumes() -> None:
        results_volume.commit()
        hf_volume.commit()

    cfg = session.SessionConfig(mode=mode, results_dir=Path(RESULTS), gateway_python=GATEWAY_PY,
                                gpu_label=f"{GPU} (Modal)", gpu_price_per_hour=GPU_PRICE_PER_HOUR,
                                git_commit=git_commit, git_dirty=git_dirty, on_variant_done=commit_volumes)
    statuses = session.run_session(cfg, run_id)
    results_volume.commit()
    return statuses


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    if mode not in gpu_session.RUN_PROFILES:
        raise SystemExit(f"--mode must be one of {list(gpu_session.RUN_PROFILES)}")
    commit, dirty = gpu_session.git_state()
    if dirty and mode == "full":
        raise SystemExit("commit your changes first: full-run results must map to an exact commit")
    run_id = f"{datetime.now(timezone.utc):%Y-%m-%dT%H%M}_{GPU.lower()}_{mode}"
    print(f"run_id={run_id} commit={(commit or '?')[:10]} dirty={dirty}")
    for status in run_session.remote(mode, run_id, commit, dirty):
        print(json.dumps(status))
    print(f"\nfetch results: modal volume get synapse-bench-results {run_id} benchmarks/results/{run_id}")
