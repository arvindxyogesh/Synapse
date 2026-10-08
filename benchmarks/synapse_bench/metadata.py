"""Everything needed to reproduce (or distrust) a result, captured
automatically next to it: when, what code, what hardware, what server
versions, what settings. Each probe is best-effort -- a missing piece of
information is recorded as missing, never guessed."""

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
# Never written into results files.
SECRET_ARGS = {"api_key", "admin_key"}


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=REPO_ROOT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict:
    # Remote runs (e.g. a Modal container) get a copy of the code without
    # .git; the launcher passes the commit it shipped through these instead.
    if os.environ.get("SYNAPSE_GIT_COMMIT"):
        return {
            "commit": os.environ["SYNAPSE_GIT_COMMIT"],
            "dirty": os.environ.get("SYNAPSE_GIT_DIRTY") == "true",
            "source": "launcher",
        }
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        # Uncommitted changes mean the commit hash alone doesn't describe the
        # code that ran -- flagged so such results aren't published as-is.
        "dirty": bool(status) if status is not None else None,
    }


def gpu_info() -> list[dict] | None:
    """One entry per GPU visible to nvidia-smi, or None if there is none."""
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None
    gpus = []
    for line in out.splitlines():
        name, driver, mem_mib = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "driver_version": driver, "memory_total_mib": int(mem_mib)})
    return gpus


def http_json(url: str, headers: dict | None = None) -> dict | list | None:
    try:
        resp = httpx.get(url, headers=headers or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def public_args(args: dict) -> dict:
    return {k: v for k, v in args.items() if k not in SECRET_ARGS}


def collect(args: dict, workload_path: Path, gateway_url: str | None, api_key: str | None,
            direct_url: str | None) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "client": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "gpu": gpu_info(),
        "workload": {"path": _display_path(workload_path), "sha256": file_sha256(workload_path)},
        "gateway": None if not gateway_url else {
            "health": http_json(f"{gateway_url}/health"),
            "models": http_json(f"{gateway_url}/v1/models", {"Authorization": f"Bearer {api_key}"}),
        },
        # vLLM's OpenAI server exposes its version at /version.
        "vllm": None if not direct_url else {"version": http_json(f"{direct_url}/version")},
        "args": public_args(args),
    }
