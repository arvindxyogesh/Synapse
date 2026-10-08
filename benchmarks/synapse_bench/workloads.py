"""Loading the committed workload files (see workloads/SOURCES.md)."""

import json
from pathlib import Path

WORKLOAD_DIR = Path(__file__).resolve().parents[1] / "workloads"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_perf_prompts(path: Path = WORKLOAD_DIR / "perf_prompts.jsonl") -> list[str]:
    return [row["prompt"] for row in read_jsonl(path)]


def load_gsm8k(path: Path = WORKLOAD_DIR / "gsm8k_test.jsonl") -> list[dict]:
    return read_jsonl(path)
