"""The committed workload files must be exactly what prepare_workloads.py
produced -- a hand edit would silently change every benchmark result."""

import hashlib
import re

from synapse_bench.workloads import WORKLOAD_DIR, load_gsm8k, load_perf_prompts


def _hashes_in_sources_md() -> dict[str, str]:
    text = (WORKLOAD_DIR / "SOURCES.md").read_text()
    return dict(re.findall(r"\| (\S+\.jsonl) \| \d+ \| `([0-9a-f]{64})` \|", text))


def test_workload_files_match_recorded_hashes():
    recorded = _hashes_in_sources_md()
    assert set(recorded) == {"perf_prompts.jsonl", "gsm8k_test.jsonl"}
    for name, expected in recorded.items():
        assert hashlib.sha256((WORKLOAD_DIR / name).read_bytes()).hexdigest() == expected, name


def test_workloads_load():
    prompts = load_perf_prompts()
    assert len(prompts) == 200 and all(p.strip() for p in prompts)
    problems = load_gsm8k()
    assert len(problems) == 1319
    assert all("####" in p["answer"] for p in problems)
