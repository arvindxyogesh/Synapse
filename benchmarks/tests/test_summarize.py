"""The published tables are generated from the committed raw results. These
tests pin the generator to those files, so a number in RESULTS.md can't
drift from the data it claims to come from."""

import json
from pathlib import Path

import pytest

from synapse_bench import summarize

RESULTS = Path(__file__).resolve().parents[1] / "results"
PERF = RESULTS / "2026-10-09T1514_perf"  # speed (after the D16 fix)
QUALITY = RESULTS / "2026-10-09T0124_full"  # GSM8K + vLLM memory
OTHER = RESULTS / "2026-10-08T2214_full"  # an earlier run on a different A40

pytestmark = pytest.mark.skipif(not PERF.exists(), reason="committed results not present")


@pytest.fixture(scope="module")
def summary():
    return summarize.build(PERF, QUALITY, OTHER)


def _raw(run, variant, target, concurrency):
    rows = json.loads((run / variant / "perf" / "summary.json").read_text())
    return next(r for r in rows if r["target"] == target and r["concurrency"] == concurrency and r["repeat"] == 0)


@pytest.mark.parametrize("variant", summarize.VARIANTS)
@pytest.mark.parametrize("target", ["direct", "gateway"])
def test_speed_numbers_come_straight_from_raw_files(summary, variant, target):
    for c in (1, 4, 16, 64):
        raw = _raw(PERF, variant, target, c)
        gen = summary["variants"][variant]["speed"]["by_target"][target][c]
        for key in ("ttft_p50_ms", "e2e_p95_ms", "aggregate_output_tok_s", "cost_per_1k_output_tokens_usd", "n_errors"):
            assert gen[key] == raw[key], (variant, target, c, key)


def test_accuracy_comes_from_the_quality_run(summary):
    for v in summarize.VARIANTS:
        raw = json.loads((QUALITY / v / "gsm8k" / "summary.json").read_text())
        assert summary["variants"][v]["quality"]["strict"] == raw["strict"]
    assert "vs_baseline" not in summary["variants"][summarize.BASELINE]["quality"]
    assert 0 <= summary["variants"]["qwen2.5-7b-awq"]["quality"]["vs_baseline"]["strict"]["mcnemar_p"] <= 1


def test_provenance_is_recorded(summary):
    p = summary["provenance"]
    assert p["speed_run"]["commit"] and p["speed_run"]["dirty"] is False
    assert p["quality_and_memory_run"]["folder"] == QUALITY.name
    assert p["speed_settings"]["fixed_length_outputs"] is True


def test_cross_card_only_covers_variants_measured_twice(summary):
    assert set(summary["cross_card"]["variants"]) == {"qwen2.5-7b-bf16", "qwen2.5-7b-gptq-int8"}


def test_markdown_renders_every_section(summary):
    md = summarize.markdown(summary)
    for heading in ("## Memory", "## Accuracy", "## Speed, directly against vLLM",
                    "## Speed, through the Synapse gateway", "## Run-to-run spread", "## Card-to-card check"):
        assert heading in md
    assert "Do not edit by hand" in md


README = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_results_block_matches_generated(summary):
    """The README's results tables are generated; this fails if anyone edits
    them by hand or the data changes without re-running summarize."""
    text = README.read_text()
    block = text[text.index(summarize.README_BEGIN) + len(summarize.README_BEGIN):text.index(summarize.README_END)]
    assert block.strip() == summarize.readme_results(summary).strip()


def test_readme_results_numbers_trace_to_raw_files(summary):
    block = summarize.readme_results(summary)
    awq = json.loads((PERF / "qwen2.5-7b-awq" / "perf" / "summary.json").read_text())
    c64 = next(r for r in awq if r["target"] == "direct" and r["concurrency"] == 64 and r["repeat"] == 0)
    assert f"{c64['aggregate_output_tok_s']:,.0f} tok/s" in block
