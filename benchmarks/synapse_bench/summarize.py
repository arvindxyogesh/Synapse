"""Build the published results from the committed raw result folders.

    python -m synapse_bench.summarize \\
        --perf results/2026-10-09T1514_perf --quality results/2026-10-09T0124_full \\
        --out results

Writes results/summary.json (read by the dashboard) and results/RESULTS.md
(tables quoted by the README). Every number in them is computed here from
files in the repo; nothing is typed in by hand. Provenance (run folders,
commits, GPU, vLLM version, settings) is written next to the numbers.
"""

import argparse
import json
import sys
from pathlib import Path

from synapse_bench.compare import compare
from synapse_bench.workloads import read_jsonl

VARIANTS = ["qwen2.5-7b-bf16", "qwen2.5-7b-gptq-int8", "qwen2.5-7b-awq"]  # baseline first
LABELS = {"qwen2.5-7b-bf16": "16-bit (BF16)", "qwen2.5-7b-gptq-int8": "8-bit (GPTQ)",
          "qwen2.5-7b-awq": "4-bit (AWQ)"}
BASELINE = "qwen2.5-7b-bf16"


def _load(path: Path):
    return json.loads(Path(path).read_text())


def speed(perf_dir: Path, variant: str) -> dict:
    """Main speed rows (repeat 0) by target and concurrency, plus the spread
    across the main run and its extra repeats (gateway, two levels)."""
    rows = _load(perf_dir / variant / "perf" / "summary.json")
    by_target: dict = {}
    for r in rows:
        if r["repeat"] == 0:
            by_target.setdefault(r["target"], {})[r["concurrency"]] = {
                k: r[k] for k in ("n_ok", "n_total", "n_errors", "ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms",
                                  "e2e_p95_ms", "decode_tok_s_p50", "aggregate_output_tok_s", "requests_per_s",
                                  "cost_per_1k_output_tokens_usd", "mean_output_tokens", "cached_prompt_tokens")
            }
    repeats_file = perf_dir / variant / "perf_repeats" / "summary.json"
    spread = {}
    if repeats_file.exists():
        all_runs = [r for r in rows if r["target"] == "gateway"] + _load(repeats_file)
        for c in sorted({r["concurrency"] for r in _load(repeats_file)}):
            runs = [r for r in all_runs if r["concurrency"] == c]
            spread[c] = {key: {"min": min(r[key] for r in runs), "max": max(r[key] for r in runs), "n_runs": len(runs)}
                         for key in ("ttft_p50_ms", "e2e_p50_ms", "aggregate_output_tok_s")}
    return {"by_target": by_target, "gateway_spread": spread}


def memory(quality_dir: Path, variant: str) -> dict:
    log = _load(quality_dir / variant / "perf" / "metadata.json")["vllm_log"]
    return {"weights_gib": log["weights_gib"], "kv_cache_memory_gib": log["kv_cache_memory_gib"],
            "kv_cache_tokens": log["kv_cache_tokens"], "max_concurrency": log["max_concurrency"],
            "quant_kernel": log["quant_kernel"]}


def quality(quality_dir: Path, variant: str) -> dict:
    s = _load(quality_dir / variant / "gsm8k" / "summary.json")
    out = {"n": s["n"], "n_errors": s["n_errors"], "n_possibly_truncated": s["n_possibly_truncated"],
           "strict": s["strict"], "flexible": s["flexible"]}
    if variant != BASELINE:
        base = read_jsonl(quality_dir / BASELINE / "gsm8k" / "answers.jsonl")
        mine = read_jsonl(quality_dir / variant / "gsm8k" / "answers.jsonl")
        out["vs_baseline"] = {kind: compare(base, mine, kind) for kind in ("strict", "flexible")}
    return out


def provenance(perf_dir: Path, quality_dir: Path) -> dict:
    def run(d: Path) -> dict:
        session = _load(d / "session.json")
        meta = _load(d / BASELINE / "perf" / "metadata.json")
        return {"folder": d.name, "commit": session["git_commit"], "dirty": session["git_dirty"],
                "gpu": session["gpu_label"], "gpu_memory_utilization": session["gpu_memory_utilization"],
                "vllm_version": session["vllm_version"], "vllm_args": session["vllm_args"],
                "variants": session["variants"], "started_utc": meta["timestamp_utc"],
                "gpu_price_per_hour": session["gpu_price_per_hour"]}

    perf_meta = _load(perf_dir / BASELINE / "perf" / "metadata.json")["args"]
    return {"speed_run": run(perf_dir), "quality_and_memory_run": run(quality_dir),
            "speed_settings": {k: perf_meta[k] for k in ("concurrency", "requests", "warmup", "max_tokens")}
            | {"fixed_length_outputs": not perf_meta["no_fixed_length"]}}


def cross_card(perf_dir: Path, other_dir: Path) -> dict:
    """Direct-to-vLLM throughput for the variants measured in both runs: the
    same benchmark on a different physical GPU, at a different time."""
    out = {"other_run": other_dir.name, "other_gpu": _load(other_dir / "session.json")["gpu_label"], "variants": {}}
    for v in VARIANTS:
        theirs_file = other_dir / v / "perf" / "summary.json"
        if not theirs_file.exists():
            continue
        ours = {r["concurrency"]: r for r in _load(perf_dir / v / "perf" / "summary.json")
                if r["target"] == "direct" and r["repeat"] == 0}
        theirs = {r["concurrency"]: r for r in _load(theirs_file) if r["target"] == "direct" and r["repeat"] == 0}
        out["variants"][v] = {c: {"this_run": ours[c]["aggregate_output_tok_s"],
                                  "other_run": theirs[c]["aggregate_output_tok_s"]}
                              for c in sorted(ours) if c in theirs}
    return out


def build(perf_dir: Path, quality_dir: Path, cross_card_dir: Path | None = None) -> dict:
    summary = {
        "provenance": provenance(perf_dir, quality_dir),
        "variants": {v: {"label": LABELS[v], "speed": speed(perf_dir, v), "memory": memory(quality_dir, v),
                         "quality": quality(quality_dir, v)} for v in VARIANTS},
    }
    if cross_card_dir is not None:
        summary["cross_card"] = cross_card(perf_dir, cross_card_dir)
    return summary


# -- Markdown ---------------------------------------------------------------------


def _f(x, fmt=".0f"):
    return "–" if x is None else format(x, fmt)


def markdown(summary: dict) -> str:
    p, variants = summary["provenance"], summary["variants"]
    sp, q = p["speed_run"], p["quality_and_memory_run"]
    lines = [
        "# Quantization benchmark results",
        "",
        "Generated by `python -m synapse_bench.summarize` from the raw result folders in this directory. "
        "Do not edit by hand.",
        "",
        "- **Model:** Qwen2.5-7B-Instruct, three weight precisions (pinned revisions in each run's `session.json`).",
        f"- **Hardware:** {sp['gpu']}.",
        f"- **Serving:** vLLM {sp['vllm_version']}, `{' '.join(sp['vllm_args'])}`.",
        f"- **Speed run:** `{sp['folder']}`, commit `{sp['commit'][:7]}`, started {sp['started_utc']}.",
        f"- **Accuracy and memory run:** `{q['folder']}`, commit `{q['commit'][:7]}`, started {q['started_utc']}.",
        f"- **Speed settings:** {p['speed_settings']['requests']} measured requests per level after "
        f"{p['speed_settings']['warmup']} warm-up, exactly {p['speed_settings']['max_tokens']} output tokens each "
        f"(`ignore_eos`), closed loop, Synapse cache bypassed, vLLM prefix caching off.",
        "",
        "## Memory (from vLLM's startup log)",
        "",
        "| Variant | Weights | KV cache | KV cache tokens | Max concurrent 4,096-token requests | Kernel |",
        "|---|---|---|---|---|---|",
    ]
    for d in variants.values():
        m = d["memory"]
        kernel = ", ".join(k.split(" for ")[0] for k in m["quant_kernel"]) if m["quant_kernel"] else "(16-bit, none)"
        kernel = ", ".join(dict.fromkeys(kernel.split(", ")))
        lines.append(f"| {d['label']} | {m['weights_gib']:.2f} GiB | {m['kv_cache_memory_gib']:.2f} GiB | "
                     f"{m['kv_cache_tokens']:,} | {m['max_concurrency']['requests']:.0f} | {kernel} |")

    lines += ["", "## Accuracy (GSM8K test set, greedy, exact match)", "",
              "| Variant | n | Strict (last `\\boxed{}`) [95% CI] | Flexible (last number) [95% CI] | "
              "vs 16-bit, strict: difference, McNemar p |", "|---|---|---|---|---|"]
    for d in variants.values():
        qq = d["quality"]
        s, f = qq["strict"], qq["flexible"]
        cmp = qq.get("vs_baseline", {}).get("strict")
        vs = "baseline" if cmp is None else f"{cmp['difference_b_minus_a'] * 100:+.1f} pts, p = {cmp['mcnemar_p']:.2f}"
        lines.append(f"| {d['label']} | {qq['n']} | {s['accuracy']:.1%} [{s['ci95_low']:.1%}, {s['ci95_high']:.1%}] | "
                     f"{f['accuracy']:.1%} [{f['ci95_low']:.1%}, {f['ci95_high']:.1%}] | {vs} |")

    sections = (("direct", "Speed, directly against vLLM"), ("gateway", "Speed, through the Synapse gateway"))
    for target, title in sections:
        lines += ["", f"## {title}", "",
                  "| Variant | Concurrency | TTFT p50 / p95 (ms) | E2E p50 / p95 (ms) | "
                  "Decode tok/s per request (p50) | Throughput (output tok/s) | Requests/s | "
                  "$ per 1K output tokens | Failed |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for d in variants.values():
            for c, r in sorted(d["speed"]["by_target"][target].items(), key=lambda kv: int(kv[0])):
                lines.append(
                    f"| {d['label']} | {c} | {_f(r['ttft_p50_ms'])} / {_f(r['ttft_p95_ms'])} | "
                    f"{_f(r['e2e_p50_ms'])} / {_f(r['e2e_p95_ms'])} | {_f(r['decode_tok_s_p50'], '.1f')} | "
                    f"{_f(r['aggregate_output_tok_s'], '.1f')} | {_f(r['requests_per_s'], '.2f')} | "
                    f"{_f(r['cost_per_1k_output_tokens_usd'], '.5f')} | {r['n_errors']}/{r['n_total']} |")

    lines += ["", "## Run-to-run spread (gateway, same job, 3 runs per level)", "",
              "| Variant | Concurrency | TTFT p50 range (ms) | E2E p50 range (ms) | Throughput range (tok/s) |",
              "|---|---|---|---|---|"]
    for d in variants.values():
        for c, s in sorted(d["speed"]["gateway_spread"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {d['label']} | {c} | {s['ttft_p50_ms']['min']:.0f}–{s['ttft_p50_ms']['max']:.0f} | "
                         f"{s['e2e_p50_ms']['min']:.0f}–{s['e2e_p50_ms']['max']:.0f} | "
                         f"{s['aggregate_output_tok_s']['min']:.1f}–{s['aggregate_output_tok_s']['max']:.1f} |")
    if "cross_card" in summary:
        cc = summary["cross_card"]
        lines += ["", "## Card-to-card check (direct to vLLM, two different A40s)", "",
                  f"Same benchmark, same settings, on a different physical GPU: `{cc['other_run']}` "
                  f"({cc['other_gpu']}).", "",
                  "| Variant | Concurrency | Throughput, this run (tok/s) | Throughput, other run (tok/s) "
                  "| Difference |",
                  "|---|---|---|---|---|"]
        for v, levels in cc["variants"].items():
            for c, t in sorted(levels.items(), key=lambda kv: int(kv[0])):
                diff = (t["other_run"] / t["this_run"] - 1) * 100
                lines.append(f"| {LABELS[v]} | {c} | {t['this_run']:.1f} | {t['other_run']:.1f} | {diff:+.1f}% |")
    lines += ["", f"Cost per 1K output tokens uses ${sp['gpu_price_per_hour']}/hour: the Great Lakes billed rate "
                  "for this job shape (1 A40, 8 cores, 64 GB), assuming the GPU stays this busy.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perf", type=Path, required=True, help="speed run folder")
    p.add_argument("--quality", type=Path, required=True, help="run folder with valid GSM8K + vLLM memory data")
    p.add_argument("--cross-card", type=Path, help="an earlier run on a different GPU, to compare direct throughput")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    summary = build(args.perf, args.quality, args.cross_card)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "RESULTS.md").write_text(markdown(summary))
    print(f"wrote {args.out / 'summary.json'} and {args.out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
