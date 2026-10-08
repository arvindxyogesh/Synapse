"""Speed benchmark: latency, throughput and cost of one model behind the
gateway (and, optionally, of the same model queried directly) at several
concurrency levels.

    python -m synapse_bench.run_perf \\
        --model qwen2.5-7b-awq \\
        --direct-url http://localhost:8002 --direct-model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --gpu-price-per-hour 0.86 \\
        --out results/2026-10-20_l40s/qwen2.5-7b-awq

Reads the gateway API key from SYNAPSE_API_KEY (never written to results).
Writes to --out:
    metadata.json   when/what/where (see metadata.py)
    requests.jsonl  one raw record per measured request
    summary.json    one summary per (target, repeat, concurrency level)

Exits non-zero, after saving everything, if any response failed an
integrity check (e.g. the cache wasn't bypassed) -- such a run must not be
published.
"""

import argparse
import asyncio
import contextlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from synapse_bench import metadata
from synapse_bench.gpu import GpuMemorySampler, parse_vllm_log
from synapse_bench.loadgen import INTEGRITY_ERRORS, LevelResult, Target, run_level
from synapse_bench.metrics import summarize_level
from synapse_bench.workloads import WORKLOAD_DIR, load_perf_prompts

# Above this, the load generator itself may be the bottleneck (DECISIONS.md D5).
CLIENT_CPU_WARNING = 0.8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="gateway model name (as registered in models.toml)")
    p.add_argument("--gateway-url", default="http://localhost:8000")
    p.add_argument("--direct-url", help="also benchmark this vLLM server directly (base URL, no /v1)")
    p.add_argument("--direct-model", help="the model name the direct server expects (its upstream name)")
    p.add_argument("--concurrency", default="1,4,16,64", help="comma-separated levels")
    p.add_argument("--requests", type=int, default=100, help="measured requests per level")
    p.add_argument("--warmup", type=int, default=10, help="discarded warm-up requests per level")
    p.add_argument("--repeat", type=int, default=1, help="run every level this many times")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--no-fixed-length", action="store_true",
                   help="don't send ignore_eos (needed for non-vLLM backends; output lengths then vary)")
    p.add_argument("--gpu-price-per-hour", type=float, help="USD/hour, for cost per 1K tokens")
    p.add_argument("--timeout", type=float, default=300.0, help="per-request timeout, seconds")
    p.add_argument("--prompts", type=Path, default=WORKLOAD_DIR / "perf_prompts.jsonl")
    p.add_argument("--gpu-sample", action="store_true", help="record peak nvidia-smi memory per level")
    p.add_argument("--gpu-index", type=int, action="append",
                   help="only record this GPU's memory (nvidia-smi index; repeatable). Default: all GPUs")
    p.add_argument("--vllm-log", type=Path, help="vLLM server log to parse for weight / KV-cache memory")
    p.add_argument("--out", type=Path, required=True, help="results directory (must not exist yet)")
    args = p.parse_args(argv)
    if args.direct_url and not args.direct_model:
        p.error("--direct-url needs --direct-model")
    args.levels = [int(c) for c in args.concurrency.split(",")]
    return args


def build_targets(args: argparse.Namespace, api_key: str) -> dict[str, Target]:
    extra = {} if args.no_fixed_length else {"ignore_eos": True}
    targets = {
        "gateway": Target(
            url=f"{args.gateway_url.rstrip('/')}/v1/chat/completions",
            model=args.model,
            headers={"Authorization": f"Bearer {api_key}", "x-synapse-cache": "bypass"},
            extra_body=extra,
            via_gateway=True,
        )
    }
    if args.direct_url:
        targets["direct"] = Target(
            url=f"{args.direct_url.rstrip('/')}/v1/chat/completions",
            model=args.direct_model,
            extra_body=extra,
            via_gateway=False,
        )
    return targets


def summary_row(target_name: str, repeat: int, level: LevelResult, gpu_price: float | None) -> dict:
    s = summarize_level(level.records, level.concurrency, level.wall_time_s, gpu_price)
    return {"target": target_name, "repeat": repeat, "client_cpu_fraction": level.client_cpu_fraction,
            **s.to_dict()}


def print_row(row: dict) -> None:
    def f(v, fmt):
        return format(v, fmt) if v is not None else "-"

    print(
        f"{row['target']:<8} rep={row['repeat']} c={row['concurrency']:<3} ok={row['n_ok']}/{row['n_total']} "
        f"ttft p50={f(row['ttft_p50_ms'], '.0f')}ms p95={f(row['ttft_p95_ms'], '.0f')}ms  "
        f"e2e p50={f(row['e2e_p50_ms'], '.0f')}ms p95={f(row['e2e_p95_ms'], '.0f')}ms  "
        f"agg={f(row['aggregate_output_tok_s'], '.1f')} tok/s  client_cpu={row['client_cpu_fraction']:.2f}",
        flush=True,
    )


async def run(args: argparse.Namespace, api_key: str) -> int:
    prompts = load_perf_prompts(args.prompts)
    if args.warmup + args.requests > len(prompts):
        print(f"warning: warmup+requests ({args.warmup + args.requests}) > {len(prompts)} prompts; "
              "some warm-up prompts will repeat in the measured run", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=False)
    meta = metadata.collect(vars(args) | {"api_key": api_key}, args.prompts, args.gateway_url, api_key,
                            args.direct_url)
    # Weight memory and KV-cache capacity, from the server's own startup log
    # (see gpu.py for why nvidia-smi alone can't show these).
    meta["vllm_log"] = parse_vllm_log(args.vllm_log.read_text()) if args.vllm_log else None
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    rows, integrity_failures = [], 0
    with open(args.out / "requests.jsonl", "w") as raw:
        for target_name, target in build_targets(args, api_key).items():
            for repeat in range(args.repeat):
                for concurrency in args.levels:
                    indices = set(args.gpu_index) if args.gpu_index else None
                    sampler = GpuMemorySampler(gpu_indices=indices) if args.gpu_sample else None
                    with sampler or contextlib.nullcontext():
                        level = await run_level(target, prompts, concurrency, args.requests, args.max_tokens,
                                                warmup_requests=args.warmup, timeout_s=args.timeout)
                    for r in level.records:
                        raw.write(json.dumps({"target": target_name, "repeat": repeat, "concurrency": concurrency,
                                              **asdict(r)}) + "\n")
                    row = summary_row(target_name, repeat, level, args.gpu_price_per_hour)
                    row["gpu_memory"] = sampler.result() if sampler else None
                    rows.append(row)
                    print_row(row)
                    if row["client_cpu_fraction"] > CLIENT_CPU_WARNING:
                        print(f"warning: client CPU {row['client_cpu_fraction']:.2f} -- the load generator may be "
                              "the bottleneck at this level", file=sys.stderr)
                    if row["cached_prompt_tokens"]:
                        print(f"warning: the backend served {row['cached_prompt_tokens']} prompt tokens from its own "
                              "prompt cache at this level -- TTFT is not comparable (start vLLM with "
                              "--no-enable-prefix-caching)", file=sys.stderr)
                    integrity_failures += sum(1 for r in level.records if r.error in INTEGRITY_ERRORS)

    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    if integrity_failures:
        print(f"ERROR: {integrity_failures} responses failed integrity checks (see requests.jsonl). "
              "These results are not valid measurements.", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("SYNAPSE_API_KEY")
    if not api_key:
        print("set SYNAPSE_API_KEY to a gateway API key", file=sys.stderr)
        return 1
    return asyncio.run(run(args, api_key))


if __name__ == "__main__":
    sys.exit(main())
