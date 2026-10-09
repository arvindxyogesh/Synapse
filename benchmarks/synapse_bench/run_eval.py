"""Quality eval: GSM8K exact match for one model behind the gateway.

    python -m synapse_bench.run_eval --model qwen2.5-7b-awq \\
        --out results/2026-10-20_l40s/qwen2.5-7b-awq/gsm8k

Greedy decoding (temperature 0), cache bypassed, outputs NOT fixed-length
(the model must stop by itself -- see DECISIONS.md D8). Reads the API key
from SYNAPSE_API_KEY. Writes metadata.json, answers.jsonl (one row per
question, including the full response) and summary.json.

Scoring is conservative: a question whose request failed counts as wrong,
never as skipped, so errors can't raise the accuracy. Exits non-zero if any
response came from the mock provider or the cache.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from synapse_bench import metadata
from synapse_bench.gsm8k import build_prompt, score
from synapse_bench.metrics import wilson_interval
from synapse_bench.workloads import WORKLOAD_DIR, load_gsm8k


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--gateway-url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=32, help="questions in flight at once")
    # 512 cut off ~2% of the 7B's genuine (non-looping) solutions in the first
    # full run, scoring them wrong for length rather than math (DECISIONS D15).
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, help="only the first N questions (for dry runs; never for reported results)")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--dataset", type=Path, default=WORKLOAD_DIR / "gsm8k_test.jsonl")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


async def answer_one(client: httpx.AsyncClient, url: str, headers: dict, model: str, problem: dict,
                     max_tokens: int) -> dict:
    row = {"id": problem["id"], "response": None, "completion_tokens": None, "provider": None, "error": None}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(problem["question"])}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            row["error"] = f"HTTP {resp.status_code}"
        else:
            data = resp.json()
            row["response"] = data["choices"][0]["message"]["content"]
            row["completion_tokens"] = (data.get("usage") or {}).get("completion_tokens")
            row["provider"] = data.get("provider")
            if data.get("provider") in ("mock", "cache") or data.get("cached"):
                row["error"] = f"answered by {data.get('provider')}"
    except httpx.TimeoutException:
        row["error"] = "timeout"
    except httpx.HTTPError as exc:
        row["error"] = type(exc).__name__

    graded = score(row["response"] or "", problem["answer"])
    if row["error"]:
        graded["correct_strict"] = graded["correct_flexible"] = False
    # The gateway reports finish_reason "stop" even when a reply hit the cap,
    # so a reply that used exactly max_tokens may have been cut off mid-answer.
    row["possibly_truncated"] = row["completion_tokens"] == max_tokens
    return row | graded


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    summary = {"n": n, "n_errors": sum(1 for r in rows if r["error"]),
               "n_possibly_truncated": sum(1 for r in rows if r["possibly_truncated"])}
    for kind in ("strict", "flexible"):
        correct = sum(1 for r in rows if r[f"correct_{kind}"])
        low, high = wilson_interval(correct, n)
        summary[kind] = {"correct": correct, "accuracy": correct / n, "ci95_low": low, "ci95_high": high}
    return summary


async def run(args: argparse.Namespace, api_key: str) -> int:
    problems = load_gsm8k(args.dataset)
    if args.limit:
        problems = problems[: args.limit]

    args.out.mkdir(parents=True, exist_ok=False)
    meta = metadata.collect(vars(args) | {"api_key": api_key}, args.dataset, args.gateway_url, api_key, None)
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    url = f"{args.gateway_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "x-synapse-cache": "bypass"}
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        async def guarded(problem: dict) -> dict:
            async with semaphore:
                return await answer_one(client, url, headers, args.model, problem, args.max_tokens)

        rows = await asyncio.gather(*(guarded(p) for p in problems))

    with open(args.out / "answers.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = summarize(rows) | {"model": args.model, "limit": args.limit}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    s, fl = summary["strict"], summary["flexible"]
    print(f"{args.model}: n={summary['n']} errors={summary['n_errors']} "
          f"possibly_truncated={summary['n_possibly_truncated']}\n"
          f"  strict   {s['accuracy']:.1%} [{s['ci95_low']:.1%}, {s['ci95_high']:.1%}]\n"
          f"  flexible {fl['accuracy']:.1%} [{fl['ci95_low']:.1%}, {fl['ci95_high']:.1%}]")

    invalid = sum(1 for r in rows if (r["error"] or "").startswith("answered by"))
    if invalid:
        print(f"ERROR: {invalid} answers came from the mock provider or cache; results are invalid.", file=sys.stderr)
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
