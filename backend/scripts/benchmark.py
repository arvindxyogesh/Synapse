#!/usr/bin/env python3
"""Benchmark Synapse's semantic cache against a live gateway instance.

Sends a labeled, realistic mix of exact-repeat / paraphrased / "confuser"
prompts through POST /v1/chat/completions and reports:

  - precision/recall of the cache's hit-vs-miss decisions (does a
    paraphrase of an already-seen question correctly hit the cache, and
    does a genuinely different-but-similar question correctly NOT hit it)
  - latency: cache miss (real model inference) vs cache hit
  - cost: incurred vs. what a cache-less gateway would have spent, using
    the gateway's own cost model

This requires a running gateway backed by a real model (Ollama) -- run it
against MOCK_MODE and every "latency" and "cost" number is meaningless
(the mock provider is near-instant and free), though cache precision/
recall is still valid since that only depends on the caching layer.

Usage:
    python backend/scripts/benchmark.py \\
        --gateway-url http://localhost:8000 \\
        --admin-key change-me-admin-key \\
        --model llama3 \\
        --requests 300
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass

import httpx

from benchmark_data import CLUSTERS, CONFUSERS


@dataclass
class RequestRecord:
    prompt: str
    cluster_id: str | None  # None for a confuser
    expected_hit: bool
    actual_hit: bool
    latency_ms: float
    cost_usd: float
    status_code: int


def build_traffic(n: int, confuser_rate: float, seed: int) -> list[dict]:
    """Build a shuffled traffic plan of n requests. Each cluster/confuser is
    weighted so popular topics repeat (Zipf-ish), like real FAQ traffic."""
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(len(CLUSTERS))]

    plan = []
    for _ in range(n):
        if rng.random() < confuser_rate:
            c = rng.choice(CONFUSERS)
            plan.append({"cluster_id": None, "prompt": c["prompt"]})
        else:
            cluster = rng.choices(CLUSTERS, weights=weights, k=1)[0]
            phrasing = rng.choice(cluster["paraphrases"])
            plan.append({"cluster_id": cluster["id"], "prompt": phrasing})
    rng.shuffle(plan)
    return plan


def get_or_create_api_key(client: httpx.Client, admin_key: str) -> str:
    resp = client.post(
        "/v1/admin/keys",
        json={"name": "benchmark"},
        headers={"x-admin-key": admin_key},
    )
    resp.raise_for_status()
    return resp.json()["api_key"]


def run(args: argparse.Namespace) -> None:
    client = httpx.Client(base_url=args.gateway_url, timeout=120)

    health = client.get("/health")
    health.raise_for_status()

    api_key = args.api_key or get_or_create_api_key(client, args.admin_key)
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    plan = build_traffic(args.requests, args.confuser_rate, args.seed)
    seen_clusters: set[str] = set()
    records: list[RequestRecord] = []

    print(f"Sending {len(plan)} requests to {args.gateway_url} (model={args.model})...")
    for i, item in enumerate(plan, 1):
        cluster_id = item["cluster_id"]
        prompt = item["prompt"]
        # A paraphrase counts as an expected-hit only from its *second*
        # appearance onward; confusers are always expected to miss.
        expected_hit = cluster_id is not None and cluster_id in seen_clusters
        if cluster_id is not None:
            seen_clusters.add(cluster_id)

        start = time.perf_counter()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": args.model, "messages": [{"role": "user", "content": prompt}]},
            headers=auth_headers,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            records.append(
                RequestRecord(prompt, cluster_id, expected_hit, False, latency_ms, 0.0, resp.status_code)
            )
            print(f"  [{i}/{len(plan)}] HTTP {resp.status_code}: {resp.text[:200]}")
            continue

        body = resp.json()
        actual_hit = bool(body["cached"])
        records.append(
            RequestRecord(prompt, cluster_id, expected_hit, actual_hit, latency_ms, body["cost_usd"], 200)
        )
        if i % 25 == 0 or i == len(plan):
            print(f"  [{i}/{len(plan)}] ...")

    report(records, args)


def report(records: list[RequestRecord], args: argparse.Namespace) -> None:
    ok = [r for r in records if r.status_code == 200]
    if len(ok) < len(records):
        print(f"\nWARNING: {len(records) - len(ok)} requests failed and are excluded below.")

    tp = sum(1 for r in ok if r.expected_hit and r.actual_hit)
    fn = sum(1 for r in ok if r.expected_hit and not r.actual_hit)
    fp = sum(1 for r in ok if not r.expected_hit and r.actual_hit)
    tn = sum(1 for r in ok if not r.expected_hit and not r.actual_hit)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    miss_latencies = [r.latency_ms for r in ok if not r.actual_hit]
    hit_latencies = [r.latency_ms for r in ok if r.actual_hit]
    avg_miss = statistics.mean(miss_latencies) if miss_latencies else float("nan")
    avg_hit = statistics.mean(hit_latencies) if hit_latencies else float("nan")
    p95_miss = statistics.quantiles(miss_latencies, n=20)[18] if len(miss_latencies) >= 20 else max(miss_latencies, default=float("nan"))

    cost_incurred = sum(r.cost_usd for r in ok)
    # Hits are recorded at cost_usd == 0. Approximate the cost a cache-less
    # gateway would have paid for each hit using this run's average miss
    # cost as a stand-in for "what this would have cost without the cache."
    avg_miss_cost = statistics.mean([r.cost_usd for r in ok if not r.actual_hit]) if miss_latencies else 0.0
    cost_saved = sum(avg_miss_cost for r in ok if r.actual_hit)
    cost_without_cache = cost_incurred + cost_saved

    print("\n" + "=" * 60)
    print("CACHE HIT/MISS DECISION QUALITY")
    print("=" * 60)
    print(f"  requests analyzed : {len(ok)}")
    print(f"  precision         : {precision:.1%}  (of cache hits, how many were actually the same question)")
    print(f"  recall            : {recall:.1%}  (of repeat questions, how many the cache actually caught)")
    print(f"  f1                : {f1:.1%}")
    print(f"  confusion matrix  : TP={tp} FP={fp} FN={fn} TN={tn}")

    print("\n" + "=" * 60)
    print("LATENCY")
    print("=" * 60)
    print(f"  avg cache-miss (real inference) : {avg_miss:.0f} ms")
    print(f"  p95 cache-miss                  : {p95_miss:.0f} ms")
    print(f"  avg cache-hit                   : {avg_hit:.0f} ms")
    if avg_hit and avg_hit == avg_hit:  # not NaN
        print(f"  speedup on a cache hit           : {avg_miss / avg_hit:.1f}x")

    print("\n" + "=" * 60)
    print("COST (estimated, via the gateway's reference pricing table)")
    print("=" * 60)
    print(f"  cost incurred (with cache)      : ${cost_incurred:.5f}")
    print(f"  cost without cache (estimated)  : ${cost_without_cache:.5f}")
    if cost_without_cache:
        print(f"  cost reduction                  : {(1 - cost_incurred / cost_without_cache):.1%}")

    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(
            {
                "requests": len(ok),
                "model": args.model,
                "cache_precision": precision,
                "cache_recall": recall,
                "cache_f1": f1,
                "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                "avg_miss_latency_ms": avg_miss,
                "p95_miss_latency_ms": p95_miss,
                "avg_hit_latency_ms": avg_hit,
                "cost_incurred_usd": cost_incurred,
                "cost_without_cache_usd": cost_without_cache,
                "records": [asdict(r) for r in records],
            },
            f,
            indent=2,
        )
    print(f"\nFull results written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--admin-key", default="change-me-admin-key")
    parser.add_argument("--api-key", default=None, help="Skip key creation and use this gateway API key directly")
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--confuser-rate", type=float, default=0.15, help="Fraction of traffic that is a confuser")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    try:
        run(args)
    except httpx.ConnectError:
        print(f"Could not connect to {args.gateway_url} -- is the gateway running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
