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

This requires a running gateway backed by a real model (Ollama or vLLM) -- run it
against MOCK_MODE and every "latency" and "cost" number is meaningless
(the mock provider is near-instant and free), though cache precision/
recall is still valid since that only depends on the caching layer.

Usage:
    python backend/scripts/benchmark.py \\
        --gateway-url http://localhost:8000 \\
        --admin-key change-me-admin-key \\
        --model llama3 \\
        --requests 300

Pass --rounds N (N > 1) to send the traffic plan repeatedly against the
same persistent gateway and watch the *adaptive cache threshold*
(app/threshold_controller.py) converge across rounds -- each round's
precision/recall/threshold/estimated-FP-rate is printed and written to the
output JSON, so you can chart precision recovering as the controller
tightens the threshold in response to shadow-verified false positives.
For a convergence run that's actually visible in a handful of rounds
(rather than the default threshold already being strict enough to avoid
false positives in the first place), start the gateway with a
deliberately loose starting point and a higher sampling rate, e.g.:

    CACHE_SIMILARITY_THRESHOLD=0.75 SHADOW_VERIFY_SAMPLE_RATE=1.0 \\
        uvicorn app.main:app

    python backend/scripts/benchmark.py --rounds 6 --requests 150
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
    # True if this *exact* prompt string was already sent earlier in the
    # run. Those are trivial exact-hash cache hits (see cache.py's
    # exact-match check) that say nothing about the *similarity* decision
    # -- with a small, fixed prompt universe, traffic saturates into
    # almost-all-exact-repeats after enough requests, which would
    # otherwise quietly inflate precision/recall into a meaningless 100%.
    # compute_metrics() reports the non-exact-repeat subset separately so
    # that doesn't happen unnoticed.
    is_exact_repeat: bool = False


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


def fetch_threshold_state(client: httpx.Client, model: str) -> dict | None:
    resp = client.get("/v1/stats/cache-threshold")
    resp.raise_for_status()
    for row in resp.json():
        if row["model"] == model:
            return row
    return None


def send_round(
    client: httpx.Client,
    auth_headers: dict,
    model: str,
    plan: list[dict],
    seen_clusters: set[str],
    seen_exact_prompts: set[str],
    round_no: int,
) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    print(f"Round {round_no}: sending {len(plan)} requests (model={model})...")
    for i, item in enumerate(plan, 1):
        cluster_id = item["cluster_id"]
        prompt = item["prompt"]
        # A paraphrase counts as an expected-hit only from its *second*
        # appearance onward (tracked across the whole run, not per round --
        # the cache persists across rounds against a live gateway).
        # Confusers are expected to miss *unless* this exact confuser text
        # was already sent before (there are only 15 of them, drawn with
        # replacement, so exact repeats are common over a few hundred
        # requests -- and a repeat of the identical string is a legitimate
        # exact-hash cache hit, not a false positive).
        is_exact_repeat = prompt in seen_exact_prompts
        if cluster_id is not None:
            expected_hit = cluster_id in seen_clusters
            seen_clusters.add(cluster_id)
        else:
            expected_hit = is_exact_repeat
        seen_exact_prompts.add(prompt)

        start = time.perf_counter()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            headers=auth_headers,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            records.append(
                RequestRecord(
                    prompt, cluster_id, expected_hit, False, latency_ms, 0.0, resp.status_code, is_exact_repeat
                )
            )
            print(f"  [{i}/{len(plan)}] HTTP {resp.status_code}: {resp.text[:200]}")
            continue

        body = resp.json()
        actual_hit = bool(body["cached"])
        records.append(
            RequestRecord(
                prompt, cluster_id, expected_hit, actual_hit, latency_ms, body["cost_usd"], 200, is_exact_repeat
            )
        )
        if i % 25 == 0 or i == len(plan):
            print(f"  [{i}/{len(plan)}] ...")

    return records


def _precision_recall_f1(rows: list[RequestRecord]) -> dict:
    tp = sum(1 for r in rows if r.expected_hit and r.actual_hit)
    fn = sum(1 for r in rows if r.expected_hit and not r.actual_hit)
    fp = sum(1 for r in rows if not r.expected_hit and r.actual_hit)
    tn = sum(1 for r in rows if not r.expected_hit and not r.actual_hit)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n": len(rows),
    }


def compute_metrics(records: list[RequestRecord]) -> dict:
    ok = [r for r in records if r.status_code == 200]
    failed = len(records) - len(ok)

    overall = _precision_recall_f1(ok)
    # The subset that actually exercises the similarity decision -- exact
    # repeats of an already-seen string are trivial exact-hash hits (see
    # cache.py) and say nothing about threshold correctness. Reported
    # separately so "100% precision" can't quietly just mean "the traffic
    # ran out of new things to ask."
    novel = _precision_recall_f1([r for r in ok if not r.is_exact_repeat])

    tp, fn, fp, tn = (overall["confusion_matrix"][k] for k in ("tp", "fn", "fp", "tn"))
    precision, recall, f1 = overall["precision"], overall["recall"], overall["f1"]

    miss_latencies = [r.latency_ms for r in ok if not r.actual_hit]
    hit_latencies = [r.latency_ms for r in ok if r.actual_hit]
    avg_miss = statistics.mean(miss_latencies) if miss_latencies else float("nan")
    avg_hit = statistics.mean(hit_latencies) if hit_latencies else float("nan")
    p95_miss = (
        statistics.quantiles(miss_latencies, n=20)[18]
        if len(miss_latencies) >= 20
        else max(miss_latencies, default=float("nan"))
    )

    cost_incurred = sum(r.cost_usd for r in ok)
    avg_miss_cost = statistics.mean([r.cost_usd for r in ok if not r.actual_hit]) if miss_latencies else 0.0
    cost_saved = sum(avg_miss_cost for r in ok if r.actual_hit)
    cost_without_cache = cost_incurred + cost_saved

    return {
        "requests": len(ok),
        "failed": failed,
        "cache_precision": precision,
        "cache_recall": recall,
        "cache_f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "novel_only": novel,  # excludes trivial exact-repeat hits -- see _precision_recall_f1 call above
        "avg_miss_latency_ms": avg_miss,
        "p95_miss_latency_ms": p95_miss,
        "avg_hit_latency_ms": avg_hit,
        "cost_incurred_usd": cost_incurred,
        "cost_without_cache_usd": cost_without_cache,
    }


def print_round_summary(round_no: int, metrics: dict, threshold_state: dict | None) -> None:
    cm = metrics["confusion_matrix"]
    threshold_bits = ""
    if threshold_state is not None:
        threshold_bits = (
            f"  threshold={threshold_state['threshold']:.3f}"
            f"  est.fp_rate={threshold_state['estimated_false_positive_rate']:.1%}"
            f"  verified={threshold_state['verified_samples']}"
        )
    print(
        f"round {round_no:>2}: precision={metrics['cache_precision']:.1%}  "
        f"recall={metrics['cache_recall']:.1%}  f1={metrics['cache_f1']:.1%}  "
        f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}{threshold_bits}"
    )
    novel = metrics["novel_only"]
    if novel["n"]:
        ncm = novel["confusion_matrix"]
        print(
            f"          novel-only (excl. exact repeats, n={novel['n']}): "
            f"precision={novel['precision']:.1%}  recall={novel['recall']:.1%}  "
            f"TP={ncm['tp']} FP={ncm['fp']} FN={ncm['fn']} TN={ncm['tn']}"
        )
    else:
        print("          novel-only: n=0 -- every request this round was an exact repeat of an earlier prompt")


def run(args: argparse.Namespace) -> None:
    client = httpx.Client(base_url=args.gateway_url, timeout=120)

    health = client.get("/health")
    health.raise_for_status()

    api_key = args.api_key or get_or_create_api_key(client, args.admin_key)
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    seen_clusters: set[str] = set()
    seen_exact_prompts: set[str] = set()
    rounds: list[dict] = []
    all_records: list[RequestRecord] = []

    for round_no in range(1, args.rounds + 1):
        plan = build_traffic(args.requests, args.confuser_rate, args.seed + round_no)
        records = send_round(client, auth_headers, args.model, plan, seen_clusters, seen_exact_prompts, round_no)
        all_records.extend(records)
        metrics = compute_metrics(records)
        threshold_state = fetch_threshold_state(client, args.model)
        print_round_summary(round_no, metrics, threshold_state)
        rounds.append(
            {
                "round": round_no,
                "threshold_state": threshold_state,
                **metrics,
                "records": [asdict(r) for r in records],
            }
        )

    final = rounds[-1]
    # Aggregated across every round: a converged late round can easily have
    # zero cache misses at all (nothing left to compare latency/cost
    # against), so the last round alone isn't a meaningful latency/cost
    # summary even though it's the right place to read final precision/
    # recall/threshold from.
    aggregate = compute_metrics(all_records)

    print("\n" + "=" * 60)
    print(f"FINAL RESULT (round {final['round']}/{args.rounds}; latency/cost aggregated across all rounds)")
    print("=" * 60)
    print(f"  precision (final round)  : {final['cache_precision']:.1%}")
    print(f"  recall (final round)     : {final['cache_recall']:.1%}")
    print(f"  f1 (final round)         : {final['cache_f1']:.1%}")
    if aggregate["avg_hit_latency_ms"] == aggregate["avg_hit_latency_ms"]:  # not NaN
        print(f"  avg cache-miss latency   : {aggregate['avg_miss_latency_ms']:.0f} ms")
        print(f"  avg cache-hit latency    : {aggregate['avg_hit_latency_ms']:.0f} ms")
        if aggregate["avg_hit_latency_ms"]:
            speedup = aggregate["avg_miss_latency_ms"] / aggregate["avg_hit_latency_ms"]
            print(f"  speedup on a cache hit   : {speedup:.1f}x")
    if aggregate["cost_without_cache_usd"]:
        reduction = 1 - aggregate["cost_incurred_usd"] / aggregate["cost_without_cache_usd"]
        print(f"  cost reduction           : {reduction:.1%}")
    if final["threshold_state"] is not None:
        print(f"  adaptive threshold       : {final['threshold_state']['threshold']:.3f}")
        print(f"  estimated FP rate        : {final['threshold_state']['estimated_false_positive_rate']:.1%}")

    if args.rounds > 1:
        print("\nPer-round precision/recall/threshold (see output JSON for full series):")
        for r in rounds:
            print_round_summary(r["round"], r, r["threshold_state"])

    with open(args.output, "w") as f:
        json.dump({"model": args.model, "aggregate": aggregate, "rounds": rounds}, f, indent=2)
    print(f"\nFull results written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--admin-key", default="change-me-admin-key")
    parser.add_argument("--api-key", default=None, help="Skip key creation and use this gateway API key directly")
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--requests", type=int, default=300, help="Requests per round")
    parser.add_argument("--rounds", type=int, default=1, help="Repeat the traffic plan N times to show convergence")
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
