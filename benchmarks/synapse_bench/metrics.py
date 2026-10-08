"""Pure functions that turn raw request records into reported numbers.

Kept free of I/O so each formula can be unit-tested against hand-computed
values. Definitions (also in benchmarks/README.md):

- TTFT: client send -> first streamed chunk containing text.
- End-to-end (e2e) latency: client send -> [DONE].
- Decode speed (per request): (output_tokens - 1) / (e2e - ttft). The first
  token is excluded because its time is already counted in TTFT; what's left
  is the token-by-token generation phase.
- Aggregate throughput: total output tokens of successful requests / wall
  time of the whole level. This is what a GPU "produces" at that
  concurrency, and what cost is computed from.
- Cost per 1K output tokens: GPU $/hour / (aggregate tok/s * 3600) * 1000,
  i.e. assuming the GPU is kept exactly this busy for the whole hour.
"""

import math
from collections import Counter
from dataclasses import asdict, dataclass

from synapse_bench.records import RequestRecord


def percentile(values: list[float], p: float) -> float:
    """p-th percentile (0-100) with linear interpolation between the two
    nearest ranks -- the same definition as numpy's default."""
    if not values:
        raise ValueError("percentile of an empty list")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def decode_tokens_per_second(record: RequestRecord) -> float | None:
    """Per-request generation speed after the first token, or None when it
    can't be computed (failed request, fewer than 2 tokens, or no time
    between first token and end)."""
    if not record.ok or record.output_tokens is None or record.ttft_s is None or record.e2e_s is None:
        return None
    decode_time = record.e2e_s - record.ttft_s
    if record.output_tokens < 2 or decode_time <= 0:
        return None
    return (record.output_tokens - 1) / decode_time


def cost_per_1k_output_tokens(gpu_price_per_hour: float, aggregate_tokens_per_second: float) -> float | None:
    if aggregate_tokens_per_second <= 0:
        return None
    tokens_per_hour = aggregate_tokens_per_second * 3600
    return gpu_price_per_hour / tokens_per_hour * 1000


@dataclass
class LevelSummary:
    concurrency: int
    n_total: int
    n_ok: int
    n_errors: int
    error_types: dict[str, int]
    wall_time_s: float
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    e2e_p50_ms: float | None
    e2e_p95_ms: float | None
    mean_output_tokens: float | None
    decode_tok_s_p50: float | None
    aggregate_output_tok_s: float | None
    requests_per_s: float | None
    gpu_price_per_hour: float | None
    cost_per_1k_output_tokens_usd: float | None
    # Sum of backend-reported cached prompt tokens; None if the backend never
    # reports them. Should be 0 for a valid TTFT comparison (DECISIONS.md D12).
    cached_prompt_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _p_ms(values_s: list[float], p: float) -> float | None:
    return percentile(values_s, p) * 1000 if values_s else None


def summarize_level(
    records: list[RequestRecord],
    concurrency: int,
    wall_time_s: float,
    gpu_price_per_hour: float | None = None,
) -> LevelSummary:
    """Summary of one concurrency level. Latency percentiles cover successful
    requests only; failures are counted (and typed) separately, never
    silently dropped."""
    ok = [r for r in records if r.ok]
    errors = Counter(r.error or "unknown" for r in records if not r.ok)

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2es = [r.e2e_s for r in ok if r.e2e_s is not None]
    output_tokens = [r.output_tokens for r in ok if r.output_tokens is not None]
    decode_speeds = [s for s in (decode_tokens_per_second(r) for r in ok) if s is not None]

    reported_cached = [r.cached_prompt_tokens for r in records if r.cached_prompt_tokens is not None]
    aggregate = sum(output_tokens) / wall_time_s if wall_time_s > 0 and output_tokens else None
    cost = None
    if gpu_price_per_hour is not None and aggregate:
        cost = cost_per_1k_output_tokens(gpu_price_per_hour, aggregate)

    return LevelSummary(
        concurrency=concurrency,
        n_total=len(records),
        n_ok=len(ok),
        n_errors=len(records) - len(ok),
        error_types=dict(errors),
        wall_time_s=wall_time_s,
        ttft_p50_ms=_p_ms(ttfts, 50),
        ttft_p95_ms=_p_ms(ttfts, 95),
        e2e_p50_ms=_p_ms(e2es, 50),
        e2e_p95_ms=_p_ms(e2es, 95),
        mean_output_tokens=sum(output_tokens) / len(output_tokens) if output_tokens else None,
        decode_tok_s_p50=percentile(decode_speeds, 50) if decode_speeds else None,
        aggregate_output_tok_s=aggregate,
        requests_per_s=len(ok) / wall_time_s if wall_time_s > 0 and ok else None,
        gpu_price_per_hour=gpu_price_per_hour,
        cost_per_1k_output_tokens_usd=cost,
        cached_prompt_tokens=sum(reported_cached) if reported_cached else None,
    )


# -- Quality statistics ------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% (by default) Wilson score interval for a proportion. Better than
    the textbook p +- z*sqrt(p(1-p)/n) near 0% or 100%, where that one can
    run past the [0, 1] range."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def mcnemar_exact_p(only_a_correct: int, only_b_correct: int) -> float:
    """Two-sided exact McNemar test for two models answering the *same*
    questions. Only the questions where they disagree carry information:
    if the models were equally good, each disagreement would be a coin flip,
    so the p-value is a two-sided binomial test with p = 0.5 on those."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
