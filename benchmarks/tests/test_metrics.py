"""Every reported number is computed here, so each formula is checked
against values worked out by hand (shown in the comments)."""

import pytest

from synapse_bench.metrics import (
    cost_per_1k_output_tokens,
    decode_tokens_per_second,
    mcnemar_exact_p,
    percentile,
    summarize_level,
    wilson_interval,
)
from synapse_bench.records import RequestRecord


def _rec(i, ok=True, ttft=0.1, e2e=1.1, out=101, error=None):
    return RequestRecord(
        index=i, ok=ok, status_code=200 if ok else 502, error=error, start_s=0.0,
        ttft_s=ttft if ok else None, e2e_s=e2e if ok else None,
        prompt_tokens=10, output_tokens=out if ok else None, x_cache="bypass", provider="vllm",
    )


# -- percentile ---------------------------------------------------------------


def test_percentile_interpolates_like_numpy():
    values = [4, 1, 3, 2]  # unsorted on purpose
    assert percentile(values, 50) == 2.5  # halfway between 2 and 3
    assert percentile(values, 95) == pytest.approx(3.85)  # rank 2.85 -> 3 + 0.85 * (4 - 3)
    assert percentile(values, 0) == 1 and percentile(values, 100) == 4


def test_percentile_single_value_and_errors():
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


# -- per-request decode speed ------------------------------------------------


def test_decode_speed_excludes_first_token():
    # 101 tokens; first arrives at 0.1 s, last at 1.1 s -> 100 tokens in 1.0 s.
    assert decode_tokens_per_second(_rec(0)) == pytest.approx(100.0)


@pytest.mark.parametrize("record", [_rec(0, ok=False, error="x"), _rec(0, out=1), _rec(0, ttft=1.1, e2e=1.1)])
def test_decode_speed_undefined_cases(record):
    assert decode_tokens_per_second(record) is None


# -- cost ---------------------------------------------------------------------


def test_cost_per_1k_tokens():
    # $1/h at 1000 tok/s = 3.6M tokens/h -> $1 / 3600 per 1K tokens.
    assert cost_per_1k_output_tokens(1.0, 1000.0) == pytest.approx(1 / 3600)
    assert cost_per_1k_output_tokens(1.0, 0.0) is None


# -- level summary ------------------------------------------------------------


def test_summarize_level_counts_errors_and_ignores_them_in_latency():
    records = [_rec(0, ttft=0.1, e2e=1.1), _rec(1, ttft=0.3, e2e=2.1),
               _rec(2, ok=False, error="HTTP 502"), _rec(3, ok=False, error="HTTP 502")]
    s = summarize_level(records, concurrency=2, wall_time_s=4.0, gpu_price_per_hour=2.0)

    assert (s.n_total, s.n_ok, s.n_errors) == (4, 2, 2)
    assert s.error_types == {"HTTP 502": 2}
    assert s.ttft_p50_ms == pytest.approx(200.0)  # midpoint of 100 ms and 300 ms
    assert s.e2e_p50_ms == pytest.approx(1600.0)
    assert s.mean_output_tokens == 101
    # Decode speeds: 100/1.0 = 100 and 100/1.8 = 55.6 -> median 77.8
    assert s.decode_tok_s_p50 == pytest.approx((100 + 100 / 1.8) / 2)
    assert s.aggregate_output_tok_s == pytest.approx(202 / 4.0)
    assert s.requests_per_s == pytest.approx(0.5)
    assert s.cost_per_1k_output_tokens_usd == pytest.approx(2.0 / (50.5 * 3600) * 1000)


def test_summarize_level_all_failed_has_no_latency_numbers():
    s = summarize_level([_rec(0, ok=False, error="timeout")], concurrency=1, wall_time_s=1.0, gpu_price_per_hour=1.0)
    assert s.n_ok == 0
    assert s.e2e_p50_ms is None and s.aggregate_output_tok_s is None and s.cost_per_1k_output_tokens_usd is None


# -- quality statistics -------------------------------------------------------


def test_wilson_interval_known_value():
    low, high = wilson_interval(5, 10)  # textbook example: 5/10 -> (0.2366, 0.7634)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)


def test_wilson_interval_stays_in_range_at_extremes():
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and 0 < high < 0.2
    low, high = wilson_interval(20, 20)
    assert 0.8 < low < 1 and high == 1.0


def test_mcnemar_exact_known_values():
    assert mcnemar_exact_p(0, 0) == 1.0  # no disagreements -> no evidence of a difference
    assert mcnemar_exact_p(0, 5) == pytest.approx(2 * 0.5**5)  # 0.0625: not significant at 0.05
    assert mcnemar_exact_p(1, 9) == pytest.approx(2 * (1 + 10) / 1024)  # 0.0215
    assert mcnemar_exact_p(9, 1) == mcnemar_exact_p(1, 9)  # symmetric
    assert mcnemar_exact_p(10, 10) == 1.0
