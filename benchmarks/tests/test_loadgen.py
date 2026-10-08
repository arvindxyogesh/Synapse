"""Load generator against a fake streaming backend, so timing, concurrency
and failure handling can be checked without any model server."""

import httpx
import pytest

from synapse_bench.loadgen import Target, run_level, send_one

TARGET = Target(url="http://gateway/v1/chat/completions", model="m", headers={"x-synapse-cache": "bypass"})


async def _one(target=TARGET):
    async with httpx.AsyncClient() as client:
        return await send_one(client, target, "hello", max_tokens=8, index=0, level_start=0.0)


async def test_measures_ttft_and_e2e(fake):
    fake.prefill_s, fake.per_token_s, fake.n_tokens = 0.10, 0.02, 6
    record = await _one()
    assert record.ok, record.error
    # TTFT ~ prefill (0.10 s); e2e ~ prefill + 5 gaps of 0.02 s = 0.20 s.
    assert 0.09 <= record.ttft_s < 0.15
    assert 0.19 <= record.e2e_s < 0.30
    assert record.output_tokens == 6 and record.prompt_tokens == 3
    assert (record.x_cache, record.provider) == ("bypass", "vllm")


async def test_request_body_asks_for_usage_and_streaming(fake):
    target = Target(url=TARGET.url, model="qwen-awq", extra_body={"ignore_eos": True})
    await _one(target)
    body = fake.bodies[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.0 and body["max_tokens"] == 8
    assert body["model"] == "qwen-awq" and body["ignore_eos"] is True


@pytest.mark.parametrize(
    "setup, expected_error",
    [
        (dict(status=502), "HTTP 502"),
        (dict(send_done=False), "incomplete stream (no [DONE])"),
        (dict(x_cache="miss"), "cache not bypassed"),
        (dict(provider="mock"), "answered by mock"),
        (dict(send_usage=False), "no usage in stream"),
    ],
)
async def test_failures_become_typed_error_records(fake, setup, expected_error):
    for name, value in setup.items():
        setattr(fake, name, value)
    record = await _one()
    assert not record.ok
    assert record.error == expected_error
    assert record.e2e_s is None and record.output_tokens is None  # never half-counted


async def test_direct_to_vllm_does_not_require_gateway_headers(fake):
    fake.x_cache, fake.provider = None, None  # a bare vLLM server sends neither
    record = await _one(Target(url="http://vllm/v1/chat/completions", model="m", via_gateway=False))
    assert record.ok, record.error


async def test_timeout_is_recorded_not_raised(fake):
    fake.raise_timeout = True
    record = await _one()
    assert (record.ok, record.error) == (False, "timeout")


async def test_run_level_holds_concurrency_exactly(fake):
    result = await run_level(TARGET, prompts=[f"p{i}" for i in range(50)], concurrency=4, n_requests=20,
                             max_tokens=8)
    assert fake.max_in_flight == 4
    assert [r.index for r in result.records] == list(range(20))
    assert all(r.ok for r in result.records)
    assert result.wall_time_s > 0 and result.client_cpu_fraction >= 0


async def test_run_level_prompt_order_is_deterministic_and_warmup_is_separate(fake):
    prompts = [f"p{i}" for i in range(10)]
    result = await run_level(TARGET, prompts, concurrency=1, n_requests=5, max_tokens=8, warmup_requests=2)
    assert len(result.records) == 5  # warm-up requests are not recorded
    # Warm-up used the last two prompts; measured requests p0..p4, in order.
    assert fake.prompts_seen == ["p9", "p8", "p0", "p1", "p2", "p3", "p4"]


async def test_run_level_prompt_offset(fake):
    await run_level(TARGET, [f"p{i}" for i in range(10)], concurrency=1, n_requests=3, max_tokens=8,
                    prompt_offset=8)
    assert fake.prompts_seen == ["p8", "p9", "p0"]


async def test_run_level_rejects_bad_arguments():
    with pytest.raises(ValueError):
        await run_level(TARGET, ["p"], concurrency=0, n_requests=1, max_tokens=8)
    with pytest.raises(ValueError):
        await run_level(TARGET, [], concurrency=1, n_requests=1, max_tokens=8)
