"""run_eval and compare against a fake (non-streaming) gateway whose answers
we control, so the expected accuracy is known exactly."""

import json

import httpx
import pytest

import synapse_bench.metadata as metadata
import synapse_bench.run_eval as run_eval
from synapse_bench.compare import compare
from synapse_bench.gsm8k import build_prompt, reference_answer
from synapse_bench.workloads import load_gsm8k

_RealAsyncClient = httpx.AsyncClient
PROBLEMS = load_gsm8k()[:6]
GOLD_BY_PROMPT = {build_prompt(p["question"]): reference_answer(p["answer"]) for p in PROBLEMS}


class FakeGateway:
    """Answers question i correctly when i is in `correct`, wrong otherwise."""

    def __init__(self, correct: set[int], provider="vllm", fail: set[int] = frozenset(), tokens=40):
        self.correct, self.provider, self.fail, self.tokens = correct, provider, fail, tokens
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        prompt = body["messages"][0]["content"]
        i = list(GOLD_BY_PROMPT).index(prompt)
        if i in self.fail:
            return httpx.Response(502, json={"detail": "down"})
        gold = GOLD_BY_PROMPT[prompt]
        text = f"Reasoning...\n#### {gold}" if i in self.correct else "Reasoning...\n#### 999999"
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": self.tokens},
            "provider": self.provider, "cached": False,
        })


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(metadata, "http_json", lambda url, headers=None: {"stub": url})
    monkeypatch.setenv("SYNAPSE_API_KEY", "llmgw_test")


def _use(monkeypatch, gateway: FakeGateway):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(gateway.handler)
        return _RealAsyncClient(*args, **kwargs)

    monkeypatch.setattr(run_eval.httpx, "AsyncClient", factory)


def _run(tmp_path, name="run", *extra):
    out = tmp_path / name
    code = run_eval.main(["--model", "m", "--limit", "6", "--concurrency", "3", "--out", str(out), *extra])
    return code, out


def _answers(out):
    return [json.loads(line) for line in (out / "answers.jsonl").read_text().splitlines()]


def test_accuracy_matches_known_answers(monkeypatch, tmp_path):
    gateway = FakeGateway(correct={0, 1, 2, 3})
    _use(monkeypatch, gateway)
    code, out = _run(tmp_path)
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n"] == 6 and summary["strict"]["correct"] == 4
    assert summary["strict"]["accuracy"] == pytest.approx(4 / 6)
    assert summary["strict"]["ci95_low"] < 4 / 6 < summary["strict"]["ci95_high"]
    # Requests: greedy, cache bypassed, NOT fixed-length.
    assert all(b["temperature"] == 0.0 and "ignore_eos" not in b and b["max_tokens"] == 512 for b in gateway.bodies)


def test_failed_requests_count_as_wrong(monkeypatch, tmp_path):
    _use(monkeypatch, FakeGateway(correct=set(range(6)), fail={0, 1}))
    _, out = _run(tmp_path)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n"] == 6 and summary["n_errors"] == 2
    assert summary["strict"]["correct"] == 4  # not 4/4 -- errors stay in the denominator


def test_answers_from_mock_invalidate_the_run(monkeypatch, tmp_path):
    _use(monkeypatch, FakeGateway(correct=set(range(6)), provider="mock"))
    code, out = _run(tmp_path)
    assert code == 2
    assert all(r["error"] == "answered by mock" and not r["correct_strict"] for r in _answers(out))


def test_flags_replies_that_hit_max_tokens(monkeypatch, tmp_path):
    _use(monkeypatch, FakeGateway(correct=set(range(6)), tokens=64))
    _, out = _run(tmp_path, "run", "--max-tokens", "64")
    assert json.loads((out / "summary.json").read_text())["n_possibly_truncated"] == 6


def test_compare_counts_disagreements_and_tests_them(monkeypatch, tmp_path):
    _use(monkeypatch, FakeGateway(correct={0, 1, 2, 3, 4}))
    _, out_a = _run(tmp_path, "a")
    _use(monkeypatch, FakeGateway(correct={0, 1, 5}))
    _, out_b = _run(tmp_path, "b")

    result = compare(_answers(out_a), _answers(out_b))
    assert (result["both_correct"], result["only_a_correct"], result["only_b_correct"],
            result["neither_correct"]) == (2, 3, 1, 0)
    assert result["difference_b_minus_a"] == pytest.approx((1 - 3) / 6)
    assert result["mcnemar_p"] == pytest.approx(2 * (1 + 4) / 16)  # 4 disagreements, split 3:1 -> p = 0.625


def test_compare_refuses_different_question_sets():
    a = [{"id": "q1", "correct_strict": True}]
    b = [{"id": "q2", "correct_strict": True}]
    with pytest.raises(ValueError):
        compare(a, b)
