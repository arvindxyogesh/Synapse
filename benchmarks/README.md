# Synapse benchmark harness

Measures what weight quantization buys and costs when serving one model
through vLLM behind the Synapse gateway: **speed** (latency, throughput,
GPU memory, cost) and **quality** (GSM8K accuracy). The design and its
trade-offs are explained in [`../DECISIONS.md`](../DECISIONS.md), D8–D13,
and the overall plan is in [`../docs/PLAN.md`](../docs/PLAN.md).

> **Status:** the harness is built and tested, and has been dry-run on a
> laptop against Ollama (those numbers are not published). No GPU results
> exist yet. They'll appear under `results/` with full metadata, and the
> top-level README will quote only those files.

## Install

```bash
cd benchmarks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # no GPU or server needed
```

The harness only needs `httpx`. It runs on any machine that can reach the
gateway, not necessarily the GPU box.

## Run

Every command reads the gateway API key from `SYNAPSE_API_KEY`. The key is
never written to any output file. Each command refuses to overwrite an
existing `--out` directory.

**Speed**, through the gateway and directly against vLLM, to measure the
gateway's own overhead:

```bash
python -m synapse_bench.run_perf \
    --model qwen2.5-7b-awq \
    --direct-url http://localhost:8002 --direct-model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --concurrency 1,4,16,64 --requests 100 --warmup 10 --max-tokens 256 \
    --gpu-price-per-hour <USD/h> --gpu-sample --vllm-log vllm-awq.log \
    --out results/<date>_<gpu>/qwen2.5-7b-awq/perf
```

**Quality**, on the full GSM8K test set:

```bash
python -m synapse_bench.run_eval --model qwen2.5-7b-awq \
    --out results/<date>_<gpu>/qwen2.5-7b-awq/gsm8k
python -m synapse_bench.compare results/.../qwen2.5-7b-bf16/gsm8k results/.../qwen2.5-7b-awq/gsm8k
```

**Required vLLM flags for valid numbers:** `--no-enable-prefix-caching`
(otherwise later runs reuse earlier prompts' work, see D12) and
`--enable-prompt-tokens-details` (so the harness can verify that). Use the
same `--gpu-memory-utilization` and `--max-model-len` for every variant
(see D11).

## What gets measured (exact definitions)

| Metric | Definition |
|---|---|
| TTFT | client send → first streamed chunk containing text |
| E2E latency | client send → `[DONE]` |
| p50 / p95 | percentiles over successful requests (linear interpolation, as numpy) |
| Decode speed (per request) | `(output_tokens − 1) / (e2e − ttft)`; the first token is excluded because its time is in TTFT |
| Aggregate throughput | total output tokens of successful requests / wall time of the level |
| Requests/s | successful requests / wall time of the level |
| Cost per 1K output tokens | `gpu_$_per_hour / (aggregate_tok_s × 3600) × 1000`, assuming the GPU stays this busy |
| Errors | counted and typed per level, never dropped |
| GPU memory | weight memory, KV-cache size and max concurrency from vLLM's log, plus `nvidia-smi` peak (see D11 for why both) |
| GSM8K accuracy | exact match on the last `\boxed{}` (strict) and the last number (flexible); failed requests count as wrong; 95% Wilson interval; paired McNemar test between variants |

Load is **closed-loop**: N workers each send their next request as soon as
the previous one finishes, so concurrency is exactly N for the whole level.
Outputs are **fixed-length** (`ignore_eos`) for speed runs and
**free-length** for quality runs (D8).

## Validity checks built in

A run exits with code 2 (after saving everything) if any response:
- didn't come back with `x-cache: bypass`, meaning the gateway's semantic
  cache wasn't bypassed;
- was answered by the mock provider or the cache;
- carried no token usage, so tokens/s couldn't be computed.

It also warns when:
- the backend served prompt tokens from its own cache (D12);
- the load generator's CPU use suggests the client was the bottleneck (D5).

## Output files

```
<out>/metadata.json   time (UTC), git commit + dirty flag, client, GPU, workload hash,
                      gateway /health and /v1/models, vLLM version and parsed log, all args
<out>/requests.jsonl  one raw record per request (perf) / per question (eval)
<out>/summary.json    the numbers above, computed from requests.jsonl
```

## Not yet validated

- `gpu.py`'s vLLM log parser has been tested on example lines in known
  formats, not yet on a real log from the vLLM version that will be used.
  It returns `None` rather than guessing, and is checked against the real
  log before any memory number is reported.
- `--gpu-sample` has never run next to a real GPU.

## Workloads

See [`workloads/SOURCES.md`](workloads/SOURCES.md) for sources, licenses and
hashes. `perf_prompts.jsonl` (200 Dolly instructions, CC BY-SA 3.0) is
**short-prompt heavy**: the median prompt is 63 characters, about 15 tokens.
So with 256-token outputs these runs mostly measure *decode*, and TTFT
mostly reflects queueing rather than prompt processing. That's typical of
chat traffic, but it isn't a long-document workload, and results shouldn't
be read as one.
