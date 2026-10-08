# Synapse upgrade plan: quantization benchmark + live demo

Status: **approved 2026-10-08** with all recommendations: Qwen2.5-7B-Instruct (BF16, AWQ, GPTQ-Int8), one L40S 48 GB, live demo option B, and removal of README numbers that have no raw data. Spending money and creating accounts still need explicit approval at each step.
Baseline checked on 2026-10-08: `main` @ 546778a, `ruff` clean, 66/66 backend tests pass (Python 3.12), GitHub CI green.

---

## 0. What the code actually does vs. your description

Most of your description matches. These are the differences, ordered by how much they matter for this project.

| # | Finding | Where | Why it matters here |
|---|---|---|---|
| 1 | **Silent fallback to the mock provider.** Any HTTP error from Ollama/vLLM (timeout, OOM, 500) is swallowed and a canned mock reply is served instead, logged as `provider="mock"`. | `providers.py:195-237` | A benchmark run where vLLM crashed would record *very fast* "responses". This is the biggest threat to honest numbers. It has to fail loudly. |
| 2 | **One global backend, no per-model routing.** `PROVIDER` + one base URL apply to every request; the `model` string is just passed through. | `config.py:21-25`, `providers.py:188` | Part 1 needs each quantized variant registered as its own model with its own backend URL. |
| 3 | **`max_tokens` is accepted but never forwarded** to Ollama or vLLM. | `schemas.py:15`, `providers.py` | Without it, output length is uncontrolled, which makes tokens/s comparisons meaningless. |
| 4 | **Synchronous work inside async handlers.** Cache lookup (sentence-transformers embedding on CPU + sync Redis) and SQLAlchemy writes run on the event loop. | `gateway.py:97-130`, `cache.py:344` | Under concurrency this serializes requests in the gateway. I expect it to show up as gateway overhead at high concurrency. We'll measure it, not guess. |
| 5 | **A new `httpx.AsyncClient` per request**, 60 s timeout. | `providers.py:55,106,132` | No connection reuse. Long generations at high concurrency could hit 60 s. |
| 6 | **The cache can't be turned off** per request or per config. | `gateway.py:104` | The perf runs need cache off, and the bonus run needs on vs. off. |
| 7 | **Postgres: `/v1/stats/timeseries` uses SQLite's `strftime`.** Tests run on SQLite, so CI never sees it. | `stats.py:49` | Postgres has no `strftime`, so this endpoint very likely errors on the docker-compose (Postgres) stack. I'll confirm against a real Postgres before fixing. |
| 8 | **Dashboard costs are "illustrative reference rates"**, not derived from hardware. | `pricing.py` | Fine for the gateway, but the benchmark's $/1K tokens will be computed from a stated GPU $/hr instead. |
| 9 | **The README has numbers with no raw data in the repo** (53.4 s cold start, ~470 ms, 621 ms, 18.3x, ~2500x, "8x H200"). | `README.md` | Under your rule they must be re-run with saved results or removed. I propose removing them and pointing to new, reproducible results. |
| 10 | Frontend has a `vitest run` script but no tests, and CI doesn't run it. | `frontend/package.json` | The dashboard data-shaping code needs tests, and CI should run them. |
| 11 | Pinned deps (`psycopg2-binary==2.9.9`, `numpy==1.26.4`) don't build on Python 3.14 (your Mac's default). | `requirements*.txt` | CI uses 3.11, so it's fine there. Locally use 3.11/3.12. I'll note this in CONTRIBUTING. |

Everything else matches: FastAPI + SQLAlchemy + Alembic, Redis ANN cache with linear-scan fallback, per-key rate limits/quotas, the shadow-verification threshold controller (state in Redis), React/TS/Vite/Tailwind/Recharts, GitHub Actions CI.

---

## 1. Model choice

**Recommendation: `Qwen/Qwen2.5-7B-Instruct`**, in three variants, all **first-party checkpoints from the Qwen team** (verified on Hugging Face today: all Apache-2.0, none gated):

| Variant | Checkpoint | Role |
|---|---|---|
| 16-bit (BF16) | `Qwen/Qwen2.5-7B-Instruct` | baseline |
| 4-bit AWQ | `Qwen/Qwen2.5-7B-Instruct-AWQ` | main quantized variant |
| 8-bit GPTQ (optional) | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8` | middle point, if time allows |

Why this model:
- **The quantized checkpoints were made by the people who trained the model.** That removes the "who quantized it, with what calibration data?" question a third-party checkpoint raises. Interviewers will ask.
- **Not gated** (Llama 3.x needs a license click-through and an HF token), so anyone can reproduce the run.
- **Size fits the story.** ~7.6B params × 2 bytes ≈ **15 GB of weights in BF16** vs. roughly **5–6 GB in 4-bit AWQ**. These are arithmetic estimates; the real numbers will come from vLLM's load logs. On a 48 GB GPU both fit with plenty of KV-cache room. On 24 GB, BF16 leaves much less room for KV cache, which makes the memory trade-off visible.
- Well supported by vLLM, and strong enough at GSM8K that quantization damage is measurable rather than lost in noise.

Alternative considered: Qwen3-8B (newer). Rejected for now because its default "thinking" mode adds long reasoning traces that complicate both the throughput and the exact-match numbers. Explainable, but not worth it in two weeks.

## 2. Hardware

**Recommendation: one NVIDIA L40S (48 GB) on an hourly GPU cloud** (RunPod, Lambda, or similar; I'll compare prices on the day and ask you before creating any account).
- A datacenter inference card, more representative of what Fireworks/Together run than a consumer 4090.
- 48 GB fits BF16 comfortably, so BF16 vs. AWQ is compared on *speed*, not just "one of them didn't fit."
- Cheaper fallback: a 24 GB card (RTX 4090 / A10G). This also makes the KV-cache trade-off sharper. Your call (see questions at the end).

**Budget estimate:** about 4–6 GPU-hours total (model downloads, 2–3 variants × perf + eval + cache runs, one re-run for safety). At the roughly $1/hr L40S rates I've seen this would be **~$5–10**, but I'll confirm the actual price before booking. Variants run **one at a time** on the same GPU with identical vLLM flags. Running them side by side would split the memory and contaminate each other's numbers.

## 3. Benchmark design

### 3a. Gateway changes it needs (each tested and committed separately)
1. **Fail loudly:** fallback to mock becomes opt-in (`MOCK_FALLBACK=true`, keeping today's zero-setup demo behavior via `.env.example`). Off for registered vLLM models. The benchmark also asserts `provider != "mock"` on every response and aborts otherwise.
2. **Model registry:** a small `models.yaml` maps a model name to `{provider, base_url, upstream_model, gpu_price_per_hour?}`. Unknown names fall back to today's `PROVIDER` behavior, so nothing existing breaks. Gateway names: `qwen2.5-7b-bf16`, `qwen2.5-7b-awq`, `qwen2.5-7b-gptq-int8`.
3. **Forward `max_tokens`** to both backends.
4. **Cache bypass header** `x-synapse-cache: bypass` skips lookup *and* store, including the embedding call, so cache-off numbers aren't paying for an embedding they don't use.
5. **Record TTFT** for streamed requests: new nullable `ttft_ms` column in `request_logs` (Alembic migration), so per-variant metrics exist in the gateway too, not only in the script.
6. **One shared `httpx.AsyncClient`** per backend (connection pooling), with a configurable timeout.

Deliberately **not** fixing #4 (sync work on the event loop) *before* the first run. We measure gateway overhead first (see 3c), then decide whether to fix it and re-measure. That before/after is a good interview story either way.

### 3b. Performance workload
- **Prompts:** 200 instructions sampled with a fixed seed from `databricks-dolly-15k` (CC BY-SA 3.0), committed as JSONL so the run needs no download.
- **Fixed output length:** `max_tokens=256` plus vLLM's `ignore_eos=true`, so every request generates exactly 256 tokens in every variant. Otherwise a variant that writes shorter answers would look "faster." Cost: `ignore_eos` is a vLLM extension, so the gateway forwards it as an allow-listed passthrough field. That trade-off goes into DECISIONS.md.
- `temperature=0`, streaming on (needed for TTFT), cache bypassed.
- **Concurrency levels:** 1, 4, 16, 64 (closed loop: N workers, each sending its next request when the previous one finishes). At least 100 measured requests per level, after 10 discarded warm-up requests (the first requests trigger CUDA graph capture and compile and are not representative).
- **Two paths per level:** through the gateway, and **directly to vLLM**. The difference is the gateway's own overhead, reported as-is.
- **Repetition:** 3 repeats at c=1 and c=16 to report run-to-run spread, so we know which differences are real.

### 3c. Metrics (exact definitions go in the code and README)
| Metric | Definition |
|---|---|
| TTFT | client send → first non-empty content chunk |
| E2E latency p50 / p95 | client send → `[DONE]` |
| Output tok/s per request | `(output_tokens − 1) / (e2e − ttft)` (decode speed) |
| Aggregate output tok/s | total output tokens / wall time of the level |
| Requests/s | completed requests / wall time |
| Peak GPU memory | see 3d, three numbers, not one |
| Cost per 1K output tokens | `gpu_$_per_hr / (aggregate_tok_s × 3600) × 1000`, at the stated $/hr, assuming the GPU is fully used. Stated as an upper-bound-efficiency estimate. |
| Errors | count and type per level; never silently dropped |

### 3d. GPU memory: the trap to avoid
vLLM **pre-allocates** `--gpu-memory-utilization` (default 0.9) of the GPU at startup: model weights first, then **all remaining space is reserved for the KV cache**. So `nvidia-smi` shows about 90% for *every* variant, and "peak memory" read that way is meaningless. We'll report:
1. **Weight memory**, from vLLM's startup log ("Loading model weights took X GiB").
2. **KV-cache capacity**: blocks/tokens vLLM could allocate, and its reported max concurrency at our sequence length. **This is where 4-bit quantization actually pays off:** smaller weights leave more room for KV cache, which means more simultaneous requests.
3. **nvidia-smi peak** (sampled every 200 ms during the run), reported for completeness with the explanation above.

Same `--gpu-memory-utilization`, `--max-model-len 4096` and `--max-num-seqs` for every variant, recorded in the results.

### 3e. Quality
- **GSM8K test set, all 1,319 questions** (MIT license). Small enough that we don't need to sample.
- Greedy decoding, `max_tokens=512`, a fixed zero-shot chat prompt asking for the final answer after `####`. **Exact match** on the extracted final number (commas and `$` normalized). The extraction code gets unit tests on tricky cases.
- Report accuracy, **n**, and a **95% Wilson interval**. Because every variant answers the *same* questions, also report a **paired McNemar test** between BF16 and each quantized variant. That's the honest way to say whether a 1–2 point drop is real or noise. (With n=1,319, the interval is roughly ±2.5 points, so small differences may well come out "not significant". If so, that's what we report.)
- Run through the gateway, cache bypassed, so it also tests the routing.

### 3f. Bonus: cache × quantization
Replay the existing labelled paraphrase traffic (`scripts/benchmark_data.py`) against BF16 and AWQ with the cache **on** vs. **off**. Report the hit rate, p50/p95, and effective tok/s per variant. Expected shape (to be tested, not assumed): caching dominates latency on hits regardless of variant, so quantization only matters on misses.

### 3g. Reproducibility metadata (auto-captured into every result file)
Date (UTC), git commit, GPU model and driver, CUDA, vLLM version and full launch flags, HF model revision SHA, gateway settings, GPU $/hr used, seed, prompt-file hash. Raw per-request records → `benchmarks/results/<date>_<gpu>/` as JSONL + a summary CSV/JSON. The dashboard and README read **only** from these files.

### 3h. Code layout and tests
```
benchmarks/
  workloads/dolly_200.jsonl, gsm8k_test.jsonl
  synapse_bench/
    metrics.py      # pure functions: percentiles, tok/s, cost, Wilson, McNemar
    loadgen.py      # async closed-loop load generator, streaming client
    gpu.py          # nvidia-smi sampler + vLLM log parser
    gsm8k.py        # prompt + answer extraction + scoring
    run_perf.py / run_eval.py / run_cache.py   # CLIs
  scripts/run_gpu_session.sh   # starts each vLLM variant in turn, runs everything
  results/
```
Tests (in CI, no GPU): metric math on hand-computed fixtures, answer extraction, a load-generator run against a fake streaming server (checks TTFT is measured from the right chunk, errors are counted, not dropped), the vLLM log parser on a saved log sample, and the "abort on mock provider" guard.

Before spending any money, I'll do a **full dry run on your Mac** against Ollama (CPU, small model) to shake out bugs. Those numbers are discarded, never published.

---

## 4. Making it live

| Option | What a reviewer gets | Monthly cost | Risks / trade-offs |
|---|---|---|---|
| **A. Static site only** (GitHub Pages / Cloudflare Pages): dashboard + benchmark page from committed JSON; playground replays recorded responses | Charts + a fake-but-labelled chat | $0 | Nothing is actually live. A reviewer can tell. |
| **B. One small CPU VPS** running everything: Caddy (HTTPS + static frontend) → gateway + Redis + Postgres + Ollama with `Qwen2.5-0.5B-Instruct` | One link: real gateway, real cache hits, real streaming, plus the GPU benchmark page from saved results | roughly $5–8 (Hetzner-class 2–4 vCPU / 4 GB; price confirmed before signup) | Slow tokens on CPU (that's the point of the label), needs abuse protection, and a VM to maintain |
| **C. Serverless GPU** (Modal / RunPod serverless, scale to zero) running real vLLM | The real 7B model live | pay per second; unpredictable | 30–90 s cold starts, and **public endpoint + per-second GPU billing = bill risk** from bots |

**Recommendation: B, with A's replay mode as the automatic fallback** if the backend is unreachable. One link, the real gateway path is live, the cost is predictable, and the GPU results are clearly labelled as recorded.
Safety for a public endpoint, using features Synapse already has:
- A public, read-only **demo key**, rate-limited (e.g. 10 req/min) and with a monthly quota.
- `max_tokens` capped server-side.
- Admin endpoints disabled in demo mode, and the API Keys page hidden.
- Postgres and Redis not exposed.
- A banner: "Live demo: 0.5B model on 2 vCPUs. Benchmark numbers: L40S run on <date>, see methodology."

## 5. README rewrite
Two-sentence summary → live link → GIF (playground showing a cache miss then hit, plus the benchmark page) → Mermaid architecture diagram (renders on GitHub) → run locally → benchmark method → results table generated from the summary file by a script (so no number is typed by hand) → known limitations. The old claims from finding #9 are removed or moved to a "history" note with no numbers.

## 6. DECISIONS.md
Started in milestone 1 and updated with every commit that makes a real trade-off. Each entry: the decision, the options, why, what it costs, how we'd know if it was wrong. It also gets a short "vLLM & GPU memory primer":
- **AWQ is W4A16.** Weights are stored in 4 bits, but the math runs in 16-bit. So it mainly speeds up *memory-bound* decoding at low concurrency, and may help less, or even hurt, at high concurrency where the GPU becomes compute-bound. That's a hypothesis our c=1…64 sweep tests directly.
- vLLM reads the quantization method from the checkpoint's `config.json`, so `--quantization` is usually unnecessary. On Ampere and newer GPUs it picks the faster Marlin kernels automatically (`awq_marlin`), and the log line confirms it.
- `--dtype` is the *activation/compute* dtype, not the weight format.
- What `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs` and CUDA graphs (`--enforce-eager` off) do.

## 7. Milestones (≈ 2 weeks of your time)

| # | Days | Deliverable | Needs money/account? |
|---|---|---|---|
| M1 | 1–2 | Honesty fixes: fail-loud fallback, forward `max_tokens`, cache bypass header, Postgres-safe timeseries (+ a Postgres test in CI), shared httpx client. DECISIONS.md started. | No |
| M2 | 3–4 | Model registry, `ttft_ms` column + migration, `ignore_eos` passthrough | No |
| M3 | 4–6 | Benchmark package + GSM8K eval + tests in CI; full dry run on your Mac via Ollama | No |
| M4 | 7 | **GPU session**: run BF16, AWQ (+ GPTQ-Int8) → commit raw results | **Yes: ask first** |
| M5 | 8–9 | Dashboard "Benchmarks" page (latency, throughput, memory, accuracy per variant) + vitest tests in CI; (optional) gateway event-loop fix + re-measure | No (re-measure needs a short GPU session; ask first) |
| M6 | 10–12 | Demo mode + deploy (option B) + replay fallback | **Yes: ask first** |
| M7 | 13–14 | README rewrite, GIF, final DECISIONS.md pass, limitations | No |

Work happens on a branch, one PR per milestone (small commits inside), and CI stays green throughout.

## 8. Questions for you
1. **Model:** Qwen2.5-7B-Instruct OK, and include the optional GPTQ-Int8 variant?
2. **GPU:** L40S 48 GB (recommended) or a cheaper 24 GB card? Any provider you already have an account with or credits on?
3. **Live demo:** option B (≈$5–8/mo VPS) OK in principle? I'll still ask before creating the account. Do you own a domain, or is a provider subdomain fine?
4. **README cleanup:** OK to remove the existing numbers that have no raw data in the repo (finding #9)?
