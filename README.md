# Synapse

A memory layer for your LLM calls: a full-stack gateway that sits in front
of open-weight LLMs (served locally via [Ollama](https://ollama.com)), adds
**semantic response caching** with ANN vector search, **streaming**
completions, **per-key rate limits and usage quotas**, and a live
**observability dashboard**. Everything in this stack is free and
open-source — no paid API keys required to run it end-to-end.

```
                 ┌─────────────┐        ┌──────────────┐
  client  ─────▶ │   FastAPI    │──────▶ │    Ollama     │  (local, free,
 (API key)       │   gateway    │        │ open models   │   open-weight)
                 │ rate limits  │◀───────┤ llama3/mistral│
                 │ + quotas     │        └──────────────┘
                 └───┬──────┬───┘
                     │      │
         semantic    │      │  request/cost/latency log
         cache (ANN) ▼      ▼
                 ┌────────┐ ┌────────────┐
                 │ Redis  │ │ Postgres   │  (Alembic-managed schema)
                 │ +vector│ └────────────┘
                 └────────┘      ▲
                     ▲            │ stats API
                     │            │
                 ┌────────────────┐
                 │ React dashboard │  live charts, chat playground
                 │  (Vite + TS)    │  (streaming), API key management
                 └────────────────┘
```

## Why

Every call to `POST /v1/chat/completions` is checked against a semantic
cache before it reaches the model: an exact-hash check first, then a
vector similarity search against recent embeddings for that model. A cache
hit skips inference entirely — zero cost, near-zero latency — and the
dashboard shows exactly how much that's saving in real time. Responses can
also be streamed token-by-token (SSE), and gateway keys can carry a
per-minute rate limit and a monthly cost quota.

## Adaptive cache threshold (self-tuning correctness)

A semantic cache has one structural risk: at a fixed similarity threshold,
a "close enough" match can be *wrong* -- e.g. reusing the answer to "How do
I cancel my subscription?" for "How do I pause my subscription instead of
cancelling?" is a real, damaging false positive, not just a missed
optimization. Turning the threshold up to be safe throws away hit rate;
turning it down to get more hits risks more of these.

Rather than pick one fixed threshold and hope, `app/threshold_controller.py`
tunes it online, per model, with a small closed-loop controller:

1. A sample of cache hits (`SHADOW_VERIFY_SAMPLE_RATE`, default 20%) is
   shadow-verified in the background, after the response has already been
   served (so it never adds latency): an **independent LLM-judge**
   (`app/judge.py`) is asked whether the prompt that originally produced
   the cached response and the prompt that just hit it are actually
   asking the same thing. This is a genuinely different signal from the
   embedding similarity that produced the hit in the first place, so it
   catches errors the embedding model itself missed. It falls back to a
   deterministic stopword-filtered token-overlap heuristic when there's no
   real model to ask (mock mode / Ollama unreachable) -- same
   real-model-with-deterministic-fallback shape as `app/embeddings.py`, so
   the whole thing is testable and demoable without a GPU.
2. An EWMA of the observed false-positive rate drives the threshold up
   (stricter) when it's above target, or back down (looser, more hits)
   once it's comfortably under target -- in small, bounded, cooldown-gated
   steps, the same "hold an operating metric near a target" shape as an
   SLO-adaptive controller, applied here to cache *correctness* instead of
   latency.

State and results are visible at `GET /v1/stats/cache-threshold` and on
the dashboard.

### Honest status of the evidence so far

`backend/scripts/benchmark.py --rounds` (see its docstring to reproduce)
demonstrates the *mechanism* end-to-end -- shadow verification firing,
false positives being detected, the threshold moving in response, and
recovering -- but every number produced against this environment used
`MOCK_MODE` and the **hash-embedding fallback**, not a real embedding
model: `sentence-transformers` needs to download weights from
huggingface.co on first use, which this sandbox's network policy blocks
outright (confirmed directly, not assumed -- see `app/embeddings.py`,
which now logs a warning and reports its active backend on `GET /health`
specifically because this failure used to be silent). That matters,
concretely: the benchmark also reports precision/recall on the
**"novel-only"** subset -- excludes exact-repeat prompts, which trivially
hit and inflate the headline numbers once a small traffic universe
saturates -- and on that harder, honest slice, the hash fallback's recall
on genuine differently-worded paraphrases collapses to roughly 0-40%. The
bag-of-words hash embedder mostly can't tell that "How do I reset my
password?" and "I forgot my password, how can I reset it?" mean the same
thing; only close-to-verbatim repeats reliably hit.

So: the control loop (shadow verification catching real errors, the
threshold correctly tightening/loosening in response) is verified and
reproducible. Whether it holds up with *real* embeddings and a *real*
LLM-judge -- where the interesting failure modes are different (judge
noise/inconsistency, real embeddings' own precision/recall curve) -- is
not yet measured and needs a real network + a real Ollama-served model,
i.e. GPU is optional but real network access and a real model are not.

Two real bugs were also found and fixed while building this (not just the
network-blocked one above): an EWMA tuned too fast (alpha=0.3, ~3-sample
memory) silently underestimated a real ~15% false-positive rate down to
~0%, and the benchmark harness's own ground truth mislabeled repeated
confuser prompts as false positives. Both fixed in the current code.

## Stack

| Layer | Tech |
|---|---|
| Model serving | [Ollama](https://ollama.com) (local, open-weight models) with an automatic mock-provider fallback |
| Backend | FastAPI, SQLAlchemy, Alembic |
| Cache | Redis, with ANN vector search (Query Engine / RediSearch) when available, falling back to a linear scan otherwise |
| Cache correctness | Adaptive per-model similarity threshold, tuned online by LLM-judge shadow verification |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (local, falls back to a hashing embedding if unavailable) |
| Storage | Postgres (SQLite for local dev without Docker) |
| Frontend | React + TypeScript (Vite), Tailwind CSS, Recharts |
| CI | GitHub Actions (ruff + pytest, eslint + tsc + vite build) |

## Quickstart

### Option A — Docker Compose (Postgres + Redis + backend + frontend)

```bash
cp .env.example .env
docker compose up --build
```

- Dashboard: http://localhost:5173
- API: http://localhost:8000 (docs at `/docs`)

`docker compose` runs `redis/redis-stack-server`, which bundles the Query
Engine module the semantic cache uses for ANN vector search, and the
backend image runs `alembic upgrade head` on startup before serving traffic.

If [Ollama](https://ollama.com) isn't running locally, the gateway
automatically serves mock responses instead of erroring out — you can still
exercise the whole cache/cost/dashboard pipeline with zero model setup. To
use real open-weight models: `ollama pull llama3` then `ollama serve`.

### Option B — run backend/frontend directly

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head   # creates ./gateway.db (SQLite) via migrations
uvicorn app.main:app --reload

# frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Plain `redis-server` (no Query Engine module) works fine here too — the
cache detects the module is missing and transparently falls back to a
linear-scan similarity search.

### Try it

```bash
# 1. create a gateway API key (ADMIN_KEY defaults to "change-me-admin-key").
#    rate_limit_per_minute / monthly_quota_usd are both optional (omit or
#    null = unlimited).
curl -X POST localhost:8000/v1/admin/keys \
  -H "x-admin-key: change-me-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "local-dev", "rate_limit_per_minute": 60, "monthly_quota_usd": 5}'
# -> {"api_key": "llmgw_..."}

# 2. send a chat completion through the gateway
curl -X POST localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer llmgw_..." \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [{"role": "user", "content": "hello"}]}'

# 3. send the exact same request again -> "cached": true, cost_usd: 0, x-cache: hit

# 4. or stream it token-by-token (Server-Sent Events, OpenAI-chunk shaped)
curl -N -X POST localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer llmgw_..." \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [{"role": "user", "content": "hello"}], "stream": true}'
```

Then open the dashboard to watch request volume, cache hit rate, cost
saved, and latency update live, use **Playground** to chat through the
gateway directly (streaming or not), and **API Keys** to create/revoke keys
and set or clear their rate limit and quota.

Exceeding a key's rate limit or quota returns `429` (rate limit responses
carry a `Retry-After` header).

## Tests

```bash
cd backend && pip install -r requirements-dev.txt && ruff check app tests && pytest
cd frontend && npm install && npm run lint && npm run build
```

Both run in CI on every push (`.github/workflows/ci.yml`). Tests build
their schema straight from the SQLAlchemy models (no Alembic involved) and
run against fakeredis, so the ANN cache branch is covered separately with a
mocked Redis client (`tests/test_cache_ann.py`) since fakeredis doesn't
implement the vector search commands.

## Project layout

```
backend/
  alembic/            migrations (env.py reads DATABASE_URL from Settings)
  app/
    main.py           FastAPI app + router wiring
    providers.py       Ollama client + mock fallback, incl. streaming
    cache.py           semantic cache: exact match, ANN vector search
                        (Redis Query Engine) with linear-scan fallback
    embeddings.py      sentence-transformers embedder + hashing fallback
    threshold_controller.py  adaptive per-model cache similarity threshold
    judge.py            LLM-judge (+ heuristic fallback) shadow verification
    background.py       fire-and-forget helper for shadow verification
    pricing.py         cost estimation per model
    auth.py            API key issuance/verification
    ratelimit.py        per-key rate limiting (req/min) + monthly $ quotas
    redis_client.py     shared Redis connection (cache + rate limiter)
    api/
      gateway.py       POST /v1/chat/completions (streaming + non-streaming)
      admin.py         API key CRUD + rate limit/quota updates (admin-key protected)
      stats.py         summary / timeseries / provider breakdown / request log
                        + cache-threshold (adaptive controller state)
  tests/
  scripts/
    benchmark.py       cache precision/recall/F1 vs a live gateway, with
                        --rounds to show the adaptive threshold converging
frontend/
  src/
    pages/             Dashboard (incl. adaptive threshold panel),
                        Playground (live chat), Requests, ApiKeys
    api/client.ts      typed fetch wrapper
```

## Notes / possible next steps

Highest-value next step, concretely: re-run
`backend/scripts/benchmark.py --rounds` on a machine with real network
access and Ollama, with `pip install -r requirements.txt` (the full one,
which includes `sentence-transformers`) so `GET /health` reports
`"embedder_backend": "sentence-transformers"` instead of
`"hash-fallback"` -- then compare the "novel-only" precision/recall
against the hash-fallback numbers documented above. A GPU speeds up
Ollama inference but isn't required for this specific comparison; what's
required is that huggingface.co isn't blocked.

Other next steps:
- Swap linear TTL-window rate limiting for a sliding-window/token-bucket
  algorithm if bursts right at the minute boundary start to matter.
- Multi-provider support beyond Ollama (e.g. vLLM, llama.cpp server).
- Alerting on quota/rate-limit thresholds instead of just blocking at 100%.
- A controlled ANN-vs-linear-scan latency/throughput benchmark as cache
  size grows (1k/10k/100k entries) -- the vector index exists, but its
  payoff at scale isn't measured yet.
- Load/concurrency testing (p50/p95/p99 under N concurrent users), ideally
  against real GPU-served Ollama so the numbers reflect real inference
  queueing under load, not the mock provider's near-instant response.
