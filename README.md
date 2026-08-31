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

## Stack

| Layer | Tech |
|---|---|
| Model serving | [Ollama](https://ollama.com) (local, open-weight models) with an automatic mock-provider fallback |
| Backend | FastAPI, SQLAlchemy, Alembic |
| Cache | Redis, with ANN vector search (Query Engine / RediSearch) when available, falling back to a linear scan otherwise |
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
    pricing.py         cost estimation per model
    auth.py            API key issuance/verification
    ratelimit.py        per-key rate limiting (req/min) + monthly $ quotas
    redis_client.py     shared Redis connection (cache + rate limiter)
    api/
      gateway.py       POST /v1/chat/completions (streaming + non-streaming)
      admin.py         API key CRUD + rate limit/quota updates (admin-key protected)
      stats.py         summary / timeseries / provider breakdown / request log
  tests/
frontend/
  src/
    pages/             Dashboard, Playground (live chat), Requests, ApiKeys
    api/client.ts      typed fetch wrapper
```

## Notes / possible next steps

- Swap linear TTL-window rate limiting for a sliding-window/token-bucket
  algorithm if bursts right at the minute boundary start to matter.
- Multi-provider support beyond Ollama (e.g. vLLM, llama.cpp server).
- Alerting on quota/rate-limit thresholds instead of just blocking at 100%.
