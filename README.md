# Synapse

A memory layer for your LLM calls: a full-stack gateway that sits in front
of open-weight LLMs (served locally via [Ollama](https://ollama.com)), adds
**semantic response caching**, tracks cost/latency per request, and exposes
a live **observability dashboard**. Everything in this stack is free and
open-source — no paid API keys required to run it end-to-end.

```
                 ┌─────────────┐        ┌──────────────┐
  client  ─────▶ │   FastAPI    │──────▶ │    Ollama     │  (local, free,
 (API key)       │   gateway    │        │ open models   │   open-weight)
                 │              │◀───────┤ llama3/mistral│
                 └───┬──────┬───┘        └──────────────┘
                     │      │
         semantic    │      │  request/cost/latency log
         cache       ▼      ▼
                 ┌────────┐ ┌────────────┐
                 │ Redis  │ │ Postgres   │
                 └────────┘ └────────────┘
                     ▲
                     │ stats API
                     │
                 ┌────────────────┐
                 │ React dashboard │  live charts, request explorer,
                 │  (Vite + TS)    │  API key management
                 └────────────────┘
```

## Why

Every call to `POST /v1/chat/completions` is checked against a semantic
cache before it reaches the model: an exact-hash check first, then a
cosine-similarity comparison against recent embeddings for that model. A
cache hit skips inference entirely — zero cost, near-zero latency — and the
dashboard shows exactly how much that's saving in real time.

## Stack

| Layer | Tech |
|---|---|
| Model serving | [Ollama](https://ollama.com) (local, open-weight models) with an automatic mock-provider fallback |
| Backend | FastAPI, SQLAlchemy |
| Cache | Redis (semantic cache) |
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
uvicorn app.main:app --reload

# frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Try it

```bash
# 1. create a gateway API key (ADMIN_KEY defaults to "change-me-admin-key")
curl -X POST localhost:8000/v1/admin/keys \
  -H "x-admin-key: change-me-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "local-dev"}'
# -> {"api_key": "llmgw_..."}

# 2. send a chat completion through the gateway
curl -X POST localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer llmgw_..." \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [{"role": "user", "content": "hello"}]}'

# 3. send the exact same request again -> "cached": true, cost_usd: 0, x-cache: hit
```

Then open the dashboard to watch request volume, cache hit rate, cost
saved, and latency update live.

## Tests

```bash
cd backend && pip install -r requirements-dev.txt && ruff check app tests && pytest
cd frontend && npm install && npm run lint && npm run build
```

Both run in CI on every push (`.github/workflows/ci.yml`).

## Project layout

```
backend/
  app/
    main.py         FastAPI app + router wiring
    providers.py     Ollama client + mock fallback
    cache.py         semantic cache (exact + embedding similarity, via Redis)
    embeddings.py    sentence-transformers embedder + hashing fallback
    pricing.py       cost estimation per model
    auth.py          API key issuance/verification
    api/
      gateway.py     POST /v1/chat/completions
      admin.py       API key CRUD (admin-key protected)
      stats.py       summary / timeseries / provider breakdown / request log
  tests/
frontend/
  src/
    pages/           Dashboard, Requests, ApiKeys
    api/client.ts    typed fetch wrapper
```

## Notes / next steps

This is a working scaffold, not a finished product. Natural next steps:
- Swap the Redis linear-scan similarity search for a real ANN index
  (pgvector or RediSearch) once cache size matters.
- Add Alembic migrations instead of `create_all`-on-startup.
- Add streaming responses (SSE) for chat completions.
- Add per-key rate limiting and usage quotas.
