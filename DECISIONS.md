# Decisions

Real trade-offs made in this project, in plain language, newest last. Each
entry says what was decided, what else was on the table, why, and what it
costs. If an entry turns out to be wrong, it gets a follow-up entry rather
than being edited away.

---

## D1. A failing backend returns an error, not a fake answer

**Date:** 2026-10-08

**Context.** The gateway used to catch *any* HTTP error from Ollama/vLLM
(connection refused, timeout, a 500 from an out-of-memory crash) and quietly
answer with a canned mock response instead. That made `docker compose up`
demo nicely with no model running, but a mock answer takes about 1 ms, so a
backend that crashed halfway through a benchmark would have made the results
look *better*: faster responses and no errors.

**Options.**
1. Keep the silent fallback and have the benchmark filter out `provider="mock"` rows afterwards.
2. Remove the fallback entirely.
3. Make it opt-in (`MOCK_FALLBACK`, default off), and log a warning whenever it fires.

**Decision.** Option 3. The app now returns **HTTP 502** when the backend fails.
The failed call is also written to `request_logs` with `status="error"`, so
error rates are visible per model instead of vanishing. `docker-compose.yml`
and `.env.example` turn the fallback on, so the zero-setup demo still works.

**Why not 1.** It relies on every consumer of the data remembering to filter.
The gateway's own dashboard didn't, so a failure would still have shown up as
a latency *improvement* there.

**Cost.** A plain `uvicorn` run with no model now returns 502s instead of mock
text, which is a small surprise for someone trying it the first time. The
README says so.

**Related details.**
- **Streaming.** The first chunk is now fetched *before* the HTTP response
  starts, because once a 200 status line has gone out you can't change it to
  a 502. A failure *after* the first chunk ends the stream without the
  `data: [DONE]` marker, which a client (and the benchmark) can detect as an
  incomplete response.
- **Dashboard latency.** Latency stats in `/v1/stats/summary` now count
  successful responses only, so a burst of fast 502s (or slow timeouts)
  doesn't skew "how fast are answers".
- **Shadow verification.** The LLM-judge already caught exceptions and fell
  back to its word-overlap heuristic, so cache shadow verification behaves
  the same as before when the backend is down.

---

## D2. Forward `max_tokens`, and skip the cache for requests that set it

**Date:** 2026-10-08

**Context.** The API accepted `max_tokens` but never passed it to Ollama or
vLLM, so every reply ran until the model decided to stop. For a benchmark
that's fatal: if one model variant writes longer answers, its tokens/s and
latency aren't comparable with another's. It's now forwarded (`max_tokens`
for vLLM, `options.num_predict` for Ollama, which uses a different name).

**The new problem this creates.** The semantic cache is keyed on
*(model, prompt)* only. Once `max_tokens` actually works, a reply cut off at
5 tokens could be cached and then served to a later request for the same
prompt that asked for no limit. That's a wrong answer, not just a slow one.

**Options.**
1. Add `max_tokens` to the cache key. That splits the cache into many small
   pieces (one per cap value), and a request capped at 300 still couldn't
   reuse a perfectly good 120-token answer stored under cap 256.
2. Store a reply only if the backend says it finished naturally
   (`finish_reason == "stop"`, not `"length"`). This is the most precise
   option, but it needs `finish_reason` plumbed through both providers and
   the streaming path.
3. Requests that set `max_tokens` skip the cache: no lookup, no store.

**Decision.** Option 3 for now, because it's simple and obviously correct.
The response says so: `x-cache: bypass`.

**Cost.** Clients that always send `max_tokens` (many SDK users do) get no
caching at all. Option 2 is the better long-term fix and is a known
follow-up.

**Related, pre-existing limitation (not fixed here).** `temperature` isn't in
the cache key either, so a `temperature=0` request can be served a reply that
was sampled at `temperature=0.7`. It's noted here so it isn't forgotten. The
benchmark bypasses the cache for its performance runs, so it isn't affected.

---

## D3. Turning the cache off: a per-request header

**Date:** 2026-10-08

**Context.** The benchmark needs to measure *the model*, with the cache out
of the way. The bonus experiment then needs cache on vs. off side by side.

**Options.**
1. A global setting (`CACHE_ENABLED=false`). That needs a gateway restart to
   flip, and it's easy to forget it's off in a demo afterwards.
2. A field in the request body. That isn't part of the OpenAI request shape,
   and some SDKs reject or strip unknown fields.
3. A request header, `x-synapse-cache: bypass`.

**Decision.** Option 3. Every OpenAI SDK lets you add headers
(`extra_headers=...`), so it doesn't touch the request format, and it
applies to one request at a time, so cache-on and cache-off traffic can run
against the same gateway. A bypassed request skips the lookup, the store
*and* the embedding computation, so cache-off timings don't include work
whose result would be thrown away. The response says `x-cache: bypass`, which
lets the benchmark check every request really skipped the cache rather than
trusting that it did.

**Small guard.** Any value other than `bypass` gets a 400. A typo like
`bypas` would otherwise silently leave the cache *on* and contaminate a run.

---

## D4. Fix the Postgres-only dashboard bug by bucketing in Python, and test on Postgres in CI

**Date:** 2026-10-08

**Context.** `/v1/stats/timeseries` (the dashboard's main chart) grouped rows
by hour using `strftime()`, a SQLite function. Postgres doesn't have it, so
on the docker-compose stack the endpoint failed with
`function strftime(unknown, timestamp with time zone) does not exist`
(reproduced against `postgres:16-alpine` before fixing). CI never noticed
because the tests only ran on SQLite, and no test covered that endpoint.

**Options.**
1. Pick SQL per database (`date_trunc` on Postgres, `strftime` on SQLite).
   This keeps the work in the database, but it means two code paths, and
   each one is only tested on one database.
2. Load the rows for the window and group them by hour in Python, which is
   what `/v1/stats/summary` already does.

**Decision.** Option 2. One code path, identical behavior on both databases,
and easy to read. Timestamps are converted to UTC before bucketing, because
Postgres returns timezone-aware values and SQLite returns naive ones.

**Cost.** Every row in the window (up to 30 days) is loaded into memory. At
portfolio scale (thousands of rows) that's milliseconds. At millions of rows
it wouldn't be, and the fix would be option 1 or a pre-aggregated table.

**Preventing a repeat.** A new CI job, `backend-postgres`, runs the whole test
suite against a real Postgres 16 service container. It also runs
`alembic upgrade head` (migrations apply cleanly) and `alembic check`
(migrations match the models, so a forgotten migration fails CI). Locally:
`TEST_DATABASE_URL=postgresql://... pytest`.

**Also changed.** Timeseries latency now averages successful responses only,
the same as `/summary` (D1).

---

## D5. Keep one HTTP client per request for now; make the backend timeout configurable

**Date:** 2026-10-08

**Context.** `app/providers.py` creates a new `httpx.AsyncClient` for every
call to the model backend. The usual advice is to reuse one client, so open
connections get reused (connection pooling) instead of reconnecting every
time. The plan included making that switch.

**What I measured first.** `benchmarks/micro/httpx_client_reuse.py` sends
400 requests to an instant-response stub server at concurrency 1, 16 and 64,
with a new client per request vs. one shared client. Six runs on a MacBook,
raw output in `benchmarks/micro/results_2026-10-08_macbook.txt`:
- At concurrency 1 and 16 the results were consistent: the shared client is
  faster by under 1 ms and by about 25 ms (p50) respectively.
- At concurrency 64 the results **didn't reproduce**. The same settings gave
  a shared-client p50 of about 183–207 ms in three runs and about 41–44 ms in
  two others.

The likely cause is that the client, the server and 64 concurrent requests
all shared one laptop CPU, so the test mostly measured Python's own
scheduling, not connection reuse.

**Decision.** Don't change the client on the strength of an unreliable
measurement. The real benchmark (milestone 3) sends the same traffic both
*through the gateway* and *directly to vLLM* on the GPU machine, and the
difference between the two is the gateway's total overhead. If that overhead
turns out to matter, connection reuse is one of the first things to try,
measured the same way.

**What did change.** The backend timeout was hard-coded to 60 s and is now
`BACKEND_TIMEOUT_SECONDS` (still 60 by default). At high concurrency a long
generation can sit in vLLM's queue past 60 s, and the benchmark should be
able to raise the timeout rather than record those requests as errors.

**Lesson carried into the benchmark design.** At high concurrency, *the load
generator itself* can be the bottleneck. The milestone 3 harness will record
its own CPU usage, and will run on a machine with enough cores to keep up, so
a slow client isn't reported as a slow server.
