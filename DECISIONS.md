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
