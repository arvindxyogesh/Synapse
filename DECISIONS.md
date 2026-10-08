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

---

## D6. A model registry file, with the old single-backend behavior as the fallback

**Date:** 2026-10-08

**Context.** The benchmark needs the 16-bit, 4-bit AWQ and 8-bit GPTQ
versions of Qwen2.5-7B-Instruct behind one gateway, each as its own model
name, so requests can be routed to any of them and metrics are recorded per
variant. Before this change there was one global backend (`PROVIDER` plus one
URL), and the `model` string was just passed through to it.

**Why each variant needs its own server.** A vLLM server loads one set of
weights at startup, so a 16-bit and a 4-bit checkpoint are two separate
`vllm serve` processes on two ports. The gateway has to know which port
serves which name. (For the benchmark they run one at a time on the same
GPU, as explained in docs/PLAN.md. The registry still lists all three, so
whichever one is up can be reached by name.)

**Options.**
1. Environment variables per model (`MODEL_QWEN_AWQ_URL=...`). That gets
   unreadable past two models, and model names don't map cleanly onto
   environment variable names.
2. A database table, editable through the admin API. It's dynamic, but it
   needs a migration, endpoints and UI, and it's far more than "which port
   serves which model" requires.
3. A small TOML file (`MODEL_REGISTRY_PATH`), parsed with Python's built-in
   `tomllib`, so no new dependency.

**Decision.** Option 3. Each entry is just `provider`, `base_url` and an
optional `upstream_model` (the name the backend itself uses, e.g.
`Qwen/Qwen2.5-7B-Instruct-AWQ`, so clients can use a short name).
Unregistered names keep the old behavior, so nothing that worked before
breaks. The file is checked strictly at startup: an unknown provider, a URL
that isn't http(s), or a misspelled setting (`base_ur`) stops the gateway
with a clear error. A silently ignored typo would send traffic to the wrong
backend and mislabel the results.

**Cost.** Changing the registry needs a gateway restart. That's fine for a
benchmark and a demo; it wouldn't be for a production system that adds
models often (that's where option 2 would win).

**Also added.** `GET /v1/models`, the OpenAI-compatible list (what
`client.models.list()` calls), tested with the real `openai` SDK.

---

## D7. Logging time to first token (TTFT): what exactly is measured

**Date:** 2026-10-08

**Context.** For a chat product, two latencies matter and they can move in
opposite directions. **TTFT** is how long the user stares at a blank screen.
**End-to-end latency** is how long until the whole answer is there.
Quantization can change them differently: TTFT is dominated by *prefill*
(processing the prompt, which is compute-heavy), while the rest is *decode*
(one token at a time, which is limited by memory bandwidth). The gateway only
logged end-to-end latency, so it now logs TTFT too (`request_logs.ttft_ms`,
Alembic migration `d0caa44f82d4`).

**What the number means.** From the moment the handler starts (after auth
and rate limiting) to the moment the first chunk *with actual text* is handed
to the response stream. Chunks with no text (some backends send an empty
first chunk carrying only the role) don't count, because the user sees
nothing yet.

**Choices made along the way.**
- **Streaming only.** A non-streamed reply arrives all at once, so it has no
  separate "first token". Those rows store `NULL`, not 0 and not a copy of
  the total latency, so averages over TTFT can't be quietly dragged by
  requests that never had one.
- **Gateway-side vs. client-side.** This is the gateway's view. A client
  also pays network time, so its TTFT is a bit higher. The benchmark records
  its own client-side TTFT as the headline number, and the logged one is
  what the gateway can report per model on its own. Comparing the two is a
  check on the gateway's overhead.
- **Cache hits get a TTFT too.** It's what a user actually experiences on a
  hit, and it shows how much faster a hit starts than a miss.

**Migration safety.** The column is nullable, so adding it to a table that
already has rows needs no backfill. It was tested upgrade, then downgrade,
then upgrade on SQLite, and CI applies it to Postgres and runs `alembic
check` there.

---

## D8. Fixed-length outputs for the speed benchmark (`ignore_eos`)

**Date:** 2026-10-08

**Context.** Tokens per second and end-to-end latency only compare fairly if
every variant generates *the same number of tokens*. Quantization changes a
model's outputs slightly, so given the same prompt the 4-bit variant might
stop after 180 tokens where the 16-bit one writes 240. Its end-to-end latency
would look better for a reason that has nothing to do with speed. `max_tokens`
alone only sets a ceiling; the model can still stop earlier.

**What `ignore_eos` does.** A model signals "I'm done" by generating a
special end-of-sequence (EOS) token. vLLM's `ignore_eos` option tells it to
keep going anyway until `max_tokens` is reached. Every request then produces
exactly `max_tokens` tokens, and speed is measured on identical amounts of
work.

**Options.**
1. Don't force a length. Report tokens/s from each variant's actual token
   counts and accept that latency comparisons are skewed. Simple, but the
   headline latency numbers would be misleading.
2. Pick prompts that always produce long answers and hope they hit the cap.
   This is unreliable, and still uncontrolled.
3. Pass vLLM's `ignore_eos` through the gateway.

**Decision.** Option 3 for the *speed* runs. `ignore_eos` isn't part of the
OpenAI API; it's a vLLM extension, and the OpenAI SDK sends it via
`extra_body`. The gateway accepts it as an explicit field and forwards it
only to vLLM. It returns 400 if `max_tokens` is missing (otherwise the model
would generate until the context window is full) or if the model is served
by Ollama, which has no equivalent, so the benchmark can't *think* it's
running fixed-length outputs when it isn't.

**What it does NOT apply to.** The *quality* evaluation (GSM8K) runs without
`ignore_eos`. There the model must stop naturally, because padding an answer
with extra tokens after it should have stopped would corrupt the
answer-extraction step. Output text after the EOS point is meaningless filler,
which is fine for measuring speed and wrong for measuring correctness.

**Cost.** One non-standard request field, documented in the README.

---

## D9. Streamed responses report token usage (OpenAI's `stream_options.include_usage`)

**Date:** 2026-10-08

**Context.** The benchmark measures output tokens per second, so it needs to
know how many tokens each streamed reply contained. Non-streamed responses
carry a `usage` object. Streamed ones from Synapse didn't. Counting the text
chunks isn't a substitute: a chunk can hold several tokens, or part of one.

**Options.**
1. Re-tokenize the received text on the client with the model's tokenizer.
   That adds a heavy dependency, needs the right tokenizer per model, and
   still isn't guaranteed to match what the server actually generated.
2. Always append a usage chunk to every stream. That could surprise clients
   that assume every chunk has at least one `choices` entry.
3. Follow the OpenAI spec: when a request sets
   `stream_options: {"include_usage": true}`, send one extra chunk at the
   end with empty `choices` and a `usage` object.

**Decision.** Option 3. It's what the real `openai` SDK already
understands, and it's tested with it. Clients that don't ask get exactly the
stream they got before. The counts come from the backend itself (vLLM's
exact numbers), the same ones the gateway already logs.

**Note for interviews.** The token counts are only as exact as the backend's.
vLLM reports exact counts. When a backend doesn't, the gateway falls back to
a rough characters ÷ 4 estimate (`providers._estimate_tokens`). The benchmark
targets vLLM, so its tokens/s numbers use exact counts.

---

## D10. How quality is measured: GSM8K, two scorings, and a paired test

**Date:** 2026-10-08

**Why measure quality at all.** Quantization shrinks the weights by
rounding them. The question is whether that rounding costs accuracy, and a
faster model that's wrong more often isn't a win by default. Speed numbers
without a quality number next to them can't support a recommendation.

**Why GSM8K.** These are 1,319 grade-school math word problems with one
numeric answer each.
- *Exact-match scoring.* The answer is a number, so grading is a string
  comparison after normalization. No judge model or human rater is needed,
  and anyone can rerun it and get the same score.
- *Sensitive to small errors.* Multi-step arithmetic fails if any step goes
  wrong, so it's a place where quantization damage should show if it exists.
- *Small enough to run in full.* All 1,319 problems take minutes on one GPU,
  so there's no question of which sample was chosen.
- *Known limitation.* It's one task type. A model that holds up on GSM8K
  could still degrade on, say, long-document summarization. The README will
  say "accuracy on GSM8K", not "quality".

**Two scorings, both reported.** *Strict* takes the number after the model's
final `####`, the format the prompt asks for. *Flexible* takes the last
number anywhere in the reply. If quantization made the model worse at
following the format but not at math, strict drops while flexible doesn't.
That distinction is worth seeing rather than averaging away.

**Conservative rules.**
- A question whose request failed (HTTP error, timeout) counts as **wrong**,
  not skipped. Dropping failures would raise the accuracy of a variant that
  fails more often.
- Synapse's non-streaming responses always say `finish_reason: "stop"`, even
  when the reply hit `max_tokens`. So replies that used exactly `max_tokens`
  are flagged as *possibly truncated* and counted in the summary.
  (`max_tokens=512` is generous for GSM8K. If many replies hit it, the cap
  needs raising, not the numbers explaining away.)
- Any answer that came from the mock provider or the cache invalidates the
  run (exit code 2).

**Confidence intervals and the paired test.** With 1,319 questions, a 95%
interval on an accuracy around 85% is roughly ±2 points (Wilson interval).
So two variants scoring 85.1% and 84.3% may well be indistinguishable.
Because both variants answer the *same* questions, the right test only looks
at the questions where they *disagree*. If they were equally good, those
disagreements would split about 50/50, and McNemar's exact test asks how
surprising the observed split is. A large p-value means "no detectable
difference at this sample size". That isn't proof they're equal, and the
README will word it that way.

**Settings.** Greedy decoding (`temperature=0`), so the result doesn't depend
on sampling luck. A zero-shot prompt (no worked examples), which is simpler
to explain, though it gives lower absolute scores than the few-shot setups
published leaderboards often use. So these numbers compare *variants with
each other*, not this model with published scores.

---

## D11. GPU memory: report what vLLM reserved, not just what `nvidia-smi` shows

**Date:** 2026-10-08

**The trap.** It's natural to expect "the 4-bit model uses less GPU memory"
to show up in `nvidia-smi`. With vLLM it doesn't. At startup vLLM loads the
weights, then claims a fixed fraction of the whole GPU
(`--gpu-memory-utilization`, default 0.9) and fills everything beyond the
weights with **KV cache**: the stored attention keys and values for every
token of every request in flight. So `nvidia-smi` reads about 90% for every
variant, before any traffic. Reporting that as "peak memory" would say
16-bit and 4-bit cost the same memory, which is technically true and
completely misleading.

**What quantization actually changes.** Smaller weights leave more of that
fixed budget for KV cache. More KV cache means more tokens of context can be
held at once, so more requests can run concurrently before vLLM has to queue
them. For a serving system that's the real memory benefit: capacity, not a
smaller footprint.

**What gets recorded.**
1. From vLLM's startup log (`--vllm-log`, parsed by `synapse_bench/gpu.py`):
   - weight memory (GiB);
   - KV cache memory (GiB) and size (tokens);
   - the maximum concurrency vLLM computed for our sequence length;
   - which quantized kernel it chose.
2. `nvidia-smi` peak per concurrency level (`--gpu-sample`, polled every
   200 ms), for completeness. It's expected to be nearly flat, and will be
   reported with the explanation above.

All variants run with the **same** `--gpu-memory-utilization` and
`--max-model-len`, so their KV-cache numbers are directly comparable.

**Honest caveat.** vLLM's log wording has changed between versions. The
parser handles the phrasings known when it was written, and returns `None`
for anything it can't find; it never estimates. Before any memory number is
published, milestone 4 checks the parser against the real server log from
the GPU run.

---

## D12. The dry run found a measurement trap: the backend's own prompt cache

**Date:** 2026-10-08

**What happened.** Before renting a GPU, the whole harness was run on a
MacBook against Ollama with `qwen2.5:0.5b`. Those numbers are throwaway and
aren't published; the point was to find bugs. The gateway-vs-direct
comparison at concurrency 1 showed the gateway's TTFT p95 about 10x higher
than direct. Taken at face value, that's a damning "gateway overhead" result.

**It was wrong.** The slow gateway requests were exactly the four longest
prompts (1,137–3,416 characters). The direct run happened *second*, and
Ollama keeps recently processed prompts in a cache, so the direct run got
those long prompts pre-processed. Swapping the order flipped the result:
cold direct was 107–259 ms on those prompts, and the gateway running second
was 20–23 ms. The gap was run order, not the gateway.

**Why it matters for the real run.** vLLM has the same mechanism
(*automatic prefix caching*): if a new request starts with the same tokens
as an earlier one, vLLM reuses the stored KV cache instead of recomputing
it. It's a real production optimization, but in a benchmark that sends the
same prompts at every concurrency level, to every target, it makes whatever
runs later look faster.

**Fix.**
1. *Prevent:* the GPU run starts vLLM with `--no-enable-prefix-caching`, so
   every request pays its full prompt cost and order can't matter.
2. *Detect:* the harness records the backend's own count of cached prompt
   tokens (`usage.prompt_tokens_details.cached_tokens`) whenever it's
   reported. It sums them per level and prints a warning if any are non-zero.
   This was checked against Ollama, where the warning fired (3,104 cached
   tokens). vLLM only reports this field when started with
   `--enable-prompt-tokens-details`, so the GPU run uses that flag too, to
   *prove* the cache was off rather than assume it.

**Limitation.** The gateway doesn't pass `prompt_tokens_details` through, so
only the direct target reports it. Both targets hit the same vLLM server,
though, so a direct run showing 0 cached tokens confirms the server's cache
was off for both.

**The broader lesson.** A comparison can be confounded by anything that
remembers earlier requests. This setup has three such layers: Synapse's
semantic cache (bypassed with a header), the backend's prompt cache (now
disabled and checked), and warm-up (handled with discarded warm-up
requests).
