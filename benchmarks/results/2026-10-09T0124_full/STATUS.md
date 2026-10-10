# Run 2 — Great Lakes job 63571760

- **Hardware:** one NVIDIA A40 (48 GB), Great Lakes `spgpu`, whole GPU (node gl1522).
- **Code:** commit `4f42b29` (clean). vLLM 0.31.0. Started 2026-10-09 01:24 UTC; 3 h 16 min; $0.70.
- **Settings:** `session.json` (vLLM flags, environment, pinned model revisions).

| Part | Status |
|---|---|
| GSM8K accuracy (all three variants, n=1,319, cap 2,048) | **Valid.** 0 request errors; 1 possibly-truncated answer in total (AWQ). |
| GPU memory from vLLM's log (weights, KV cache, max concurrency, kernel) | **Valid.** |
| Direct-to-vLLM speed numbers, all concurrency levels | **Valid.** 0 failures, exactly 256 output tokens each, 0 cached prompt tokens. |
| Gateway speed numbers | **Not valid.** Concurrency 64 failed 68–75% of requests and lower levels may be inflated, due to a gateway bug since fixed (DECISIONS.md D16). Re-measured after the fix in `2026-10-09T1514_perf`. |
