# Speed run — Great Lakes job 63609971

- **Hardware:** one NVIDIA A40 (48 GB), Great Lakes `spgpu`, whole GPU (node gl1522).
- **Code:** commit `00f5b3a` (clean; includes the D16 fix). vLLM 0.31.0. Started 2026-10-09 15:14 UTC; 2 h 27 min; ~$0.53 (8,804 s at $0.217/h).
- **Profile:** `perf` = the `full` speed settings without GSM8K (accuracy comes from run `2026-10-09T0124_full`).

| Part | Status |
|---|---|
| Gateway and direct speed, all variants, concurrency 1/4/16/64 | **Valid.** 0 failed requests anywhere (gateway at 64 included), exactly 256 output tokens each, 0 cached prompt tokens, 0 gateway tracebacks. |
| Repeats (gateway, concurrency 1 and 16, 2 extra runs each) | **Valid.** |
| GPU memory from vLLM's log | Valid, and matches `2026-10-09T0124_full`; the published table uses that run. |

These are the speed numbers published in `RESULTS.md`.
