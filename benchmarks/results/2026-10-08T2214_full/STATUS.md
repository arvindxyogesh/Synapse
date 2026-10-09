# Run 1 — Great Lakes job 63559791 — SUPERSEDED, kept as raw data

- **Hardware:** one NVIDIA A40 (48 GB), Great Lakes `spgpu`, whole GPU (node gl1518).
- **Code:** commit `128e91c` (clean). vLLM 0.31.0. Started 2026-10-08 22:14 UTC; 2 h 23 min; $0.52.

| Part | Status |
|---|---|
| `qwen2.5-7b-awq` | **Failed before measuring anything** (port hand-over bug, DECISIONS.md D15). |
| GSM8K (bf16, GPTQ) | **Not valid as results**: 512-token cap cut off ~2% of genuine solutions (D15). Superseded by run 2 at 2,048. |
| Gateway speed numbers | **Not valid**: affected by the DB-pool bug at high concurrency (D16). |
| Direct-to-vLLM speed numbers (bf16, GPTQ) | Valid measurements. Kept as a second, independent run on a *different* A40, to show run-to-run / card-to-card spread. |
