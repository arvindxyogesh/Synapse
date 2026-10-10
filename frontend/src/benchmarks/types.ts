// Shape of benchmarks/results/summary.json, written by
// benchmarks/synapse_bench/summarize.py. JSON object keys are strings, so
// concurrency levels appear as "1", "4", "16", "64".

export type Target = "direct" | "gateway";

export interface SpeedRow {
  n_ok: number;
  n_total: number;
  n_errors: number;
  ttft_p50_ms: number | null;
  ttft_p95_ms: number | null;
  e2e_p50_ms: number | null;
  e2e_p95_ms: number | null;
  decode_tok_s_p50: number | null;
  aggregate_output_tok_s: number | null;
  requests_per_s: number | null;
  cost_per_1k_output_tokens_usd: number | null;
  mean_output_tokens: number | null;
  cached_prompt_tokens: number | null;
}

export interface Accuracy {
  correct: number;
  accuracy: number;
  ci95_low: number;
  ci95_high: number;
}

export interface Comparison {
  difference_b_minus_a: number;
  mcnemar_p: number;
  only_a_correct: number;
  only_b_correct: number;
}

export interface VariantSummary {
  label: string;
  speed: {
    by_target: Record<Target, Record<string, SpeedRow>>;
    gateway_spread: Record<string, Record<string, { min: number; max: number; n_runs: number }>>;
  };
  memory: {
    weights_gib: number;
    kv_cache_memory_gib: number;
    kv_cache_tokens: number;
    max_concurrency: { tokens_per_request: number; requests: number };
    quant_kernel: string[] | null;
  };
  quality: {
    n: number;
    n_errors: number;
    n_possibly_truncated: number;
    strict: Accuracy;
    flexible: Accuracy;
    vs_baseline?: { strict: Comparison; flexible: Comparison };
  };
}

export interface RunInfo {
  folder: string;
  commit: string;
  dirty: boolean;
  gpu: string;
  gpu_memory_utilization: number;
  vllm_version: string;
  vllm_args: string[];
  started_utc: string;
  gpu_price_per_hour: number | null;
}

export interface BenchmarkSummary {
  provenance: {
    speed_run: RunInfo;
    quality_and_memory_run: RunInfo;
    speed_settings: { concurrency: string; requests: number; warmup: number; max_tokens: number;
      fixed_length_outputs: boolean };
  };
  variants: Record<string, VariantSummary>;
  cross_card?: { other_run: string; other_gpu: string;
    variants: Record<string, Record<string, { this_run: number; other_run: number }>> };
}
