// Pure functions that reshape summary.json into chart rows. Kept free of
// React so they can be unit-tested (data.test.ts).

import type { BenchmarkSummary, SpeedRow, Target } from "./types";

export type SpeedMetric = keyof Pick<
  SpeedRow,
  | "aggregate_output_tok_s"
  | "decode_tok_s_p50"
  | "ttft_p50_ms"
  | "ttft_p95_ms"
  | "e2e_p50_ms"
  | "e2e_p95_ms"
  | "cost_per_1k_output_tokens_usd"
>;

export const SPEED_METRICS: { key: SpeedMetric; label: string; unit: string; higherIsBetter: boolean }[] = [
  { key: "aggregate_output_tok_s", label: "Throughput", unit: "output tok/s", higherIsBetter: true },
  { key: "decode_tok_s_p50", label: "Decode speed per request (p50)", unit: "tok/s", higherIsBetter: true },
  { key: "ttft_p50_ms", label: "Time to first token, p50", unit: "ms", higherIsBetter: false },
  { key: "ttft_p95_ms", label: "Time to first token, p95", unit: "ms", higherIsBetter: false },
  { key: "e2e_p50_ms", label: "End-to-end latency, p50", unit: "ms", higherIsBetter: false },
  { key: "e2e_p95_ms", label: "End-to-end latency, p95", unit: "ms", higherIsBetter: false },
  { key: "cost_per_1k_output_tokens_usd", label: "Cost per 1K output tokens", unit: "USD", higherIsBetter: false },
];

/** Variant keys in the order they're stored (16-bit baseline first). Colours
 * follow this order and are attached to the variant, never to its rank. */
export function variantKeys(summary: BenchmarkSummary): string[] {
  return Object.keys(summary.variants);
}

/** Concurrency levels, numerically sorted, as the strings used in the JSON. */
export function concurrencyLevels(summary: BenchmarkSummary): string[] {
  const first = Object.values(summary.variants)[0];
  return Object.keys(first.speed.by_target.direct).sort((a, b) => Number(a) - Number(b));
}

export type ChartRow = { concurrency: string } & Record<string, number | string | null>;

/** One row per concurrency level, one column per variant. */
export function speedRows(summary: BenchmarkSummary, target: Target, metric: SpeedMetric): ChartRow[] {
  return concurrencyLevels(summary).map((c) => {
    const row: ChartRow = { concurrency: c };
    for (const key of variantKeys(summary)) {
      row[key] = summary.variants[key].speed.by_target[target][c]?.[metric] ?? null;
    }
    return row;
  });
}

/** Throughput lost by going through the gateway instead of straight to vLLM,
 * in percent of the direct throughput (positive = gateway is slower). */
export function gatewayOverheadRows(summary: BenchmarkSummary): ChartRow[] {
  return concurrencyLevels(summary).map((c) => {
    const row: ChartRow = { concurrency: c };
    for (const key of variantKeys(summary)) {
      const t = summary.variants[key].speed.by_target;
      const direct = t.direct[c]?.aggregate_output_tok_s;
      const gateway = t.gateway[c]?.aggregate_output_tok_s;
      row[key] = direct && gateway != null ? ((direct - gateway) / direct) * 100 : null;
    }
    return row;
  });
}

export interface MemoryRow {
  key: string;
  label: string;
  weights: number;
  kvCache: number;
  maxConcurrency: number;
}

export function memoryRows(summary: BenchmarkSummary): MemoryRow[] {
  return variantKeys(summary).map((key) => {
    const m = summary.variants[key].memory;
    return { key, label: summary.variants[key].label, weights: m.weights_gib, kvCache: m.kv_cache_memory_gib,
      maxConcurrency: m.max_concurrency.requests };
  });
}

export interface AccuracyRow {
  key: string;
  label: string;
  accuracy: number; // percent
  low: number; // percent
  high: number; // percent
  /** [below, above] distances from the point, as Recharts' ErrorBar expects. */
  error: [number, number];
  n: number;
  vsBaseline: { differencePts: number; p: number } | null;
}

export function accuracyRows(summary: BenchmarkSummary, kind: "strict" | "flexible"): AccuracyRow[] {
  return variantKeys(summary).map((key) => {
    const q = summary.variants[key].quality;
    const a = q[kind];
    const cmp = q.vs_baseline?.[kind];
    const pct = (x: number) => x * 100;
    return {
      key,
      label: summary.variants[key].label,
      accuracy: pct(a.accuracy),
      low: pct(a.ci95_low),
      high: pct(a.ci95_high),
      error: [pct(a.accuracy - a.ci95_low), pct(a.ci95_high - a.accuracy)],
      n: q.n,
      vsBaseline: cmp ? { differencePts: pct(cmp.difference_b_minus_a), p: cmp.mcnemar_p } : null,
    };
  });
}

/** Ratio of the fastest variant to the baseline for a metric at one level,
 * e.g. "4-bit decodes 3.1x faster than 16-bit at concurrency 1". */
export function speedupVsBaseline(summary: BenchmarkSummary, target: Target, metric: SpeedMetric,
                                  concurrency: string, variant: string): number | null {
  const [baseline] = variantKeys(summary);
  const base = summary.variants[baseline].speed.by_target[target][concurrency]?.[metric];
  const mine = summary.variants[variant].speed.by_target[target][concurrency]?.[metric];
  return base && mine != null ? mine / base : null;
}
