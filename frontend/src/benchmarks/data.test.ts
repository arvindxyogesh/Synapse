import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
  accuracyRows,
  concurrencyLevels,
  gatewayOverheadRows,
  memoryRows,
  speedRows,
  speedupVsBaseline,
  variantKeys,
} from "./data";
import realSummary from "./summary.json";
import type { BenchmarkSummary, SpeedRow } from "./types";

function speed(throughput: number, ttft: number): SpeedRow {
  return {
    n_ok: 100, n_total: 100, n_errors: 0, ttft_p50_ms: ttft, ttft_p95_ms: ttft * 2, e2e_p50_ms: 1000,
    e2e_p95_ms: 1100, decode_tok_s_p50: 50, aggregate_output_tok_s: throughput, requests_per_s: 1,
    cost_per_1k_output_tokens_usd: 0.001, mean_output_tokens: 256, cached_prompt_tokens: 0,
  };
}

function variant(label: string, direct: [number, number], gateway: [number, number], acc: number) {
  return {
    label,
    speed: {
      by_target: {
        // Deliberately out of numeric order: "16" before "4".
        direct: { "1": speed(direct[0], 20), "16": speed(direct[1], 200), "4": speed(direct[0] * 3, 60) },
        gateway: { "1": speed(gateway[0], 50), "16": speed(gateway[1], 400), "4": speed(gateway[0] * 3, 120) },
      },
      gateway_spread: {},
    },
    memory: { weights_gib: 10, kv_cache_memory_gib: 30, kv_cache_tokens: 1000,
      max_concurrency: { tokens_per_request: 4096, requests: 120.4 }, quant_kernel: null },
    quality: { n: 1000, n_errors: 0, n_possibly_truncated: 0,
      strict: { correct: acc * 1000, accuracy: acc, ci95_low: acc - 0.02, ci95_high: acc + 0.01 },
      flexible: { correct: acc * 1000, accuracy: acc, ci95_low: acc - 0.02, ci95_high: acc + 0.01 } },
  };
}

const fixture = {
  provenance: {} as BenchmarkSummary["provenance"],
  variants: {
    base: variant("16-bit", [100, 1000], [100, 800], 0.9),
    quant: {
      ...variant("4-bit", [300, 2000], [290, 1500], 0.88),
      quality: {
        ...variant("4-bit", [300, 2000], [290, 1500], 0.88).quality,
        vs_baseline: {
          strict: { difference_b_minus_a: -0.02, mcnemar_p: 0.3, only_a_correct: 30, only_b_correct: 10 },
          flexible: { difference_b_minus_a: -0.02, mcnemar_p: 0.3, only_a_correct: 30, only_b_correct: 10 },
        },
      },
    },
  },
} as BenchmarkSummary;

describe("benchmark data shaping", () => {
  it("keeps variants in stored order (baseline first) and sorts concurrency numerically", () => {
    expect(variantKeys(fixture)).toEqual(["base", "quant"]);
    expect(concurrencyLevels(fixture)).toEqual(["1", "4", "16"]);
  });

  it("builds one row per concurrency level with a column per variant", () => {
    expect(speedRows(fixture, "direct", "aggregate_output_tok_s")).toEqual([
      { concurrency: "1", base: 100, quant: 300 },
      { concurrency: "4", base: 300, quant: 900 },
      { concurrency: "16", base: 1000, quant: 2000 },
    ]);
    expect(speedRows(fixture, "gateway", "ttft_p50_ms")[2]).toEqual({ concurrency: "16", base: 400, quant: 400 });
  });

  it("computes gateway overhead as percent of direct throughput lost", () => {
    const rows = gatewayOverheadRows(fixture);
    expect(rows[0].base).toBeCloseTo(0); // 100 -> 100
    expect(rows[2].base).toBeCloseTo(20); // 1000 -> 800
    expect(rows[2].quant).toBeCloseTo(25); // 2000 -> 1500
  });

  it("exposes memory per variant", () => {
    expect(memoryRows(fixture)[0]).toEqual({ key: "base", label: "16-bit", weights: 10, kvCache: 30,
      maxConcurrency: 120.4 });
  });

  it("turns accuracy into percent with asymmetric error bars", () => {
    const [base, quant] = accuracyRows(fixture, "strict");
    expect(base.accuracy).toBeCloseTo(90);
    expect(base.error[0]).toBeCloseTo(2);
    expect(base.error[1]).toBeCloseTo(1);
    expect(base.vsBaseline).toBeNull();
    expect(quant.vsBaseline?.differencePts).toBeCloseTo(-2);
    expect(quant.vsBaseline?.p).toBe(0.3);
  });

  it("computes speedup relative to the baseline", () => {
    expect(speedupVsBaseline(fixture, "direct", "aggregate_output_tok_s", "1", "quant")).toBeCloseTo(3);
  });
});

describe("the real results", () => {
  it("are an exact copy of benchmarks/results/summary.json (run `npm run sync-results` if this fails)", () => {
    const source = readFileSync(resolve(__dirname, "../../../benchmarks/results/summary.json"), "utf8");
    const copy = readFileSync(resolve(__dirname, "summary.json"), "utf8");
    expect(copy).toBe(source);
  });

  it("contain the three variants at four concurrency levels, with no failed requests", () => {
    const summary = realSummary as unknown as BenchmarkSummary;
    expect(variantKeys(summary)).toEqual(["qwen2.5-7b-bf16", "qwen2.5-7b-gptq-int8", "qwen2.5-7b-awq"]);
    expect(concurrencyLevels(summary)).toEqual(["1", "4", "16", "64"]);
    for (const v of Object.values(summary.variants)) {
      for (const target of ["direct", "gateway"] as const) {
        for (const row of Object.values(v.speed.by_target[target])) expect(row.n_errors).toBe(0);
      }
    }
  });
});
