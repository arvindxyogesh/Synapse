import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  LabelList,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import StatCard from "../components/StatCard";
import {
  accuracyRows,
  gatewayOverheadRows,
  memoryRows,
  SPEED_METRICS,
  speedRows,
  speedupVsBaseline,
  variantKeys,
  type ChartRow,
  type SpeedMetric,
} from "../benchmarks/data";
import summaryJson from "../benchmarks/summary.json";
import type { BenchmarkSummary, Target } from "../benchmarks/types";

const summary = summaryJson as unknown as BenchmarkSummary;

// Categorical slots 1-3 of the reference palette, dark steps; validated as a
// set against this card surface (#0f172a): all pairs pass CVD and contrast.
// Colour follows the variant (stored order), never its rank in a chart.
const SERIES_COLORS = ["#3987e5", "#d95926", "#199e70"];
const SURFACE = "#0f172a";
const GRID = "#1e293b";
const AXIS_TEXT = "#94a3b8";
const MUTED_FILL = "#475569";

const REPO_RESULTS = "https://github.com/arvindxyogesh/Synapse/blob/main/benchmarks/results";

const tooltipStyle = { background: SURFACE, border: `1px solid ${GRID}`, color: "#e2e8f0", fontSize: 12 };

function colorOf(key: string): string {
  return SERIES_COLORS[variantKeys(summary).indexOf(key)] ?? MUTED_FILL;
}

function labelOf(key: string): string {
  return summary.variants[key]?.label ?? key;
}

function fmt(value: number | null | undefined, metric: SpeedMetric): string {
  if (value == null) return "–";
  if (metric === "cost_per_1k_output_tokens_usd") return `$${value.toFixed(5)}`;
  if (metric.endsWith("_ms")) return `${Math.round(value).toLocaleString()} ms`;
  return value >= 100 ? Math.round(value).toLocaleString() : value.toFixed(1);
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
      <h2 className="text-sm font-medium text-slate-200">{title}</h2>
      {subtitle && <p className="mt-1 text-xs text-slate-400">{subtitle}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Toggle<T extends string>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-md border border-slate-700 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`rounded px-3 py-1 text-xs ${value === o.value ? "bg-slate-700 text-slate-50" : "text-slate-400 hover:text-slate-200"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// Legend text stays in text colour; the coloured swatch carries identity.
const legendFormatter = (value: string) => <span className="text-slate-300">{labelOf(value)}</span>;

function DataTable({ rows, columns, format }: {
  rows: ChartRow[]; columns: string[]; format: (v: number | null) => string;
}) {
  return (
    <table className="mt-3 w-full text-left text-xs text-slate-300">
      <thead className="text-slate-400">
        <tr>
          <th className="py-1 pr-4 font-medium">Concurrency</th>
          {columns.map((c) => <th key={c} className="py-1 pr-4 font-medium">{labelOf(c)}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.concurrency} className="border-t border-slate-800">
            <td className="py-1 pr-4">{r.concurrency}</td>
            {columns.map((c) => <td key={c} className="py-1 pr-4">{format(r[c] as number | null)}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** True below Tailwind's `sm` breakpoint (640 px). */
function useIsNarrow(): boolean {
  const query = "(max-width: 639px)";
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && window.matchMedia(query).matches);
  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setNarrow(mql.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

function SeriesLines({ rows, keys, format }: { rows: ChartRow[]; keys: string[]; format: (v: number) => string }) {
  // On phones the end-of-line labels would squeeze the plot to a third of the
  // card; the legend below still names every line there.
  const narrow = useIsNarrow();
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={rows} margin={{ top: 8, right: narrow ? 12 : 96, bottom: 8, left: narrow ? 0 : 8 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="concurrency" tick={{ fontSize: 11, fill: AXIS_TEXT }} stroke={GRID}
               label={{ value: "Concurrent requests", position: "insideBottom", offset: -4, fill: AXIS_TEXT, fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11, fill: AXIS_TEXT }} stroke={GRID} width={narrow ? 56 : 72}
               tickFormatter={(v: number) => format(v)} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v: number, name: string) => [format(v), labelOf(name)]}
                 labelFormatter={(c) => `Concurrency ${c}`} />
        <Legend formatter={legendFormatter} wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
        {keys.map((k) => (
          <Line key={k} type="linear" dataKey={k} stroke={colorOf(k)} strokeWidth={2}
                dot={{ r: 4, fill: colorOf(k), stroke: SURFACE, strokeWidth: 2 }} activeDot={{ r: 6 }}
                isAnimationActive={false}>
            {/* Direct label at the last point only (identity never relies on colour alone). */}
            <LabelList dataKey={k} content={({ x, y, index }) =>
              !narrow && index === rows.length - 1 ? (
                <text x={Number(x) + 10} y={Number(y) + 4} fill="#cbd5e1" fontSize={11}>{labelOf(k)}</text>
              ) : null} />
          </Line>
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function Benchmarks() {
  const [metric, setMetric] = useState<SpeedMetric>("aggregate_output_tok_s");
  const [target, setTarget] = useState<Target>("direct");
  const [accuracyKind, setAccuracyKind] = useState<"strict" | "flexible">("strict");
  const [showTables, setShowTables] = useState(false);

  const keys = variantKeys(summary);
  const [baseline, , fastest] = keys;
  const metricInfo = SPEED_METRICS.find((m) => m.key === metric)!;
  const speed = useMemo(() => speedRows(summary, target, metric), [target, metric]);
  const overhead = useMemo(() => gatewayOverheadRows(summary), []);
  const memory = useMemo(() => memoryRows(summary), []);
  const accuracy = useMemo(() => accuracyRows(summary, accuracyKind), [accuracyKind]);

  const p = summary.provenance;
  const decodeSpeedup = speedupVsBaseline(summary, "direct", "decode_tok_s_p50", "1", fastest);
  const throughputSpeedup = speedupVsBaseline(summary, "direct", "aggregate_output_tok_s", "64", fastest);
  const fastestAcc = accuracyRows(summary, "strict").find((r) => r.key === fastest)!;
  const memBase = memory.find((m) => m.key === baseline)!;
  const memFast = memory.find((m) => m.key === fastest)!;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-50">Quantization benchmark</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Qwen2.5-7B-Instruct served by vLLM at three weight precisions, measured through the Synapse gateway and
          directly, on one {p.speed_run.gpu.split(",")[0]}. Recorded results from the runs listed below — this page
          doesn't call a GPU.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label={`${labelOf(fastest)} decode speed`} value={decodeSpeedup ? `${decodeSpeedup.toFixed(1)}×` : "–"}
                  hint={`vs ${labelOf(baseline)}, one request at a time`} />
        <StatCard label={`${labelOf(fastest)} throughput`} value={throughputSpeedup ? `${throughputSpeedup.toFixed(1)}×` : "–"}
                  hint={`vs ${labelOf(baseline)}, 64 concurrent requests`} />
        <StatCard label="GSM8K accuracy change"
                  value={fastestAcc.vsBaseline ? `${fastestAcc.vsBaseline.differencePts >= 0 ? "+" : ""}${fastestAcc.vsBaseline.differencePts.toFixed(1)} pts` : "–"}
                  hint={fastestAcc.vsBaseline
                    ? `${labelOf(fastest)} vs ${labelOf(baseline)}; p = ${fastestAcc.vsBaseline.p.toFixed(2)}${fastestAcc.vsBaseline.p >= 0.05 ? ", not a detectable difference" : ""}`
                    : undefined} />
        <StatCard label="Max concurrent requests"
                  value={`${Math.round(memBase.maxConcurrency)} → ${Math.round(memFast.maxConcurrency)}`}
                  hint={`KV-cache room, ${labelOf(baseline)} → ${labelOf(fastest)} (4,096-token requests)`} />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label className="text-xs text-slate-400" htmlFor="metric">Metric</label>
        <select id="metric" value={metric} onChange={(e) => setMetric(e.target.value as SpeedMetric)}
                className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200">
          {SPEED_METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
        </select>
        <Toggle label="Measured" value={target} onChange={setTarget}
                options={[{ value: "direct", label: "Direct to vLLM" }, { value: "gateway", label: "Through gateway" }]} />
        <label className="ml-auto flex items-center gap-2 text-xs text-slate-400">
          <input type="checkbox" checked={showTables} onChange={(e) => setShowTables(e.target.checked)} />
          Show tables
        </label>
      </div>

      <Card title={`${metricInfo.label} (${metricInfo.unit})`}
            subtitle={`${metricInfo.higherIsBetter ? "Higher" : "Lower"} is better. ${target === "direct" ? "Straight to vLLM." : "Through the Synapse gateway (cache bypassed)."} Every request generates exactly ${p.speed_settings.max_tokens} tokens.`}>
        <SeriesLines rows={speed} keys={keys} format={(v) => fmt(v, metric)} />
        {showTables && <DataTable rows={speed} columns={keys} format={(v) => fmt(v, metric)} />}
      </Card>

      <Card title="Throughput lost to the gateway (%)"
            subtitle="(direct − through gateway) ÷ direct output tokens/s. The gateway is a single Python process re-streaming every token; its cost grows with concurrency.">
        <SeriesLines rows={overhead} keys={keys} format={(v) => `${v.toFixed(1)}%`} />
        {showTables && <DataTable rows={overhead} columns={keys} format={(v) => (v == null ? "–" : `${v.toFixed(1)}%`)} />}
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="GPU memory (GiB)"
              subtitle={`vLLM reserves ${Math.round(p.quality_and_memory_run.gpu_memory_utilization * 100)}% of the GPU and fills what the weights leave with KV cache, so smaller weights mean more requests in flight.`}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={memory} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }}>
              <CartesianGrid stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: AXIS_TEXT }} stroke={GRID} unit=" GiB" />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11, fill: AXIS_TEXT }} stroke={GRID} width={96} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#1e293b55" }}
                       formatter={(v: number, name: string) => [`${v.toFixed(2)} GiB`, name === "weights" ? "Weights" : "KV cache"]} />
              <Legend wrapperStyle={{ fontSize: 12 }} content={() => (
                <div className="flex justify-center gap-5 pt-2 text-xs text-slate-300">
                  <span className="flex items-center gap-1.5">
                    {keys.map((k) => <span key={k} className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: colorOf(k) }} />)}
                    Weights
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: MUTED_FILL }} />
                    KV cache
                  </span>
                </div>
              )} />
              <Bar dataKey="weights" stackId="m" stroke={SURFACE} strokeWidth={2} isAnimationActive={false}>
                {memory.map((m) => <Cell key={m.key} fill={colorOf(m.key)} />)}
              </Bar>
              <Bar dataKey="kvCache" name="kvCache" stackId="m" fill={MUTED_FILL} stroke={SURFACE} strokeWidth={2}
                   radius={[0, 4, 4, 0]} isAnimationActive={false}>
                <LabelList dataKey="maxConcurrency" position="right" fill="#cbd5e1" fontSize={11}
                           formatter={(v: number) => `${Math.round(v)} req`} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {showTables && (
            <table className="mt-3 w-full text-left text-xs text-slate-300">
              <thead className="text-slate-400"><tr><th className="py-1 font-medium">Variant</th><th className="font-medium">Weights</th><th className="font-medium">KV cache</th><th className="font-medium">Max concurrent</th></tr></thead>
              <tbody>{memory.map((m) => (
                <tr key={m.key} className="border-t border-slate-800"><td className="py-1">{m.label}</td><td>{m.weights.toFixed(2)} GiB</td><td>{m.kvCache.toFixed(2)} GiB</td><td>{Math.round(m.maxConcurrency)}</td></tr>
              ))}</tbody>
            </table>
          )}
        </Card>

        <Card title="GSM8K accuracy (%), 95% interval"
              subtitle={`All ${accuracy[0].n.toLocaleString()} test questions, greedy decoding. Overlapping intervals and a paired McNemar test say whether a gap is real.`}>
          <div className="mb-2">
            <Toggle label="Scoring" value={accuracyKind} onChange={setAccuracyKind}
                    options={[{ value: "strict", label: "Strict (\\boxed{})" }, { value: "flexible", label: "Flexible (last number)" }]} />
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ top: 8, right: 48, bottom: 4, left: 8 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis type="category" dataKey="label" allowDuplicatedCategory={false} tick={{ fontSize: 11, fill: AXIS_TEXT }} stroke={GRID} />
              <YAxis type="number" dataKey="accuracy" domain={[84, 96]} tick={{ fontSize: 11, fill: AXIS_TEXT }} stroke={GRID} unit="%" width={48} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: GRID }}
                       content={({ payload }) => {
                         const r = payload?.[0]?.payload as (typeof accuracy)[number] | undefined;
                         if (!r) return null;
                         return (
                           <div style={tooltipStyle} className="rounded p-2">
                             <div className="font-medium">{r.label}</div>
                             <div>{r.accuracy.toFixed(1)}% [{r.low.toFixed(1)}, {r.high.toFixed(1)}], n = {r.n}</div>
                             {r.vsBaseline && <div>vs {labelOf(baseline)}: {r.vsBaseline.differencePts.toFixed(1)} pts, p = {r.vsBaseline.p.toFixed(2)}</div>}
                           </div>
                         );
                       }} />
              <Scatter data={accuracy} isAnimationActive={false}
                       shape={(props: { cx?: number; cy?: number; fill?: string }) => (
                         <circle cx={props.cx} cy={props.cy} r={7} fill={props.fill} stroke={SURFACE} strokeWidth={2} />
                       )}>
                {accuracy.map((r) => <Cell key={r.key} fill={colorOf(r.key)} />)}
                <ErrorBar dataKey="error" width={12} strokeWidth={2} stroke="#64748b" direction="y" />
                <LabelList dataKey="accuracy" position="right" offset={16} fill="#e2e8f0" fontSize={12}
                           formatter={(v: number) => `${v.toFixed(1)}%`} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <p className="mt-2 text-xs text-slate-500">The axis starts at 84% to show the intervals; these are points, not bars.</p>
        </Card>
      </div>

      <Card title="Where these numbers come from">
        <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-xs text-slate-300 md:grid-cols-2">
          <div><dt className="text-slate-500">Speed run</dt><dd>{p.speed_run.folder} · commit {p.speed_run.commit.slice(0, 7)} · {p.speed_run.started_utc.slice(0, 16).replace("T", " ")} UTC</dd></div>
          <div><dt className="text-slate-500">Accuracy &amp; memory run</dt><dd>{p.quality_and_memory_run.folder} · commit {p.quality_and_memory_run.commit.slice(0, 7)}</dd></div>
          <div><dt className="text-slate-500">Hardware</dt><dd>{p.speed_run.gpu}</dd></div>
          <div><dt className="text-slate-500">Serving</dt><dd>vLLM {p.speed_run.vllm_version} · <code className="text-slate-400">{p.speed_run.vllm_args.join(" ")}</code></dd></div>
          <div><dt className="text-slate-500">Load</dt><dd>{p.speed_settings.requests} measured requests per level after {p.speed_settings.warmup} warm-up, closed loop, cache bypassed</dd></div>
          <div><dt className="text-slate-500">Cost basis</dt><dd>{p.speed_run.gpu_price_per_hour ? `$${p.speed_run.gpu_price_per_hour}/hour (Great Lakes billed rate for this job)` : "–"}</dd></div>
        </dl>
        <p className="mt-3 text-xs text-slate-400">
          Full tables, method and raw data: <a className="text-emerald-400 hover:underline" href={`${REPO_RESULTS}/RESULTS.md`}>RESULTS.md</a>.
        </p>
      </Card>
    </div>
  );
}
