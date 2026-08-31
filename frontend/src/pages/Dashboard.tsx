import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../api/client";
import StatCard from "../components/StatCard";
import type { CacheThresholdState, ProviderBreakdown, StatsSummary, TimeseriesPoint } from "../types";

export default function Dashboard() {
  const [summary, setSummary] = useState<StatsSummary | null>(null);
  const [timeseries, setTimeseries] = useState<TimeseriesPoint[]>([]);
  const [providers, setProviders] = useState<ProviderBreakdown[]>([]);
  const [thresholds, setThresholds] = useState<CacheThresholdState[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [s, t, p, th] = await Promise.all([
          api.getSummary(),
          api.getTimeseries(),
          api.getProviderBreakdown(),
          api.getCacheThresholdState(),
        ]);
        if (!cancelled) {
          setSummary(s);
          setTimeseries(t);
          setProviders(p);
          setThresholds(th);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    }

    load();
    const interval = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
        Couldn't reach the gateway API ({error}). Is the backend running?
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Requests (24h)" value={summary ? String(summary.total_requests) : "…"} />
        <StatCard
          label="Cache hit rate"
          value={summary ? `${(summary.cache_hit_rate * 100).toFixed(0)}%` : "…"}
        />
        <StatCard
          label="Cost (24h)"
          value={summary ? `$${summary.total_cost_usd.toFixed(4)}` : "…"}
        />
        <StatCard
          label="Cost saved"
          value={summary ? `$${summary.cost_saved_usd.toFixed(4)}` : "…"}
          hint="via cache hits"
        />
        <StatCard label="Avg latency" value={summary ? `${summary.avg_latency_ms.toFixed(0)}ms` : "…"} />
        <StatCard label="p95 latency" value={summary ? `${summary.p95_latency_ms.toFixed(0)}ms` : "…"} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <h2 className="mb-4 text-sm font-medium text-slate-300">Requests over time</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={timeseries}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: "#94a3b8" }} hide />
              <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b" }} />
              <Line type="monotone" dataKey="requests" stroke="#34d399" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="cache_hits" stroke="#818cf8" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <h2 className="mb-4 text-sm font-medium text-slate-300">Requests by provider</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={providers}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="provider" tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b" }} />
              <Bar dataKey="requests" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {thresholds.length > 0 && (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <h2 className="mb-1 text-sm font-medium text-slate-300">Adaptive cache threshold</h2>
          <p className="mb-4 text-xs text-slate-500">
            Per-model similarity threshold, self-tuned from LLM-judge shadow verification of a sample of cache
            hits (see app/threshold_controller.py).
          </p>
          <table className="w-full text-left text-sm">
            <thead className="text-slate-400">
              <tr>
                <th className="py-1 font-medium">Model</th>
                <th className="py-1 font-medium">Threshold</th>
                <th className="py-1 font-medium">Est. false-positive rate</th>
                <th className="py-1 font-medium">Target</th>
                <th className="py-1 font-medium">Verified samples</th>
                <th className="py-1 font-medium">Last move</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {thresholds.map((t) => (
                <tr key={t.model} className="text-slate-200">
                  <td className="py-1.5">{t.model}</td>
                  <td className="py-1.5">{t.threshold.toFixed(3)}</td>
                  <td className="py-1.5">
                    <span className={t.estimated_false_positive_rate > t.target_false_positive_rate ? "text-amber-400" : "text-emerald-400"}>
                      {(t.estimated_false_positive_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-1.5 text-slate-400">{(t.target_false_positive_rate * 100).toFixed(0)}%</td>
                  <td className="py-1.5 text-slate-400">{t.verified_samples}</td>
                  <td className="py-1.5 text-slate-400">
                    {t.last_direction === "up" && <span className="text-amber-400">tightened ↑</span>}
                    {t.last_direction === "down" && <span className="text-emerald-400">loosened ↓</span>}
                    {!t.last_direction && "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
