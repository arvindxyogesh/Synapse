import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { RequestLogEntry } from "../types";

export default function Requests() {
  const [rows, setRows] = useState<RequestLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getRequests(50, 0)
      .then(setRows)
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) {
    return <div className="text-sm text-red-300">Couldn't load requests ({error}).</div>;
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="w-full text-left text-sm">
        <thead className="bg-slate-900 text-slate-400">
          <tr>
            <th className="px-4 py-2 font-medium">Time</th>
            <th className="px-4 py-2 font-medium">Provider</th>
            <th className="px-4 py-2 font-medium">Model</th>
            <th className="px-4 py-2 font-medium">Cached</th>
            <th className="px-4 py-2 font-medium">Tokens</th>
            <th className="px-4 py-2 font-medium">Cost</th>
            <th className="px-4 py-2 font-medium">Latency</th>
            <th className="px-4 py-2 font-medium">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {rows.map((row) => (
            <tr key={row.id} className="text-slate-200">
              <td className="px-4 py-2 text-slate-400">{new Date(row.created_at).toLocaleTimeString()}</td>
              <td className="px-4 py-2">{row.provider}</td>
              <td className="px-4 py-2">{row.model}</td>
              <td className="px-4 py-2">
                {row.cached ? <span className="text-emerald-400">hit</span> : <span className="text-slate-500">miss</span>}
              </td>
              <td className="px-4 py-2">{row.prompt_tokens + row.completion_tokens}</td>
              <td className="px-4 py-2">${row.cost_usd.toFixed(5)}</td>
              <td className="px-4 py-2">{row.latency_ms.toFixed(0)}ms</td>
              <td className="px-4 py-2">{row.status}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={8} className="px-4 py-6 text-center text-slate-500">
                No requests yet. Send one to /v1/chat/completions to see it here.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
