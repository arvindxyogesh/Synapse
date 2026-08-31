import { useState } from "react";

import { api } from "../api/client";
import type { ApiKey } from "../types";

export default function ApiKeys() {
  const [adminKey, setAdminKey] = useState("");
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [newKeyName, setNewKeyName] = useState("");
  const [rateLimit, setRateLimit] = useState("");
  const [monthlyQuota, setMonthlyQuota] = useState("");
  const [justCreated, setJustCreated] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh(key: string) {
    try {
      setKeys(await api.listApiKeys(key));
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleCreate() {
    if (!newKeyName.trim()) return;
    try {
      const created = await api.createApiKey(
        adminKey,
        newKeyName.trim(),
        rateLimit.trim() ? Number(rateLimit) : null,
        monthlyQuota.trim() ? Number(monthlyQuota) : null,
      );
      setJustCreated(created.api_key);
      setNewKeyName("");
      setRateLimit("");
      setMonthlyQuota("");
      await refresh(adminKey);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleRevoke(id: string) {
    try {
      await api.revokeApiKey(adminKey, id);
      await refresh(adminKey);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleClearLimit(id: string, field: "rate" | "quota") {
    try {
      await api.updateApiKeyLimits(adminKey, id, {
        clear_rate_limit: field === "rate",
        clear_monthly_quota: field === "quota",
      });
      await refresh(adminKey);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <label className="block text-sm text-slate-400">Admin key</label>
        <div className="mt-1 flex gap-2">
          <input
            type="password"
            value={adminKey}
            onChange={(e) => setAdminKey(e.target.value)}
            className="flex-1 rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
            placeholder="ADMIN_KEY from backend .env"
          />
          <button
            onClick={() => refresh(adminKey)}
            className="rounded bg-emerald-600 px-3 py-2 text-sm font-medium hover:bg-emerald-500"
          >
            Load keys
          </button>
        </div>
      </div>

      {error && <div className="text-sm text-red-300">{error}</div>}

      {justCreated && (
        <div className="rounded border border-emerald-900 bg-emerald-950/40 p-3 text-sm">
          New key (shown once): <code className="text-emerald-300">{justCreated}</code>
        </div>
      )}

      <div>
        <label className="block text-sm text-slate-400">Create a new gateway API key</label>
        <div className="mt-1 grid grid-cols-1 gap-2 sm:grid-cols-[2fr_1fr_1fr_auto]">
          <input
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
            placeholder="e.g. local-dev"
          />
          <input
            value={rateLimit}
            onChange={(e) => setRateLimit(e.target.value)}
            type="number"
            min={1}
            className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
            placeholder="req/min (optional)"
          />
          <input
            value={monthlyQuota}
            onChange={(e) => setMonthlyQuota(e.target.value)}
            type="number"
            min={0}
            step="any"
            className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
            placeholder="$/month (optional)"
          />
          <button
            onClick={handleCreate}
            className="rounded bg-emerald-600 px-3 py-2 text-sm font-medium hover:bg-emerald-500"
          >
            Create
          </button>
        </div>
      </div>

      <table className="w-full text-left text-sm">
        <thead className="text-slate-400">
          <tr>
            <th className="py-2 font-medium">Name</th>
            <th className="py-2 font-medium">Prefix</th>
            <th className="py-2 font-medium">Status</th>
            <th className="py-2 font-medium">Rate limit</th>
            <th className="py-2 font-medium">Quota (spent / limit)</th>
            <th className="py-2 font-medium" />
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {keys.map((key) => (
            <tr key={key.id}>
              <td className="py-2">{key.name}</td>
              <td className="py-2 text-slate-400">{key.key_prefix}…</td>
              <td className="py-2">{key.revoked ? "revoked" : "active"}</td>
              <td className="py-2 text-slate-300">
                {key.rate_limit_per_minute ? (
                  <span className="flex items-center gap-2">
                    {key.rate_limit_per_minute}/min
                    <button
                      onClick={() => handleClearLimit(key.id, "rate")}
                      className="text-xs text-slate-500 hover:text-slate-300"
                    >
                      clear
                    </button>
                  </span>
                ) : (
                  <span className="text-slate-600">unlimited</span>
                )}
              </td>
              <td className="py-2 text-slate-300">
                {key.monthly_quota_usd ? (
                  <span className="flex items-center gap-2">
                    ${key.quota_spent_usd.toFixed(4)} / ${key.monthly_quota_usd.toFixed(2)}
                    <button
                      onClick={() => handleClearLimit(key.id, "quota")}
                      className="text-xs text-slate-500 hover:text-slate-300"
                    >
                      clear
                    </button>
                  </span>
                ) : (
                  <span className="text-slate-600">unlimited</span>
                )}
              </td>
              <td className="py-2 text-right">
                {!key.revoked && (
                  <button onClick={() => handleRevoke(key.id)} className="text-red-400 hover:text-red-300">
                    Revoke
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
