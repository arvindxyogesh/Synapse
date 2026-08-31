import type {
  ApiKey,
  CacheThresholdState,
  ProviderBreakdown,
  RequestLogEntry,
  StatsSummary,
  TimeseriesPoint,
} from "../types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init);
  if (!res.ok) {
    throw new Error(`${init?.method ?? "GET"} ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  baseUrl: BASE_URL,

  getSummary: (hours = 24) => request<StatsSummary>(`/v1/stats/summary?hours=${hours}`),
  getTimeseries: (hours = 24) => request<TimeseriesPoint[]>(`/v1/stats/timeseries?hours=${hours}`),
  getProviderBreakdown: (hours = 24) => request<ProviderBreakdown[]>(`/v1/stats/providers?hours=${hours}`),
  getRequests: (limit = 50, offset = 0) =>
    request<RequestLogEntry[]>(`/v1/stats/requests?limit=${limit}&offset=${offset}`),
  getCacheThresholdState: () => request<CacheThresholdState[]>("/v1/stats/cache-threshold"),

  listApiKeys: (adminKey: string) => request<ApiKey[]>("/v1/admin/keys", { headers: { "x-admin-key": adminKey } }),
  createApiKey: (
    adminKey: string,
    name: string,
    rateLimitPerMinute: number | null,
    monthlyQuotaUsd: number | null,
  ) =>
    request<{ id: string; name: string; api_key: string }>("/v1/admin/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-admin-key": adminKey },
      body: JSON.stringify({
        name,
        rate_limit_per_minute: rateLimitPerMinute,
        monthly_quota_usd: monthlyQuotaUsd,
      }),
    }),
  revokeApiKey: (adminKey: string, id: string) =>
    request<ApiKey>(`/v1/admin/keys/${id}/revoke`, { method: "POST", headers: { "x-admin-key": adminKey } }),
  updateApiKeyLimits: (
    adminKey: string,
    id: string,
    body: {
      rate_limit_per_minute?: number | null;
      monthly_quota_usd?: number | null;
      clear_rate_limit?: boolean;
      clear_monthly_quota?: boolean;
    },
  ) =>
    request<ApiKey>(`/v1/admin/keys/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", "x-admin-key": adminKey },
      body: JSON.stringify(body),
    }),
};
