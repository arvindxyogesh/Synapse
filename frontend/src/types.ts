export interface StatsSummary {
  total_requests: number;
  cache_hit_rate: number;
  total_cost_usd: number;
  cost_saved_usd: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
}

export interface TimeseriesPoint {
  bucket: string;
  requests: number;
  cache_hits: number;
  cost_usd: number;
  avg_latency_ms: number;
}

export interface ProviderBreakdown {
  provider: string;
  requests: number;
  cost_usd: number;
}

export interface RequestLogEntry {
  id: string;
  provider: string;
  model: string;
  cached: boolean;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  latency_ms: number;
  status: string;
  created_at: string;
}

export interface ApiKey {
  id: string;
  name: string;
  key_prefix: string;
  revoked: boolean;
}
