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
  rate_limit_per_minute: number | null;
  monthly_quota_usd: number | null;
  quota_spent_usd: number;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CacheThresholdState {
  model: string;
  threshold: number;
  estimated_false_positive_rate: number;
  verified_samples: number;
  target_false_positive_rate: number;
  last_direction: "up" | "down" | null;
  last_adjusted_at: number | null;
}
