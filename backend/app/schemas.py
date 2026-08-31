from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="llama3", description="Open-weight model name, e.g. llama3, mistral")
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    provider: str
    cached: bool
    choices: list[ChatCompletionChoice]
    usage: Usage
    cost_usd: float
    latency_ms: float


class ApiKeyCreateRequest(BaseModel):
    name: str
    rate_limit_per_minute: int | None = Field(default=None, ge=1, description="Requests/minute; null = unlimited")
    monthly_quota_usd: float | None = Field(default=None, ge=0, description="Estimated $/month; null = unlimited")


class ApiKeyUpdateRequest(BaseModel):
    rate_limit_per_minute: int | None = Field(default=None, ge=1)
    monthly_quota_usd: float | None = Field(default=None, ge=0)
    clear_rate_limit: bool = False
    clear_monthly_quota: bool = False


class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str
    api_key: str  # returned once, at creation time only
    rate_limit_per_minute: int | None
    monthly_quota_usd: float | None


class ApiKeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    revoked: bool
    rate_limit_per_minute: int | None
    monthly_quota_usd: float | None
    quota_spent_usd: float = 0.0

    model_config = ConfigDict(from_attributes=True)


class StatsSummary(BaseModel):
    total_requests: int
    cache_hit_rate: float
    total_cost_usd: float
    cost_saved_usd: float
    avg_latency_ms: float
    p95_latency_ms: float


class TimeseriesPoint(BaseModel):
    bucket: str
    requests: int
    cache_hits: int
    cost_usd: float
    avg_latency_ms: float


class ProviderBreakdown(BaseModel):
    provider: str
    requests: int
    cost_usd: float


class RequestLogOut(BaseModel):
    id: str
    provider: str
    model: str
    cached: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    status: str
    created_at: str
