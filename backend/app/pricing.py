"""Cost accounting for locally-served, open-weight models.

There's no per-token API charge for local inference (Ollama or vLLM) -- the "cost" here
is an estimated compute cost so the dashboard has a meaningful $ metric to
show savings against (what an equivalent hosted API would have charged).
Numbers are illustrative reference points, not billing-accurate.
"""

# USD per 1K tokens, (prompt, completion), based on published open-weight
# hosted-inference reference pricing at comparable model sizes.
REFERENCE_RATES_PER_1K: dict[str, tuple[float, float]] = {
    "llama3": (0.0002, 0.0002),
    "llama3.1": (0.0002, 0.0002),
    "mistral": (0.00015, 0.00015),
    "mixtral": (0.0006, 0.0006),
    "phi3": (0.0001, 0.0001),
}
DEFAULT_RATE_PER_1K = (0.0002, 0.0002)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = REFERENCE_RATES_PER_1K.get(model, DEFAULT_RATE_PER_1K)
    return (prompt_tokens / 1000) * prompt_rate + (completion_tokens / 1000) * completion_rate
