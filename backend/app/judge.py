"""LLM-judge used to shadow-verify a sample of semantic-cache hits: asks
whether the query that's currently arriving and the query that produced a
cached response are actually asking the same thing. This is an
*independent* signal from the embedding similarity that produced the hit in
the first place, so it can catch false positives the embedding model
itself missed (e.g. "How do I cancel my subscription?" vs "How do I pause
my subscription instead of cancelling?" -- close in embedding space, very
different correct answers).

Falls back to a deterministic token-overlap heuristic whenever there's no
real model to ask (mock mode, or the configured provider unreachable) -- the same
real-model-with-deterministic-fallback shape already used by
app/embeddings.py, so the cache stays testable and demoable without a GPU.
"""

import re

from app.providers import run_completion

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common English function words. Raw token overlap is dominated by these on
# exactly the pairs that matter most -- e.g. "How do I cancel my
# subscription?" vs "How do I pause my subscription instead of
# cancelling?" share 5 of the shorter prompt's 6 tokens (how/do/i/my/
# subscription) despite asking for opposite actions. Stripping them first
# means overlap reflects the actual content words instead.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "be", "been", "being",
    "how", "what", "when", "where", "why", "who", "which",
    "do", "does", "did", "can", "could", "will", "would", "should", "shall",
    "i", "you", "your", "my", "me", "we", "us", "our",
    "to", "of", "for", "in", "on", "at", "by", "with", "and", "or", "but",
    "this", "that", "these", "those", "it", "its",
    "have", "has", "had", "get", "got",
}


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS}


def _heuristic_same_intent(prompt_a: str, prompt_b: str) -> bool:
    a, b = _content_tokens(prompt_a), _content_tokens(prompt_b)
    if not a or not b:
        # One side is nothing but stopwords -- an empty-set overlap would
        # be meaningless, so compare the raw (unfiltered) tokens instead.
        a, b = _tokens(prompt_a), _tokens(prompt_b)
    if not a or not b:
        return prompt_a.strip().lower() == prompt_b.strip().lower()
    overlap = len(a & b) / len(a | b)
    return overlap >= 0.5


_JUDGE_PROMPT = """You are auditing a cache for a question-answering system. \
Two questions are shown below. Answer with exactly one word, YES or NO: \
would both questions be correctly answered by the *same* answer?

Question A: {a}
Question B: {b}

Answer (YES or NO):"""


async def judge_same_intent(model: str, prompt_a: str, prompt_b: str) -> bool:
    """True if prompt_a and prompt_b are judged to have the same answer."""
    if prompt_a.strip().lower() == prompt_b.strip().lower():
        return True

    judge_messages = [{"role": "user", "content": _JUDGE_PROMPT.format(a=prompt_a, b=prompt_b)}]
    try:
        text, _, _, provider = await run_completion(model, judge_messages, temperature=0.0)
    except Exception:
        return _heuristic_same_intent(prompt_a, prompt_b)

    if provider == "mock":
        # No real model answered this -- the canned mock response isn't a
        # judgment, so fall back to the heuristic instead of misreading it.
        return _heuristic_same_intent(prompt_a, prompt_b)

    return text.strip().upper().startswith("YES")
