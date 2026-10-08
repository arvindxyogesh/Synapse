"""GSM8K: prompt, answer extraction and exact-match scoring.

GSM8K reference answers end with "#### <number>". The model is asked to end
its reply the same way, then scored two ways:

- strict: the number after the model's last "####". Measures "solved it AND
  followed the requested format".
- flexible: the last number anywhere in the reply. Forgives a model that
  solved the problem but wrote "The answer is 18." or "\\boxed{18}".

Both are reported. The gap between them is itself informative: if
quantization hurt format-following more than math, strict drops but
flexible doesn't. Numbers are normalized before comparing ("1,000", "$1000"
and "1000.00" all equal 1000), since GSM8K answers are numeric and the
question is whether the value is right, not how it was typed.
"""

import re
from decimal import Decimal, InvalidOperation

PROMPT_TEMPLATE = (
    "Solve the following math problem. Think step by step, then give the final answer "
    'on its own line in the form "#### <number>".\n\n'
    "Problem: {question}"
)

# A number as people write them: optional minus, digits with optional
# thousands commas, optional decimals. Also matches ".5".
_NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?")


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def normalize_number(text: str) -> str | None:
    """Canonical string for a numeric answer, or None if it isn't one:
    "1,000" -> "1000", "$18.00" -> "18", "-3.50" -> "-3.5", "0.5" -> "0.5"."""
    cleaned = text.strip().replace(",", "").replace("$", "").rstrip(".").strip()
    if not cleaned:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text) if any(ch.isdigit() for ch in m.group(0))]


def reference_answer(answer_field: str) -> str:
    """The gold answer from a GSM8K `answer` field (text after the last ####)."""
    gold = normalize_number(answer_field.rsplit("####", 1)[-1])
    if gold is None:
        raise ValueError(f"GSM8K answer has no number after ####: {answer_field[-80:]!r}")
    return gold


def extract_strict(response: str) -> str | None:
    """First number after the model's last '####', or None."""
    if "####" not in response:
        return None
    numbers = _numbers(response.rsplit("####", 1)[-1])
    return normalize_number(numbers[0]) if numbers else None


def extract_flexible(response: str) -> str | None:
    """Last number anywhere in the response, or None."""
    numbers = _numbers(response)
    return normalize_number(numbers[-1]) if numbers else None


def score(response: str, answer_field: str) -> dict:
    gold = reference_answer(answer_field)
    strict, flexible = extract_strict(response), extract_flexible(response)
    return {
        "gold": gold,
        "pred_strict": strict,
        "pred_flexible": flexible,
        "correct_strict": strict == gold,
        "correct_flexible": flexible == gold,
    }
