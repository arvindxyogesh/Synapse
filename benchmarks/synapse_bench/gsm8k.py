"""GSM8K: prompt, answer extraction and exact-match scoring.

GSM8K reference answers end with "#### <number>". The model is asked to put
its final answer in \\boxed{...} -- the format Qwen models are trained to use
for math and the one Qwen's own math evaluations ask for (see DECISIONS.md
D13 for why this replaced "####") -- then scored two ways:

- strict: the number inside the model's last \\boxed{...}. Measures "solved
  it AND followed the requested format".
- flexible: the last number anywhere in the reply. Forgives a model that
  solved the problem but wrote "The answer is 18." with no box.

Both are reported. The gap between them is itself informative: if
quantization hurt format-following more than math, strict drops but
flexible doesn't. Numbers are normalized before comparing ("1,000", "$1000"
and "1000.00" all equal 1000), since GSM8K answers are numeric and the
question is whether the value is right, not how it was typed.
"""

import re
from decimal import Decimal, InvalidOperation

PROMPT_TEMPLATE = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

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


# \frac{a}{b} (or \dfrac / \tfrac), with an optional minus sign in front of
# the fraction or inside the numerator.
_FRACTION = re.compile(r"(-?)\s*\\[dt]?frac\{(-?\d+)\}\{(\d+)\}")


def last_boxed(response: str) -> str | None:
    """Contents of the last \\boxed{...}, braces balanced (so
    \\boxed{\\frac{1}{2}} gives "\\frac{1}{2}"), or None."""
    start = response.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(response)):
        if response[j] == "{":
            depth += 1
        elif response[j] == "}":
            depth -= 1
            if depth == 0:
                return response[i:j]
    return None  # unclosed box, e.g. the reply was cut off


def extract_strict(response: str) -> str | None:
    """The number inside the last \\boxed{...}, or None. Handles \\frac{a}{b}
    and trailing units/LaTeX like "18 \\text{ meters}" (first number wins)."""
    content = last_boxed(response)
    if content is None:
        return None
    fraction = _FRACTION.search(content)
    if fraction:
        value = Decimal(fraction.group(2)) / Decimal(fraction.group(3))
        if fraction.group(1):
            value = -value
        return normalize_number(str(value))
    numbers = _numbers(content)
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
