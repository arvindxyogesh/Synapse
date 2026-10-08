"""Answer extraction is where accuracy numbers quietly go wrong, so it gets
the tricky cases spelled out."""

import pytest

from synapse_bench.gsm8k import (
    build_prompt,
    extract_flexible,
    extract_strict,
    normalize_number,
    reference_answer,
    score,
)
from synapse_bench.workloads import load_gsm8k


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("18", "18"),
        ("1,000", "1000"),
        ("$1,234.50", "1234.5"),
        ("18.00", "18"),
        ("-3.50", "-3.5"),
        (".5", "0.5"),
        ("42.", "42"),  # sentence-ending period
        (" 7 ", "7"),
        ("", None),
        ("eighteen", None),
        ("1.2.3", None),
    ],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_reference_answer_from_gsm8k_field():
    assert reference_answer("She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7") == "7"
    assert reference_answer("...\n#### 1,000") == "1000"
    with pytest.raises(ValueError):
        reference_answer("no marker here")


@pytest.mark.parametrize(
    "response, strict, flexible",
    [
        ("3 + 4 = 7\n#### 7", "7", "7"),
        ("The answer is $1,250.\n#### $1,250", "1250", "1250"),
        # No #### -> strict fails, flexible takes the last number.
        ("So she pays 18 dollars in total.", None, "18"),
        ("\\boxed{42}", None, "42"),
        # Only the LAST #### counts (models sometimes restate the format).
        ("Format: #### <number>. First try #### 5, corrected: #### 6", "6", "6"),
        # Text after the final answer doesn't change strict.
        ("#### 12\nHope this helps! Step 3 was hard.", "12", "3"),
        ("#### (no number)", None, None),
        ("-5 degrees\n#### -5", "-5", "-5"),
        ("", None, None),
    ],
)
def test_extraction(response, strict, flexible):
    assert extract_strict(response) == strict
    assert extract_flexible(response) == flexible


def test_score_marks_correctness_both_ways():
    result = score("Total is 18.\n#### 18.00", "...\n#### 18")
    assert result == {"gold": "18", "pred_strict": "18", "pred_flexible": "18",
                      "correct_strict": True, "correct_flexible": True}
    result = score("I think it's 18", "...\n#### 18")
    assert (result["correct_strict"], result["correct_flexible"]) == (False, True)


def test_every_reference_answer_in_the_dataset_parses():
    # Guards against a dataset row the scorer can't read, which would
    # otherwise surface as a silently wrong "incorrect".
    golds = [reference_answer(p["answer"]) for p in load_gsm8k()]
    assert len(golds) == 1319


def test_prompt_asks_for_the_scored_format():
    prompt = build_prompt("What is 2+2?")
    assert "What is 2+2?" in prompt and '"#### <number>"' in prompt
