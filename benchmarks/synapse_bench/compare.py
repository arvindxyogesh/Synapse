"""Paired comparison of two GSM8K runs (e.g. 16-bit vs 4-bit) on the same
questions.

    python -m synapse_bench.compare results/.../qwen2.5-7b-bf16/gsm8k results/.../qwen2.5-7b-awq/gsm8k

Because both models answered the *same* questions, only the questions where
they disagree say anything about which is better. McNemar's test asks: if
the two were equally accurate, how surprising is this split of
disagreements? A large p-value means the accuracy gap is within what chance
alone would produce on this many questions -- "no detectable difference",
which is not the same as "no difference".
"""

import argparse
import json
import sys
from pathlib import Path

from synapse_bench.metrics import mcnemar_exact_p
from synapse_bench.workloads import read_jsonl


def compare(rows_a: list[dict], rows_b: list[dict], kind: str = "strict") -> dict:
    a = {r["id"]: r[f"correct_{kind}"] for r in rows_a}
    b = {r["id"]: r[f"correct_{kind}"] for r in rows_b}
    if set(a) != set(b):
        raise ValueError("the two runs didn't answer the same questions; a paired comparison needs identical sets")
    both = sum(1 for q in a if a[q] and b[q])
    only_a = sum(1 for q in a if a[q] and not b[q])
    only_b = sum(1 for q in a if b[q] and not a[q])
    n = len(a)
    return {
        "scoring": kind,
        "n": n,
        "accuracy_a": (both + only_a) / n,
        "accuracy_b": (both + only_b) / n,
        "difference_b_minus_a": (only_b - only_a) / n,
        "both_correct": both,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "neither_correct": n - both - only_a - only_b,
        "mcnemar_p": mcnemar_exact_p(only_a, only_b),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", type=Path, help="run_eval output directory (the baseline)")
    p.add_argument("run_b", type=Path, help="run_eval output directory (the variant)")
    args = p.parse_args(argv)
    rows_a, rows_b = read_jsonl(args.run_a / "answers.jsonl"), read_jsonl(args.run_b / "answers.jsonl")
    result = {kind: compare(rows_a, rows_b, kind) for kind in ("strict", "flexible")}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
