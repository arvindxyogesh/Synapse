#!/usr/bin/env python3
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

results_dir = Path(sys.argv[1])
by_threshold = defaultdict(list)

for f in sorted(results_dir.glob("result_t*_s*.json")):
    m = re.match(r"result_t([\d.]+)_s(\d+)\.json", f.name)
    if not m:
        continue
    threshold = float(m.group(1))
    with open(f) as fh:
        by_threshold[threshold].append(json.load(fh))

def fmt(vals):
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.1%} +/- {std:.1%}"

print(f"{'threshold':>10} {'precision':>18} {'recall':>18} {'f1':>18} {'cost reduction':>18}  n_seeds")
print("-" * 100)
for t in sorted(by_threshold):
    runs = by_threshold[t]
    precision = [r["cache_precision"] for r in runs]
    recall = [r["cache_recall"] for r in runs]
    f1 = [r["cache_f1"] for r in runs]
    cost_reduction = [
        1 - r["cost_incurred_usd"] / r["cost_without_cache_usd"]
        for r in runs if r["cost_without_cache_usd"]
    ]
    print(f"{t:>10} {fmt(precision):>18} {fmt(recall):>18} {fmt(f1):>18} {fmt(cost_reduction):>18}  {len(runs)}")
