#!/usr/bin/env bash
set -uo pipefail
cd ~/Synapse/backend

THRESHOLDS=(0.85 0.90 0.92 0.95)
SEEDS=(1 2 3)
RESULTS_DIR=sweep_results
mkdir -p "$RESULTS_DIR"

for T in "${THRESHOLDS[@]}"; do
  echo "=== threshold=$T ==="
  for S in "${SEEDS[@]}"; do
    echo "  seed=$S"

    pkill -f run_fake_redis.py >/dev/null 2>&1
    sleep 1
    nohup python scripts/run_fake_redis.py --port 6379 > /dev/null 2>&1 &
    sleep 1

    pkill -f "uvicorn app.main:app" >/dev/null 2>&1
    sleep 1
    CACHE_SIMILARITY_THRESHOLD=$T nohup uvicorn app.main:app --port 8000 \
      > "$RESULTS_DIR/gateway_t${T}_s${S}.log" 2>&1 &
    sleep 2

    python scripts/benchmark.py --model llama3 --requests 300 --seed "$S" \
      --output "$RESULTS_DIR/result_t${T}_s${S}.json" \
      > "$RESULTS_DIR/bench_t${T}_s${S}.log" 2>&1
  done
done

pkill -f "uvicorn app.main:app" >/dev/null 2>&1
pkill -f run_fake_redis.py >/dev/null 2>&1

echo "Sweep done. Aggregating..."
python scripts/aggregate_sweep.py "$RESULTS_DIR"
