#!/bin/bash
# Benchmark session on one Great Lakes A40 (University of Michigan HPC).
#
#   sbatch benchmarks/greatlakes/run_session.sbatch                               # smoke (default)
#   sbatch --time=08:00:00 --export=ALL,MODE=full benchmarks/greatlakes/run_session.sbatch
#
# Slurm gives this job one whole GPU (no other jobs on it), unlike a shared
# lab node. One-time setup (envs, models) is benchmarks/greatlakes/setup.sh.
#SBATCH --job-name=synapse-bench
#SBATCH --account=si650f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/si650f26_class_root/si650f26_class/savyo/synapse-bench/slurm-logs/%x-%j.out

set -euo pipefail
MODE="${MODE:-smoke}"
source /scratch/si650f26_class_root/si650f26_class/savyo/synapse-bench/env.sh
cd "$B/repo"

# What this job actually sees -- recorded so the GPU identity is never assumed.
echo "job=$SLURM_JOB_ID node=$(hostname) mode=$MODE CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
GPU_NAME="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# Great Lakes billed rate for THIS job shape (1 A40 + 8 cores + 64 GB), from
# the spgpu partition's TRESBillingWeights (per minute; 10,000,000 units = $1):
#   (18078.70 + 8 * 4519.68 + 64 * 376.64) * 60 / 1e7 = $0.470/hour.
# Used only to turn throughput into cost per 1K tokens; checked against
# Slurm's actual charge (sacct) after the smoke run.
PRICE_PER_HOUR=0.470

# Other jobs can share this node (not this GPU), so pick ports unlikely to clash.
PORT_OFFSET=$(( 20000 + SLURM_JOB_ID % 10000 ))

# No --cuda-device: Slurm already restricts the job to its GPU, and overriding
# CUDA_VISIBLE_DEVICES here could point at the wrong one.
"$B/envs/gateway/bin/python" benchmarks/gpu_session.py \
    --mode "$MODE" \
    --results-dir "$B/results" \
    --gateway-python "$B/envs/gateway/bin/python" \
    --gpu-label "$GPU_NAME (Great Lakes spgpu, job $SLURM_JOB_ID, whole GPU)" \
    --gpu-price-per-hour "$PRICE_PER_HOUR" \
    --gpu-memory-utilization 0.90 \
    --vllm-bin "$B/envs/vllm/bin/vllm" \
    --redis-bin "$B/envs/vllm/bin/redis-server" \
    --port-offset "$PORT_OFFSET"
