#!/bin/bash
# One-time setup on Great Lakes (login node: downloads and installs only, no GPU).
set -euo pipefail
source /scratch/si650f26_class_root/si650f26_class/savyo/synapse-bench/env.sh
cd $B
echo "== micromamba $(date)"
[ -x bin/micromamba ] || curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C . bin/micromamba
bin/micromamba --version
echo "== clone $(date)"
[ -d repo ] || git clone -q https://github.com/arvindxyogesh/Synapse.git repo
git -C repo fetch -q origin && git -C repo checkout -q m4-gpu-run && git -C repo pull -q && git -C repo rev-parse HEAD
echo "== vllm env $(date)"
[ -d envs/vllm ] || bin/micromamba create -q -y -p envs/vllm -c conda-forge python=3.12 redis-server pip
envs/vllm/bin/pip install -q vllm==0.31.0
echo "vllm installed: $(envs/vllm/bin/pip show vllm | grep ^Version)"
echo "== gateway env $(date)"
[ -d envs/gateway ] || bin/micromamba create -q -y -p envs/gateway -c conda-forge python=3.12 pip
envs/gateway/bin/pip install -q -r repo/backend/requirements-dev.txt
echo "== models (pinned revisions) $(date)"
envs/vllm/bin/python - <<PY
from huggingface_hub import snapshot_download
for repo, rev in [("Qwen/Qwen2.5-7B-Instruct-AWQ", "b25037543e9394b818fdfca67ab2a00ecc7dd641"),
                  ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
                  ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", "6711f2b6fc95545efe6018c469eca6832e28cefd")]:
    print(repo, snapshot_download(repo, revision=rev), flush=True)
PY
echo "== done $(date)"
