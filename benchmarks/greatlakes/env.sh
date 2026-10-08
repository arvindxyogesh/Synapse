# Source before anything Synapse-benchmark related on Great Lakes: keeps every
# download, cache and temp file under scratch (home is full and is not used).
# Note: Great Lakes purges /scratch data older than 60 days.
export B=/scratch/si650f26_class_root/si650f26_class/savyo/synapse-bench
export HF_HOME=$B/hf
export PIP_CACHE_DIR=$B/cache/pip
export XDG_CACHE_HOME=$B/cache/xdg
export VLLM_CACHE_ROOT=$B/cache/vllm
export TRITON_CACHE_DIR=$B/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$B/cache/torchinductor
export MAMBA_ROOT_PREFIX=$B/cache/mamba
export CONDA_PKGS_DIRS=$B/cache/mamba/pkgs
export TMPDIR=$B/tmp

# Libraries that default to $HOME and ignore the XDG variables above
# (found by checking home after the first runs):
export FLASHINFER_WORKSPACE_BASE=$B/cache/flashinfer-base   # FlashInfer kernel cache -> $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer
export HUMMING_TMP_DIR=$B/cache/humming/tmp                 # humming-kernels (a vLLM dependency)
export HUMMING_CACHE_DIR=$B/cache/humming/cache
export VLLM_CONFIG_ROOT=$B/cache/vllm-config                # vLLM writes usage_stats.json here
