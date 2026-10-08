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
