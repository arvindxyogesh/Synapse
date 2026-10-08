"""GPU memory: what to measure, and why `nvidia-smi` alone misleads.

vLLM reserves memory up front. At startup it loads the weights, then claims
(by default) 90% of the GPU's memory (`--gpu-memory-utilization 0.9`) and
fills everything left after the weights with KV cache -- the per-token
attention state of the requests being served. So `nvidia-smi` shows ~90%
used for *every* variant, before a single request arrives. A 4-bit model
doesn't show up as "less memory used"; it shows up as *more KV cache*,
which means more requests can run at once.

So two sources are recorded:
1. vLLM's own startup log (parse_vllm_log): weight memory, KV cache size,
   and the max concurrency vLLM computed. These are the numbers that
   actually differ between variants.
2. nvidia-smi peak during each run (GpuMemorySampler), for completeness --
   expected to be nearly flat, and reported with that explanation.

The log's wording has changed between vLLM versions; the patterns below
cover the phrasings known when this was written. Any field that isn't found
is returned as None -- never estimated.
"""

import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field


def _num(text: str) -> float:
    return float(text.replace(",", ""))


# (field, pattern, converter). The last match in the log wins.
_LOG_PATTERNS: list[tuple[str, re.Pattern, Callable[[re.Match], object]]] = [
    # "Model loading took 5.4337 GiB and 3.2 seconds" (newer)
    # "Loading model weights took 14.2487 GB" (older)
    ("weights_gib", re.compile(r"(?:Model loading|Loading model weights) took ([\d.]+) Gi?B"),
     lambda m: _num(m.group(1))),
    # "Available KV cache memory: 25.63 GiB"
    ("kv_cache_memory_gib", re.compile(r"Available KV cache memory: ([\d.]+) GiB"), lambda m: _num(m.group(1))),
    # "GPU KV cache size: 480,000 tokens"
    ("kv_cache_tokens", re.compile(r"GPU KV cache size: ([\d,]+) tokens"), lambda m: int(_num(m.group(1)))),
    # "# GPU blocks: 2048, # CPU blocks: 512" (older; one block = block_size tokens, default 16)
    ("gpu_blocks", re.compile(r"# GPU blocks: ([\d,]+)"), lambda m: int(_num(m.group(1)))),
    # "Maximum concurrency for 4,096 tokens per request: 117.19x"
    ("max_concurrency", re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"),
     lambda m: {"tokens_per_request": int(_num(m.group(1))), "requests": _num(m.group(2))}),
    # Which quantized-matmul kernel was chosen, e.g. "... Using awq_marlin kernel."
    ("quant_kernel_line", re.compile(r"[^\n]*\b(?:awq_marlin|gptq_marlin|marlin)\b[^\n]*kernel[^\n]*", re.IGNORECASE),
     lambda m: m.group(0).strip()),
    ("vllm_version", re.compile(r"vLLM API server version ([\w.+-]+)"), lambda m: m.group(1)),
]


def parse_vllm_log(text: str) -> dict:
    """Memory facts from a vLLM server log. Missing fields are None."""
    result: dict = {name: None for name, _, _ in _LOG_PATTERNS}
    for name, pattern, convert in _LOG_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            result[name] = convert(matches[-1])
    return result


# -- nvidia-smi sampling --------------------------------------------------------

NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]


def parse_memory_used(output: str) -> dict[int, int]:
    """{gpu_index: MiB used} from NVIDIA_SMI_QUERY output."""
    used = {}
    for line in output.strip().splitlines():
        index, mib = (part.strip() for part in line.split(","))
        used[int(index)] = int(mib)
    return used


def _query_nvidia_smi() -> str:
    return subprocess.run(NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=5, check=True).stdout


@dataclass
class GpuMemorySampler:
    """Context manager: polls GPU memory in a background thread and keeps the
    peak per GPU. On a machine without nvidia-smi it does nothing and
    `available` is False."""

    interval_s: float = 0.2
    query: Callable[[], str] = _query_nvidia_smi
    # Only record these GPU indices (nvidia-smi numbering). On a shared
    # machine, other users' GPUs must not end up in our "peak memory".
    gpu_indices: set[int] | None = None
    available: bool = field(init=False, default=False)
    peak_mib: dict[int, int] = field(init=False, default_factory=dict)
    samples: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.available = self.query is not _query_nvidia_smi or shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        for index, mib in parse_memory_used(self.query()).items():
            if self.gpu_indices is not None and index not in self.gpu_indices:
                continue
            self.peak_mib[index] = max(self.peak_mib.get(index, 0), mib)
        self.samples += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except (subprocess.SubprocessError, OSError, ValueError):
                pass  # one failed poll shouldn't end the run; `samples` shows how many succeeded
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuMemorySampler":
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=5)
            try:
                self._sample_once()  # one last reading at the end of the run
            except (subprocess.SubprocessError, OSError, ValueError):
                pass

    def result(self) -> dict | None:
        if not self.available:
            return None
        return {"peak_mib": dict(self.peak_mib), "samples": self.samples, "interval_s": self.interval_s}

