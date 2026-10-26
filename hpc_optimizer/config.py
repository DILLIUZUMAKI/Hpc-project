from dataclasses import asdict, dataclass, field
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {
    "matmul": {"name": "Dense matrix multiplication", "default": 256, "min": 1, "max": 1024, "unit": "matrix dimension"},
    "jacobi": {"name": "Jacobi 2D stencil", "default": 256, "min": 3, "max": 1024, "unit": "grid dimension"},
    "monte_carlo": {"name": "Monte Carlo Pi", "default": 1000000, "min": 1, "max": 100000000, "unit": "samples"},
    "histogram": {"name": "Histogram (256 bins)", "default": 1000000, "min": 1, "max": 100000000, "unit": "items"},
    "bfs": {"name": "Breadth-first search", "default": 20000, "min": 1, "max": 200000, "unit": "vertices"},
}


def cpu_limit():
    """Respect affinity and Linux cgroup quotas; Docker capacity is checked separately."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n)


def hardware_info():
    result = {"logical_cpus": os.cpu_count(), "available_cpu_budget": cpu_limit()}
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        result["cpu_model"] = next(line.split(":",1)[1].strip() for line in lines if line.startswith("model name"))
    except (OSError, StopIteration):
        result["cpu_model"] = "unavailable"
    try:
        total = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:"))
        result["visible_memory_kib"] = int(total.split()[1])
    except (OSError, StopIteration, ValueError):
        result["visible_memory_kib"] = None
    return result


@dataclass
class ExperimentConfig:
    benchmark: str = "matmul"
    paradigm: str = "openmp"
    size: int = 256
    steps: int = 20
    seed: int = 42
    counts: list[int] = field(default_factory=lambda: [1, 2])
    repeats: int = 5
    warmups: int = 1
    iterations: int = 2
    timeout: int = 30
    memory_mb: int = 1024
    profiler: str = "basic"
    min_improvement: float = 0.02
    use_ai: bool = False
    allow_source_transfer: bool = False
    image: str = "hpc-optimizer-sandbox:1.0"

    def validate(self, available_cpus=None):
        if self.benchmark not in BENCHMARKS or self.paradigm not in {"openmp", "mpi"}:
            raise ValueError("Choose a listed benchmark and OpenMP or MPI.")
        b = BENCHMARKS[self.benchmark]
        bounds = [("size", b["min"], b["max"]), ("steps", 1, 1000), ("seed", 0, 2**32 - 1),
                  ("repeats", 3, 30), ("warmups", 0, 3), ("iterations", 0, 3),
                  ("timeout", 1, 120), ("memory_mb", 512, 8192)]
        for key, low, high in bounds:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer from {low} to {high}.")
        if not self.counts or any(type(p) is not int or p < 1 for p in self.counts):
            raise ValueError("Select positive integer thread/process counts.")
        self.counts = sorted(set(self.counts))
        limit = cpu_limit() if available_cpus is None else available_cpus
        if max(self.counts) > limit:
            raise ValueError(f"Requested {max(self.counts)} workers; available CPU budget is {limit}.")
        if self.profiler not in {"basic", "perf", "cachegrind"}:
            raise ValueError("Unknown profiler.")
        if not 0 <= self.min_improvement <= 0.5:
            raise ValueError("Minimum improvement must be between 0 and 50%.")
        if self.use_ai and not self.allow_source_transfer:
            raise ValueError("Confirm source-code transfer before requesting AI generation.")
        if not self.image or self.image.startswith("-") or any(c.isspace() for c in self.image):
            raise ValueError("Invalid sandbox image name.")
        return self

    def to_dict(self):
        return asdict(self)
