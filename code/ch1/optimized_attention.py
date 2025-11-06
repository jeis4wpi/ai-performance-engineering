"""optimized attention - Optimized. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class AttentionBenchmark(Benchmark):
    """Optimized implementation."""

    def __init__(self):
        self.device = resolve_device()

    def setup(self) -> None:
        """Setup: Initialize resources."""
        pass

    def benchmark_fn(self) -> None:
        """Benchmark: Run computation."""
        pass

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        pass

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return AttentionBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nResult: {result.mean_ms:.3f} ms")
