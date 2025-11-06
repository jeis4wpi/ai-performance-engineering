"""optimized_coalescing.py - Coalesced memory access pattern (optimized).

Demonstrates proper memory access patterns that enable memory coalescing.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
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
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class OptimizedCoalescingBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol.
    
    Demonstrates coalesced memory access by accessing consecutive elements.
    This enables memory coalescing into single 128-byte transactions.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 10_000_000  # 10M elements
    
    def setup(self) -> None:
        """Setup: Initialize tensors (EXCLUDED from timing)."""
        torch.manual_seed(42)
        # Create input tensor
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        # Preallocate output tensor
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Coalesced memory access pattern.
        
        Accesses consecutive memory locations, enabling coalescing.
        Warps can combine 32 consecutive accesses into single transaction.
        """
        torch.cuda.nvtx.range_push("optimized_coalescing_coalesced")
        try:
            # Coalesced access: threads access consecutive elements
            # This enables memory coalescing into single 128-byte transactions
            # All threads in a warp access consecutive memory, allowing GPU to
            # combine them into efficient memory transactions
            self.output = self.input * 2.0
        finally:
            torch.cuda.nvtx.range_pop()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor is None"
        if self.input is None:
            return "Input tensor is None"
        if self.output.shape[0] != self.N:
            return f"Output shape mismatch: expected {self.N}, got {self.output.shape[0]}"
        # Check that output values are reasonable (input * 2.0)
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCoalescingBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Coalescing Benchmark Results:")
    print(f"  Mean time: {result.mean_ms:.3f} ms")
    print(f"  Std dev: {result.std_ms:.3f} ms")
    print(f"  Min time: {result.min_ms:.3f} ms")
    print(f"  Max time: {result.max_ms:.3f} ms")

