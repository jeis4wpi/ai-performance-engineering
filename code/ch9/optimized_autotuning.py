"""optimized_autotuning.py - Optimized with autotuning in kernel efficiency/arithmetic intensity context.

Demonstrates autotuning for optimal kernel configurations.
Autotuning: Uses autotuning to automatically find optimal kernel parameters.
Improves performance through automatic kernel tuning.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available
from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedAutotuningBenchmark(Benchmark):
    """Optimized: Autotuning for optimal kernel configurations.
    
    Autotuning: Uses autotuning to automatically find optimal kernel parameters.
    Improves performance through automatic kernel tuning.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with autotuning."""
        torch.manual_seed(42)
        # Optimization: Autotuning - automatic kernel tuning
        # Autotuning finds optimal kernel configurations automatically
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Autotuning: Use torch.compile with max-autotune for autotuning
        # Autotuning automatically explores kernel configurations
        self.model = torch.compile(self.model, mode="max-autotune")
        
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with autotuning."""
        torch.cuda.nvtx.range_push("optimized_autotuning")
        try:
            with torch.no_grad():
                # Optimization: Autotuning
                # Uses autotuned kernel configurations
                # Autotuning: automatically finds optimal parameters
                output = self.model(self.input)
                
                # Optimization: Autotuning benefits
                # - Automatic kernel configuration optimization
                # - Optimal parameter selection
                # - Improved performance through autotuning
                # - Better kernel efficiency
                _ = output.sum()
        finally:
            torch.cuda.nvtx.range_pop()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedAutotuningBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedAutotuningBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Autotuning")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

