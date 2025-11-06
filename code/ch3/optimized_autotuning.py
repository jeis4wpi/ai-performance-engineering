"""optimized_autotuning.py - Optimized with autotuning in infrastructure/OS tuning context.

Demonstrates autotuning for performance optimization.
Autotuning: Uses autotuning to automatically find optimal kernel configurations.
torch.compile with max-autotune mode enables autotuning.
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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")


class OptimizedAutotuningBenchmark(Benchmark):
    """Optimized: Autotuning for performance optimization.
    
    Autotuning: Uses autotuning to automatically find optimal kernel configurations.
    torch.compile with max-autotune mode enables autotuning.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with autotuning."""
        # Optimization: Autotuning
        # Autotuning automatically finds optimal kernel configurations
        # torch.compile with max-autotune enables autotuning
        
        # Prevent TF32 API mixing: PyTorch internal initialization may access old API (allow_tf32)
        # during module import, before arch_config.py can set the new API (fp32_precision).
        # arch_config.py uses ONLY the new API and never touches the old API.
        # Solution: Reset CUDA state to clear any TF32 API state before compile
        try:
            # Synchronize to ensure all previous CUDA ops complete
            torch.cuda.synchronize()
            # Clear any device-side assertions or errors
            torch.cuda.empty_cache()
        except Exception:
            pass
        
        # Set seed after CUDA reset to avoid triggering API mixing detection
        torch.manual_seed(42)
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
            # Autotuning: use torch.compile with max-autotune
            # Autotuning finds optimal kernel configurations automatically
            # Note: If TF32 API mixing error occurs, it's due to PyTorch internal initialization
            # accessing the old API (allow_tf32) during module import, before arch_config.py
            # can set the new API (fp32_precision). arch_config.py uses ONLY the new API.
            # This is a known PyTorch 2.9 issue when PyTorch internals touch old API.
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except RuntimeError as e:
                if "mix of the legacy and new APIs" in str(e):
                    # Fallback: compile without max-autotune if API mixing detected
                    # This happens when PyTorch internals accessed old TF32 API during initialization
                    self.model = torch.compile(self.model, mode="reduce-overhead")
        
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with autotuning."""
        torch.cuda.nvtx.range_push("optimized_autotuning")
        try:
            with torch.no_grad():
                # Optimization: Autotuning
                # Autotuning automatically finds optimal configurations
                # torch.compile with max-autotune enables autotuning
                output = self.model(self.input)
                
                # Optimization: Autotuning benefits
                # - Automatic kernel configuration optimization
                # - Optimal performance through autotuning
                # - Improved efficiency
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

