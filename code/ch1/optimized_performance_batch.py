"""optimized_performance_batch.py - Optimized performance benchmark with larger batch size.

Demonstrates how larger batch sizes improve GEMM efficiency.
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
)


def resolve_device() -> torch.device:
    """Return a usable device, falling back to CPU if CUDA is unavailable or unsupported."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:
        print(f"WARNING: CUDA unavailable or unsupported ({exc}); falling back to CPU.")
        return torch.device("cpu")


class OptimizedPerformanceBatchBenchmark(Benchmark):
    """Benchmark implementation with larger batch size optimization."""
    
    def __init__(self, batch_size: int = 256):
        self.device = resolve_device()
        self.batch_size = batch_size
        self.model = None
        self.data = None
        self.target = None
    
    def setup(self) -> None:
        """Setup: initialize model and data with larger batch."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        ).to(self.device)
        
        self.data = torch.randn(self.batch_size, 256, device=self.device)
        self.target = torch.randint(0, 10, (self.batch_size,), device=self.device)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        torch.cuda.nvtx.range_push("optimized_performance_batch")
        try:
            logits = self.model(self.data)
            loss = torch.nn.functional.cross_entropy(logits, self.target)
            loss.backward()
        finally:
            torch.cuda.nvtx.range_pop()
    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.data, self.target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.data is None:
            return "Data not initialized"
        if self.target is None:
            return "Target not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != self.batch_size:
                    return f"Output batch size mismatch: expected {self.batch_size}, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
                if self.target.shape[0] != self.batch_size:
                    return f"Target batch size mismatch: expected {self.batch_size}, got {self.target.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedPerformanceBatchBenchmark(batch_size=256)


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedPerformanceBatchBenchmark(batch_size=256)
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Performance with Larger Batch Size")
    print("=" * 70)
    print(f"Batch size: {benchmark.batch_size}")
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

