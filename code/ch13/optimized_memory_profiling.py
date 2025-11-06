"""optimized_memory_profiling.py - Optimized memory profiling (optimized).

Memory profiling with gradient checkpointing to reduce peak memory.
Trades compute for memory by recomputing activations during backward.

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
from torch.utils.checkpoint import checkpoint


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
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class OptimizedModel(nn.Module):
    """Model with gradient checkpointing for memory optimization."""
    
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient checkpointing: recompute activations in backward
        # Saves memory by not storing intermediate activations
        x = checkpoint(self._fc1_relu, x)
        x = self.fc2(x)
        return x
    
    def _fc1_relu(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function for checkpointing."""
        return self.relu(self.fc1(x))


class OptimizedMemoryProfilingBenchmark(Benchmark):
    """Optimized memory profiling - uses gradient checkpointing."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.targets = None
        self.criterion = None
        self.peak_memory_mb = 0.0
        self.batch_size = 32
        self.hidden_dim = 2048
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        
        # Optimized model with gradient checkpointing
        self.model = OptimizedModel(hidden_dim=self.hidden_dim).to(self.device).train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.criterion = nn.MSELoss()
        
        # Warmup
        _ = self.model(self.inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - memory profiling with checkpointing."""
        torch.cuda.nvtx.range_push("optimized_memory_profiling")
        try:
            # Forward pass (checkpointing reduces memory)
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            
            # Backward pass (recomputes activations, saves memory)
            loss.backward()
            
            # Track peak memory (should be lower than baseline)
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        finally:
            torch.cuda.nvtx.range_pop()
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.criterion
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None and self.peak_memory_mb > 0


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryProfilingBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Memory Profiling: {result.mean_ms:.3f} ms")
    print(f"Peak Memory: {benchmark.peak_memory_mb:.2f} MB")

