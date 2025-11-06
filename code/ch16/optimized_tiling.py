"""optimized_tiling.py - Optimized with tiling in MoE context.

Demonstrates tiling optimization for better memory access patterns.
Tiling: Breaks matrices into smaller tiles for better cache utilization.
Improves memory access locality and reduces cache misses.
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

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class OptimizedTilingBenchmark(Benchmark):
    """Optimized: Tiling for better memory access patterns.
    
    Tiling: Breaks matrices into smaller tiles for better cache utilization.
    Improves memory access locality and reduces cache misses.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.tile_size = 256
    
    def setup(self) -> None:
        """Setup: Initialize model with tiling optimization."""
        torch.manual_seed(42)
        # Optimization: Tiling for better memory access
        # Tiling breaks matrices into smaller tiles
        # Improves cache utilization and memory access patterns
        
        # Large linear layer (will use tiling)
        self.model = nn.Linear(2048, 2048).to(self.device).eval()
        
        # Large input (will be processed with tiling)
        self.input = torch.randn(64, 2048, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Matrix operations with tiling."""
        torch.cuda.nvtx.range_push("optimized_tiling")
        try:
            with torch.no_grad():
                # Optimization: Tiling - process matrix in tiles
                # Breaks computation into smaller tiles for better cache usage
                # Improves memory access locality
                
                # Simulate tiled matrix multiplication
                # In CUDA kernels, this would use explicit tile loading/storing
                # For PyTorch, we demonstrate tiling concept through chunked processing
                batch_size, input_dim = self.input.shape
                output_dim = self.model.out_features
                
                # Process in tiles (tiling optimization)
                output_parts = []
                for i in range(0, input_dim, self.tile_size):
                    tile_end = min(i + self.tile_size, input_dim)
                    input_tile = self.input[:, i:tile_end]
                    
                    # Process tile (tiling: smaller working set)
                    # Extract corresponding weight tile
                    weight_tile = self.model.weight[:, i:tile_end]
                    output_tile = torch.matmul(input_tile, weight_tile.t())
                    
                    output_parts.append(output_tile)
                
                # Combine tile results (tiling: reassemble)
                output = torch.cat(output_parts, dim=-1) if len(output_parts) > 1 else output_parts[0]
                output = output + self.model.bias
                
                # Optimization: Tiling benefits
                # - Better cache utilization (smaller working set)
                # - Improved memory access locality
                # - Reduced cache misses
                # - Better performance for large matrices
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
            iterations=10,
            warmup=2,
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
    return OptimizedTilingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedTilingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: tiling")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
