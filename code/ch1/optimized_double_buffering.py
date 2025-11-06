"""optimized_double_buffering.py - Overlapped memory transfer and computation (optimized).

Demonstrates double buffering where memory transfer and computation overlap.
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


class OptimizedDoubleBufferingBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol.
    
    Demonstrates double buffering: overlap memory transfers with computation.
    Uses two buffers and async streams to hide transfer latency.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.host_buffer_a = None
        self.host_buffer_b = None
        self.device_buffer_a = None
        self.device_buffer_b = None
        self.result_buffer_a = None
        self.result_buffer_b = None
        self.N = 10_000_000  # 10M elements
        self.stream_transfer = None
        self.stream_compute = None
        self.current_buffer = 0
    
    def setup(self) -> None:
        """Setup: Initialize buffers and streams (EXCLUDED from timing)."""
        torch.manual_seed(42)
        # Create pinned host memory for efficient async transfers
        self.host_buffer_a = torch.randn(self.N, pin_memory=True)
        self.host_buffer_b = torch.randn(self.N, pin_memory=True)
        # Preallocate device buffers
        self.device_buffer_a = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.device_buffer_b = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.result_buffer_a = torch.empty(self.N, pin_memory=True)
        self.result_buffer_b = torch.empty(self.N, pin_memory=True)
        # Create separate streams for transfer and compute
        self.stream_transfer = torch.cuda.Stream()
        self.stream_compute = torch.cuda.Stream()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Double buffering with overlapped transfers and computation.
        
        This pattern achieves better performance by:
        1. Transferring buffer A while computing on buffer B (overlap)
        2. Transferring buffer B while computing on buffer A (overlap)
        Memory transfers and computation happen concurrently.
        """
        torch.cuda.nvtx.range_push("optimized_double_buffering_overlapped")
        try:
            # Double buffering pattern: overlap transfers with computation
            for iteration in range(2):  # Process two batches
                # Determine which buffers to use
                if iteration % 2 == 0:
                    host_in = self.host_buffer_a
                    device_in = self.device_buffer_a
                    device_out = self.device_buffer_b
                    host_out = self.result_buffer_b
                else:
                    host_in = self.host_buffer_b
                    device_in = self.device_buffer_b
                    device_out = self.device_buffer_a
                    host_out = self.result_buffer_a
                
                # Start async H2D transfer in transfer stream
                with torch.cuda.stream(self.stream_transfer):
                    torch.cuda.nvtx.range_push("H2D_transfer_async")
                    try:
                        device_in.copy_(host_in, non_blocking=True)
                    finally:
                        torch.cuda.nvtx.range_pop()
                
                # While transfer is happening, compute on previous buffer in compute stream
                if iteration > 0:
                    with torch.cuda.stream(self.stream_compute):
                        torch.cuda.nvtx.range_push("computation_overlapped")
                        try:
                            prev_device_out = self.device_buffer_b if iteration % 2 == 0 else self.device_buffer_a
                            prev_device_out = prev_device_out * 2.0 + 1.0
                        finally:
                            torch.cuda.nvtx.range_pop()
                
                # Synchronize transfer stream before using transferred data
                self.stream_transfer.synchronize()
                
                # Compute on newly transferred data
                with torch.cuda.stream(self.stream_compute):
                    torch.cuda.nvtx.range_push("computation")
                    try:
                        device_out = device_in * 2.0 + 1.0
                    finally:
                        torch.cuda.nvtx.range_pop()
                
                # Start async D2H transfer while next iteration can begin
                with torch.cuda.stream(self.stream_transfer):
                    torch.cuda.nvtx.range_push("D2H_transfer_async")
                    try:
                        host_out.copy_(device_out, non_blocking=True)
                    finally:
                        torch.cuda.nvtx.range_pop()
            
            # Final synchronization
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_buffer_a = None
        self.host_buffer_b = None
        self.device_buffer_a = None
        self.device_buffer_b = None
        self.result_buffer_a = None
        self.result_buffer_b = None
        if self.stream_transfer is not None:
            del self.stream_transfer
        if self.stream_compute is not None:
            del self.stream_compute
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.result_buffer_a is None:
            return "Result buffer A is None"
        if self.result_buffer_b is None:
            return "Result buffer B is None"
        if self.result_buffer_a.shape[0] != self.N:
            return f"Result buffer A shape mismatch: expected {self.N}, got {self.result_buffer_a.shape[0]}"
        if self.result_buffer_b.shape[0] != self.N:
            return f"Result buffer B shape mismatch: expected {self.N}, got {self.result_buffer_b.shape[0]}"
        # Check that result values are reasonable
        if not torch.isfinite(self.result_buffer_a).all():
            return "Result buffer A contains non-finite values"
        if not torch.isfinite(self.result_buffer_b).all():
            return "Result buffer B contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedDoubleBufferingBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Double Buffering Benchmark Results:")
    print(f"  Mean time: {result.mean_ms:.3f} ms")
    print(f"  Std dev: {result.std_ms:.3f} ms")
    print(f"  Min time: {result.min_ms:.3f} ms")
    print(f"  Max time: {result.max_ms:.3f} ms")

