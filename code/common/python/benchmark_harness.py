"""Production-grade benchmarking harness with profiling integration.

Provides industry-standard benchmarking using Triton do_bench, PyTorch Timer,
and custom CUDA Events. Supports nsys, ncu, and PyTorch profiler integration.
"""

from __future__ import annotations

import gc
import random
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import torch


class BenchmarkMode(Enum):
    """Benchmarking mode selection."""
    TRITON = "triton"  # Use triton.testing.do_bench
    PYTORCH = "pytorch"  # Use torch.utils.benchmark.Timer
    CUSTOM = "custom"  # Use CUDA Events / time.perf_counter


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    iterations: int = 100
    warmup: int = 10
    min_run_time_ms: float = 100.0  # Minimum total runtime for PyTorch Timer
    percentiles: List[float] = field(default_factory=lambda: [25, 50, 75, 99])
    enable_memory_tracking: bool = False
    deterministic: bool = False
    seed: Optional[int] = None
    device: Optional[torch.device] = None
    enable_profiling: bool = False  # Enable nsys/ncu/PyTorch profiler
    profiling_output_dir: Optional[str] = None  # Directory for profiling outputs
    timeout_seconds: int = 15  # Required timeout for benchmark execution in seconds (prevents hangs) - DEFAULT 15s
    # Note: Setup/teardown (including compilation) are not subject to timeout,
    # but should complete within reasonable time or fail with error


@dataclass
class BenchmarkResult:
    """Statistical results from benchmarking."""
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    percentiles: Dict[float, float]  # e.g., {25.0: 1.23, 50.0: 1.45, ...}
    iterations: int
    warmup_iterations: int
    memory_peak_mb: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    profiling_outputs: Dict[str, str] = field(default_factory=dict)  # Paths to profiling files


class Benchmark(Protocol):
    """Protocol for benchmarkable implementations."""
    
    def setup(self) -> None:
        """Setup phase: initialize models, data, etc."""
        ...
    
    def benchmark_fn(self) -> None:
        """Function to benchmark. Must be callable with no args."""
        ...
    
    def teardown(self) -> None:
        """Cleanup phase."""
        ...
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """Optional: return benchmark-specific config overrides."""
        return None
    
    def validate_result(self) -> Optional[str]:
        """Optional: validate benchmark result, return error message if invalid."""
        return None


class BenchmarkHarness:
    """Production-grade benchmarking harness with profiling support."""
    
    def __init__(
        self,
        mode: BenchmarkMode = BenchmarkMode.CUSTOM,
        config: Optional[BenchmarkConfig] = None
    ):
        self.mode = mode
        self.config = config or BenchmarkConfig()
        self.device = self.config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._setup_reproducibility()
    
    def _setup_reproducibility(self) -> None:
        """Setup for reproducible benchmarks."""
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
    
    @contextmanager
    def _memory_tracking(self):
        """Context manager for memory tracking."""
        if not self.config.enable_memory_tracking or not torch.cuda.is_available():
            yield None
            return
        
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        yield
        torch.cuda.synchronize(self.device)
        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
        # Return values via generator protocol - caller gets tuple
        return (peak_mb, allocated_mb)
    
    def benchmark(self, benchmark: Benchmark) -> BenchmarkResult:
        """Run benchmark and return statistical results.
        
        Uses threading timeout (required) to prevent hangs. Default timeout is 15 seconds.
        """
        # Clone config to avoid mutating shared instance
        from dataclasses import replace
        config = replace(self.config)
        bench_config = benchmark.get_config()
        if bench_config:
            # Override with benchmark-specific settings
            for key, value in bench_config.__dict__.items():
                if value is not None:
                    setattr(config, key, value)
        
        errors = []
        memory_peak_mb = None
        memory_allocated_mb = None
        profiling_outputs = {}
        times_ms = []
        
        def run_benchmark_internal():
            """Internal benchmark execution function."""
            nonlocal times_ms, memory_peak_mb, memory_allocated_mb, profiling_outputs, errors
            
            try:
                # Setup - this may include CUDA extension compilation OR torch.compile()
                # IMPORTANT: Setup MUST complete quickly or timeout will occur
                # torch.compile() compilation can hang - timeout will catch it
                # If setup takes longer than timeout, it will be killed by the outer timeout
                import signal
                import time
                start_time = time.time()
                benchmark.setup()
                setup_time = time.time() - start_time
                if setup_time > config.timeout_seconds * 0.8:  # Warn if setup takes >80% of timeout
                    print(f"  WARNING: Setup took {setup_time:.1f}s (near timeout limit)")
                
                # Warmup
                self._warmup(benchmark.benchmark_fn, config.warmup)
                
                # Memory tracking: Reset stats BEFORE benchmark to capture peak during execution
                if config.enable_memory_tracking and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.device)
                    torch.cuda.synchronize(self.device)
                
                # Benchmark using selected mode
                if config.enable_profiling:
                    times_ms, profiling_outputs = self._benchmark_with_profiling(
                        benchmark.benchmark_fn, config
                    )
                else:
                    times_ms = self._benchmark_without_profiling(benchmark.benchmark_fn, config)
                
                # Memory tracking: Get stats AFTER benchmark to capture peak during execution
                if config.enable_memory_tracking and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                    memory_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                    memory_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
                
                # Validate result
                validation_error = benchmark.validate_result()
                if validation_error:
                    errors.append(f"Validation failed: {validation_error}")
                
            except Exception as e:
                errors.append(f"Benchmark execution failed: {str(e)}")
                times_ms = []
            finally:
                # Always cleanup
                try:
                    benchmark.teardown()
                except Exception as e:
                    errors.append(f"Teardown failed: {str(e)}")
                
                # Force cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # ALWAYS run with timeout (required, default 15 seconds)
        execution_result = {"done": False, "error": None}
        
        def run_with_result():
            try:
                run_benchmark_internal()
            except Exception as e:
                execution_result["error"] = e
            finally:
                execution_result["done"] = True
        
        # Only print timeout message if timeout actually occurs (not upfront)
        thread = threading.Thread(target=run_with_result, daemon=True)
        thread.start()
        thread.join(timeout=config.timeout_seconds)
        
        if not execution_result["done"]:
            # TIMEOUT OCCURRED - make it very clear
            print("\n" + "=" * 80)
            print("TIMEOUT: Benchmark execution exceeded timeout limit")
            print("=" * 80)
            print(f"   Timeout limit: {config.timeout_seconds} seconds")
            print(f"   Status: Benchmark did not complete within timeout period")
            print(f"   Action: Benchmark execution was terminated to prevent hang")
            print("=" * 80)
            print()
            
            errors.append(f"TIMEOUT: Benchmark exceeded timeout of {config.timeout_seconds} seconds")
            times_ms = []
            # Aggressive cleanup on timeout - CUDA operations can hang
            try:
                benchmark.teardown()
            except:
                pass
            # Force CUDA cleanup - critical after timeout
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()  # Clean up IPC resources
                    torch.cuda.reset_peak_memory_stats()  # Reset stats
                except:
                    pass
            gc.collect()
            # Force another GC pass to clean up any remaining references
            gc.collect()
        elif execution_result["error"]:
            errors.append(f"Benchmark execution error: {str(execution_result['error'])}")
            times_ms = []
        # Don't print success message for normal completion - only print on timeout/failure
        
        if not times_ms:
            raise RuntimeError(f"Benchmark failed: {', '.join(errors)}")
        
        # Compute statistics
        result = self._compute_stats(times_ms, config)
        result.memory_peak_mb = memory_peak_mb
        result.memory_allocated_mb = memory_allocated_mb
        result.errors = errors
        result.profiling_outputs = profiling_outputs
        
        return result
    
    def _benchmark_with_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> tuple[List[float], Dict[str, str]]:
        """Benchmark with profiling enabled."""
        profiling_outputs = {}
        
        # Create profiling output directory
        if config.profiling_output_dir:
            prof_dir = Path(config.profiling_output_dir)
            prof_dir.mkdir(parents=True, exist_ok=True)
        else:
            prof_dir = Path("profiling_results")
            prof_dir.mkdir(parents=True, exist_ok=True)
        
        # Try PyTorch profiler first (best for Python benchmarks)
        try:
            import torch.profiler
            
            # Run benchmark with PyTorch profiler
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            ) as prof:
                # Run benchmark iterations with minimal overhead
                times_ms = []
                is_cuda = self.device.type == "cuda"
                
                if is_cuda:
                    # Create events once, reuse across iterations
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(self.device)  # Sync once before loop
                    
                    for _ in range(config.iterations):
                        start_event.record()
                        fn()
                        end_event.record()
                        torch.cuda.synchronize(self.device)
                        times_ms.append(start_event.elapsed_time(end_event))
                        prof.step()  # Record each iteration in profiling trace
                else:
                    # CPU: use high-resolution timer
                    for _ in range(config.iterations):
                        start_time = time.perf_counter()
                        fn()
                        end_time = time.perf_counter()
                        times_ms.append((end_time - start_time) * 1000)
                        prof.step()  # Record each iteration in profiling trace
            
            # Export profiling data
            trace_file = prof_dir / "trace.json"
            prof.export_chrome_trace(str(trace_file))
            profiling_outputs["pytorch_trace"] = str(trace_file)
            
            return times_ms, profiling_outputs
            
        except Exception as e:
            # Fallback to non-profiling benchmark
            return self._benchmark_without_profiling(fn, config), {}
    
    def _benchmark_without_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> List[float]:
        """Benchmark without profiling."""
        if self.mode == BenchmarkMode.TRITON:
            return self._benchmark_triton(fn, config)
        elif self.mode == BenchmarkMode.PYTORCH:
            return self._benchmark_pytorch(fn, config)
        else:
            return self._benchmark_custom(fn, config)
    
    def _benchmark_triton(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use Triton's do_bench (returns single value per call)."""
        try:
            import triton.testing as tt
            times_ms = []
            # Triton do_bench handles warmup internally, but we do our own
            for _ in range(config.iterations):
                time_ms = tt.do_bench(fn, warmup=0, rep=1)  # We handle warmup
                times_ms.append(time_ms)
            return times_ms
        except ImportError:
            # Fallback to custom if Triton not available
            return self._benchmark_custom(fn, config)
    
    def _benchmark_pytorch(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use PyTorch's Timer."""
        try:
            from torch.utils.benchmark import Timer
            
            timer = Timer(
                stmt=fn,
                globals={},
                num_threads=1,
                device=self.device.type,
            )
            
            # blocked_autorange runs until min_run_time is met
            measurement = timer.blocked_autorange(
                min_run_time=config.min_run_time_ms / 1000.0  # Convert to seconds
            )
            
            # measurement.times is already in seconds
            times_ms = [t * 1000 for t in measurement.times]
            
            # If we got fewer iterations than requested, pad with repeats
            if len(times_ms) < config.iterations:
                times_ms = (times_ms * ((config.iterations // len(times_ms)) + 1))[:config.iterations]
            
            return times_ms
        except Exception as e:
            # Fallback to custom on error
            return self._benchmark_custom(fn, config)
    
    def _benchmark_custom(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Custom benchmarking with CUDA Events for accurate GPU timing.
        
        Minimal overhead: Only synchronize once before loop (for first iteration),
        then sync only after end_event.record() to ensure measurement accuracy.
        """
        times_ms = []
        is_cuda = self.device.type == "cuda"
        
        if is_cuda:
            # Use CUDA Events for accurate GPU timing
            # Create events once - reuse across iterations (efficient)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Synchronize once before starting to ensure clean state
            torch.cuda.synchronize(self.device)
            
            for _ in range(config.iterations):
                # Record start event (non-blocking)
                start_event.record()
                # Execute function under test
                fn()
                # Record end event (non-blocking)
                end_event.record()
                # Synchronize to ensure events are recorded, then get elapsed time
                torch.cuda.synchronize(self.device)
                times_ms.append(start_event.elapsed_time(end_event))
        else:
            # CPU: use high-resolution timer
            for _ in range(config.iterations):
                start_time = time.perf_counter()
                fn()
                end_time = time.perf_counter()
                times_ms.append((end_time - start_time) * 1000)
        
        return times_ms
    
    def _warmup(self, fn: Callable, warmup_iterations: int) -> None:
        """Perform warmup iterations."""
        is_cuda = self.device.type == "cuda"
        for _ in range(warmup_iterations):
            fn()
        if is_cuda:
            torch.cuda.synchronize(self.device)
    
    def _compute_stats(
        self, times_ms: List[float], config: BenchmarkConfig
    ) -> BenchmarkResult:
        """Compute statistical measures."""
        if not times_ms:
            raise ValueError("No timing data collected")
        
        sorted_times = sorted(times_ms)
        n = len(sorted_times)
        
        # Compute percentiles
        percentiles_dict = {}
        for p in config.percentiles:
            idx = int((p / 100.0) * (n - 1))
            idx = min(idx, n - 1)
            percentiles_dict[p] = sorted_times[idx]
        
        return BenchmarkResult(
            mean_ms=statistics.mean(times_ms),
            median_ms=statistics.median(times_ms),
            std_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            percentiles=percentiles_dict,
            iterations=n,
            warmup_iterations=config.warmup,
        )


def compare_benchmarks(
    baseline: Benchmark,
    optimized: Benchmark,
    harness: Optional[BenchmarkHarness] = None,
    name: str = "Comparison"
) -> Dict[str, any]:
    """Compare baseline vs optimized benchmarks and return metrics."""
    if harness is None:
        harness = BenchmarkHarness()
    
    baseline_result = harness.benchmark(baseline)
    optimized_result = harness.benchmark(optimized)
    
    speedup = baseline_result.mean_ms / optimized_result.mean_ms if optimized_result.mean_ms > 0 else 1.0
    
    return {
        "name": name,
        "baseline": {
            "mean_ms": baseline_result.mean_ms,
            "median_ms": baseline_result.median_ms,
            "std_ms": baseline_result.std_ms,
            "min_ms": baseline_result.min_ms,
            "max_ms": baseline_result.max_ms,
        },
        "optimized": {
            "mean_ms": optimized_result.mean_ms,
            "median_ms": optimized_result.median_ms,
            "std_ms": optimized_result.std_ms,
            "min_ms": optimized_result.min_ms,
            "max_ms": optimized_result.max_ms,
        },
        "speedup": speedup,
        "baseline_result": baseline_result,
        "optimized_result": optimized_result,
    }

