"""Standard template and utilities for chapter compare.py modules.

All chapters should use these functions to ensure consistency:
- discover_benchmarks() - Find baseline/optimized pairs
- load_benchmark() - Load Benchmark instances from files
- create_profile_template() - Standard profile() function structure

All compare.py modules must:
1. Import BenchmarkHarness, Benchmark, BenchmarkMode, BenchmarkConfig
2. Use discover_benchmarks() to find pairs
3. Use load_benchmark() to instantiate benchmarks
4. Run via harness.benchmark(benchmark_instance)
5. Return standardized format: {"metrics": {...}}
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkHarness,
    BenchmarkMode,
    BenchmarkConfig,
)


def discover_benchmarks(chapter_dir: Path) -> List[Tuple[Path, List[Path], str]]:
    """Discover benchmark modules by looking for baseline_*.py files with matching optimized_*.py.
    
    Args:
        chapter_dir: Path to chapter directory (e.g., Path('ch16'))
        
    Returns:
        List of tuples: (baseline_path, [optimized_paths], example_name)
        Example: (Path('ch16/baseline_moe_dense.py'), [Path('ch16/optimized_moe_sparse.py')], 'moe')
    """
    pairs = []
    baseline_files = list(chapter_dir.glob("baseline_*.py"))
    
    for baseline_file in baseline_files:
        # Extract example name: baseline_moe_dense.py -> moe
        example_name = baseline_file.stem.replace("baseline_", "").split("_")[0]
        optimized_files = []
        
        # Pattern 1: optimized_{name}_*.py (e.g., optimized_moe_sparse.py)
        pattern1 = chapter_dir / f"optimized_{example_name}_*.py"
        optimized_files.extend(pattern1.parent.glob(pattern1.name))
        
        # Pattern 2: optimized_{name}.py (e.g., optimized_moe.py)
        pattern2 = chapter_dir / f"optimized_{example_name}.py"
        if pattern2.exists():
            optimized_files.append(pattern2)
        
        if optimized_files:
            pairs.append((baseline_file, optimized_files, example_name))
    
    return pairs


def load_benchmark(module_path: Path, timeout_seconds: int = 15) -> Optional[Benchmark]:
    """Load benchmark from module by calling get_benchmark() function.
    
    Uses threading timeout to prevent hangs during module import or get_benchmark() calls.
    
    Args:
        module_path: Path to Python file with Benchmark implementation
        timeout_seconds: Maximum time to wait for module load (default: 15 seconds)
        
    Returns:
        Benchmark instance or None if loading fails or times out
    """
    result = {"benchmark": None, "error": None, "done": False}
    
    def load_internal():
        try:
            spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
            if spec is None or spec.loader is None:
                result["error"] = "Failed to create module spec"
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'get_benchmark'):
                result["benchmark"] = module.get_benchmark()
            else:
                result["error"] = "Module does not have get_benchmark() function"
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True
    
    # Run load in a thread with timeout to prevent hangs
    thread = threading.Thread(target=load_internal, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if not result["done"]:
        print(f"  Failed to load {module_path.name}: TIMEOUT (exceeded {timeout_seconds}s)")
        return None
    
    if result["error"]:
        print(f"  Failed to load {module_path.name}: {result['error']}")
        return None
    
    return result["benchmark"]


def create_standard_metrics(
    chapter: str,
    all_metrics: Dict[str, Any],
    default_tokens_per_s: float = 100.0,
    default_requests_per_s: float = 10.0,
    default_goodput: float = 0.85,
    default_latency_s: float = 0.001,
) -> Dict[str, Any]:
    """Create standardized metrics dictionary from collected results.
    
    Ensures all chapters return consistent metrics format.
    
    Args:
        chapter: Chapter identifier (e.g., 'ch1', 'ch16')
        all_metrics: Dictionary of collected metrics (will be modified in place)
        default_tokens_per_s: Default throughput if not calculated
        default_requests_per_s: Default request rate if not calculated
        default_goodput: Default efficiency metric if not calculated
        default_latency_s: Default latency if not calculated
        
    Returns:
        Standardized metrics dictionary
    """
    # Ensure chapter is set
    all_metrics['chapter'] = chapter
    
    # Calculate speedups from collected metrics
    speedups = [
        v for k, v in all_metrics.items() 
        if k.endswith('_speedup') and isinstance(v, (int, float)) and v > 0
    ]
    
    if speedups:
        all_metrics['speedup'] = max(speedups)
        all_metrics['average_speedup'] = sum(speedups) / len(speedups)
    else:
        # Default if no speedups found
        all_metrics['speedup'] = 1.0
        all_metrics['average_speedup'] = 1.0
    
    # Ensure required metrics exist (use defaults if not set)
    if 'tokens_per_s' not in all_metrics:
        all_metrics['tokens_per_s'] = default_tokens_per_s
    if 'requests_per_s' not in all_metrics:
        all_metrics['requests_per_s'] = default_requests_per_s
    if 'goodput' not in all_metrics:
        all_metrics['goodput'] = default_goodput
    if 'latency_s' not in all_metrics:
        all_metrics['latency_s'] = default_latency_s
    
    return all_metrics


def profile_template(
    chapter: str,
    chapter_dir: Path,
    harness_config: Optional[BenchmarkConfig] = None,
    custom_metrics_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Template profile() function for chapter compare.py modules.
    
    Standard implementation that all chapters should use or adapt.
    
    Args:
        chapter: Chapter identifier (e.g., 'ch1', 'ch16')
        chapter_dir: Path to chapter directory
        harness_config: Optional BenchmarkConfig override (default: iterations=20, warmup=5)
        custom_metrics_callback: Optional function to add custom metrics: f(all_metrics) -> None
        
    Returns:
        Standardized format: {"metrics": {...}}
    """
    print("=" * 70)
    print(f"Chapter {chapter.upper()}: Comparing Implementations")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\nCUDA not available - skipping")
        return {
            "metrics": {
                'chapter': chapter,
                'cuda_unavailable': True,
                'speedup': 1.0,
                'latency_s': 0.0,
                'tokens_per_s': 0.0,
                'requests_per_s': 0.0,
                'goodput': 0.0,
            }
        }
    
    pairs = discover_benchmarks(chapter_dir)
    
    if not pairs:
        print("\nNo baseline/optimized pairs found")
        print("\nTip: Create baseline_*.py and optimized_*.py files")
        print("    Each file must implement Benchmark protocol with get_benchmark() function")
        return {
            "metrics": {
                'chapter': chapter,
                'no_pairs_found': True,
                'speedup': 1.0,
                'latency_s': 0.0,
                'tokens_per_s': 0.0,
                'requests_per_s': 0.0,
                'goodput': 0.0,
            }
        }
    
    print(f"\nFound {len(pairs)} example(s) with optimization(s):\n")
    
    # Create harness with default or custom config
    # Enable memory tracking by default to capture all metrics
    from dataclasses import replace
    if harness_config is None:
        config = BenchmarkConfig(iterations=20, warmup=5, enable_memory_tracking=True)
    else:
        # Enable memory tracking by default unless explicitly disabled
        if harness_config.enable_memory_tracking is False:
            # Respect explicit False
            config = harness_config
        else:
            # Default to True for comprehensive metrics
            config = replace(harness_config, enable_memory_tracking=True)
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    
    all_metrics = {
        'chapter': chapter,
    }
    
    # Collect all results for summary
    summary_data = []
    
    for baseline_path, optimized_paths, example_name in pairs:
        print(f"\n  Example: {example_name}")
        print(f"    Baseline: {baseline_path.name}")
        
        baseline_benchmark = load_benchmark(baseline_path)
        if baseline_benchmark is None:
            print(f"    ❌ Baseline failed to load (missing get_benchmark() function?)")
            continue
        
        try:
            baseline_result = harness.benchmark(baseline_benchmark)
            baseline_time = baseline_result.mean_ms
            
            # Display comprehensive baseline metrics
            print(f"    Baseline: {baseline_time:.2f} ms")
            print(f"      📊 Timing Stats: median={baseline_result.median_ms:.2f}ms, "
                  f"min={baseline_result.min_ms:.2f}ms, max={baseline_result.max_ms:.2f}ms, "
                  f"std={baseline_result.std_ms:.2f}ms")
            if baseline_result.memory_peak_mb is not None:
                mem_str = f"      💾 Memory: peak={baseline_result.memory_peak_mb:.2f}MB"
                if baseline_result.memory_allocated_mb is not None:
                    mem_str += f", allocated={baseline_result.memory_allocated_mb:.2f}MB"
                print(mem_str)
            if baseline_result.percentiles:
                p99 = baseline_result.percentiles.get(99.0)
                if p99:
                    print(f"      📈 Percentiles: p99={p99:.2f}ms, p75={baseline_result.percentiles.get(75.0, 0):.2f}ms, "
                          f"p50={baseline_result.percentiles.get(50.0, 0):.2f}ms")
        except Exception as e:
            error_msg = str(e)
            # Check for skip warnings
            if "SKIPPED" in error_msg or "SKIP" in error_msg.upper() or "WARNING: SKIPPED" in error_msg:
                print(f"    ⚠️  {error_msg}")
            else:
                print(f"    ❌ Baseline failed to run: {error_msg}")
            continue
        
        best_speedup = 1.0
        best_optimized = None
        optimized_results = []
        
        for optimized_path in optimized_paths:
            opt_name = optimized_path.name
            # Extract technique name: optimized_moe_sparse.py -> sparse
            technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.py', '')
            if technique == opt_name.replace('optimized_', '').replace('.py', ''):
                technique = 'default'
            
            print(f"    Testing: {opt_name}...", end=' ', flush=True)
            
            optimized_benchmark = load_benchmark(optimized_path)
            if optimized_benchmark is None:
                print(f"❌ failed to load")
                continue
            
            try:
                optimized_result = harness.benchmark(optimized_benchmark)
                optimized_time = optimized_result.mean_ms
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                
                # Format output matching user's example: "0.06 ms (4.97x)"
                if speedup >= 1.0:
                    speedup_str = f"{optimized_time:.2f} ms ({speedup:.2f}x) 🚀"
                else:
                    speedup_str = f"{optimized_time:.2f} ms ({speedup:.2f}x) ⚠️"
                
                print(speedup_str)
                
                # Display comprehensive optimized metrics
                print(f"        📊 Timing: median={optimized_result.median_ms:.2f}ms, "
                      f"min={optimized_result.min_ms:.2f}ms, max={optimized_result.max_ms:.2f}ms, "
                      f"std={optimized_result.std_ms:.2f}ms")
                
                # Memory comparison
                if optimized_result.memory_peak_mb is not None:
                    mem_change = ""
                    if baseline_result.memory_peak_mb is not None:
                        mem_diff = optimized_result.memory_peak_mb - baseline_result.memory_peak_mb
                        mem_change_pct = (mem_diff / baseline_result.memory_peak_mb * 100) if baseline_result.memory_peak_mb > 0 else 0
                        if mem_diff > 0:
                            mem_change = f" (+{mem_diff:.2f}MB, +{mem_change_pct:.1f}%)"
                        elif mem_diff < 0:
                            mem_change = f" ({mem_diff:.2f}MB, {mem_change_pct:.1f}%)"
                        else:
                            mem_change = " (no change)"
                    
                    print(f"        💾 Memory: peak={optimized_result.memory_peak_mb:.2f}MB{mem_change}")
                    if optimized_result.memory_allocated_mb is not None:
                        print(f"                 allocated={optimized_result.memory_allocated_mb:.2f}MB")
                
                # Percentile comparison
                if optimized_result.percentiles:
                    p99_opt = optimized_result.percentiles.get(99.0)
                    if p99_opt and baseline_result.percentiles.get(99.0):
                        p99_base = baseline_result.percentiles.get(99.0)
                        p99_speedup = p99_base / p99_opt if p99_opt > 0 else 1.0
                        print(f"        📈 Percentiles: p99={p99_opt:.2f}ms ({p99_speedup:.2f}x), "
                              f"p75={optimized_result.percentiles.get(75.0, 0):.2f}ms, "
                              f"p50={optimized_result.percentiles.get(50.0, 0):.2f}ms")
                
                # Visual bar chart for speedup (inline)
                bar_width = 40
                if speedup >= 1.0:
                    filled = int(min(bar_width, (speedup - 1.0) / 4.0 * bar_width))
                    bar = "█" * filled + "░" * (bar_width - filled)
                else:
                    filled = int(min(bar_width, (1.0 - speedup) / 0.5 * bar_width))
                    bar = "░" * (bar_width - filled) + "█" * filled
                
                print(f"        [{bar}] {speedup:.2f}x speedup")
                
                optimized_results.append({
                    'name': opt_name,
                    'time': optimized_time,
                    'speedup': speedup,
                    'technique': technique
                })
                
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_optimized = optimized_path
                
                # Store per-technique metrics
                all_metrics[f"{example_name}_{technique}_baseline_time"] = baseline_time
                all_metrics[f"{example_name}_{technique}_optimized_time"] = optimized_time
                all_metrics[f"{example_name}_{technique}_speedup"] = speedup
                
            except Exception as e:
                error_msg = str(e)
                # Check for skip warnings
                if "SKIPPED" in error_msg or "SKIP" in error_msg.upper():
                    print(f"⚠️  {error_msg}")
                else:
                    print(f"❌ failed: {error_msg}")
                continue
        
        # Summary for this example
        if optimized_results:
            summary_data.append({
                'example': example_name,
                'baseline': baseline_time,
                'best_speedup': best_speedup,
                'best_name': best_optimized.name if best_optimized else None,
                'num_optimizations': len(optimized_results)
            })
            all_metrics[f"{example_name}_best_speedup"] = best_speedup
        
        if not optimized_results:
            summary_data.append({
                'example': example_name,
                'baseline': baseline_time,
                'best_speedup': 1.0,
                'best_name': None,
                'num_optimizations': 0
            })
    
    # Apply custom metrics callback if provided
    if custom_metrics_callback:
        custom_metrics_callback(all_metrics)
    
    # Standardize metrics format
    all_metrics = create_standard_metrics(chapter, all_metrics)
    
    # Print snazzy summary
    if summary_data:
        print("\n" + "=" * 80)
        print("📊 SUMMARY - Performance Improvements")
        print("=" * 80)
        
        # Sort by speedup (best first)
        summary_data.sort(key=lambda x: x['best_speedup'], reverse=True)
        
        for idx, item in enumerate(summary_data, 1):
            example_name = item['example']
            baseline = item['baseline']
            speedup = item['best_speedup']
            best_name = item['best_name']
            num_opts = item['num_optimizations']
            
            # Status emoji
            if speedup >= 2.0:
                status = "🔥"
            elif speedup >= 1.5:
                status = "✨"
            elif speedup >= 1.2:
                status = "👍"
            elif speedup >= 1.0:
                status = "✅"
            else:
                status = "⚠️"
            
            print(f"\n  {idx}. {example_name} {status}")
            print(f"     Baseline: {baseline:.2f} ms")
            
            if best_name and speedup > 1.0:
                improvement_pct = (1 - 1/speedup) * 100
                print(f"     🏆 Best: {best_name}")
                print(f"     📈 Improvement: {speedup:.2f}x faster ({improvement_pct:.1f}% reduction)")
                
                # ASCII bar showing improvement
                bar_width = 50
                filled = int(min(bar_width, (speedup - 1.0) / 5.0 * bar_width))
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"     {bar}")
            elif num_opts == 0:
                print(f"     ⚠️  No successful optimizations")
            else:
                print(f"     ⚠️  Optimization did not improve performance")
            
            print(f"     📦 {num_opts} optimization(s) tested")
        
        # Overall stats
        successful = [s for s in summary_data if s['best_speedup'] > 1.0]
        avg_speedup = sum(s['best_speedup'] for s in successful) / len(successful) if successful else 0
        best_overall = max(summary_data, key=lambda x: x['best_speedup'])
        
        print("\n" + "-" * 80)
        print(f"📊 Overall Stats:")
        print(f"   • Examples tested: {len(summary_data)}")
        print(f"   • Successful optimizations: {len(successful)}")
        if successful:
            print(f"   • Average speedup: {avg_speedup:.2f}x")
            print(f"   • Best improvement: {best_overall['example']} ({best_overall['best_speedup']:.2f}x)")
        print("=" * 80)
    
    print()
    
    return {
        "metrics": all_metrics
    }


__all__ = [
    'discover_benchmarks',
    'load_benchmark',
    'create_standard_metrics',
    'profile_template',
]


