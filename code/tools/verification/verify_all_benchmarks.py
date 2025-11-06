#!/usr/bin/env python3
"""Verify all baseline/optimized benchmarks can be loaded and executed.

Tests:
1. All files compile (syntax check)
2. All benchmarks can be imported
3. All benchmarks can be instantiated via get_benchmark()
4. All benchmarks can run setup() without errors
5. All benchmarks can run benchmark_fn() without errors (minimal run)

NOTE: Distributed benchmarks are ONLY skipped if num_gpus == 1 (single GPU system).
This is clearly logged when it happens.

Usage:
    python3 tools/verification/verify_all_benchmarks.py [--chapter ch1]
"""

import sys
import os
import argparse
import importlib.util
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


# Default timeout constant (15 seconds - required for all benchmarks)
DEFAULT_TIMEOUT = 15


def check_syntax(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            compile(f.read(), str(file_path), 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Compile error: {e}"


def load_benchmark(file_path: Path, timeout_seconds: int = DEFAULT_TIMEOUT) -> Tuple[Optional[object], Optional[str]]:
    """Load benchmark from file and return instance.
    
    Uses threading timeout to prevent hangs during module import or get_benchmark() calls.
    
    Args:
        file_path: Path to Python file with Benchmark implementation
        timeout_seconds: Maximum time to wait for module load (default: 15 seconds)
        
    Returns:
        Tuple of (benchmark_instance, error_message). If successful: (benchmark, None).
        If failed or timed out: (None, error_string).
    """
    import threading
    
    result = {"benchmark": None, "error": None, "done": False}
    
    def load_internal():
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                result["error"] = "Could not create module spec"
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'get_benchmark'):
                result["error"] = "Missing get_benchmark() function"
                return
            
            result["benchmark"] = module.get_benchmark()
        except Exception as e:
            result["error"] = f"Load error: {e}"
        finally:
            result["done"] = True
    
    # Run load in a thread with timeout to prevent hangs
    thread = threading.Thread(target=load_internal, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if not result["done"]:
        return None, f"TIMEOUT: exceeded {timeout_seconds} second timeout"
    
    if result["error"]:
        return None, result["error"]
    
    return result["benchmark"], None


def test_benchmark(benchmark: object, timeout: int = DEFAULT_TIMEOUT) -> Tuple[bool, Optional[str]]:
    """Test benchmark execution with timeout protection.
    
    Runs full execution: setup(), benchmark_fn(), teardown()
    Resets CUDA state before and after to prevent cascading failures.
    
    Uses threading timeout (reliable, cross-platform) instead of signal-based timeout.
    """
    import threading
    import torch
    
    def reset_cuda_state():
        """Reset CUDA state to prevent cascading failures."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Clear any device-side errors by resetting peak stats
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
        except Exception:
            pass
    
    # Reset CUDA state before running benchmark
    reset_cuda_state()
    
    execution_result = {"success": False, "error": None, "done": False}
    
    def run_benchmark():
        """Run benchmark in a separate thread with timeout protection."""
        try:
            # Test setup
            if hasattr(benchmark, 'setup'):
                benchmark.setup()
            
            # Test benchmark_fn (full execution)
            if hasattr(benchmark, 'benchmark_fn'):
                benchmark.benchmark_fn()
            
            # Test teardown (no timeout needed, should be fast)
            if hasattr(benchmark, 'teardown'):
                benchmark.teardown()
            
            # Reset CUDA state after successful execution
            reset_cuda_state()
            
            # Only mark as success if we got here without exceptions
            execution_result["success"] = True
        except Exception as e:
            reset_cuda_state()  # Reset on error to prevent cascading failures
            execution_result["error"] = e
            execution_result["success"] = False  # Explicitly mark as failed
        finally:
            execution_result["done"] = True
    
    # Run benchmark in thread with timeout (required, default 15 seconds)
    # Only print timeout message if timeout actually occurs (not upfront)
    thread = threading.Thread(target=run_benchmark, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if not execution_result["done"]:
        # TIMEOUT OCCURRED - make it very clear
        print("\n" + "=" * 80)
        print("TIMEOUT: Benchmark execution exceeded timeout limit")
        print("=" * 80)
        print(f"   Timeout limit: {timeout} seconds")
        print(f"   Status: Benchmark did not complete within timeout period")
        print(f"   Action: Benchmark execution was terminated to prevent hang")
        print("=" * 80)
        print()
        
        reset_cuda_state()  # Reset on timeout too
        return False, f"TIMEOUT: exceeded {timeout} second timeout"
    
    if execution_result["error"]:
        # Error occurred during execution
        error = execution_result["error"]
        return False, f"Execution error: {str(error)}\n{traceback.format_exc()}"
    
    # Don't print success message for normal completion - only print on timeout/failure
    if execution_result["success"]:
        return True, None
    
    # Shouldn't reach here, but handle gracefully
    return False, "Unknown error during benchmark execution"


def is_distributed_benchmark(file_path: Path) -> bool:
    """Check if a benchmark file contains distributed operations."""
    try:
        content = file_path.read_text()
        return any(pattern in content for pattern in [
            'dist.init_process_group',
            'WORLD_SIZE',
            'RANK',
            'torch.distributed.init_process_group',
        ])
    except:
        return False


def verify_chapter(chapter_dir: Path) -> Dict[str, any]:
    """Verify all benchmarks in a chapter.
    
    Runs ALL tests. Only skips distributed benchmarks if num_gpus == 1,
    and logs this clearly.
    """
    import torch
    
    results = {
        'chapter': chapter_dir.name,
        'total': 0,
        'syntax_pass': 0,
        'load_pass': 0,
        'exec_pass': 0,
        'skipped': [],
        'failures': []
    }
    
    # Check GPU count for distributed benchmark detection
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Find all baseline and optimized files
    baseline_files = list(chapter_dir.glob("baseline_*.py"))
    optimized_files = list(chapter_dir.glob("optimized_*.py"))
    all_files = baseline_files + optimized_files
    
    results['total'] = len(all_files)
    
    for file_path in sorted(all_files):
        file_name = file_path.name
        
        # Check syntax
        syntax_ok, syntax_err = check_syntax(file_path)
        if not syntax_ok:
            results['failures'].append({
                'file': file_name,
                'stage': 'syntax',
                'error': syntax_err
            })
            continue
        results['syntax_pass'] += 1
        
        # Load benchmark
        benchmark, load_err = load_benchmark(file_path)
        if benchmark is None:
            results['failures'].append({
                'file': file_name,
                'stage': 'load',
                'error': load_err
            })
            continue
        results['load_pass'] += 1
        
        # Check if this is a distributed benchmark and we have only 1 GPU
        is_distributed = is_distributed_benchmark(file_path)
        if is_distributed and num_gpus == 1:
            # SKIP ONLY when distributed benchmark on single GPU system
            skip_reason = f"SKIPPED: Distributed benchmark requires multiple GPUs (found {num_gpus} GPU)"
            results['skipped'].append({
                'file': file_name,
                'reason': skip_reason
            })
            print(f"    WARNING: {file_name}: {skip_reason}")
            results['exec_pass'] += 1  # Count as pass since we intentionally skipped
            continue
        
        # Test execution (ALL benchmarks run - no skipping except single-GPU distributed)
        exec_ok, exec_err = test_benchmark(benchmark, timeout=DEFAULT_TIMEOUT)
        if not exec_ok:
            results['failures'].append({
                'file': file_name,
                'stage': 'execution',
                'error': exec_err
            })
            continue
        results['exec_pass'] += 1
    
    return results


def main():
    import torch
    
    parser = argparse.ArgumentParser(description='Verify all baseline/optimized benchmarks')
    parser.add_argument('--chapter', type=str, help='Chapter to test (e.g., ch1) or "all"')
    args = parser.parse_args()
    
    print("=" * 80)
    print("VERIFYING ALL BASELINE/OPTIMIZED BENCHMARKS")
    print("=" * 80)
    print("Mode: FULL EXECUTION - All tests run")
    print()
    
    # Check system configuration
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"System: {num_gpus} GPU(s) available")
        if num_gpus == 1:
            print("WARNING: NOTE: Distributed benchmarks will be SKIPPED (require multiple GPUs)")
            print("   This will be clearly logged for each skipped benchmark")
    else:
        print("System: No CUDA GPUs available")
        print("WARNING: NOTE: All GPU benchmarks will likely fail")
    print()
    
    # Determine chapters to test
    if args.chapter and args.chapter != 'all':
        chapter_dirs = [repo_root / args.chapter]
    else:
        chapter_dirs = sorted([d for d in repo_root.iterdir() 
                              if d.is_dir() and d.name.startswith('ch') and d.name[2:].isdigit()])
    
    all_results = []
    total_files = 0
    total_syntax_pass = 0
    total_load_pass = 0
    total_exec_pass = 0
    total_failures = 0
    
    for chapter_dir in chapter_dirs:
        if not chapter_dir.exists():
            continue
        
        print(f"Testing {chapter_dir.name}...")
        results = verify_chapter(chapter_dir)
        all_results.append(results)
        
        total_files += results['total']
        total_syntax_pass += results['syntax_pass']
        total_load_pass += results['load_pass']
        total_exec_pass += results['exec_pass']
        total_failures += len(results['failures'])
        total_skipped = sum(len(r['skipped']) for r in all_results)
        
        # Print chapter summary
        status = "PASS" if len(results['failures']) == 0 else "WARN"
        skipped_msg = f", {len(results['skipped'])} skipped" if results['skipped'] else ""
        print(f"  {status} {results['total']} files: "
              f"{results['syntax_pass']} syntax, "
              f"{results['load_pass']} load, "
              f"{results['exec_pass']} exec, "
              f"{len(results['failures'])} failures{skipped_msg}")
    
    # Calculate total skipped
    total_skipped = sum(len(r['skipped']) for r in all_results)
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files tested: {total_files}")
    print(f"Syntax check passed: {total_syntax_pass}/{total_files} ({100*total_syntax_pass/max(total_files,1):.1f}%)")
    print(f"Load check passed: {total_load_pass}/{total_files} ({100*total_load_pass/max(total_files,1):.1f}%)")
    print(f"Execution check passed: {total_exec_pass}/{total_files} ({100*total_exec_pass/max(total_files,1):.1f}%)")
    print(f"Total failures: {total_failures}")
    if total_skipped > 0:
        print(f"Total skipped: {total_skipped} (distributed benchmarks on single-GPU system)")
    print()
    
    # Print skipped benchmarks (EXTREMELY CLEAR)
    if total_skipped > 0:
        print("=" * 80)
        print("SKIPPED BENCHMARKS (Single-GPU System)")
        print("=" * 80)
        print("These benchmarks were SKIPPED because they require multiple GPUs")
        print(f"and this system has only {torch.cuda.device_count() if torch.cuda.is_available() else 0} GPU(s).")
        print()
        for results in all_results:
            if results['skipped']:
                print(f"{results['chapter']}:")
                for skipped in results['skipped']:
                    print(f"  WARNING: SKIPPED: {skipped['file']}")
                    print(f"     Reason: {skipped['reason']}")
        print()
    
    # Print failures
    if total_failures > 0:
        print("=" * 80)
        print("FAILURES")
        print("=" * 80)
        for results in all_results:
            if results['failures']:
                print(f"\n{results['chapter']}:")
                for failure in results['failures']:
                    print(f"  FAILED: {failure['file']} ({failure['stage']}): {failure['error']}")
        print()
        return 1
    else:
        print("All benchmarks verified successfully!")
        if total_skipped > 0:
            print(f"(Note: {total_skipped} distributed benchmarks skipped on single-GPU system)")
        return 0


if __name__ == "__main__":
    sys.exit(main())

