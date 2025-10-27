"""
Triton 3.5 TMA (Tensor Memory Accelerator) for Blackwell GPUs

Demonstrates TMA descriptor support for bulk memory transfers on Blackwell.
Requires SM 10.0, CUDA 13+, and Triton 3.5+.
"""

import torch
import triton
import triton.language as tl
import triton.testing
from typing import Tuple


# ============================================================================
# TMA-Based Matrix Copy Kernel
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def tma_copy_2d_kernel(
    src_ptr,
    dst_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D matrix copy using TMA tensor descriptors via make_tensor_descriptor()."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    
    # Compute offsets for boundary checking
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    
    # Create tensor descriptors for TMA hardware
    src_desc = tl.make_tensor_descriptor(
        src_ptr,
        shape=[M, N],
        strides=[stride_m, stride_n],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    
    dst_desc = tl.make_tensor_descriptor(
        dst_ptr,
        shape=[M, N],
        strides=[stride_m, stride_n],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    
    if (m0 + BLOCK_M <= M) and (n0 + BLOCK_N <= N):
        data = src_desc.load([m0, n0])
    else:
        row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_N), dtype=offs_m.dtype)
        col_offsets = offs_n[None, :] + tl.zeros((BLOCK_M, BLOCK_N), dtype=offs_n.dtype)
        data = tl.load(
            src_desc,
            offsets=(row_offsets, col_offsets),
            boundary_check=(0, 1),
            padding_option="zero",
        )
    
    if (m0 + BLOCK_M <= M) and (n0 + BLOCK_N <= N):
        dst_desc.store([m0, n0], data)
    else:
        row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_N), dtype=offs_m.dtype)
        col_offsets = offs_n[None, :] + tl.zeros((BLOCK_M, BLOCK_N), dtype=offs_n.dtype)
        tl.store(
            dst_desc,
            data,
            offsets=(row_offsets, col_offsets),
            boundary_check=(0, 1),
        )


def tma_copy_2d(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy 2D tensors using TMA descriptors."""
    assert src.is_contiguous() and dst.is_contiguous(), "Tensors must be contiguous for TMA"
    assert src.shape == dst.shape, f"Shape mismatch: {src.shape} != {dst.shape}"
    
    M, N = src.shape
    
    # Use large block sizes for efficient 128-byte cache line usage
    MAX_BLOCK_M = 128  # From autotune configs
    MAX_BLOCK_N = 128
    
    grid = (triton.cdiv(M, MAX_BLOCK_M), triton.cdiv(N, MAX_BLOCK_N))
    tma_copy_2d_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
    )


# ============================================================================
# TMA-Optimized GEMM with Descriptor Load
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def tma_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Matrix multiplication using TMA tensor descriptors."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    
    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    
    C_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k0 in range(0, K, BLOCK_K):
        a = A_desc.load([m0, k0])
        b = B_desc.load([k0, n0])
        acc += tl.dot(a, b, out_dtype=tl.float32)
    

    C_desc.store([m0, n0], acc)


def tma_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using TMA tensor descriptors."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible dimensions: {K} != {K2}"
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    MAX_BLOCK_M = 128
    MAX_BLOCK_N = 256
    grid = (triton.cdiv(M, MAX_BLOCK_M), triton.cdiv(N, MAX_BLOCK_N))
    tma_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C


# ============================================================================
# Benchmarking and Validation
# ============================================================================

def benchmark_tma_vs_standard(
    sizes: list[int] = [1024, 2048, 4096, 8192],
    dtype: torch.dtype = torch.float16,
    num_iters: int = 100,
) -> dict:
    """Benchmark TMA operations against standard implementations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping benchmarks")
        return {}
    
    props = torch.cuda.get_device_properties(0)
    is_blackwell = props.major == 10 and props.minor == 0
    
    print("\n" + "="*70)
    print("TMA Performance Benchmark (Triton 3.5 + Blackwell)")
    print("="*70)
    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Blackwell Detected: {'YES' if is_blackwell else 'NO'}")
    print(f"Memory: {props.total_memory / 1e9:.2f} GB")
    print("="*70 + "\n")
    
    results = {}
    
    for size in sizes:
        print(f"\n{'='*70}")
        print(f"Matrix Size: {size}x{size}")
        print(f"{'='*70}")
        
        # Test 1: Matrix Copy
        print("\n[1/2] Testing Matrix Copy (TMA vs Standard)...")
        src = torch.randn(size, size, device=device, dtype=dtype)
        dst_tma = torch.empty_like(src)
        dst_std = torch.empty_like(src)
        
        tma_time = triton.testing.do_bench(lambda: tma_copy_2d(src, dst_tma)) / 1000.0
        std_time = triton.testing.do_bench(lambda: dst_std.copy_(src)) / 1000.0
        
        bytes_transferred = size * size * src.element_size() * 2
        tma_bw = bytes_transferred / tma_time / 1e12
        std_bw = bytes_transferred / std_time / 1e12
        speedup_copy = std_time / tma_time
        
        print(f"  TMA Copy:      {tma_time*1e6:.2f} µs ({tma_bw:.2f} TB/s)")
        print(f"  Standard Copy: {std_time*1e6:.2f} µs ({std_bw:.2f} TB/s)")
        print(f"  Speedup:       {speedup_copy:.2f}x")
        
        # Test 2: Matrix Multiplication
        print("\n[2/2] Testing GEMM (TMA vs Standard)...")
        A = torch.randn(size, size, device=device, dtype=dtype)
        B = torch.randn(size, size, device=device, dtype=dtype)
        
        tma_gemm_time = triton.testing.do_bench(lambda: tma_gemm(A, B)) / 1000.0
        torch_gemm_time = triton.testing.do_bench(lambda: torch.matmul(A.float(), B.float())) / 1000.0
        

        C_tma = tma_gemm(A, B)
        C_torch = torch.matmul(A.float(), B.float())
        
        flops = 2 * size ** 3
        tma_tflops = flops / tma_gemm_time / 1e12
        torch_tflops = flops / torch_gemm_time / 1e12
        speedup_gemm = torch_gemm_time / tma_gemm_time
        
        print(f"  TMA GEMM:      {tma_gemm_time*1e3:.2f} ms ({tma_tflops:.2f} TFLOPS)")
        print(f"  PyTorch GEMM:  {torch_gemm_time*1e3:.2f} ms ({torch_tflops:.2f} TFLOPS)")
        print(f"  Speedup:       {speedup_gemm:.2f}x")
        
        max_diff = torch.abs(C_tma - C_torch).max().item()
        print(f"  Max Difference: {max_diff:.2e}")
        
        results[size] = {
            'copy_speedup': speedup_copy,
            'copy_bandwidth_tma': tma_bw,
            'copy_bandwidth_std': std_bw,
            'gemm_speedup': speedup_gemm,
            'gemm_tflops_tma': tma_tflops,
            'gemm_tflops_torch': torch_tflops,
            'correctness': max_diff < 1e-2,
        }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    avg_copy_speedup = sum(r['copy_speedup'] for r in results.values()) / len(results)
    avg_gemm_speedup = sum(r['gemm_speedup'] for r in results.values()) / len(results)
    max_bw = max(r['copy_bandwidth_tma'] for r in results.values())
    max_tflops = max(r['gemm_tflops_tma'] for r in results.values())
    
    print(f"Average Copy Speedup:  {avg_copy_speedup:.2f}x")
    print(f"Average GEMM Speedup:  {avg_gemm_speedup:.2f}x")
    print(f"Peak Bandwidth:        {max_bw:.2f} TB/s")
    print(f"Peak TFLOPS:           {max_tflops:.2f}")
    print(f"All Tests Passed:      {'YES' if all(r['correctness'] for r in results.values()) else 'NO'}")
    
    if is_blackwell:
        hbm3e_peak = 7.8  # TB/s for B200
        utilization = (max_bw / hbm3e_peak) * 100
        print(f"HBM3e Utilization:     {utilization:.1f}%")
    
    print("="*70)
    
    return results


def demonstrate_tma_features():
    """Demonstrate Blackwell TMA capabilities."""
    print("\n" + "="*70)
    print("Triton 3.5 TMA for Blackwell - Feature Demonstration")
    print("="*70)
    
    print("\n[1] TMA Descriptor Overview")
    print("  - Hardware-accelerated bulk memory transfers")
    print("  - 128-byte aligned for HBM3e optimization")
    print("  - Asynchronous execution with minimal CPU overhead")
    print("  - L2 cache management for large transfers")
    print("  - Up to 7.8 TB/s bandwidth on B200")
    
    print("\n[2] When to Use TMA")
    print("  Best for:")
    print("    - Large contiguous memory transfers (>64KB)")
    print("    - Matrix operations with regular access patterns")
    print("    - Bulk copies between global and shared memory")
    print("  Avoid for:")
    print("    - Small scattered loads (<1KB)")
    print("    - Irregular access patterns")
    
    print("\n[3] Performance Guidelines")
    print("  - Block size: 128x128 or larger")
    print("  - Pipeline depth: 4-5 stages on Blackwell")
    print("  - Warps: 8 for optimal occupancy")
    print("  - Expected speedup: 1.3-1.5x over manual loads")
    
    print("="*70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main benchmark suite for Triton 3.5 TMA on Blackwell."""
    print("\n" + "="*70)
    print("TRITON 3.5 TMA FOR BLACKWELL")
    print("Tensor Memory Accelerator Optimization")
    print("="*70)
    
    demonstrate_tma_features()
    
    if torch.cuda.is_available():
        print("\nRunning performance benchmarks...")
        results = benchmark_tma_vs_standard(
            sizes=[2048, 4096, 8192],
            num_iters=50,
        )
        print("\nBenchmarks complete!")
    else:
        print("\nCUDA not available - skipping benchmarks")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

