# Benchmark Test Results Summary

**Generated:** 2025-11-06 00:09:51

## Overall Summary

- **Chapters tested:** 1/1
- **Chapters skipped:** 0 (CUDA unavailable)
- **Chapters with no benchmarks:** 0
- **Total benchmarks:** 19
- **Successful:** 19
- **Failed:** 0
- **Average speedup:** 18.47x
- **Best speedup:** 200.96x
- **Worst speedup:** 1.00x

## Per-Chapter Summary

| Chapter | Status | Benchmarks | Successful | Failed | Avg Speedup | Max Speedup |
|---------|--------|------------|------------|--------|-------------|-------------|
| ch1 | PASS | 19 | 19 | 0 | 34.50x | 200.96x |

## Detailed Results

### CH1

**nvlink**
- Baseline: `baseline_nvlink.py` (1.06 ms)
- `optimized_nvlink.py`: 0.01 ms (200.96x speedup)
- Best speedup: 200.96x

**double**
- Baseline: `baseline_double_buffering.py` (3.79 ms)
- `optimized_double_buffering.py`: 7.03 ms (0.54x speedup)

**continuous**
- Baseline: `baseline_continuous_batching.py` (0.01 ms)
- `optimized_continuous_batching.py`: 0.01 ms (0.54x speedup)

**cutlass**
- Baseline: `baseline_cutlass.py` (0.00 ms)
- `optimized_cutlass.py`: 0.00 ms (0.98x speedup)

**kv**
- Baseline: `baseline_kv_cache.py` (0.00 ms)
- `optimized_kv_cache.py`: 0.00 ms (1.00x speedup)

**speculative**
- Baseline: `baseline_speculative_decoding.py` (0.42 ms)
- `optimized_speculative_decoding.py`: 0.00 ms (98.22x speedup)
- Best speedup: 98.22x

**moe**
- Baseline: `baseline_moe.py` (0.00 ms)
- `optimized_moe.py`: 0.00 ms (1.11x speedup)
- Best speedup: 1.11x

**shared**
- Baseline: `baseline_shared_memory.py` (0.00 ms)
- `optimized_shared_memory.py`: 0.00 ms (0.98x speedup)

**coalescing**
- Baseline: `baseline_coalescing.py` (0.16 ms)
- `optimized_coalescing.py`: 0.42 ms (0.38x speedup)

**warp**
- Baseline: `baseline_warp_specialization.py` (0.00 ms)
- `optimized_warp_divergence.py`: 0.00 ms (0.93x speedup)
- `optimized_warp_specialization.py`: 0.00 ms (0.95x speedup)

**ilp**
- Baseline: `baseline_ilp_basic.py` (0.00 ms)
- `optimized_ilp_basic.py`: 0.01 ms (0.72x speedup)

**warp**
- Baseline: `baseline_warp_divergence.py` (0.00 ms)
- `optimized_warp_divergence.py`: 0.00 ms (1.03x speedup)
- `optimized_warp_specialization.py`: 0.00 ms (0.99x speedup)
- Best speedup: 1.03x

**performance**
- Baseline: `baseline_performance.py` (1.46 ms)
- `optimized_performance_pinned.py`: 0.80 ms (1.83x speedup)
- `optimized_performance_batch.py`: 0.45 ms (3.27x speedup)
- `optimized_performance_graphs.py`: 0.18 ms (8.30x speedup)
- Best speedup: 8.30x

**disaggregated**
- Baseline: `baseline_disaggregated.py` (0.00 ms)
- `optimized_disaggregated.py`: 0.00 ms (0.98x speedup)

**nccl**
- Baseline: `baseline_nccl.py` (0.01 ms)
- `optimized_nccl.py`: 0.00 ms (1.34x speedup)
- Best speedup: 1.34x

**guided**
- Baseline: `baseline_guided_decoding.py` (0.00 ms)
- `optimized_guided_decoding.py`: 0.01 ms (0.56x speedup)

**attention**
- Baseline: `baseline_attention.py` (0.12 ms)
- `optimized_attention.py`: 0.00 ms (27.88x speedup)
- Best speedup: 27.88x

**arithmetic** *(CUDA)*
- Baseline: `baseline_arithmetic_intensity.cu` (467.69 ms)
- `optimized_arithmetic_intensity_combined.cu`: 472.20 ms (0.99x speedup)

**gemm** *(CUDA)*
- Baseline: `baseline_gemm.cu` (357.43 ms)
- `optimized_gemm_batched.cu`: 325.16 ms (1.10x speedup)
- `optimized_gemm_strided.cu`: 325.18 ms (1.10x speedup)
- Best speedup: 1.10x


