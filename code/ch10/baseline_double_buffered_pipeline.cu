// baseline_double_buffered_pipeline.cu -- GEMM without pipeline (baseline).

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

template<int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_tiled_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    cg::thread_block cta = cg::this_thread_block();
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    extern __shared__ float shared[];
    float* A_tile = shared;
    float* B_tile = shared + TILE_M * TILE_K;

    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    float sum = 0.0f;

    const int total_tiles = (K + TILE_K - 1) / TILE_K;
    for (int tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
        const int k_base = tile_idx * TILE_K;

        // Load tiles sequentially (no overlap)
        if (thread_row < TILE_M && thread_col < TILE_K) {
            int global_row = block_row + thread_row;
            int global_col = k_base + thread_col;
            A_tile[thread_row * TILE_K + thread_col] =
                (global_row < M && global_col < K) ?
                A[global_row * K + global_col] : 0.0f;
        }

        if (thread_row < TILE_K && thread_col < TILE_N) {
            int global_row = k_base + thread_row;
            int global_col = block_col + thread_col;
            B_tile[thread_row * TILE_N + thread_col] =
                (global_row < K && global_col < N) ?
                B[global_row * N + global_col] : 0.0f;
        }

        cta.sync();

        // Compute
        for (int kk = 0; kk < TILE_K; ++kk) {
            if (k_base + kk < K) {
                sum += A_tile[thread_row * TILE_K + kk] * B_tile[kk * TILE_N + thread_col];
            }
        }

        cta.sync();
    }

    if (block_row + thread_row < M && block_col + thread_col < N) {
        C[(block_row + thread_row) * N + (block_col + thread_col)] = sum;
    }
}

int main() {
    const int M = 512, N = 512, K = 512;
    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    
    constexpr int TILE_M = 32, TILE_N = 32, TILE_K = 32;
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    size_t smem = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        gemm_tiled_naive_kernel<TILE_M, TILE_N, TILE_K><<<grid, block, smem>>>(
            d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GEMM naive (baseline): %.2f ms\n", ms / iterations);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

