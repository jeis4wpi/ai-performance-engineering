// baseline_async_prefetch.cu -- Fallback kernel without TMA (baseline).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template <int TILE_SIZE>
__global__ void async_prefetch_fallback_kernel(
    const float* data,
    float* out,
    int tiles) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;

    for (int t = 0; t < tiles; ++t) {
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            smem[i] = data[t * TILE_SIZE + i];
        }
        __syncthreads();

        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            float v = smem[i] * 2.0f;
            smem[i] = v;
            out[t * TILE_SIZE + i] = v;
        }
        __syncthreads();
    }
}

int main() {
    constexpr int TILE_SIZE = 256;
    const int tiles = 1000;
    const size_t n = TILE_SIZE * tiles;
    const size_t bytes = n * sizeof(float);
    const size_t smem_bytes = TILE_SIZE * sizeof(float);
    
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    
    // Initialize
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        async_prefetch_fallback_kernel<TILE_SIZE><<<1, 256, smem_bytes>>>(d_in, d_out, tiles);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    printf("Fallback kernel (baseline): %.2f ms\n", avg_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    
    return 0;
}

