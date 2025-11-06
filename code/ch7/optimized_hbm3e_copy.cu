// optimized_hbm3e_copy.cu -- 256-byte bursts with cache streaming (optimized).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "../common/headers/cuda_helpers.cuh"

// Optimized: 256-byte bursts with cache streaming (Blackwell HBM3e optimal)
__global__ void hbm3e_optimized_copy_kernel(float4* dst, const float4* src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Process 256 bytes per iteration (16 float4s)
    // This matches HBM3e burst size
    constexpr int BURST_SIZE = 16;  // 16 * 16 bytes = 256 bytes
    
    for (size_t base = tid * BURST_SIZE; base < n; base += stride * BURST_SIZE) {
        // Unroll loop for 256-byte burst
        #pragma unroll
        for (int i = 0; i < BURST_SIZE; i++) {
            size_t idx = base + i;
            if (idx < n) {
                // Use cache streaming modifier for HBM3e
                // .cs (cache-streaming) bypasses L2 for write-only patterns
                #if __CUDA_ARCH__ >= 1000  // Blackwell
                // PTX inline assembly for cache streaming
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];" 
                    : "=f"(reinterpret_cast<float*>(&dst[idx])[0]),
                      "=f"(reinterpret_cast<float*>(&dst[idx])[1]),
                      "=f"(reinterpret_cast<float*>(&dst[idx])[2]),
                      "=f"(reinterpret_cast<float*>(&dst[idx])[3])
                    : "l"(&src[idx]));
                #else
                dst[idx] = src[idx];
                #endif
            }
        }
    }
}

int main() {
    const size_t size_bytes = 256 * 1024 * 1024;  // 256 MB
    const size_t n_floats = size_bytes / sizeof(float);
    const size_t n_float4 = n_floats / 4;
    
    float4* d_src = nullptr;
    float4* d_dst = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_src, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, size_bytes));
    
    // Initialize
    CUDA_CHECK(cudaMemset(d_src, 1, size_bytes));
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        hbm3e_optimized_copy_kernel<<<256, 256>>>(d_dst, d_src, n_float4);
        CUDA_CHECK_LAST_ERROR();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    double bw = (size_bytes * 2 / (avg_ms / 1000.0)) / 1e9;
    
    printf("HBM3e optimized (256-byte bursts): %.2f ms, %.2f GB/s\n", avg_ms, bw);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}

