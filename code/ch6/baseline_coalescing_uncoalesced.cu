// coalescing_example.cu
// Demonstrates memory coalescing: uncoalesced vs coalesced access patterns
// Shows how memory access patterns affect bandwidth utilization
//
// Key concepts:
// - Coalesced access: threads in a warp access consecutive memory locations
//   (e.g., thread 0->0, thread 1->1, thread 2->2) -> single 128-byte transaction
// - Uncoalesced access: threads access scattered locations
//   (e.g., thread 0->0, thread 1->stride, thread 2->2*stride) -> multiple transactions

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../common/headers/profiling_helpers.cuh"

//------------------------------------------------------
// Uncoalesced memory access pattern
// Each thread accesses memory with a large stride
// This causes each thread to generate separate memory transactions
__global__ void uncoalesced_copy(float* output, const float* input, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int access_idx = idx * stride;  // Non-consecutive access
    
    if (access_idx < N) {
        output[access_idx] = input[access_idx];
    }
}

//------------------------------------------------------
// Coalesced memory access pattern
// Threads access consecutive memory locations
// This allows the warp to combine accesses into a single 128-byte transaction
__global__ void coalesced_copy(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = input[idx];  // Consecutive access
    }
}

//------------------------------------------------------
// Helper function to measure kernel execution time
float measure_kernel_time(
    void (*kernel)(float*, const float*, int, int),
    float* d_output,
    const float* d_input,
    int N,
    int stride,
    cudaStream_t stream
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel<<<((N + stride - 1) / stride + 255) / 256, 256, 0, stream>>>(
        d_output, d_input, N, stride);
    cudaStreamSynchronize(stream);
    
    // Measure
    cudaEventRecord(start, stream);
    {
        PROFILE_KERNEL_LAUNCH("uncoalesced_copy");
        kernel<<<((N + stride - 1) / stride + 255) / 256, 256, 0, stream>>>(
            d_output, d_input, N, stride);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
}

float measure_coalesced_time(
    void (*kernel)(float*, const float*, int),
    float* d_output,
    const float* d_input,
    int N,
    cudaStream_t stream
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel<<<(N + 255) / 256, 256, 0, stream>>>(d_output, d_input, N);
    cudaStreamSynchronize(stream);
    
    // Measure
    cudaEventRecord(start, stream);
    {
        PROFILE_KERNEL_LAUNCH("coalesced_copy");
        kernel<<<(N + 255) / 256, 256, 0, stream>>>(d_output, d_input, N);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
}

int main() {
    const int N = 10'000'000;  // 10M elements (40 MB)
    const int stride = 32;     // Large stride for uncoalesced access
    
    printf("========================================\n");
    printf("Memory Coalescing Example\n");
    printf("========================================\n");
    printf("Problem size: %d elements (%.2f MB)\n", N, N * sizeof(float) / 1024.0f / 1024.0f);
    printf("Uncoalesced stride: %d\n\n", stride);
    
    // Allocate pinned host memory
    float* h_input = nullptr;
    float* h_output_uncoalesced = nullptr;
    float* h_output_coalesced = nullptr;
    
    cudaMallocHost(&h_input, N * sizeof(float));
    cudaMallocHost(&h_output_uncoalesced, N * sizeof(float));
    cudaMallocHost(&h_output_coalesced, N * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    cudaMallocAsync(&d_input, N * sizeof(float), stream);
    cudaMallocAsync(&d_output, N * sizeof(float), stream);
    
    // Copy input to device
    {
        PROFILE_MEMORY_COPY("H2D copy");
        cudaMemcpyAsync(d_input, h_input, N * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);
    
    // Test uncoalesced access (poor pattern)
    printf("1. Uncoalesced memory access (stride=%d):\n", stride);
    float uncoalesced_time = measure_kernel_time(
        uncoalesced_copy, d_output, d_input, N, stride, stream);
    
    float uncoalesced_bandwidth = (N * sizeof(float) / 1024.0f / 1024.0f / 1024.0f) / 
                                  (uncoalesced_time / 1000.0f);
    printf("   Time: %.3f ms\n", uncoalesced_time);
    printf("   Effective bandwidth: %.2f GB/s\n", uncoalesced_bandwidth);
    
    // Copy result back
    cudaMemcpyAsync(h_output_uncoalesced, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Test coalesced access (good pattern)
    printf("\n2. Coalesced memory access (stride=1):\n");
    float coalesced_time = measure_coalesced_time(
        coalesced_copy, d_output, d_input, N, stream);
    
    float coalesced_bandwidth = (N * sizeof(float) / 1024.0f / 1024.0f / 1024.0f) / 
                                (coalesced_time / 1000.0f);
    printf("   Time: %.3f ms\n", coalesced_time);
    printf("   Effective bandwidth: %.2f GB/s\n", coalesced_bandwidth);
    
    // Copy result back
    cudaMemcpyAsync(h_output_coalesced, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Verify correctness
    bool uncoalesced_correct = true;
    bool coalesced_correct = true;
    
    for (int i = 0; i < N; i += stride) {
        if (h_output_uncoalesced[i] != h_input[i]) {
            uncoalesced_correct = false;
            break;
        }
    }
    
    for (int i = 0; i < N; ++i) {
        if (h_output_coalesced[i] != h_input[i]) {
            coalesced_correct = false;
            break;
        }
    }
    
    printf("\n========================================\n");
    printf("Results:\n");
    printf("  Uncoalesced: %s\n", uncoalesced_correct ? "✓ Correct" : "✗ Incorrect");
    printf("  Coalesced:   %s\n", coalesced_correct ? "✓ Correct" : "✗ Incorrect");
    
    if (coalesced_time > 0 && uncoalesced_time > 0) {
        float speedup = uncoalesced_time / coalesced_time;
        printf("\n  Coalesced is %.2fx faster\n", speedup);
        printf("  Bandwidth improvement: %.2fx\n", 
               coalesced_bandwidth / uncoalesced_bandwidth);
    }
    
    printf("\nKey insight: Coalesced access allows warps to combine\n");
    printf("32 consecutive accesses into a single 128-byte transaction,\n");
    printf("maximizing memory bandwidth utilization.\n");
    printf("========================================\n");
    
    // Cleanup
    cudaFreeAsync(d_output, stream);
    cudaFreeAsync(d_input, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_output_coalesced);
    cudaFreeHost(h_output_uncoalesced);
    cudaFreeHost(h_input);
    
    return (uncoalesced_correct && coalesced_correct) ? 0 : 1;
}

