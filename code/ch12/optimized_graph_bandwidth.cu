// optimized_graph_bandwidth.cu
// Optimized version: CUDA graphs for bandwidth measurement
//
// Key concepts:
// - CUDA graphs capture kernel sequences
// - Graph replay reduces launch overhead
// - Bandwidth measurement within graph execution
// - Memory traffic analysis optimized by graph capture

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../common/headers/profiling_helpers.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Simple memory copy kernel
__global__ void memory_copy_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Memory-intensive computation kernel
__global__ void compute_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Memory-intensive: multiple reads and writes
        float val1 = data[idx];
        float val2 = data[(idx + 1) % n];
        float val3 = data[(idx + 2) % n];
        data[idx] = val1 * val2 + val3;
    }
}

// Measure bandwidth for CUDA graph
struct BandwidthResult {
    float time_ms;
    float bandwidth_gbs;
    size_t bytes_transferred;
};

BandwidthResult measure_bandwidth_graph(
    cudaGraphExec_t exec,
    cudaStream_t stream,
    int iterations,
    size_t data_size_bytes,
    const char* name
) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Measure
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    {
        PROFILE_KERNEL_LAUNCH(name);
        for (int i = 0; i < iterations; ++i) {
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Calculate bandwidth
    // For copy: 2 * size (read + write)
    // For compute: estimate based on memory accesses
    size_t total_bytes = data_size_bytes * iterations * 2;  // Read + write
    float bandwidth_gbs = (total_bytes / (1024.0f * 1024.0f * 1024.0f)) / (ms / 1000.0f);
    
    return {ms, bandwidth_gbs, total_bytes};
}

int main() {
    constexpr int N = 50'000'000;  // 50M elements (~200 MB)
    constexpr int ITERATIONS = 50;
    const size_t data_size_bytes = N * sizeof(float);
    
    printf("========================================\n");
    printf("Optimized: CUDA Graph Bandwidth Measurement\n");
    printf("========================================\n");
    printf("Problem size: %d elements (%.2f MB)\n", N, data_size_bytes / 1024.0f / 1024.0f);
    printf("Iterations: %d\n\n", ITERATIONS);
    
    // Allocate memory
    std::vector<float> h_src(N);
    std::vector<float> h_dst(N);
    for (int i = 0; i < N; ++i) {
        h_src[i] = static_cast<float>(i);
    }
    
    float* d_src = nullptr;
    float* d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, data_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, data_size_bytes));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    // Copy initial data
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), data_size_bytes,
                          cudaMemcpyHostToDevice));
    
    //------------------------------------------------------
    // Test 1: CUDA Graph with single kernel
    printf("1. CUDA Graph (single kernel):\n");
    
    cudaGraph_t graph1;
    cudaGraphExec_t exec1;
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    memory_copy_kernel<<<grid, block, 0, stream>>>(d_dst, d_src, N);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph1));
    CUDA_CHECK(cudaGraphInstantiate(&exec1, graph1, nullptr, nullptr, 0));
    
    BandwidthResult graph1_result = measure_bandwidth_graph(
        exec1, stream, ITERATIONS, data_size_bytes, "graph_single_kernel");
    printf("   Time: %.3f ms\n", graph1_result.time_ms);
    printf("   Bandwidth: %.2f GB/s\n", graph1_result.bandwidth_gbs);
    
    //------------------------------------------------------
    // Test 2: CUDA Graph with multiple kernels
    printf("\n2. CUDA Graph (multiple kernels):\n");
    
    cudaGraph_t graph2;
    cudaGraphExec_t exec2;
    
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    memory_copy_kernel<<<grid, block, 0, stream>>>(d_dst, d_src, N);
    compute_kernel<<<grid, block, 0, stream>>>(d_dst, N);
    memory_copy_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph2));
    CUDA_CHECK(cudaGraphInstantiate(&exec2, graph2, nullptr, nullptr, 0));
    
    // Estimate: 3 kernels, each with read+write = 6x data transfers
    size_t multi_kernel_bytes = data_size_bytes * 6;
    BandwidthResult graph2_result = measure_bandwidth_graph(
        exec2, stream, ITERATIONS, multi_kernel_bytes, "graph_multi_kernel");
    printf("   Time: %.3f ms\n", graph2_result.time_ms);
    printf("   Bandwidth: %.2f GB/s\n", graph2_result.bandwidth_gbs);
    printf("   Bytes transferred: %.2f GB\n", 
           graph2_result.bytes_transferred / (1024.0f * 1024.0f * 1024.0f));
    
    //------------------------------------------------------
    // Results summary
    printf("\n========================================\n");
    printf("Summary:\n");
    printf("  Graph (single):     %.2f GB/s\n", graph1_result.bandwidth_gbs);
    printf("  Graph (multi):      %.2f GB/s\n", graph2_result.bandwidth_gbs);
    
    printf("\nKey insights:\n");
    printf("  - CUDA graphs reduce launch overhead\n");
    printf("  - Graph capture allows efficient replay of kernel sequences\n");
    printf("  - Bandwidth measurement helps identify bottlenecks\n");
    printf("========================================\n");
    
    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(exec2));
    CUDA_CHECK(cudaGraphDestroy(graph2));
    CUDA_CHECK(cudaGraphExecDestroy(exec1));
    CUDA_CHECK(cudaGraphDestroy(graph1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_src));
    
    return 0;
}

