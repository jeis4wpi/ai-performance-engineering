// optimized_coalescing.cu - Coalesced memory access pattern (optimized)
// Demonstrates proper memory access patterns that enable memory coalescing

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "../common/headers/profiling_helpers.cuh"

// Coalesced memory copy kernel - threads access consecutive memory locations
__global__ void coalesced_copy(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = input[idx];  // Consecutive access enables coalescing
    }
}

int main() {
    const int N = 10'000'000;
    const size_t bytes = N * sizeof(float);
    
    float* h_input = nullptr;
    float* h_output = nullptr;
    cudaMallocHost(&h_input, bytes);
    cudaMallocHost(&h_output, bytes);
    
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    coalesced_copy<<<grid, block, 0, stream>>>(d_output, d_input, N);
    cudaStreamSynchronize(stream);
    
    // Measure
    cudaEventRecord(start, stream);
    {
        PROFILE_KERNEL_LAUNCH("coalesced_copy");
        coalesced_copy<<<grid, block, 0, stream>>>(d_output, d_input, N);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    printf("Coalesced copy: %.3f ms\n", ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFreeHost(h_output);
    cudaFreeHost(h_input);
    
    return 0;
}

