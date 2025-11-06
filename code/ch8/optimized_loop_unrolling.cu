// optimized_loop_unrolling.cu -- Unrolled loop with better ILP (optimized).

#include <cuda_runtime.h>
#include <iostream>

// Unrolled loop - better ILP
__global__ void unrolledLoop(const float* A, const float* w, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Load all elements first (independent loads)
        float a0 = A[idx * 4 + 0];
        float a1 = A[idx * 4 + 1];
        float a2 = A[idx * 4 + 2];
        float a3 = A[idx * 4 + 3];
        
        // Independent multiply operations
        float sum0 = a0 * w[0];
        float sum1 = a1 * w[1];
        float sum2 = a2 * w[2];
        float sum3 = a3 * w[3];
        
        // Final reduction
        float sum = sum0 + sum1 + sum2 + sum3;
        out[idx] = sum;
    }
}

int main() {
    const int N = 1 << 18;  // 256K elements
    const int total_elements = N * 4;
    const size_t bytes_A = total_elements * sizeof(float);
    const size_t bytes_out = N * sizeof(float);
    const size_t bytes_w = 4 * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(bytes_A);
    float* h_w = (float*)malloc(bytes_w);
    float* h_out = (float*)malloc(bytes_out);
    
    // Initialize data
    for (int i = 0; i < total_elements; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
    }
    
    for (int i = 0; i < 4; ++i) {
        h_w[i] = static_cast<float>(i + 1) / 4.0f;
    }
    
    // Allocate device memory
    float* d_A = nullptr;
    float* d_w = nullptr;
    float* d_out = nullptr;
    
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_w, bytes_w);
    cudaMalloc(&d_out, bytes_out);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, bytes_w, cudaMemcpyHostToDevice);
    
    // Launch configuration
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        unrolledLoop<<<gridSize, blockSize>>>(d_A, d_w, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_total;
    cudaEventElapsedTime(&time_total, start, stop);
    float time_avg = time_total / iterations;
    
    printf("Unrolled loop (optimized): %.3f ms\n", time_avg);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_w);
    free(h_out);
    cudaFree(d_A);
    cudaFree(d_w);
    cudaFree(d_out);
    
    return 0;
}

