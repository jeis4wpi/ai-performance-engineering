// baseline_loop_unrolling.cu -- Original loop with limited ILP (baseline).

#include <cuda_runtime.h>
#include <iostream>

// Original loop - limited ILP
__global__ void originalLoop(const float* A, const float* w, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
            float a = A[idx * 4 + k];
            sum += a * w[k]; // dependent on load a
        }
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
        originalLoop<<<gridSize, blockSize>>>(d_A, d_w, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_total;
    cudaEventElapsedTime(&time_total, start, stop);
    float time_avg = time_total / iterations;
    
    printf("Original loop (baseline): %.3f ms\n", time_avg);
    
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

