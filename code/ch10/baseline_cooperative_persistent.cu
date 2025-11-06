// baseline_cooperative_persistent.cu -- Traditional kernels (baseline).

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

// Traditional kernel - separate launches
__global__ void traditionalKernel(float* data, int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * scale + 1.0f;
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);
    
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        traditionalKernel<<<grid, block>>>(d_data, N, 2.0f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Traditional kernel (baseline): %.2f ms\n", ms / iterations);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    return 0;
}

