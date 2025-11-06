// optimized_cooperative_persistent.cu -- Persistent kernel (optimized).

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

// Persistent kernel - blocks stay alive and process multiple tasks
__device__ int g_task_index = 0;

__global__ void persistentKernel(float* data, int N, int totalTasks) {
    // Every thread loops, atomically grabbing the next task index until none remain
    while (true) {
        int idx = atomicAdd(&g_task_index, 1);
        if (idx >= totalTasks) break;
        
        // Process task
        int task_size = N / totalTasks;
        int base = idx * task_size;
        int tid = threadIdx.x;
        int stride = blockDim.x;
        
        for (int i = base + tid; i < base + task_size && i < N; i += stride) {
            data[i] = data[i] * 2.0f + 1.0f;
        }
    }
}

int main() {
    const int N = 1 << 20;
    const int totalTasks = 100;
    const size_t bytes = N * sizeof(float);
    
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    dim3 block(256);
    dim3 grid(4);  // Fewer blocks that persist
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        // Reset counter
        int zero = 0;
        cudaMemcpyToSymbol(g_task_index, &zero, sizeof(int));
        
        persistentKernel<<<grid, block>>>(d_data, N, totalTasks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Persistent kernel (optimized): %.2f ms\n", ms / iterations);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    return 0;
}

