// baseline_cluster_group.cu -- Regular launch without clusters (baseline).

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

// Regular kernel without cluster
__global__ void regular_sum_kernel(const float *in, float *out, int elems_per_block) {
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ float sdata[];

    float sum = 0.0f;
    int base = blockIdx.x * elems_per_block;
    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        sum += in[base + i];
    }

    sdata[threadIdx.x] = sum;
    cta.sync();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        cta.sync();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 1 << 20;
    const int elems_per_block = 1024;
    const int num_blocks = (N + elems_per_block - 1) / elems_per_block;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, num_blocks * sizeof(float));
    
    dim3 block(256);
    dim3 grid(num_blocks);
    size_t smem = 256 * sizeof(float);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        regular_sum_kernel<<<grid, block, smem>>>(d_in, d_out, elems_per_block);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Regular launch (baseline): %.2f ms\n", ms / iterations);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}

