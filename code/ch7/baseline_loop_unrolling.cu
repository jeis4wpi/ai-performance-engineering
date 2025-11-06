// baseline_loop_unrolling.cu -- Loop without unrolling (baseline).

#include <cuda_runtime.h>
#include <cstdio>

#include "../common/headers/cuda_helpers.cuh"

constexpr int N = 1 << 20;

__global__ void kernel_no_unroll(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    // No unroll pragma - sequential loop
    for (int i = 0; i < 16; ++i) {
      val = val * 1.001f + 0.001f;
    }
    out[idx] = val;
  }
}

int main() {
  float *h_in, *h_out;
  CUDA_CHECK(cudaMallocHost(&h_in, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));
  for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  const int iterations = 100;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    kernel_no_unroll<<<grid, block>>>(d_in, d_out, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("Loop without unroll (baseline): %.3f ms\n", ms / iterations);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

