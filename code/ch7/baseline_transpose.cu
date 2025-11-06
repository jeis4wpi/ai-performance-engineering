// transpose_naive.cu -- naive matrix transpose for Chapter 7.

#include <cuda_runtime.h>
#include <cstdio>

#include "../common/headers/cuda_helpers.cuh"

constexpr int WIDTH = 1024;

__global__ void transpose_naive(const float* in, float* out, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < width) {
    out[y * width + x] = in[x * width + y];
  }
}

int main() {
  size_t bytes = WIDTH * WIDTH * sizeof(float);
  float *h_in, *h_out;
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));
  for (int i = 0; i < WIDTH * WIDTH; ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  dim3 block(32, 32);
  dim3 grid((WIDTH + block.x - 1) / block.x, (WIDTH + block.y - 1) / block.y);
  transpose_naive<<<grid, block>>>(d_in, d_out, WIDTH);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  printf("out[0]=%.1f\n", h_out[0]);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
