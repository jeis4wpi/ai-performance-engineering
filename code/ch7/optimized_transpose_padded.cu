// transpose_padded.cu -- tiled transpose with padding for Chapter 7.

#include <cuda_runtime.h>
#include <cstdio>

#include "../common/headers/cuda_helpers.cuh"

constexpr int WIDTH = 1024;
constexpr int TILE = 32;

__global__ void transpose_padded(const float* in, float* out, int width) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < width && y < width) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();
  int trans_x = blockIdx.y * TILE + threadIdx.x;
  int trans_y = blockIdx.x * TILE + threadIdx.y;
  if (trans_x < width && trans_y < width) {
    out[trans_y * width + trans_x] = tile[threadIdx.x][threadIdx.y];
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

  dim3 block(TILE, TILE);
  dim3 grid((WIDTH + TILE - 1) / TILE, (WIDTH + TILE - 1) / TILE);
  transpose_padded<<<grid, block>>>(d_in, d_out, WIDTH);
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
