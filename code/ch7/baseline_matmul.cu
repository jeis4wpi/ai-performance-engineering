// naive_matmul.cu -- naive matrix multiplication (Chapter 7 baseline).

#include <cuda_runtime.h>
#include <cstdio>

#include "../common/headers/cuda_helpers.cuh"

constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;

__global__ void matmul_naive(const float* A, const float* B, float* C, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
  }
}

int main() {
  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  float *h_A, *h_B, *h_C;
  CUDA_CHECK(cudaMallocHost(&h_A, bytesA));
  CUDA_CHECK(cudaMallocHost(&h_B, bytesB));
  CUDA_CHECK(cudaMallocHost(&h_C, bytesC));
  for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
  for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytesA));
  CUDA_CHECK(cudaMalloc(&d_B, bytesB));
  CUDA_CHECK(cudaMalloc(&d_C, bytesC));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
  printf("C[0]=%.1f\n", h_C[0]);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));
  return 0;
}
