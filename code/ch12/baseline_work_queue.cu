// baseline_work_queue.cu -- static work distribution (baseline).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void compute_static(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int work = idx & 255;
    float sum = 0.0f;
    for (int i = 0; i < work; ++i) {
      sum += sinf(input[idx]) * cosf(input[idx]);
    }
    output[idx] = sum;
  }
}

int main() {
  constexpr int N = 1 << 20;
  std::vector<float> h_in(N);
  for (int i = 0; i < N; ++i) h_in[i] = float(i) / N;
  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  compute_static<<<grid, block>>>(d_in, d_out, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float static_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&static_ms, start, stop));

  std::printf("Static (baseline): %.2f ms\n", static_ms);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

