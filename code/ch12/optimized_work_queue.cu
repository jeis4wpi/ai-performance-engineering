// optimized_work_queue.cu -- dynamic work distribution with atomic work queue (optimized).

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

__device__ unsigned int g_index = 0;

__global__ void compute_dynamic(const float* input, float* output, int n) {
  while (true) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & (warpSize - 1);
    int leader = __ffs(mask) - 1;

    unsigned base = 0;
    if (lane == leader) {
      base = atomicAdd(&g_index, warpSize);
    }
    base = __shfl_sync(mask, base, leader);

    unsigned idx = base + lane;
    if (idx >= static_cast<unsigned>(n)) break;

    float s, c;
    __sincosf(input[idx], &s, &c);
    int work = idx & 255;
    float sum = 0.0f;
#pragma unroll 1
    for (int i = 0; i < work; ++i) {
      sum += s * c;
    }
    output[idx] = sum;
  }
}

static void reset_counter() {
  unsigned zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_index, &zero, sizeof(unsigned)));
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

  reset_counter();

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  compute_dynamic<<<grid, block>>>(d_in, d_out, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float dynamic_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&dynamic_ms, start, stop));

  std::printf("Dynamic (optimized): %.2f ms\n", dynamic_ms);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

