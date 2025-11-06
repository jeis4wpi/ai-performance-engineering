// uncoalesced_copy.cu -- demonstrate strided global memory loads.

#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void uncoalescedCopy(const float* __restrict__ in,
                                float* __restrict__ out,
                                int n,
                                int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx * stride];
  }
}

int main() {
  constexpr int N = 1 << 20;
  constexpr int STRIDE = 2;
  static_assert(STRIDE >= 1, "Stride must be positive");
  float* h_in = nullptr;
  float* h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in, N * STRIDE * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));
  for (int i = 0; i < N * STRIDE; ++i) {
    h_in[i] = static_cast<float>(i);
  }

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * STRIDE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, N * STRIDE * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  uncoalescedCopy<<<grid, block>>>(d_in, d_out, N, STRIDE);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("out[0]=%.1f out[last]=%.1f\n", h_out[0], h_out[N - 1]);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
