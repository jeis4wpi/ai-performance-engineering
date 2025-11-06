// optimized_cuda_graphs.cu -- CUDA graph replay (optimized).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void kernel_a(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = data[idx] * 1.1f + 0.1f;
}

__global__ void kernel_b(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
}

__global__ void kernel_c(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = sinf(data[idx] * 0.1f);
}

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("Optimized CUDA Graphs (graph replay) on %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
  
  if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
    printf("CUDA Graphs require compute capability 7.5 or newer.\n");
    return 0;
  }

  constexpr int N = 1 << 20;
  constexpr int ITER = 100;
  size_t bytes = N * sizeof(float);

  std::vector<float> host(N);
  for (int i = 0; i < N; ++i) host[i] = float(i) / N;

  float* device_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
  CUDA_CHECK(cudaMemcpy(device_ptr, host.data(), bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // Optimized: Graph capture and replay
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  kernel_a<<<grid, block, 0, stream>>>(device_ptr, N);
  kernel_b<<<grid, block, 0, stream>>>(device_ptr, N);
  kernel_c<<<grid, block, 0, stream>>>(device_ptr, N);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec;
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < ITER; ++i) {
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
  
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("Optimized (graph replay): %.3f ms (%.3f ms/iter)\n", ms, ms / ITER);

  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(device_ptr));
  return 0;
}

