// threshold_predicated.cu -- predicated version of thresholding.

#include <cuda_runtime.h>
#include <cstdio>

// Same workload size as baseline for fair comparison
constexpr int N = 1 << 26;  // 64M elements (same as baseline)

__global__ void threshold_predicated(const float* __restrict__ X,
                                     float* __restrict__ Y,
                                     float threshold,
                                     int N) {
  // Optimized: Vectorized loads for better memory throughput
  const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int stride = blockDim.x * gridDim.x * 4;
  
  // Process 4 elements at a time (vectorized)
  for (int i = idx; i < N - 3; i += stride) {
    // Load 4 floats at once
    float4 vec = *reinterpret_cast<const float4*>(&X[i]);
    
    // Branch-free predicated execution
    vec.x = (vec.x > threshold) ? vec.x : 0.0f;
    vec.y = (vec.y > threshold) ? vec.y : 0.0f;
    vec.z = (vec.z > threshold) ? vec.z : 0.0f;
    vec.w = (vec.w > threshold) ? vec.w : 0.0f;
    
    // Store 4 floats at once
    *reinterpret_cast<float4*>(&Y[i]) = vec;
  }
  
  // Handle remaining elements
  for (int i = idx + ((N / 4) * 4); i < N; i += stride) {
    const float x = X[i];
    Y[i] = (x > threshold) ? x : 0.0f;
  }
}

int main() {
  const float threshold = 0.5f;
  const size_t bytes = N * sizeof(float);
  
  float *h_x, *h_y;
  cudaMallocHost(&h_x, bytes);
  cudaMallocHost(&h_y, bytes);
  for (int i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_x, *d_y;
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);
  cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  
  // Warmup
  for (int i = 0; i < 5; ++i) {
    threshold_predicated<<<grid, block>>>(d_x, d_y, threshold, N);
  }
  cudaDeviceSynchronize();

  // Benchmark - multiple iterations for stable measurement
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  const int iterations = 100;
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    threshold_predicated<<<grid, block>>>(d_x, d_y, threshold, N);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  float time;
  cudaEventElapsedTime(&time, start, stop);
  float avg_time = time / iterations;

  cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
  
  printf("=== Optimized: Predicated Threshold (No Divergence) ===\n");
  printf("Array size: %d elements (%.1f MB)\n", N, bytes / 1e6);
  printf("Average kernel time: %.3f ms\n", avg_time);
  printf("Optimization: Branch-free predicated execution\n");
  printf("Benefit: Higher warp efficiency, no divergence\n");

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  return 0;
}
