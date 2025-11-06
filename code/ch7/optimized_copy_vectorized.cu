// vectorized_copy.cu -- vectorized global load demonstrating CUDA 13 vector alignment.
//
// CUDA 13 Changes:
// - Explicit alignment required: use alignas(16) for 16-byte or alignas(32) for 32-byte
// - Blackwell supports 256-bit (32-byte) per-thread loads/stores (PTX 9.0 ld.global.v8.f32)
// - For older vector types, use *_16a or *_32a variants (e.g., double4_32a)
//
// Expected Runtime: <1 second (4 MB copy with timing)

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

// CUDA 13: Explicit 16-byte alignment for 4-element float vector
struct alignas(16) Float4 {
  float x, y, z, w;
};
static_assert(alignof(Float4) == 16, "Float4 must be 16-byte aligned");

// CUDA 13/Blackwell: 32-byte alignment for 8-element float vector
// Enables 256-bit per-thread loads on Blackwell (compute capability 10.0)
struct alignas(32) Float8 {
  float elems[8];
};
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

constexpr int NUM_FLOATS = 1 << 20;
static_assert(NUM_FLOATS % 8 == 0, "NUM_FLOATS must be divisible by 8");

// 16-byte aligned vectors (4 floats)
constexpr int NUM_VEC4 = NUM_FLOATS / 4;

// 32-byte aligned vectors (8 floats) - Blackwell optimization
constexpr int NUM_VEC8 = NUM_FLOATS / 8;

// Kernel using 16-byte aligned Float4 (pre-Blackwell and Blackwell compatible)
__global__ void copyVectorized4(const Float4* __restrict__ in,
                                Float4* __restrict__ out,
                                int n_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vec) {
    out[idx] = in[idx];  // Single 16-byte load/store per thread
  }
}

// Kernel using 32-byte aligned Float8 (Blackwell optimized - single 256-bit load)
__global__ void copyVectorized8(const Float8* __restrict__ in,
                                Float8* __restrict__ out,
                                int n_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vec) {
    // On Blackwell (sm_100), this compiles to a single 256-bit load/store
    // Pre-Blackwell: compiler will split into multiple 16-byte transactions
    out[idx] = in[idx];
  }
}

int main() {
  static_assert(sizeof(Float4) == 16, "Float4 must be 16 bytes");
  static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
  
  std::printf("CUDA 13 Vector Type Alignment Example\n");
  std::printf("======================================\n");
  std::printf("Float4 size: %zu bytes (16-byte aligned)\n", sizeof(Float4));
  std::printf("Float8 size: %zu bytes (32-byte aligned)\n\n", sizeof(Float8));
  
  // Get device properties
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  bool isBlackwell = (prop.major == 10 && prop.minor == 0);
  std::printf("GPU: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
  if (isBlackwell) {
    std::printf("Blackwell GPU detected - 256-bit loads available\n\n");
  } else {
    std::printf("Pre-Blackwell GPU - 256-bit loads will be split\n\n");
  }
  
  // Allocate and initialize host memory
  float* h_data = nullptr;
  float* h_result4 = nullptr;
  float* h_result8 = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_data, NUM_FLOATS * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_result4, NUM_FLOATS * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_result8, NUM_FLOATS * sizeof(float)));
  
  for (int i = 0; i < NUM_FLOATS; ++i) {
    h_data[i] = static_cast<float>(i);
  }
  
  // Test 1: Float4 (16-byte alignment)
  std::printf("Test 1: Float4 (16-byte aligned) copy\n");
  Float4* d_in4 = nullptr;
  Float4* d_out4 = nullptr;
  // cudaMalloc returns pointers aligned to >=256 bytes, satisfying Float4 alignment.
  CUDA_CHECK(cudaMalloc(&d_in4, NUM_VEC4 * sizeof(Float4)));
  CUDA_CHECK(cudaMalloc(&d_out4, NUM_VEC4 * sizeof(Float4)));
  CUDA_CHECK(cudaMemcpy(d_in4, h_data, NUM_FLOATS * sizeof(float), cudaMemcpyHostToDevice));
  
  dim3 block4(256);
  dim3 grid4((NUM_VEC4 + block4.x - 1) / block4.x);
  
  cudaEvent_t start4, stop4;
  CUDA_CHECK(cudaEventCreate(&start4));
  CUDA_CHECK(cudaEventCreate(&stop4));
  CUDA_CHECK(cudaEventRecord(start4));
  
  copyVectorized4<<<grid4, block4>>>(d_in4, d_out4, NUM_VEC4);
  
  CUDA_CHECK(cudaEventRecord(stop4));
  CUDA_CHECK(cudaEventSynchronize(stop4));
  float time4 = 0;
  CUDA_CHECK(cudaEventElapsedTime(&time4, start4, stop4));
  
  CUDA_CHECK(cudaMemcpy(h_result4, d_out4, NUM_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  Time: %.3f ms\n", time4);
  std::printf("  Result: out[0]=%.1f out[last]=%.1f\n\n", h_result4[0], h_result4[NUM_FLOATS - 1]);
  
  // Test 2: Float8 (32-byte alignment - Blackwell optimized)
  std::printf("Test 2: Float8 (32-byte aligned) copy\n");
  Float8* d_in8 = nullptr;
  Float8* d_out8 = nullptr;
  // cudaMalloc's >=256-byte alignment ensures Float8 allocations remain 32-byte aligned.
  CUDA_CHECK(cudaMalloc(&d_in8, NUM_VEC8 * sizeof(Float8)));
  CUDA_CHECK(cudaMalloc(&d_out8, NUM_VEC8 * sizeof(Float8)));
  CUDA_CHECK(cudaMemcpy(d_in8, h_data, NUM_FLOATS * sizeof(float), cudaMemcpyHostToDevice));
  
  dim3 block8(256);
  dim3 grid8((NUM_VEC8 + block8.x - 1) / block8.x);
  
  cudaEvent_t start8, stop8;
  CUDA_CHECK(cudaEventCreate(&start8));
  CUDA_CHECK(cudaEventCreate(&stop8));
  CUDA_CHECK(cudaEventRecord(start8));
  
  copyVectorized8<<<grid8, block8>>>(d_in8, d_out8, NUM_VEC8);
  
  CUDA_CHECK(cudaEventRecord(stop8));
  CUDA_CHECK(cudaEventSynchronize(stop8));
  float time8 = 0;
  CUDA_CHECK(cudaEventElapsedTime(&time8, start8, stop8));
  
  CUDA_CHECK(cudaMemcpy(h_result8, d_out8, NUM_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  Time: %.3f ms\n", time8);
  std::printf("  Result: out[0]=%.1f out[last]=%.1f\n\n", h_result8[0], h_result8[NUM_FLOATS - 1]);
  
  if (isBlackwell) {
    float speedup = time4 / time8;
    std::printf("Speedup with 32-byte alignment: %.2fx\n", speedup);
    std::printf("(On Blackwell, 32-byte loads reduce memory transactions)\n");
  }
  
  // Cleanup
  CUDA_CHECK(cudaFree(d_in4));
  CUDA_CHECK(cudaFree(d_out4));
  CUDA_CHECK(cudaFree(d_in8));
  CUDA_CHECK(cudaFree(d_out8));
  CUDA_CHECK(cudaFreeHost(h_data));
  CUDA_CHECK(cudaFreeHost(h_result4));
  CUDA_CHECK(cudaFreeHost(h_result8));
  CUDA_CHECK(cudaEventDestroy(start4));
  CUDA_CHECK(cudaEventDestroy(stop4));
  CUDA_CHECK(cudaEventDestroy(start8));
  CUDA_CHECK(cudaEventDestroy(stop8));
  
  std::printf("\nKey Takeaways (CUDA 13):\n");
  std::printf("- Use alignas(16) for 16-byte vectors (Float4, etc.)\n");
  std::printf("- Use alignas(32) for 32-byte vectors on Blackwell (Float8, double4_32a)\n");
  std::printf("- Blackwell's 256-bit loads require 32-byte alignment\n");
  std::printf("- Pre-CUDA 13: implicit alignment; CUDA 13+: explicit required\n");
  
  return 0;
}
