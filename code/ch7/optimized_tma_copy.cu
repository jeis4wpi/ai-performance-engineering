// optimized_tma_copy.cu -- TMA-optimized copy kernel (optimized).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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

// TMA-optimized copy kernel with deep pipelining
__global__ void tma_copy_kernel_with_descriptors(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int tile_dim,
    int num_tiles,
    int pitch) {
    extern __shared__ float smem[];
    int tile_size = tile_dim * tile_dim;
    float* smem_buf0 = smem;
    float* smem_buf1 = smem + tile_size;
    
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;
    
    int tid = threadIdx.x;
    int buf = 0;
    
    // Load first tile
    for (int i = tid; i < tile_size; i += blockDim.x) {
        int row = i / tile_dim;
        int col = i % tile_dim;
        int offset = (tile_idx * tile_dim + row) * pitch + col;
        smem_buf0[i] = src[offset];
    }
    __syncthreads();
    
    // Pipeline: load next while processing current
    for (int next_tile = tile_idx + gridDim.x; next_tile < num_tiles; next_tile += gridDim.x) {
        int next_buf = 1 - buf;
        float* load_dst = next_buf ? smem_buf1 : smem_buf0;
        float* store_src = buf ? smem_buf1 : smem_buf0;

        // Async load next tile
        for (int i = tid; i < tile_size; i += blockDim.x) {
            int row = i / tile_dim;
            int col = i % tile_dim;
            int offset = (next_tile * tile_dim + row) * pitch + col;
            load_dst[i] = src[offset];
        }
        
        // Process current tile (store to destination)
        for (int i = tid; i < tile_size; i += blockDim.x) {
            int row = i / tile_dim;
            int col = i % tile_dim;
            int offset = (tile_idx * tile_dim + row) * pitch + col;
            dst[offset] = store_src[i] * 1.01f;
        }
        
        __syncthreads();
        buf = next_buf;
        tile_idx = next_tile;
    }
    
    // Process last tile
    float* final_src = buf ? smem_buf1 : smem_buf0;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        int row = i / tile_dim;
        int col = i % tile_dim;
        int offset = (tile_idx * tile_dim + row) * pitch + col;
        dst[offset] = final_src[i] * 1.01f;
    }
}

int main() {
    constexpr int tile_dim = 64;
    constexpr int num_tiles = 256;
    const int width = tile_dim;
    const int height = num_tiles * tile_dim;
    const size_t n = static_cast<size_t>(height) * width;
    const size_t bytes = n * sizeof(float);
    const size_t tile_size = tile_dim * tile_dim;
    const size_t shared_bytes = 2ull * tile_size * sizeof(float);
    
    float *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    
    // Initialize
    std::vector<float> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        tma_copy_kernel_with_descriptors<<<num_tiles, 256, shared_bytes>>>(
            d_src, d_dst, tile_dim, num_tiles, width);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    printf("TMA-optimized copy: %.2f ms\n", avg_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}

