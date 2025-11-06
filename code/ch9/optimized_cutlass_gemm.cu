// optimized_cutlass_gemm.cu -- Optimized GEMM kernel with shared memory (optimized).

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>

namespace cg = cooperative_groups;

// Optimized GEMM kernel with shared memory
__global__ void optimized_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta) {
    
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];
    cg::thread_block block = cg::this_thread_block();
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int k = 0; k < K; k += 32) {
        // Load tiles into shared memory
        if (row < M && k + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && k + threadIdx.y < K) {
            sB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        block.sync();
        
        // Compute partial dot product
        int tile_size = min(32, K - k);
        for (int i = 0; i < tile_size; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        
        block.sync();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

int main() {
    const int M = 256, N = 256, K = 256;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // Initialize matrices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (M + block_size.y - 1) / block_size.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        optimized_gemm_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_total;
    cudaEventElapsedTime(&time_total, start, stop);
    float time_avg = time_total / iterations;
    
    std::cout << "Optimized GEMM (with shared memory): " << time_avg << " ms" << std::endl;
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

