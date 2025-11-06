// kernel_fusion_kernels.cu - CUDA kernels for kernel fusion benchmarks
// Can be loaded as PyTorch CUDA extension

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include "profiling_helpers.cuh"

namespace {

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        TORCH_CHECK(_status == cudaSuccess,                                    \
                    "CUDA error at ", __FILE__, ":", __LINE__, " - ",          \
                    cudaGetErrorString(_status));                              \
    } while (0)

} // anonymous namespace

// Separate kernels (baseline): Multiple kernel launches
__global__ void kernel_add(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}

__global__ void kernel_multiply(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernel_sqrt(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
}

// Fused kernel (optimized): Single kernel combining all operations
__global__ void kernel_fused(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Fused: add + multiply + sqrt in one pass
        float val = data[idx];
        val = val + 1.0f;      // Op 1
        val = val * 2.0f;      // Op 2 (uses register, not global memory)
        val = sqrtf(val);      // Op 3 (uses register, not global memory)
        data[idx] = val;       // Single write to global memory
    }
}

void separate_kernels(torch::Tensor data, int iterations) {
    TORCH_CHECK(data.is_cuda(), "data must be CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    
    // Ensure we're on the correct device
    int device_id = data.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    int n = data.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr) - this is the legacy default stream
    // PyTorch operations on default stream will be properly synchronized
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("separate_kernels");
        for (int i = 0; i < iterations; ++i) {
            kernel_add<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
            kernel_multiply<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
            kernel_sqrt<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    // Note: No explicit synchronization here - PyTorch benchmark harness handles synchronization
}

void fused_kernel(torch::Tensor data, int iterations) {
    TORCH_CHECK(data.is_cuda(), "data must be CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    
    // Ensure we're on the correct device
    int device_id = data.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    int n = data.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Use default stream (nullptr) - this is the legacy default stream
    // PyTorch operations on default stream will be properly synchronized
    cudaStream_t stream = nullptr;
    
    {
        PROFILE_KERNEL_LAUNCH("fused_kernel");
        for (int i = 0; i < iterations; ++i) {
            kernel_fused<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    // Note: No explicit synchronization here - PyTorch benchmark harness handles synchronization
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("separate_kernels", &separate_kernels, "Separate kernel launches (baseline)");
    m.def("fused_kernel", &fused_kernel, "Fused kernel (optimized)");
}

