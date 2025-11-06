// cuda_graphs_kernels.cu - CUDA kernels for CUDA graphs benchmarks
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

__global__ void kernel_a_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = data[idx] * 1.1f + 0.1f;
}

__global__ void kernel_b_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
}

__global__ void kernel_c_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = sinf(data[idx] * 0.1f);
}

void separate_kernel_launches(torch::Tensor data, int iterations) {
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
        PROFILE_KERNEL_LAUNCH("separate_kernel_launches");
        for (int i = 0; i < iterations; ++i) {
            kernel_a_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
            kernel_b_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
            kernel_c_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    // Note: No explicit synchronization here - PyTorch benchmark harness handles synchronization
}

void graph_replay(torch::Tensor data, int iterations) {
    TORCH_CHECK(data.is_cuda(), "data must be CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "data must be float32");
    
    // Ensure we're on the correct device
    int device_id = data.device().index();
    CHECK_CUDA(cudaSetDevice(device_id));
    
    int n = data.size(0);
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Create a non-blocking stream for graph capture (required for CUDA graphs)
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    
    try {
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        kernel_a_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
        kernel_b_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
        kernel_c_kernel<<<num_blocks, threads_per_block, 0, stream>>>(data.data_ptr<float>(), n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        
        {
            PROFILE_KERNEL_LAUNCH("graph_replay");
            for (int i = 0; i < iterations; ++i) {
                CHECK_CUDA(cudaGraphLaunch(exec, stream));
            }
        }
        
        // Synchronize only the stream we're using (not the entire device)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    } catch (...) {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        throw;
    }
    
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("separate_kernel_launches", &separate_kernel_launches, "Separate kernel launches (baseline)");
    m.def("graph_replay", &graph_replay, "CUDA graph replay (optimized)");
}

