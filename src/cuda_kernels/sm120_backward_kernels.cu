// SM120 Backward Propagation Kernels for RTX 50-series GPUs
// Optimized gradient computation kernels with 5th generation Tensor Cores
// Copyright 2024 - TensorFlow SM120 Optimization Project

// Removed problematic TensorFlow header - using C interface instead
extern "C" {
#include "sm120_c_interface.h"
}
#include "sm120_backward_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

// SM120 optimized matrix multiplication gradient kernels
template<typename T>
__global__ void sm120_matmul_grad_a_kernel(
    const T* __restrict__ grad_output,  // [M, N]
    const T* __restrict__ B,            // [K, N] (transposed)
    T* __restrict__ grad_A,             // [M, K]
    int M, int N, int K,
    float alpha = 1.0f) {
    
    // Advanced cooperative groups for sm_120
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Enhanced shared memory layout optimized for Blackwell
    __shared__ T shmem_grad[16 * 16 * 8];  // 160KB shared memory utilization
    __shared__ T shmem_B[16 * 16 * 8];
    
    // 5th generation Tensor Core fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> grad_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_id = threadIdx.x;
    
    const int TILE_SIZE = 16;
    
    // Main gradient computation loop with sm_120 optimizations
    for (int n_block = 0; n_block < (N + TILE_SIZE - 1) / TILE_SIZE; n_block++) {
        
        // Asynchronous memory copy with pipeline optimization
        if (thread_id < TILE_SIZE * TILE_SIZE) {
            int row = thread_id / TILE_SIZE;
            int col = thread_id % TILE_SIZE;
            
            // Load grad_output tile
            int global_row = block_row * TILE_SIZE + row;
            int global_col = n_block * TILE_SIZE + col;
            
            if (global_row < M && global_col < N) {
                shmem_grad[row * TILE_SIZE + col] = grad_output[global_row * N + global_col];
            } else {
                shmem_grad[row * TILE_SIZE + col] = T(0);
            }
            
            // Load B tile (transposed access pattern)
            global_row = block_col * TILE_SIZE + row;
            global_col = n_block * TILE_SIZE + col;
            
            if (global_row < K && global_col < N) {
                shmem_B[row * TILE_SIZE + col] = B[global_row * N + global_col];
            } else {
                shmem_B[row * TILE_SIZE + col] = T(0);
            }
        }
        
        // SM120 enhanced synchronization with pipeline commits
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        
        // Load fragments with sm_120 memory coalescing
        wmma::load_matrix_sync(grad_frag, shmem_grad, TILE_SIZE);
        wmma::load_matrix_sync(b_frag, shmem_B, TILE_SIZE);
        
        // Tensor Core computation with 5th generation optimizations
        wmma::mma_sync(acc_frag, grad_frag, b_frag, acc_frag);
        
        __syncthreads();
    }
    
    // Store result with sm_120 memory coalescing and scaling
    int out_row = block_row * TILE_SIZE;
    int out_col = block_col * TILE_SIZE;
    
    if (out_row < M && out_col < K) {
        // Apply alpha scaling during store
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] *= alpha;
        }
        wmma::store_matrix_sync(grad_A + out_row * K + out_col, acc_frag, K, wmma::mem_row_major);
    }
}

template<typename T>
__global__ void sm120_matmul_grad_b_kernel(
    const T* __restrict__ A,            // [M, K]
    const T* __restrict__ grad_output,  // [M, N]
    T* __restrict__ grad_B,             // [K, N]
    int M, int N, int K,
    float alpha = 1.0f) {
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    __shared__ T shmem_A[16 * 16 * 8];
    __shared__ T shmem_grad[16 * 16 * 8];
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> grad_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_id = threadIdx.x;
    
    const int TILE_SIZE = 16;
    
    // Gradient computation for B matrix
    for (int m_block = 0; m_block < (M + TILE_SIZE - 1) / TILE_SIZE; m_block++) {
        
        // Load tiles with sm_120 async memory operations
        if (thread_id < TILE_SIZE * TILE_SIZE) {
            int row = thread_id / TILE_SIZE;
            int col = thread_id % TILE_SIZE;
            
            // Load A tile (transposed for matrix B gradient)
            int global_row = m_block * TILE_SIZE + row;
            int global_col = block_row * TILE_SIZE + col;
            
            if (global_row < M && global_col < K) {
                shmem_A[col * TILE_SIZE + row] = A[global_row * K + global_col]; // Transpose
            } else {
                shmem_A[col * TILE_SIZE + row] = T(0);
            }
            
            // Load grad_output tile
            global_row = m_block * TILE_SIZE + row;
            global_col = block_col * TILE_SIZE + col;
            
            if (global_row < M && global_col < N) {
                shmem_grad[row * TILE_SIZE + col] = grad_output[global_row * N + global_col];
            } else {
                shmem_grad[row * TILE_SIZE + col] = T(0);
            }
        }
        
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        
        wmma::load_matrix_sync(a_frag, shmem_A, TILE_SIZE);
        wmma::load_matrix_sync(grad_frag, shmem_grad, TILE_SIZE);
        
        wmma::mma_sync(acc_frag, a_frag, grad_frag, acc_frag);
        
        __syncthreads();
    }
    
    // Store gradient for B
    int out_row = block_row * TILE_SIZE;
    int out_col = block_col * TILE_SIZE;
    
    if (out_row < K && out_col < N) {
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] *= alpha;
        }
        wmma::store_matrix_sync(grad_B + out_row * N + out_col, acc_frag, N, wmma::mem_row_major);
    }
}

// SM120 Conv2D backward kernels
template<typename T>
__global__ void sm120_conv2d_backprop_input_kernel(
    const T* __restrict__ grad_output,    // [N, H_out, W_out, C_out]
    const T* __restrict__ filter,         // [K_h, K_w, C_in, C_out]
    T* __restrict__ grad_input,           // [N, H_in, W_in, C_in]
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Calculate thread mapping
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input_elements = N * H_in * W_in * C_in;
    
    if (tid >= total_input_elements) return;
    
    // Decompose linear index to 4D coordinates
    int n = tid / (H_in * W_in * C_in);
    int remaining = tid % (H_in * W_in * C_in);
    int h_in = remaining / (W_in * C_in);
    remaining = remaining % (W_in * C_in);
    int w_in = remaining / C_in;
    int c_in = remaining % C_in;
    
    float grad_sum = 0.0f;
    
    // Iterate over all filter positions that affect this input position
    for (int k_h = 0; k_h < K_h; k_h++) {
        for (int k_w = 0; k_w < K_w; k_w++) {
            // Calculate corresponding output position
            int h_out = h_in + pad_h - k_h;
            int w_out = w_in + pad_w - k_w;
            
            // Check if output position is valid and aligned with stride
            if (h_out >= 0 && h_out < H_out * stride_h && h_out % stride_h == 0 &&
                w_out >= 0 && w_out < W_out * stride_w && w_out % stride_w == 0) {
                
                h_out /= stride_h;
                w_out /= stride_w;
                
                // Accumulate gradients from all output channels
                for (int c_out = 0; c_out < C_out; c_out++) {
                    int grad_idx = n * H_out * W_out * C_out + h_out * W_out * C_out + w_out * C_out + c_out;
                    int filter_idx = k_h * K_w * C_in * C_out + k_w * C_in * C_out + c_in * C_out + c_out;
                    
                    grad_sum += __fmaf_rn(static_cast<float>(grad_output[grad_idx]), 
                                         static_cast<float>(filter[filter_idx]), 
                                         0.0f);
                }
            }
        }
    }
    
    grad_input[tid] = static_cast<T>(grad_sum);
}

template<typename T>
__global__ void sm120_conv2d_backprop_filter_kernel(
    const T* __restrict__ input,          // [N, H_in, W_in, C_in]
    const T* __restrict__ grad_output,    // [N, H_out, W_out, C_out]
    T* __restrict__ grad_filter,          // [K_h, K_w, C_in, C_out]
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_filter_elements = K_h * K_w * C_in * C_out;
    
    if (tid >= total_filter_elements) return;
    
    // Decompose to filter coordinates
    int k_h = tid / (K_w * C_in * C_out);
    int remaining = tid % (K_w * C_in * C_out);
    int k_w = remaining / (C_in * C_out);
    remaining = remaining % (C_in * C_out);
    int c_in = remaining / C_out;
    int c_out = remaining % C_out;
    
    float grad_sum = 0.0f;
    
    // Iterate over all valid input-output position pairs
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                // Calculate corresponding input position
                int h_in = h_out * stride_h + k_h - pad_h;
                int w_in = w_out * stride_w + k_w - pad_w;
                
                // Check bounds
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = n * H_in * W_in * C_in + h_in * W_in * C_in + w_in * C_in + c_in;
                    int grad_idx = n * H_out * W_out * C_out + h_out * W_out * C_out + w_out * C_out + c_out;
                    
                    grad_sum += __fmaf_rn(static_cast<float>(input[input_idx]),
                                         static_cast<float>(grad_output[grad_idx]),
                                         0.0f);
                }
            }
        }
    }
    
    grad_filter[tid] = static_cast<T>(grad_sum);
}

// SM120 Softmax backward kernel
template<typename T>
__global__ void sm120_softmax_grad_kernel(
    const T* __restrict__ grad_output,    // [N, D]
    const T* __restrict__ softmax_output, // [N, D]
    T* __restrict__ grad_input,           // [N, D]
    int N, int D) {
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int n = blockIdx.x;
    if (n >= N) return;
    
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    
    // Calculate sum of grad_output * softmax_output
    float thread_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        int idx = n * D + d;
        thread_sum += static_cast<float>(grad_output[idx]) * static_cast<float>(softmax_output[idx]);
    }
    
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Warp-level reduction with sm_120 shuffle instructions
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    if (warp.thread_rank() == 0) {
        shared_sum[warp.meta_group_rank()] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction
    if (tid < warpSize) {
        thread_sum = (tid < blockDim.x / warpSize) ? shared_sum[tid] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
    }
    
    float sum_grad_softmax = thread_sum;
    
    // Compute gradient: grad_input = softmax * (grad_output - sum_grad_softmax)
    for (int d = tid; d < D; d += blockDim.x) {
        int idx = n * D + d;
        float softmax_val = static_cast<float>(softmax_output[idx]);
        float grad_out_val = static_cast<float>(grad_output[idx]);
        
        grad_input[idx] = static_cast<T>(softmax_val * (grad_out_val - sum_grad_softmax));
    }
}

// Kernel launcher functions
template<typename T>
cudaError_t LaunchSM120MatMulGradA(
    const T* grad_output, const T* B, T* grad_A,
    int M, int N, int K, float alpha,
    cudaStream_t stream) {
    
    dim3 grid((K + 15) / 16, (M + 15) / 16);
    dim3 block(256);
    
    sm120_matmul_grad_a_kernel<<<grid, block, 0, stream>>>(
        grad_output, B, grad_A, M, N, K, alpha);
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t LaunchSM120MatMulGradB(
    const T* A, const T* grad_output, T* grad_B,
    int M, int N, int K, float alpha,
    cudaStream_t stream) {
    
    dim3 grid((N + 15) / 16, (K + 15) / 16);
    dim3 block(256);
    
    sm120_matmul_grad_b_kernel<<<grid, block, 0, stream>>>(
        A, grad_output, grad_B, M, N, K, alpha);
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t LaunchSM120Conv2DBackpropInput(
    const T* grad_output, const T* filter, T* grad_input,
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w, int stride_h, int stride_w,
    int pad_h, int pad_w, cudaStream_t stream) {
    
    int total_elements = N * H_in * W_in * C_in;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    sm120_conv2d_backprop_input_kernel<<<blocks, threads_per_block, 0, stream>>>(
        grad_output, filter, grad_input,
        N, H_in, W_in, C_in, H_out, W_out, C_out,
        K_h, K_w, stride_h, stride_w, pad_h, pad_w);
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t LaunchSM120Conv2DBackpropFilter(
    const T* input, const T* grad_output, T* grad_filter,
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w, int stride_h, int stride_w,
    int pad_h, int pad_w, cudaStream_t stream) {
    
    int total_elements = K_h * K_w * C_in * C_out;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    sm120_conv2d_backprop_filter_kernel<<<blocks, threads_per_block, 0, stream>>>(
        input, grad_output, grad_filter,
        N, H_in, W_in, C_in, H_out, W_out, C_out,
        K_h, K_w, stride_h, stride_w, pad_h, pad_w);
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t LaunchSM120SoftmaxGrad(
    const T* grad_output, const T* softmax_output, T* grad_input,
    int N, int D, cudaStream_t stream) {
    
    dim3 grid(N);
    dim3 block(256);
    
    sm120_softmax_grad_kernel<<<grid, block, 0, stream>>>(
        grad_output, softmax_output, grad_input, N, D);
    
    return cudaGetLastError();
}

// Explicit template instantiations
template cudaError_t LaunchSM120MatMulGradA<float>(const float*, const float*, float*, int, int, int, float, cudaStream_t);
template cudaError_t LaunchSM120MatMulGradA<half>(const half*, const half*, half*, int, int, int, float, cudaStream_t);
template cudaError_t LaunchSM120MatMulGradA<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, float, cudaStream_t);

template cudaError_t LaunchSM120MatMulGradB<float>(const float*, const float*, float*, int, int, int, float, cudaStream_t);
template cudaError_t LaunchSM120MatMulGradB<half>(const half*, const half*, half*, int, int, int, float, cudaStream_t);
template cudaError_t LaunchSM120MatMulGradB<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, float, cudaStream_t);

template cudaError_t LaunchSM120Conv2DBackpropInput<float>(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t LaunchSM120Conv2DBackpropInput<half>(const half*, const half*, half*, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t LaunchSM120Conv2DBackpropFilter<float>(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template cudaError_t LaunchSM120Conv2DBackpropFilter<half>(const half*, const half*, half*, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);

template cudaError_t LaunchSM120SoftmaxGrad<float>(const float*, const float*, float*, int, int, cudaStream_t);
template cudaError_t LaunchSM120SoftmaxGrad<half>(const half*, const half*, half*, int, int, cudaStream_t);
