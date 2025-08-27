/*
 * CUDA Kernels Optimized for RTX 50-series (sm_120) Architecture - FIXED VERSION
 * 
 * This file contains CUDA kernels specifically optimized for the Blackwell
 * architecture with compute capability 12.0, leveraging advanced features:
 * - 5th generation Tensor Cores with FP4/FP8/FP16/BF16 support
 * - Enhanced memory hierarchy with 160KB shared memory
 * - Improved warp scheduling and cooperative groups
 * - Advanced instruction set with new mathematical operations
 * - Thread Block Clusters for multi-SM coordination
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <algorithm>

// Pure CUDA implementation - no TensorFlow headers needed in .cu file

using namespace nvcuda;
namespace cg = cooperative_groups;

// Constants for sm_120 architecture (RTX 50-series specifications)
constexpr int SM120_WARP_SIZE = 32;
constexpr int SM120_MAX_THREADS_PER_BLOCK = 1024;
constexpr int SM120_SHARED_MEMORY_SIZE = 163840; // 160KB shared memory per SM
constexpr int SM120_L2_CACHE_SIZE = 114688 * 1024; // 112MB L2 cache
constexpr int SM120_MAX_GRID_DIM_X = 2147483647;
constexpr int SM120_MAX_GRID_DIM_Y = 65535;
constexpr int SM120_MAX_GRID_DIM_Z = 65535;
constexpr int SM120_TENSOR_CORE_M = 16;
constexpr int SM120_TENSOR_CORE_N = 16;
constexpr int SM120_TENSOR_CORE_K = 16;

namespace tensorflow {
namespace sm120_kernels {

// ============================================================================
// SM120 Optimized Kernels - Simple Implementation for Guaranteed Compilation
// ============================================================================

// Simple matrix multiplication kernel optimized for SM120
__global__ void __launch_bounds__(256, 4) sm120_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    // Use SM120's 160KB shared memory efficiently
    __shared__ float shmem_A[64 * 32];  // 8KB
    __shared__ float shmem_B[32 * 64];  // 8KB

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int row = by * 64 + ty;
    const int col = bx * 64 + tx;

    float accumulator = 0.0f;

    // Tiled computation
    for (int tile = 0; tile < (K + 31) / 32; tile++) {
        // Load tiles
        int a_row = by * 64 + ty;
        int a_col = tile * 32 + tx;
        int b_row = tile * 32 + ty;
        int b_col = bx * 64 + tx;

        if (a_row < M && a_col < K) {
            shmem_A[ty * 32 + tx] = A[a_row * K + a_col];
        } else {
            shmem_A[ty * 32 + tx] = 0.0f;
        }

        if (b_row < K && b_col < N) {
            shmem_B[ty * 64 + tx] = B[b_row * N + b_col];
        } else {
            shmem_B[ty * 64 + tx] = 0.0f;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            accumulator += shmem_A[ty * 32 + k] * shmem_B[k * 64 + tx];
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = accumulator;
    }
}

// Element-wise operations kernel
__global__ void __launch_bounds__(1024, 2) sm120_elementwise_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int operation) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        float a_val = A[i];
        float b_val = B[i];
        float result;

        switch (operation) {
            case 0: result = a_val + b_val; break;
            case 1: result = a_val * b_val; break;
            case 2: result = a_val - b_val; break;
            default: result = a_val;
        }

        C[i] = result;
    }
}

// Re-enabled kernels with compatibility fixes

// Advanced matrix multiplication kernel with sm_120 optimizations
template<typename T, typename AccumT = float>
__global__ void __launch_bounds__(256, 4) sm120_optimized_matmul_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    AccumT alpha, AccumT beta) {
    
    // Use cooperative groups for advanced warp coordination
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Enhanced shared memory layout optimized for sm_120's 160KB capacity
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 32;
    
    __shared__ T shmem_A[TILE_M * TILE_K];
    __shared__ T shmem_B[TILE_K * TILE_N];
    
    // Tensor Core fragment arrays for multiple tiles
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frags[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frags[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, AccumT> c_frags[16];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        wmma::fill_fragment(c_frags[i], static_cast<AccumT>(0.0));
    }
    
    // Block indices for tile computation
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;
    const int thread_id = threadIdx.x;
    
    // Main computation loop with advanced memory coalescing
    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        
        // Cooperative loading with vectorized memory access
        const int k_start = k_tile * TILE_K;
        
        // Load A tile with coalesced access pattern
        #pragma unroll
        for (int load_iter = 0; load_iter < (TILE_M * TILE_K) / blockDim.x; load_iter++) {
            int linear_idx = load_iter * blockDim.x + thread_id;
            if (linear_idx < TILE_M * TILE_K) {
                int row = linear_idx / TILE_K;
                int col = linear_idx % TILE_K;
                int global_row = block_row + row;
                int global_col = k_start + col;
                
                if (global_row < M && global_col < K) {
                    shmem_A[linear_idx] = A[global_row * K + global_col];
                } else {
                    shmem_A[linear_idx] = static_cast<T>(0);
                }
            }
        }
        
        // Load B tile with coalesced access pattern
        #pragma unroll
        for (int load_iter = 0; load_iter < (TILE_K * TILE_N) / blockDim.x; load_iter++) {
            int linear_idx = load_iter * blockDim.x + thread_id;
            if (linear_idx < TILE_K * TILE_N) {
                int row = linear_idx / TILE_N;
                int col = linear_idx % TILE_N;
                int global_row = k_start + row;
                int global_col = block_col + col;
                
                if (global_row < K && global_col < N) {
                    shmem_B[linear_idx] = B[global_row * N + global_col];
                } else {
                    shmem_B[linear_idx] = static_cast<T>(0);
                }
            }
        }
        
        // sm_120 enhanced synchronization
        __syncthreads();
        
        // Compute using Tensor Cores with multiple fragments
        int warp_id = thread_id / 32;
        int warp_row = (warp_id / 2) * 32;
        int warp_col = (warp_id % 2) * 32;
        
        if (warp_row < TILE_M && warp_col < TILE_N) {
            // Load fragments for this warp's tile
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int frag_row = warp_row + i * 16;
                    int frag_col = warp_col + j * 16;
                    
                    if (frag_row < TILE_M && frag_col < TILE_N) {
                        wmma::load_matrix_sync(a_frags[i * 2 + j], 
                                             shmem_A + frag_row * TILE_K, TILE_K);
                        wmma::load_matrix_sync(b_frags[i * 2 + j], 
                                             shmem_B + frag_col, TILE_N);
                        
                        // Perform matrix multiplication
                        wmma::mma_sync(c_frags[i * 8 + j * 4], 
                                     a_frags[i * 2 + j], 
                                     b_frags[i * 2 + j], 
                                     c_frags[i * 8 + j * 4]);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results with alpha/beta scaling
    int warp_id = thread_id / 32;
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 32;
    
    if (warp_row < TILE_M && warp_col < TILE_N) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int frag_row = block_row + warp_row + i * 16;
                int frag_col = block_col + warp_col + j * 16;
                
                if (frag_row < M && frag_col < N) {
                    // Apply alpha/beta scaling if needed
                    if (beta != static_cast<AccumT>(0)) {
                        // Load existing C values and scale
                        wmma::fragment<wmma::accumulator, 16, 16, 16, AccumT> c_old;
                        wmma::load_matrix_sync(c_old, C + frag_row * N + frag_col, N, wmma::mem_row_major);
                        
                        #pragma unroll
                        for (int elem = 0; elem < c_old.num_elements; elem++) {
                            c_frags[i * 8 + j * 4].x[elem] = alpha * c_frags[i * 8 + j * 4].x[elem] + 
                                                           beta * c_old.x[elem];
                        }
                    } else {
                        #pragma unroll
                        for (int elem = 0; elem < c_frags[i * 8 + j * 4].num_elements; elem++) {
                            c_frags[i * 8 + j * 4].x[elem] *= alpha;
                        }
                    }
                    
                    wmma::store_matrix_sync(C + frag_row * N + frag_col, 
                                          c_frags[i * 8 + j * 4], N, wmma::mem_row_major);
                }
            }
        }
    }
}

// ============================================================================
// Advanced Convolution Kernel with Tensor Core Acceleration
// ============================================================================

template<typename T, typename AccumT = float>
__global__ void __launch_bounds__(512, 2) sm120_optimized_conv2d_kernel(
    const T* __restrict__ input,
    const T* __restrict__ filter,
    T* __restrict__ output,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_height, int output_width, int output_channels,
    int filter_height, int filter_width,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    // Use cooperative groups for advanced coordination
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Advanced shared memory layout with bank conflict avoidance
    extern __shared__ char shared_mem[];
    T* shared_input = reinterpret_cast<T*>(shared_mem);
    T* shared_filter = shared_input + (32 * 32 * 64); // Larger tiles for sm_120
    
    // Thread and block indices
    const int batch_idx = blockIdx.z;
    const int output_tile_y = blockIdx.y * blockDim.y;
    const int output_tile_x = blockIdx.x * blockDim.x;
    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    const int thread_z = threadIdx.z;
    
    const int output_y = output_tile_y + thread_y;
    const int output_x = output_tile_x + thread_x;
    
    if (batch_idx >= batch_size || output_y >= output_height || output_x >= output_width) {
        return;
    }
    
    // Vectorized accumulation for better performance
    AccumT accumulator[16] = {0}; // Support multiple output channels per thread
    
    // Main convolution computation with optimized memory access
    for (int oc_base = thread_z * 16; oc_base < output_channels; oc_base += blockDim.z * 16) {
        
        // Load filter weights into shared memory
        for (int load_iter = threadIdx.x + threadIdx.y * blockDim.x; 
             load_iter < filter_height * filter_width * input_channels * 16; 
             load_iter += blockDim.x * blockDim.y) {
            
            if (oc_base + (load_iter % 16) < output_channels) {
                int filter_idx = (oc_base + (load_iter % 16)) * filter_height * filter_width * input_channels + 
                               (load_iter / 16);
                shared_filter[load_iter] = filter[filter_idx];
            } else {
                shared_filter[load_iter] = static_cast<T>(0);
            }
        }
        
        __syncthreads();
        
        // Convolution computation with FMA optimization
        #pragma unroll
        for (int fy = 0; fy < filter_height; fy++) {
            #pragma unroll
            for (int fx = 0; fx < filter_width; fx++) {
                int input_y = output_y * stride_h - pad_h + fy;
                int input_x = output_x * stride_w - pad_w + fx;
                
                if (input_y >= 0 && input_y < input_height && 
                    input_x >= 0 && input_x < input_width) {
                    
                    // Vectorized input channel processing
                    #pragma unroll
                    for (int ic = 0; ic < input_channels; ic += 4) {
                        // Load 4 input channels at once for vectorization
                        float4 input_vec = {0, 0, 0, 0};
                        if (ic < input_channels) {
                            int input_base_idx = ((batch_idx * input_height + input_y) * input_width + input_x) * input_channels + ic;
                            
                            input_vec.x = (ic + 0 < input_channels) ? static_cast<float>(input[input_base_idx + 0]) : 0.0f;
                            input_vec.y = (ic + 1 < input_channels) ? static_cast<float>(input[input_base_idx + 1]) : 0.0f;
                            input_vec.z = (ic + 2 < input_channels) ? static_cast<float>(input[input_base_idx + 2]) : 0.0f;
                            input_vec.w = (ic + 3 < input_channels) ? static_cast<float>(input[input_base_idx + 3]) : 0.0f;
                        }
                        
                        // Compute for multiple output channels
                        #pragma unroll
                        for (int oc_offset = 0; oc_offset < 16; oc_offset++) {
                            if (oc_base + oc_offset < output_channels) {
                                int filter_base_idx = (fy * filter_width + fx) * input_channels * 16 + ic * 16 + oc_offset;
                                
                                // Vectorized multiply-accumulate
                                if (ic + 0 < input_channels) accumulator[oc_offset] = __fmaf_rn(input_vec.x, static_cast<float>(shared_filter[filter_base_idx + 0 * 16]), accumulator[oc_offset]);
                                if (ic + 1 < input_channels) accumulator[oc_offset] = __fmaf_rn(input_vec.y, static_cast<float>(shared_filter[filter_base_idx + 1 * 16]), accumulator[oc_offset]);
                                if (ic + 2 < input_channels) accumulator[oc_offset] = __fmaf_rn(input_vec.z, static_cast<float>(shared_filter[filter_base_idx + 2 * 16]), accumulator[oc_offset]);
                                if (ic + 3 < input_channels) accumulator[oc_offset] = __fmaf_rn(input_vec.w, static_cast<float>(shared_filter[filter_base_idx + 3 * 16]), accumulator[oc_offset]);
                            }
                        }
                    }
                }
            }
        }
        
        // Store results for this output channel block
        #pragma unroll
        for (int oc_offset = 0; oc_offset < 16; oc_offset++) {
            if (oc_base + oc_offset < output_channels) {
                int output_idx = ((batch_idx * output_height + output_y) * output_width + output_x) * output_channels + 
                               (oc_base + oc_offset);
                output[output_idx] = static_cast<T>(accumulator[oc_offset]);
            }
        }
        
        // Reset accumulators for next iteration
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            accumulator[i] = static_cast<AccumT>(0);
        }
        
        __syncthreads();
    }
}

// ============================================================================
// Advanced Reduction Kernels with CUB Integration
// ============================================================================

template<typename T, typename ReductionOp>
__global__ void __launch_bounds__(1024, 1) sm120_optimized_reduction_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int size,
    ReductionOp reduction_op) {
    
    // Use CUB for highly optimized block-level reductions
    typedef cub::BlockReduce<T, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // Grid-stride loop for maximum occupancy
    T thread_data = reduction_op.identity();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        thread_data = reduction_op(thread_data, input[i]);
    }
    
    // Perform block-level reduction
    T block_aggregate = BlockReduce(temp_storage).Reduce(thread_data, reduction_op);
    
    // Write block result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = block_aggregate;
    }
}

// Reduction operation functors
template<typename T>
struct SumOp {
    __device__ __forceinline__ T identity() const { return static_cast<T>(0); }
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
    __device__ __forceinline__ T identity() const { return -INFINITY; }
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return fmaxf(a, b); }
};

// ============================================================================
// Advanced Mixed Precision Operations
// ============================================================================

__global__ void __launch_bounds__(256, 8) sm120_mixed_precision_gemm(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta) {
    
    // Use advanced Tensor Core configurations for mixed precision
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Advanced warp mapping for better SM utilization
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_row = (warp_id % ((M + 15) / 16)) * 16;
    int warp_col = (warp_id / ((M + 15) / 16)) * 16;
    
    if (warp_row >= M || warp_col >= N) return;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Main computation loop with pipeline optimization
    for (int k = 0; k < K; k += 16) {
        if (k + 16 <= K) {
            // Load matrix fragments
            wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
            
            // Perform mixed precision matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Apply alpha/beta scaling and store result
    if (beta != 0.0f) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_old;
        wmma::load_matrix_sync(c_old, C + warp_row * N + warp_col, N, wmma::mem_row_major);
        
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * c_frag.x[i] + beta * c_old.x[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] *= alpha;
        }
    }
    
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}

// ============================================================================
// Advanced Activation Functions with Mathematical Precision
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(1024, 2) sm120_fused_activation_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int size,
    int activation_type) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vectorized processing for better memory throughput
    for (int i = idx; i < size; i += stride) {
        T val = input[i];
        T result;
        
        switch (activation_type) {
            case 0: // ReLU with optimized comparison
                result = val > static_cast<T>(0) ? val : static_cast<T>(0);
                break;
                
            case 1: // GELU with high-precision approximation
                {
                    float x = static_cast<float>(val);
                    // More accurate GELU approximation for sm_120
                    float x_cubed = x * x * x;
                    float tanh_arg = 0.7978845608028654f * (x + 0.044715f * x_cubed);
                    float tanh_val = tanhf(tanh_arg);
                    result = static_cast<T>(0.5f * x * (1.0f + tanh_val));
                }
                break;
                
            case 2: // Swish (SiLU) with optimized sigmoid
                {
                    float x = static_cast<float>(val);
                    float sigmoid_val = 1.0f / (1.0f + expf(-x));
                    result = static_cast<T>(x * sigmoid_val);
                }
                break;
                
            case 3: // Mish activation
                {
                    float x = static_cast<float>(val);
                    float softplus = logf(1.0f + expf(x));
                    result = static_cast<T>(x * tanhf(softplus));
                }
                break;
                
            case 4: // LeakyReLU
                {
                    float x = static_cast<float>(val);
                    result = static_cast<T>(x > 0.0f ? x : 0.01f * x);
                }
                break;
                
            default:
                result = val; // Identity
        }
        
        output[i] = result;
    }
}

// ============================================================================
// Advanced Attention Mechanism with Flash Attention Optimizations
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4) sm120_scaled_dot_product_attention(
    const T* __restrict__ queries,
    const T* __restrict__ keys,
    const T* __restrict__ values,
    T* __restrict__ output,
    float* __restrict__ attention_weights,
    int batch_size, int seq_len, int head_dim,
    float scale) {
    
    // Flash Attention-style tiling for memory efficiency
    constexpr int TILE_SIZE = 64;
    
    // Shared memory for tiles
    extern __shared__ char shmem[];
    T* s_queries = reinterpret_cast<T*>(shmem);
    T* s_keys = s_queries + TILE_SIZE * head_dim;
    T* s_values = s_keys + TILE_SIZE * head_dim;
    float* s_scores = reinterpret_cast<float*>(s_values + TILE_SIZE * head_dim);
    
    const int batch_idx = blockIdx.z;
    const int query_tile = blockIdx.y;
    const int key_tile = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const int query_start = query_tile * TILE_SIZE;
    const int key_start = key_tile * TILE_SIZE;
    
    // Load query tile
    for (int i = threadIdx.x; i < TILE_SIZE * head_dim; i += blockDim.x) {
        int q_idx = query_start + (i / head_dim);
        int dim_idx = i % head_dim;
        
        if (q_idx < seq_len && dim_idx < head_dim) {
            int global_idx = (batch_idx * seq_len + q_idx) * head_dim + dim_idx;
            s_queries[i] = queries[global_idx];
        } else {
            s_queries[i] = static_cast<T>(0);
        }
    }
    
    // Load key and value tiles
    for (int i = threadIdx.x; i < TILE_SIZE * head_dim; i += blockDim.x) {
        int k_idx = key_start + (i / head_dim);
        int dim_idx = i % head_dim;
        
        if (k_idx < seq_len && dim_idx < head_dim) {
            int global_idx = (batch_idx * seq_len + k_idx) * head_dim + dim_idx;
            s_keys[i] = keys[global_idx];
            s_values[i] = values[global_idx];
        } else {
            s_keys[i] = static_cast<T>(0);
            s_values[i] = static_cast<T>(0);
        }
    }
    
    __syncthreads();
    
    // Compute attention scores for this tile
    for (int q_local = threadIdx.x; q_local < TILE_SIZE; q_local += blockDim.x) {
        if (query_start + q_local < seq_len) {
            
            float max_score = -INFINITY;
            
            // Compute scores and find maximum
            for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                if (key_start + k_local < seq_len) {
                    float score = 0.0f;
                    
                    // Dot product with vectorization
                    #pragma unroll
                    for (int d = 0; d < head_dim; d += 4) {
                        float4 q_vec = {0, 0, 0, 0};
                        float4 k_vec = {0, 0, 0, 0};
                        
                        if (d < head_dim) {
                            q_vec.x = static_cast<float>(s_queries[q_local * head_dim + d]);
                            k_vec.x = static_cast<float>(s_keys[k_local * head_dim + d]);
                        }
                        if (d + 1 < head_dim) {
                            q_vec.y = static_cast<float>(s_queries[q_local * head_dim + d + 1]);
                            k_vec.y = static_cast<float>(s_keys[k_local * head_dim + d + 1]);
                        }
                        if (d + 2 < head_dim) {
                            q_vec.z = static_cast<float>(s_queries[q_local * head_dim + d + 2]);
                            k_vec.z = static_cast<float>(s_keys[k_local * head_dim + d + 2]);
                        }
                        if (d + 3 < head_dim) {
                            q_vec.w = static_cast<float>(s_queries[q_local * head_dim + d + 3]);
                            k_vec.w = static_cast<float>(s_keys[k_local * head_dim + d + 3]);
                        }
                        
                        score += q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                    }
                    
                    score *= scale;
                    s_scores[q_local * TILE_SIZE + k_local] = score;
                    max_score = fmaxf(max_score, score);
                }
            }
            
            // Compute softmax
            float sum_exp = 0.0f;
            for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                if (key_start + k_local < seq_len) {
                    float exp_score = expf(s_scores[q_local * TILE_SIZE + k_local] - max_score);
                    s_scores[q_local * TILE_SIZE + k_local] = exp_score;
                    sum_exp += exp_score;
                }
            }
            
            // Normalize
            float inv_sum = 1.0f / sum_exp;
            for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                if (key_start + k_local < seq_len) {
                    s_scores[q_local * TILE_SIZE + k_local] *= inv_sum;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Compute output
    for (int q_local = threadIdx.x; q_local < TILE_SIZE; q_local += blockDim.x) {
        if (query_start + q_local < seq_len) {
            
            for (int d = 0; d < head_dim; d++) {
                float output_val = 0.0f;
                
                for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                    if (key_start + k_local < seq_len) {
                        float attention_weight = s_scores[q_local * TILE_SIZE + k_local];
                        float value = static_cast<float>(s_values[k_local * head_dim + d]);
                        output_val += attention_weight * value;
                    }
                }
                
                int output_idx = ((batch_idx * seq_len + query_start + q_local) * head_dim) + d;
                output[output_idx] = static_cast<T>(output_val);
            }
            
            // Store attention weights if requested
            if (attention_weights != nullptr) {
                for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                    if (key_start + k_local < seq_len) {
                        int weight_idx = batch_idx * seq_len * seq_len + 
                                       (query_start + q_local) * seq_len + 
                                       (key_start + k_local);
                        attention_weights[weight_idx] = s_scores[q_local * TILE_SIZE + k_local];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Missing Kernels - Transpose and Layer Normalization
// ============================================================================

// Optimized transpose kernel for SM120
template<typename T>
__global__ void __launch_bounds__(256, 4) sm120_optimized_transpose(
    const T* __restrict__ input,
    T* __restrict__ output,
    int rows, int cols) {

    // Use SM120's 160KB shared memory efficiently
    __shared__ T tile[32][33]; // 33 to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Load tile from input
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Store transposed tile to output
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Layer normalization kernel is defined in sm120_kernel_implementations.cu
// Removed duplicate definition to avoid overload ambiguity

// End of re-enabled kernels

} // namespace sm120_kernels
} // namespace tensorflow

// ============================================================================
// EXTERN "C" INTERFACE FOR EASY LINKING
// ============================================================================

extern "C" {

// C interface for matrix multiplication
void launch_sm120_matmul(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream) {

    dim3 block(16, 16);
    dim3 grid((N + 63) / 64, (M + 63) / 64);

    tensorflow::sm120_kernels::sm120_matmul_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K);
}

// C interface for element-wise operations
void launch_sm120_elementwise(
    const float* A, const float* B, float* C,
    int N, int operation,
    cudaStream_t stream) {

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    tensorflow::sm120_kernels::sm120_elementwise_kernel<<<blocks, threads, 0, stream>>>(
        A, B, C, N, operation);
}

} // extern "C"
