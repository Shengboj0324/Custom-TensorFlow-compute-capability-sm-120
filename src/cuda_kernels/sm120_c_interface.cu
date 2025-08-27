/*
 * C Interface Implementation for SM120 CUDA Kernels
 * 
 * This file implements the C interface functions that launch the pure CUDA kernels.
 */

#include "sm120_c_interface.h"
#include "sm120_pure_kernels.cu"
#include "sm120_rtx50_verification.cu"
#include <algorithm>

// ============================================================================
// Helper Functions
// ============================================================================

static dim3 make_block_size(int size, int max_threads) {
    int threads = std::min(size, max_threads);
    if (threads <= 32) return dim3(threads, 1, 1);
    if (threads <= 256) return dim3(16, threads / 16, 1);
    return dim3(32, threads / 32, 1);
}

static dim3 make_grid_size(int size, dim3 block) {
    int blocks = (size + block.x * block.y * block.z - 1) / (block.x * block.y * block.z);
    return dim3(std::min(blocks, 65535), 1, 1);
}

// ============================================================================
// Matrix Operations Implementation
// ============================================================================

extern "C" cudaError_t sm120_launch_matmul(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    SM120DataType dtype,
    cudaStream_t stream) {
    
    const int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_matmul_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(A),
            static_cast<const float*>(B),
            static_cast<float*>(C),
            M, N, K, alpha, beta);
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_matmul_kernel<half><<<grid, block, 0, stream>>>(
            static_cast<const half*>(A),
            static_cast<const half*>(B),
            static_cast<half*>(C),
            M, N, K, alpha, beta);
    } else {
        return cudaErrorInvalidValue;
    }
    
    return cudaGetLastError();
}

extern "C" cudaError_t sm120_launch_transpose(
    const void* input, void* output,
    int rows, int cols,
    SM120DataType dtype,
    cudaStream_t stream) {
    
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_transpose_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            rows, cols);
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_transpose_kernel<half><<<grid, block, 0, stream>>>(
            static_cast<const half*>(input),
            static_cast<half*>(output),
            rows, cols);
    } else {
        return cudaErrorInvalidValue;
    }
    
    return cudaGetLastError();
}

// ============================================================================
// Convolution Operations Implementation
// ============================================================================

extern "C" cudaError_t sm120_launch_conv2d(
    const void* input, const void* filter, void* output,
    int batch_size,
    int input_height, int input_width, int input_channels,
    int output_height, int output_width, int output_channels,
    int filter_height, int filter_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    SM120DataType dtype,
    cudaStream_t stream) {
    
    dim3 block(16, 16, 4);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size
    );
    
    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_conv2d_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(filter),
            static_cast<float*>(output),
            batch_size, input_height, input_width, input_channels,
            output_height, output_width, output_channels,
            filter_height, filter_width,
            stride_h, stride_w, pad_h, pad_w);
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_conv2d_kernel<half><<<grid, block, 0, stream>>>(
            static_cast<const half*>(input),
            static_cast<const half*>(filter),
            static_cast<half*>(output),
            batch_size, input_height, input_width, input_channels,
            output_height, output_width, output_channels,
            filter_height, filter_width,
            stride_h, stride_w, pad_h, pad_w);
    } else {
        return cudaErrorInvalidValue;
    }
    
    return cudaGetLastError();
}

// ============================================================================
// Activation Functions Implementation
// ============================================================================

extern "C" cudaError_t sm120_launch_activation(
    const void* input, void* output,
    int size,
    SM120ActivationType activation_type,
    SM120DataType dtype,
    cudaStream_t stream) {
    
    dim3 block = make_block_size(size, 1024);
    dim3 grid = make_grid_size(size, block);
    
    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_activation_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size, static_cast<int>(activation_type));
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_activation_kernel<half><<<grid, block, 0, stream>>>(
            static_cast<const half*>(input),
            static_cast<half*>(output),
            size, static_cast<int>(activation_type));
    } else {
        return cudaErrorInvalidValue;
    }
    
    return cudaGetLastError();
}

// ============================================================================
// Reduction Operations Implementation
// ============================================================================

extern "C" cudaError_t sm120_launch_reduction(
    const void* input, void* output,
    int size,
    SM120ReductionType reduction_type,
    SM120DataType dtype,
    cudaStream_t stream) {
    
    dim3 block(256);
    dim3 grid(std::min((size + block.x - 1) / block.x, 65535));
    
    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_reduction_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size, static_cast<int>(reduction_type));
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_reduction_kernel<half><<<grid, block, 0, stream>>>(
            static_cast<const half*>(input),
            static_cast<half*>(output),
            size, static_cast<int>(reduction_type));
    } else {
        return cudaErrorInvalidValue;
    }
    
    return cudaGetLastError();
}

// ============================================================================
// Normalization Operations Implementation
// ============================================================================

extern "C" cudaError_t sm120_launch_layer_norm(
    const void* input, const void* gamma, const void* beta,
    void* output, void* mean, void* variance,
    int batch_size, int feature_size,
    float epsilon,
    SM120DataType dtype,
    cudaStream_t stream) {
    
    dim3 block(256);
    dim3 grid(batch_size);

    // Dynamic shared memory size: block.x floats for reduction
    size_t shared_mem_size = block.x * sizeof(float);

    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_layer_norm_kernel<float><<<grid, block, shared_mem_size, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(gamma),
            static_cast<const float*>(beta),
            static_cast<float*>(output),
            static_cast<float*>(mean),
            static_cast<float*>(variance),
            batch_size, feature_size, epsilon);
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_layer_norm_kernel<half><<<grid, block, shared_mem_size, stream>>>(
            static_cast<const half*>(input),
            static_cast<const half*>(gamma),
            static_cast<const half*>(beta),
            static_cast<half*>(output),
            static_cast<half*>(mean),
            static_cast<half*>(variance),
            batch_size, feature_size, epsilon);
    } else {
        return cudaErrorInvalidValue;
    }
    
    return cudaGetLastError();
}

// ============================================================================
// Attention Operations Implementation
// ============================================================================

extern "C" cudaError_t sm120_launch_attention(
    const void* queries, const void* keys, const void* values,
    void* output, float* attention_weights,
    int batch_size, int seq_len, int head_dim,
    float scale,
    SM120DataType dtype,
    cudaStream_t stream) {

    dim3 block(128);
    dim3 grid(1, seq_len, batch_size);

    size_t shared_mem_size = seq_len * sizeof(float);

    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_attention_kernel<float><<<grid, block, shared_mem_size, stream>>>(
            static_cast<const float*>(queries),
            static_cast<const float*>(keys),
            static_cast<const float*>(values),
            static_cast<float*>(output),
            attention_weights,
            batch_size, seq_len, head_dim, scale);
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_attention_kernel<half><<<grid, block, shared_mem_size, stream>>>(
            static_cast<const half*>(queries),
            static_cast<const half*>(keys),
            static_cast<const half*>(values),
            static_cast<half*>(output),
            attention_weights,
            batch_size, seq_len, head_dim, scale);
    } else {
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

extern "C" bool sm120_is_supported(int device_id) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);

    if (error != cudaSuccess) {
        return false;
    }

    return (prop.major == 12 && prop.minor == 0);
}

extern "C" bool sm120_verify_rtx50_compatibility(int device_id) {
    return VerifyRTX50Performance(device_id);
}

extern "C" const char* sm120_get_rtx50_report(int device_id) {
    return GetRTX50CompatibilityReport(device_id);
}

extern "C" void sm120_get_optimal_block_size(int size, int max_threads,
                                            int* block_x, int* block_y, int* block_z) {
    int threads = std::min(size, max_threads);

    if (threads <= 32) {
        *block_x = threads;
        *block_y = 1;
        *block_z = 1;
    } else if (threads <= 256) {
        *block_x = 16;
        *block_y = threads / 16;
        *block_z = 1;
    } else {
        *block_x = 32;
        *block_y = threads / 32;
        *block_z = 1;
    }
}

extern "C" void sm120_get_optimal_grid_size(int size, int block_x, int block_y, int block_z,
                                           int* grid_x, int* grid_y, int* grid_z) {
    int total_threads = block_x * block_y * block_z;
    int blocks = (size + total_threads - 1) / total_threads;

    *grid_x = std::min(blocks, 65535);
    *grid_y = 1;
    *grid_z = 1;
}

// ============================================================================
// Advanced Operations Implementation
// ============================================================================

extern "C" cudaError_t sm120_launch_batch_norm(
    const void* input, const void* scale, const void* offset,
    const void* estimated_mean, const void* estimated_variance,
    void* output,
    int batch_size, int height, int width, int channels,
    float epsilon,
    SM120DataType dtype,
    cudaStream_t stream) {

    int total_elements = batch_size * height * width * channels;
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);

    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_batch_norm_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(scale),
            static_cast<const float*>(offset),
            static_cast<const float*>(estimated_mean),
            static_cast<const float*>(estimated_variance),
            static_cast<float*>(output),
            batch_size, height, width, channels, epsilon);
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_batch_norm_kernel<half><<<grid, block, 0, stream>>>(
            static_cast<const half*>(input),
            static_cast<const half*>(scale),
            static_cast<const half*>(offset),
            static_cast<const half*>(estimated_mean),
            static_cast<const half*>(estimated_variance),
            static_cast<half*>(output),
            batch_size, height, width, channels, epsilon);
    } else {
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

extern "C" cudaError_t sm120_launch_softmax(
    const void* logits, void* output,
    int outer_size, int axis_size, int inner_size,
    SM120DataType dtype,
    cudaStream_t stream) {

    dim3 block(256);
    dim3 grid(outer_size, inner_size);

    // Dynamic shared memory: block.x + 2 floats (data + max_val + sum_exp)
    size_t shared_mem_size = (block.x + 2) * sizeof(float);

    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_softmax_kernel<float><<<grid, block, shared_mem_size, stream>>>(
            static_cast<const float*>(logits),
            static_cast<float*>(output),
            outer_size, axis_size, inner_size);
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_softmax_kernel<half><<<grid, block, shared_mem_size, stream>>>(
            static_cast<const half*>(logits),
            static_cast<half*>(output),
            outer_size, axis_size, inner_size);
    } else {
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

extern "C" cudaError_t sm120_launch_embedding_lookup(
    const int* ids, const void* params, void* output,
    int num_ids, int vocab_size, int embed_dim,
    SM120DataType dtype,
    cudaStream_t stream) {

    dim3 block(256);
    dim3 grid(num_ids, (embed_dim + block.x - 1) / block.x);

    if (dtype == SM120_DTYPE_FLOAT32) {
        sm120_embedding_lookup_kernel<float><<<grid, block, 0, stream>>>(
            ids,
            static_cast<const float*>(params),
            static_cast<float*>(output),
            num_ids, vocab_size, embed_dim);
    } else if (dtype == SM120_DTYPE_FLOAT16) {
        sm120_embedding_lookup_kernel<half><<<grid, block, 0, stream>>>(
            ids,
            static_cast<const half*>(params),
            static_cast<half*>(output),
            num_ids, vocab_size, embed_dim);
    } else {
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

extern "C" const char* sm120_get_error_string(cudaError_t error) {
    return cudaGetErrorString(error);
}
