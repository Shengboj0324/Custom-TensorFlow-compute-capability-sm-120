/*
 * C Interface for SM120 CUDA Kernels
 * 
 * This header provides a pure C interface that can be called from any language
 * including C++, Python, and TensorFlow. No external dependencies.
 */

#ifndef SM120_C_INTERFACE_H
#define SM120_C_INTERFACE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type Definitions
// ============================================================================

typedef enum {
    SM120_ACTIVATION_RELU = 0,
    SM120_ACTIVATION_LEAKY_RELU = 1,
    SM120_ACTIVATION_ELU = 2,
    SM120_ACTIVATION_SELU = 3,
    SM120_ACTIVATION_GELU = 4,
    SM120_ACTIVATION_SWISH = 5,
    SM120_ACTIVATION_MISH = 6,
    SM120_ACTIVATION_TANH = 7,
    SM120_ACTIVATION_SIGMOID = 8
} SM120ActivationType;

typedef enum {
    SM120_REDUCTION_SUM = 0,
    SM120_REDUCTION_MEAN = 1,
    SM120_REDUCTION_MAX = 2,
    SM120_REDUCTION_MIN = 3
} SM120ReductionType;

typedef enum {
    SM120_DTYPE_FLOAT32 = 0,
    SM120_DTYPE_FLOAT16 = 1
} SM120DataType;

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * Launch SM120 matrix multiplication kernel
 * C = alpha * A * B + beta * C
 */
cudaError_t sm120_launch_matmul(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    SM120DataType dtype,
    cudaStream_t stream);

/**
 * Launch SM120 transpose kernel
 * output = transpose(input)
 */
cudaError_t sm120_launch_transpose(
    const void* input, void* output,
    int rows, int cols,
    SM120DataType dtype,
    cudaStream_t stream);

// ============================================================================
// Convolution Operations
// ============================================================================

/**
 * Launch SM120 2D convolution kernel
 */
cudaError_t sm120_launch_conv2d(
    const void* input, const void* filter, void* output,
    int batch_size,
    int input_height, int input_width, int input_channels,
    int output_height, int output_width, int output_channels,
    int filter_height, int filter_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    SM120DataType dtype,
    cudaStream_t stream);

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * Launch SM120 activation function kernel
 */
cudaError_t sm120_launch_activation(
    const void* input, void* output,
    int size,
    SM120ActivationType activation_type,
    SM120DataType dtype,
    cudaStream_t stream);

// ============================================================================
// Reduction Operations
// ============================================================================

/**
 * Launch SM120 reduction kernel
 */
cudaError_t sm120_launch_reduction(
    const void* input, void* output,
    int size,
    SM120ReductionType reduction_type,
    SM120DataType dtype,
    cudaStream_t stream);

// ============================================================================
// Normalization Operations
// ============================================================================

/**
 * Launch SM120 layer normalization kernel
 */
cudaError_t sm120_launch_layer_norm(
    const void* input, const void* gamma, const void* beta,
    void* output, void* mean, void* variance,
    int batch_size, int feature_size,
    float epsilon,
    SM120DataType dtype,
    cudaStream_t stream);

// ============================================================================
// Attention Operations
// ============================================================================

/**
 * Launch SM120 scaled dot-product attention kernel
 */
cudaError_t sm120_launch_attention(
    const void* queries, const void* keys, const void* values,
    void* output, float* attention_weights,
    int batch_size, int seq_len, int head_dim,
    float scale,
    SM120DataType dtype,
    cudaStream_t stream);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if SM120 is supported on the current device
 */
bool sm120_is_supported(int device_id);

/**
 * Verify RTX 50-series compatibility and performance
 */
bool sm120_verify_rtx50_compatibility(int device_id);

/**
 * Get RTX 50-series compatibility report
 */
const char* sm120_get_rtx50_report(int device_id);

/**
 * Get optimal block size for given problem size
 */
void sm120_get_optimal_block_size(int size, int max_threads, int* block_x, int* block_y, int* block_z);

/**
 * Get optimal grid size for given problem size and block size
 */
void sm120_get_optimal_grid_size(int size, int block_x, int block_y, int block_z, 
                                int* grid_x, int* grid_y, int* grid_z);

/**
 * Launch SM120 batch normalization kernel
 */
cudaError_t sm120_launch_batch_norm(
    const void* input, const void* scale, const void* offset,
    const void* estimated_mean, const void* estimated_variance,
    void* output,
    int batch_size, int height, int width, int channels,
    float epsilon,
    SM120DataType dtype,
    cudaStream_t stream);

/**
 * Launch SM120 softmax kernel
 */
cudaError_t sm120_launch_softmax(
    const void* logits, void* output,
    int outer_size, int axis_size, int inner_size,
    SM120DataType dtype,
    cudaStream_t stream);

/**
 * Launch SM120 embedding lookup kernel
 */
cudaError_t sm120_launch_embedding_lookup(
    const int* ids, const void* params, void* output,
    int num_ids, int vocab_size, int embed_dim,
    SM120DataType dtype,
    cudaStream_t stream);

/**
 * Get SM120 error string
 */
const char* sm120_get_error_string(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif // SM120_C_INTERFACE_H
