/*
 * Header file for sm_120 optimized CUDA kernel launchers
 * 
 * This header provides C++ interfaces for launching CUDA kernels
 * optimized for RTX 50-series GPUs with compute capability 12.0
 */

#ifndef TENSORFLOW_CORE_KERNELS_SM120_KERNEL_LAUNCHER_H_
#define TENSORFLOW_CORE_KERNELS_SM120_KERNEL_LAUNCHER_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace sm120_kernels {

// ============================================================================
// Matrix Multiplication Kernels
// ============================================================================

// Launch optimized matrix multiplication for sm_120
template<typename T>
cudaError_t LaunchSM120MatMul(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f,
    cudaStream_t stream = nullptr);

// Mixed precision matrix multiplication
cudaError_t LaunchSM120MixedPrecisionGEMM(
    const half* A, const half* B, float* C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f,
    cudaStream_t stream = nullptr);

// ============================================================================
// Convolution Kernels
// ============================================================================

template<typename T>
cudaError_t LaunchSM120Conv2D(
    const T* input, const T* filter, T* output,
    int batch_size, 
    int input_height, int input_width, int input_channels,
    int output_height, int output_width, int output_channels,
    int filter_height, int filter_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream = nullptr);

// Depthwise separable convolution optimized for sm_120
template<typename T>
cudaError_t LaunchSM120DepthwiseConv2D(
    const T* input, const T* filter, T* output,
    int batch_size,
    int input_height, int input_width, int input_channels,
    int filter_height, int filter_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream = nullptr);

// ============================================================================
// Activation Function Kernels
// ============================================================================

enum class ActivationType {
    RELU = 0,
    GELU = 1,
    SWISH = 2,
    TANH = 3,
    SIGMOID = 4
};

template<typename T>
cudaError_t LaunchSM120FusedActivation(
    const T* input, T* output,
    int size,
    ActivationType activation_type,
    cudaStream_t stream = nullptr);

// ============================================================================
// Reduction Kernels
// ============================================================================

enum class ReductionType {
    SUM = 0,
    MEAN = 1,
    MAX = 2,
    MIN = 3,
    PROD = 4
};

template<typename T>
cudaError_t LaunchSM120Reduction(
    const T* input, T* output,
    int size, int reduction_size,
    ReductionType reduction_type,
    cudaStream_t stream = nullptr);

// ============================================================================
// Attention Mechanism Kernels
// ============================================================================

template<typename T>
cudaError_t LaunchSM120ScaledDotProductAttention(
    const T* queries, const T* keys, const T* values,
    T* output, float* attention_weights,
    int batch_size, int seq_len, int head_dim,
    float scale = 1.0f,
    cudaStream_t stream = nullptr);

// Multi-head attention optimized for sm_120
template<typename T>
cudaError_t LaunchSM120MultiHeadAttention(
    const T* queries, const T* keys, const T* values,
    T* output,
    int batch_size, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream = nullptr);

// ============================================================================
// Memory Operations
// ============================================================================

template<typename T>
cudaError_t LaunchSM120Transpose(
    const T* input, T* output,
    int rows, int cols,
    cudaStream_t stream = nullptr);

template<typename T>
cudaError_t LaunchSM120MemcpyOptimized(
    const T* src, T* dst,
    int size,
    cudaStream_t stream = nullptr);

// ============================================================================
// Normalization Kernels
// ============================================================================

template<typename T>
cudaError_t LaunchSM120LayerNorm(
    const T* input, const T* gamma, const T* beta,
    T* output, T* mean, T* variance,
    int batch_size, int feature_size,
    float epsilon = 1e-5f,
    cudaStream_t stream = nullptr);

template<typename T>
cudaError_t LaunchSM120BatchNorm(
    const T* input, const T* scale, const T* offset,
    const T* estimated_mean, const T* estimated_variance,
    T* output,
    int batch_size, int height, int width, int channels,
    float epsilon = 1e-5f,
    cudaStream_t stream = nullptr);

// ============================================================================
// Utility Functions
// ============================================================================

// Check if current GPU supports sm_120 optimizations
bool IsSM120Supported();

// Get optimal block size for sm_120 kernels
dim3 GetOptimalBlockSize(int size, int max_threads = 1024);

// Get optimal grid size for sm_120 kernels
dim3 GetOptimalGridSize(int size, dim3 block_size);

// Memory bandwidth benchmark for sm_120
float BenchmarkSM120MemoryBandwidth(int size_mb = 1000);

// Compute capability detection
struct SM120Capabilities {
    int major_version;
    int minor_version;
    bool supports_tensor_cores_5th_gen;
    bool supports_enhanced_l2_cache;
    bool supports_cooperative_groups_v2;
    size_t shared_memory_per_sm;
    size_t l2_cache_size;
    int max_threads_per_sm;
};

SM120Capabilities GetSM120Capabilities(int device_id = 0);

} // namespace sm120_kernels
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_SM120_KERNEL_LAUNCHER_H_
