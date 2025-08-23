// SM120 Backward Propagation Kernels Header
// Gradient computation function declarations for RTX 50-series GPUs
// Copyright 2024 - TensorFlow SM120 Optimization Project

#ifndef SM120_BACKWARD_KERNELS_H_
#define SM120_BACKWARD_KERNELS_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Matrix multiplication gradient kernels
template<typename T>
cudaError_t LaunchSM120MatMulGradA(
    const T* grad_output, const T* B, T* grad_A,
    int M, int N, int K, float alpha = 1.0f,
    cudaStream_t stream = nullptr);

template<typename T>
cudaError_t LaunchSM120MatMulGradB(
    const T* A, const T* grad_output, T* grad_B,
    int M, int N, int K, float alpha = 1.0f,
    cudaStream_t stream = nullptr);

// Convolution gradient kernels
template<typename T>
cudaError_t LaunchSM120Conv2DBackpropInput(
    const T* grad_output, const T* filter, T* grad_input,
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w, int stride_h, int stride_w,
    int pad_h, int pad_w, cudaStream_t stream = nullptr);

template<typename T>
cudaError_t LaunchSM120Conv2DBackpropFilter(
    const T* input, const T* grad_output, T* grad_filter,
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w, int stride_h, int stride_w,
    int pad_h, int pad_w, cudaStream_t stream = nullptr);

// Activation gradient kernels
template<typename T>
cudaError_t LaunchSM120SoftmaxGrad(
    const T* grad_output, const T* softmax_output, T* grad_input,
    int N, int D, cudaStream_t stream = nullptr);

template<typename T>
cudaError_t LaunchSM120ReLUGrad(
    const T* grad_output, const T* input, T* grad_input,
    int N, cudaStream_t stream = nullptr);

template<typename T>
cudaError_t LaunchSM120GELUGrad(
    const T* grad_output, const T* input, T* grad_input,
    int N, cudaStream_t stream = nullptr);

// Normalization gradient kernels
template<typename T>
cudaError_t LaunchSM120BatchNormGrad(
    const T* grad_output, const T* input, const T* scale,
    T* grad_input, T* grad_scale, T* grad_bias,
    const T* saved_mean, const T* saved_variance,
    int N, int C, int H, int W, float epsilon = 1e-5f,
    cudaStream_t stream = nullptr);

template<typename T>
cudaError_t LaunchSM120LayerNormGrad(
    const T* grad_output, const T* input, const T* scale,
    T* grad_input, T* grad_scale, T* grad_bias,
    const T* saved_mean, const T* saved_variance,
    int N, int D, float epsilon = 1e-5f,
    cudaStream_t stream = nullptr);

// Attention gradient kernels
template<typename T>
cudaError_t LaunchSM120ScaledDotProductAttentionGrad(
    const T* grad_output, const T* query, const T* key, const T* value,
    const T* attention_weights, T* grad_query, T* grad_key, T* grad_value,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float scale, cudaStream_t stream = nullptr);

#ifdef __cplusplus
}
#endif

#endif // SM120_BACKWARD_KERNELS_H_
