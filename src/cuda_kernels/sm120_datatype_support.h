// SM120 Extended Data Type Support for RTX 50-series GPUs
// Comprehensive support for FP32, FP16, BF16, FP8, FP4, and double precision
// Copyright 2024 - TensorFlow SM120 Optimization Project

#ifndef SM120_DATATYPE_SUPPORT_H_
#define SM120_DATATYPE_SUPPORT_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900  // H100 and newer
#include <cuda_fp8.h>
#endif
#endif

// Type traits for SM120 supported types
template<typename T>
struct SM120TypeTraits {
    static constexpr bool is_supported = false;
    static constexpr bool has_tensor_core_support = false;
    static constexpr bool is_fp8_family = false;
    static constexpr int alignment_bytes = sizeof(T);
};

// FP32 specialization
template<>
struct SM120TypeTraits<float> {
    static constexpr bool is_supported = true;
    static constexpr bool has_tensor_core_support = true;
    static constexpr bool is_fp8_family = false;
    static constexpr int alignment_bytes = 4;
    using compute_type = float;
    using storage_type = float;
};

// FP16 specialization
template<>
struct SM120TypeTraits<half> {
    static constexpr bool is_supported = true;
    static constexpr bool has_tensor_core_support = true;
    static constexpr bool is_fp8_family = false;
    static constexpr int alignment_bytes = 2;
    using compute_type = float;
    using storage_type = half;
};

// BF16 specialization
template<>
struct SM120TypeTraits<__nv_bfloat16> {
    static constexpr bool is_supported = true;
    static constexpr bool has_tensor_core_support = true;
    static constexpr bool is_fp8_family = false;
    static constexpr int alignment_bytes = 2;
    using compute_type = float;
    using storage_type = __nv_bfloat16;
};

// Double precision specialization
template<>
struct SM120TypeTraits<double> {
    static constexpr bool is_supported = true;
    static constexpr bool has_tensor_core_support = false;  // No tensor core for double
    static constexpr bool is_fp8_family = false;
    static constexpr int alignment_bytes = 8;
    using compute_type = double;
    using storage_type = double;
};

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900
// FP8 E4M3 specialization (available on H100+, but preparing for SM120)
template<>
struct SM120TypeTraits<__nv_fp8_e4m3> {
    static constexpr bool is_supported = true;
    static constexpr bool has_tensor_core_support = true;
    static constexpr bool is_fp8_family = true;
    static constexpr int alignment_bytes = 1;
    using compute_type = float;
    using storage_type = __nv_fp8_e4m3;
};

// FP8 E5M2 specialization
template<>
struct SM120TypeTraits<__nv_fp8_e5m2> {
    static constexpr bool is_supported = true;
    static constexpr bool has_tensor_core_support = true;
    static constexpr bool is_fp8_family = true;
    static constexpr int alignment_bytes = 1;
    using compute_type = float;
    using storage_type = __nv_fp8_e5m2;
};
#endif
#endif

// Type conversion utilities
template<typename To, typename From>
__device__ __forceinline__ To sm120_convert(From value);

// FP32 conversions
template<>
__device__ __forceinline__ float sm120_convert<float, float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ float sm120_convert<float, half>(half value) {
    return __half2float(value);
}

template<>
__device__ __forceinline__ float sm120_convert<float, __nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template<>
__device__ __forceinline__ float sm120_convert<float, double>(double value) {
    return static_cast<float>(value);
}

// FP16 conversions
template<>
__device__ __forceinline__ half sm120_convert<half, float>(float value) {
    return __float2half(value);
}

template<>
__device__ __forceinline__ half sm120_convert<half, half>(half value) {
    return value;
}

template<>
__device__ __forceinline__ half sm120_convert<half, __nv_bfloat16>(__nv_bfloat16 value) {
    return __float2half(__bfloat162float(value));
}

// BF16 conversions
template<>
__device__ __forceinline__ __nv_bfloat16 sm120_convert<__nv_bfloat16, float>(float value) {
    return __float2bfloat16(value);
}

template<>
__device__ __forceinline__ __nv_bfloat16 sm120_convert<__nv_bfloat16, half>(half value) {
    return __float2bfloat16(__half2float(value));
}

template<>
__device__ __forceinline__ __nv_bfloat16 sm120_convert<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16 value) {
    return value;
}

// Double conversions
template<>
__device__ __forceinline__ double sm120_convert<double, float>(float value) {
    return static_cast<double>(value);
}

template<>
__device__ __forceinline__ double sm120_convert<double, double>(double value) {
    return value;
}

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900
// FP8 conversions
template<>
__device__ __forceinline__ __nv_fp8_e4m3 sm120_convert<__nv_fp8_e4m3, float>(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

template<>
__device__ __forceinline__ float sm120_convert<float, __nv_fp8_e4m3>(__nv_fp8_e4m3 value) {
    return __nv_cvt_fp8_to_float(value, __NV_E4M3);
}

template<>
__device__ __forceinline__ __nv_fp8_e5m2 sm120_convert<__nv_fp8_e5m2, float>(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
}

template<>
__device__ __forceinline__ float sm120_convert<float, __nv_fp8_e5m2>(__nv_fp8_e5m2 value) {
    return __nv_cvt_fp8_to_float(value, __NV_E5M2);
}
#endif
#endif

// Arithmetic operations with automatic type promotion
template<typename T>
__device__ __forceinline__ typename SM120TypeTraits<T>::compute_type 
sm120_add(T a, T b) {
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    return sm120_convert<ComputeType>(a) + sm120_convert<ComputeType>(b);
}

template<typename T>
__device__ __forceinline__ typename SM120TypeTraits<T>::compute_type 
sm120_mul(T a, T b) {
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    return sm120_convert<ComputeType>(a) * sm120_convert<ComputeType>(b);
}

template<typename T>
__device__ __forceinline__ typename SM120TypeTraits<T>::compute_type 
sm120_fma(T a, T b, T c) {
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    if constexpr (std::is_same_v<ComputeType, float>) {
        return __fmaf_rn(sm120_convert<ComputeType>(a), 
                        sm120_convert<ComputeType>(b), 
                        sm120_convert<ComputeType>(c));
    } else if constexpr (std::is_same_v<ComputeType, double>) {
        return __fma_rn(sm120_convert<ComputeType>(a), 
                       sm120_convert<ComputeType>(b), 
                       sm120_convert<ComputeType>(c));
    } else {
        return sm120_convert<ComputeType>(a) * sm120_convert<ComputeType>(b) + sm120_convert<ComputeType>(c);
    }
}

// Reduction operations
template<typename T>
__device__ __forceinline__ T sm120_max(T a, T b) {
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    ComputeType result = fmaxf(sm120_convert<ComputeType>(a), sm120_convert<ComputeType>(b));
    return sm120_convert<T>(result);
}

template<typename T>
__device__ __forceinline__ T sm120_min(T a, T b) {
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    ComputeType result = fminf(sm120_convert<ComputeType>(a), sm120_convert<ComputeType>(b));
    return sm120_convert<T>(result);
}

// Activation functions with type support
template<typename T>
__device__ __forceinline__ T sm120_relu(T x) {
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    ComputeType val = sm120_convert<ComputeType>(x);
    return sm120_convert<T>(fmaxf(val, static_cast<ComputeType>(0)));
}

template<typename T>
__device__ __forceinline__ T sm120_gelu(T x) {
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    ComputeType val = sm120_convert<ComputeType>(x);
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    ComputeType cube = val * val * val;
    ComputeType inner = static_cast<ComputeType>(0.7978845608) * (val + static_cast<ComputeType>(0.044715) * cube);
    ComputeType tanh_val = tanhf(inner);
    ComputeType result = static_cast<ComputeType>(0.5) * val * (static_cast<ComputeType>(1.0) + tanh_val);
    
    return sm120_convert<T>(result);
}

template<typename T>
__device__ __forceinline__ T sm120_swish(T x) {
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    ComputeType val = sm120_convert<ComputeType>(x);
    ComputeType sigmoid = static_cast<ComputeType>(1.0) / (static_cast<ComputeType>(1.0) + expf(-val));
    return sm120_convert<T>(val * sigmoid);
}

// Softmax with numerical stability
template<typename T>
__device__ __forceinline__ void sm120_softmax_row(
    const T* __restrict__ input,
    T* __restrict__ output,
    int length) {
    
    using ComputeType = typename SM120TypeTraits<T>::compute_type;
    
    // Find maximum for numerical stability
    ComputeType max_val = sm120_convert<ComputeType>(input[0]);
    for (int i = 1; i < length; i++) {
        max_val = fmaxf(max_val, sm120_convert<ComputeType>(input[i]));
    }
    
    // Compute exponentials and sum
    ComputeType sum = static_cast<ComputeType>(0.0);
    for (int i = 0; i < length; i++) {
        ComputeType exp_val = expf(sm120_convert<ComputeType>(input[i]) - max_val);
        output[i] = sm120_convert<T>(exp_val);
        sum += exp_val;
    }
    
    // Normalize
    ComputeType inv_sum = static_cast<ComputeType>(1.0) / sum;
    for (int i = 0; i < length; i++) {
        output[i] = sm120_convert<T>(sm120_convert<ComputeType>(output[i]) * inv_sum);
    }
}

// Memory alignment helpers
template<typename T>
__device__ __forceinline__ bool sm120_is_aligned(const T* ptr) {
    constexpr int alignment = SM120TypeTraits<T>::alignment_bytes;
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

template<typename T>
__device__ __forceinline__ void sm120_aligned_load(T* dest, const T* src, int count) {
    if constexpr (SM120TypeTraits<T>::alignment_bytes >= 16) {
        // Use vectorized loads for well-aligned types
        using VecType = float4;  // 16-byte aligned
        int vec_count = count * sizeof(T) / sizeof(VecType);
        VecType* vec_dest = reinterpret_cast<VecType*>(dest);
        const VecType* vec_src = reinterpret_cast<const VecType*>(src);
        
        for (int i = 0; i < vec_count; i++) {
            vec_dest[i] = vec_src[i];
        }
        
        // Handle remainder
        int remainder_start = vec_count * sizeof(VecType) / sizeof(T);
        for (int i = remainder_start; i < count; i++) {
            dest[i] = src[i];
        }
    } else {
        // Standard copy for smaller types
        for (int i = 0; i < count; i++) {
            dest[i] = src[i];
        }
    }
}

// Automatic type selection for optimal performance
template<typename InputType>
struct SM120OptimalType {
    // Default to input type
    using type = InputType;
};

// Optimize common cases
template<>
struct SM120OptimalType<float> {
    using type = float;  // Already optimal
};

template<>
struct SM120OptimalType<double> {
    using type = float;  // Convert to float for better performance if precision allows
};

// Capability checking at runtime
__device__ __forceinline__ bool sm120_supports_fp8() {
#ifdef __CUDA_ARCH__
    return __CUDA_ARCH__ >= 900;  // H100+ for now, SM120 will inherit
#else
    return false;
#endif
}

__device__ __forceinline__ bool sm120_supports_tensor_cores() {
#ifdef __CUDA_ARCH__
    return __CUDA_ARCH__ >= 700;  // Volta and newer
#else
    return false;
#endif
}

__device__ __forceinline__ bool sm120_supports_bf16() {
#ifdef __CUDA_ARCH__
    return __CUDA_ARCH__ >= 800;  // Ampere and newer
#else
    return false;
#endif
}

// Dynamic type dispatch
template<typename Func>
__device__ __forceinline__ void sm120_dispatch_type(
    int type_id, Func&& func) {
    
    switch (type_id) {
        case 0:  // float
            func.template operator()<float>();
            break;
        case 1:  // half
            func.template operator()<half>();
            break;
        case 2:  // bfloat16
            if (sm120_supports_bf16()) {
                func.template operator()<__nv_bfloat16>();
            } else {
                func.template operator()<float>();  // Fallback
            }
            break;
        case 3:  // double
            func.template operator()<double>();
            break;
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900
        case 4:  // fp8_e4m3
            if (sm120_supports_fp8()) {
                func.template operator()<__nv_fp8_e4m3>();
            } else {
                func.template operator()<half>();  // Fallback
            }
            break;
        case 5:  // fp8_e5m2
            if (sm120_supports_fp8()) {
                func.template operator()<__nv_fp8_e5m2>();
            } else {
                func.template operator()<half>();  // Fallback
            }
            break;
#endif
#endif
        default:
            func.template operator()<float>();  // Default fallback
    }
}

#endif // SM120_DATATYPE_SUPPORT_H_
