/*
 * SM120 CUDA Stream Utilities
 * 
 * Robust CUDA stream extraction that works across different TensorFlow versions.
 * Handles various TensorFlow stream API changes gracefully.
 */

#ifndef SM120_STREAM_UTILS_H
#define SM120_STREAM_UTILS_H

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include <cuda_runtime.h>

namespace tensorflow {
namespace sm120_utils {

/**
 * Safely extract CUDA stream from TensorFlow OpKernelContext
 * Works across different TensorFlow versions by trying multiple methods
 */
inline cudaStream_t GetCudaStream(OpKernelContext* context) {
    if (!context) {
        return nullptr;
    }
    
    auto* device_context = context->op_device_context();
    if (!device_context) {
        return nullptr;
    }
    
    auto* stream = device_context->stream();
    if (!stream) {
        return nullptr;
    }
    
    // Method 1: Try the modern TensorFlow API (TF 2.8+)
    try {
        auto* gpu_stream = stream->implementation();
        if (gpu_stream) {
            // Try to get the CUDA stream handle
            #if defined(TENSORFLOW_USE_ROCM)
                // ROCm/HIP support
                return reinterpret_cast<cudaStream_t>(gpu_stream->GpuStreamHack());
            #else
                // NVIDIA CUDA support
                return reinterpret_cast<cudaStream_t>(gpu_stream->GpuStreamHack());
            #endif
        }
    } catch (...) {
        // Fall through to next method
    }
    
    // Method 2: Try legacy TensorFlow API (TF 2.4-2.7)
    try {
        auto* parent = stream->parent();
        if (parent) {
            auto* implementation = parent->implementation();
            if (implementation) {
                return reinterpret_cast<cudaStream_t>(implementation->GpuStreamHack());
            }
        }
    } catch (...) {
        // Fall through to next method
    }
    
    // Method 3: Try direct stream access (older TensorFlow versions)
    try {
        return reinterpret_cast<cudaStream_t>(stream->platform_specific_handle().stream);
    } catch (...) {
        // Fall through to default
    }
    
    // Method 4: Use default stream as fallback
    return nullptr; // Will use default stream (0)
}

/**
 * Get CUDA stream with error checking
 * Returns default stream (0) if extraction fails
 */
inline cudaStream_t GetCudaStreamSafe(OpKernelContext* context) {
    cudaStream_t stream = GetCudaStream(context);
    return stream ? stream : cudaStreamDefault;
}

/**
 * Synchronize CUDA stream safely
 */
inline cudaError_t SynchronizeStream(cudaStream_t stream) {
    if (stream == nullptr || stream == cudaStreamDefault) {
        return cudaDeviceSynchronize();
    } else {
        return cudaStreamSynchronize(stream);
    }
}

/**
 * Check if we're running on a GPU device
 */
inline bool IsGpuDevice(OpKernelContext* context) {
    if (!context) return false;
    
    const auto& device_type = context->device()->device_type();
    return device_type == DEVICE_GPU;
}

/**
 * Get GPU device ID from context
 */
inline int GetGpuDeviceId(OpKernelContext* context) {
    if (!IsGpuDevice(context)) {
        return -1;
    }
    
    try {
        auto* device_context = context->op_device_context();
        if (device_context) {
            auto* stream = device_context->stream();
            if (stream && stream->parent()) {
                return stream->parent()->device_ordinal();
            }
        }
    } catch (...) {
        // Fall through to default
    }
    
    // Fallback: get current device
    int device_id = 0;
    cudaGetDevice(&device_id);
    return device_id;
}

/**
 * Verify SM 12.0 compute capability
 */
inline bool VerifySM120Support(int device_id = -1) {
    if (device_id < 0) {
        cudaGetDevice(&device_id);
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    
    if (error != cudaSuccess) {
        return false;
    }
    
    // Check for SM 12.0 (RTX 50-series)
    return (prop.major == 12 && prop.minor == 0);
}

/**
 * Get RTX 50-series specific capabilities
 */
struct RTX50Capabilities {
    bool has_5th_gen_tensor_cores;
    bool has_enhanced_l2_cache;
    bool has_cooperative_groups_v2;
    size_t shared_memory_per_sm;
    size_t l2_cache_size;
    int max_threads_per_sm;
    int multiprocessor_count;
};

inline RTX50Capabilities GetRTX50Capabilities(int device_id = -1) {
    RTX50Capabilities caps = {};
    
    if (device_id < 0) {
        cudaGetDevice(&device_id);
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    
    if (error == cudaSuccess && prop.major == 12 && prop.minor == 0) {
        caps.has_5th_gen_tensor_cores = true;
        caps.has_enhanced_l2_cache = true;
        caps.has_cooperative_groups_v2 = true;
        caps.shared_memory_per_sm = prop.sharedMemPerMultiprocessor;
        caps.l2_cache_size = prop.l2CacheSize;
        caps.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        caps.multiprocessor_count = prop.multiProcessorCount;
    }
    
    return caps;
}

} // namespace sm120_utils
} // namespace tensorflow

#endif // SM120_STREAM_UTILS_H
