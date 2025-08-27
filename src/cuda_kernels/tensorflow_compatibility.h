/*
 * TensorFlow Compatibility Header
 * 
 * This header suppresses deprecation warnings from TensorFlow headers
 * and provides compatibility definitions for SM120 kernels.
 */

#ifndef TENSORFLOW_COMPATIBILITY_H
#define TENSORFLOW_COMPATIBILITY_H

// Suppress all deprecation warnings before including TensorFlow headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// For NVCC compiler
#ifdef __CUDACC__
#pragma nv_diag_suppress 1215  // Deprecated function warnings
#pragma nv_diag_suppress 1216  // Deprecated variable warnings
#endif

// Define compatibility macros
#define TF_SUPPRESS_DEPRECATION_START \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

#define TF_SUPPRESS_DEPRECATION_END \
    _Pragma("GCC diagnostic pop")

// TensorFlow compatibility layer - define missing types before TensorFlow headers
namespace tensorflow {
    // Forward declarations for TensorFlow compatibility
    class OpKernelContext;
    class Tensor;
}

// Define missing CUDA launch config types for TensorFlow compatibility
struct Cuda2DLaunchConfig {
    dim3 block_count;
    dim3 thread_per_block;
    
    Cuda2DLaunchConfig() = default;
    Cuda2DLaunchConfig(dim3 blocks, dim3 threads) : block_count(blocks), thread_per_block(threads) {}
};

// Provide the function that TensorFlow gpu_launch_config.h expects
inline Cuda2DLaunchConfig GetCuda2DLaunchConfig(int xdim, int ydim, 
                                                int block_x_limit = 1024, 
                                                int block_y_limit = 1024) {
    int block_x = std::min(block_x_limit, xdim);
    int block_y = std::min(block_y_limit, ydim);
    
    int grid_x = (xdim + block_x - 1) / block_x;
    int grid_y = (ydim + block_y - 1) / block_y;
    
    return Cuda2DLaunchConfig(dim3(grid_x, grid_y), dim3(block_x, block_y));
}

#endif // TENSORFLOW_COMPATIBILITY_H
