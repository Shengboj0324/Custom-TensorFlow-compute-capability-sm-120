// SM120 Comprehensive Error Handling and Performance Monitoring
// Production-grade error handling with automatic fallback and diagnostics
// Copyright 2024 - TensorFlow SM120 Optimization Project

#ifndef SM120_ERROR_HANDLING_H_
#define SM120_ERROR_HANDLING_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <chrono>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>

// SM120 error checking macro with detailed diagnostics
#define SM120_CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        SM120ErrorHandler::Instance().HandleCudaError(error, #call, __FILE__, __LINE__); \
    } \
} while(0)

// SM120 performance timing macro
#define SM120_PROFILE_KERNEL(name, kernel_call) do { \
    auto timer = SM120PerformanceMonitor::Instance().StartTimer(name); \
    kernel_call; \
    SM120_CUDA_CHECK(cudaDeviceSynchronize()); \
    timer->Stop(); \
} while(0)

// Memory alignment verification
#define SM120_CHECK_ALIGNMENT(ptr, alignment) do { \
    if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0) { \
        SM120ErrorHandler::Instance().HandleAlignmentError(ptr, alignment, __FILE__, __LINE__); \
    } \
} while(0)

// Performance metrics structure
struct SM120KernelMetrics {
    float execution_time_ms;
    float memory_bandwidth_gb_s;
    float arithmetic_intensity;
    float occupancy_percent;
    int blocks_launched;
    int threads_per_block;
    size_t shared_memory_bytes;
    size_t register_count;
    
    SM120KernelMetrics() : execution_time_ms(0), memory_bandwidth_gb_s(0), 
                          arithmetic_intensity(0), occupancy_percent(0),
                          blocks_launched(0), threads_per_block(0),
                          shared_memory_bytes(0), register_count(0) {}
};

// Error recovery strategies
enum class SM120ErrorStrategy {
    FALLBACK_TO_STANDARD,    // Use standard TensorFlow operations
    RETRY_WITH_SMALLER_TILE, // Reduce tile sizes and retry
    SWITCH_TO_FP32,          // Convert to higher precision
    ABORT_OPERATION          // Fail fast with detailed error
};

// Performance tuning hints
struct SM120TuningHints {
    int preferred_tile_size;
    int optimal_block_size;
    bool use_tensor_cores;
    bool enable_async_copy;
    float memory_coalescing_factor;
    
    SM120TuningHints() : preferred_tile_size(16), optimal_block_size(256),
                        use_tensor_cores(true), enable_async_copy(true),
                        memory_coalescing_factor(1.0f) {}
};

// Timer class for performance measurement
class SM120Timer {
public:
    SM120Timer(const std::string& name);
    ~SM120Timer();
    
    void Stop();
    float GetElapsedMs() const;
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool stopped_;
};

// Singleton error handler with comprehensive diagnostics
class SM120ErrorHandler {
public:
    static SM120ErrorHandler& Instance();
    
    void HandleCudaError(cudaError_t error, const char* call, 
                        const char* file, int line);
    void HandleAlignmentError(const void* ptr, size_t alignment,
                             const char* file, int line);
    void HandleShapeError(const std::string& operation,
                         const std::vector<int>& expected_shape,
                         const std::vector<int>& actual_shape);
    
    void SetErrorStrategy(SM120ErrorStrategy strategy);
    SM120ErrorStrategy GetErrorStrategy() const;
    
    void EnableFallback(bool enable);
    bool IsFallbackEnabled() const;
    
    void LogError(const std::string& message);
    void LogWarning(const std::string& message);
    void LogInfo(const std::string& message);
    
    // Error statistics
    int GetErrorCount() const;
    int GetWarningCount() const;
    std::vector<std::string> GetRecentErrors(int count = 10) const;
    
private:
    SM120ErrorHandler() = default;
    
    mutable std::mutex mutex_;
    SM120ErrorStrategy error_strategy_ = SM120ErrorStrategy::FALLBACK_TO_STANDARD;
    bool fallback_enabled_ = true;
    int error_count_ = 0;
    int warning_count_ = 0;
    std::vector<std::string> error_log_;
    
    void AddToLog(const std::string& level, const std::string& message);
};

// Performance monitoring singleton
class SM120PerformanceMonitor {
public:
    static SM120PerformanceMonitor& Instance();
    
    std::unique_ptr<SM120Timer> StartTimer(const std::string& kernel_name);
    void RecordMetrics(const std::string& kernel_name, const SM120KernelMetrics& metrics);
    
    SM120KernelMetrics GetAverageMetrics(const std::string& kernel_name) const;
    std::unordered_map<std::string, SM120KernelMetrics> GetAllMetrics() const;
    
    void SetTuningHints(const std::string& kernel_name, const SM120TuningHints& hints);
    SM120TuningHints GetTuningHints(const std::string& kernel_name) const;
    
    void EnableProfiling(bool enable);
    bool IsProfilingEnabled() const;
    
    void ResetStatistics();
    void PrintSummary() const;
    
    // Adaptive tuning
    void UpdateTuningBasedOnPerformance();
    bool ShouldUseTensorCores(const std::string& kernel_name) const;
    int GetOptimalTileSize(const std::string& kernel_name) const;
    
private:
    SM120PerformanceMonitor() = default;
    
    mutable std::mutex mutex_;
    bool profiling_enabled_ = false;
    std::unordered_map<std::string, std::vector<SM120KernelMetrics>> metrics_history_;
    std::unordered_map<std::string, SM120TuningHints> tuning_hints_;
    
    SM120KernelMetrics CalculateAverageMetrics(
        const std::vector<SM120KernelMetrics>& history) const;
};

// Resource manager for automatic cleanup
class SM120ResourceManager {
public:
    static SM120ResourceManager& Instance();
    
    void RegisterBuffer(void* ptr, size_t size, const std::string& name);
    void UnregisterBuffer(void* ptr);
    void CleanupAll();
    
    void RegisterStream(cudaStream_t stream, const std::string& name);
    void UnregisterStream(cudaStream_t stream);
    void SynchronizeAllStreams();
    
    size_t GetTotalAllocatedMemory() const;
    int GetActiveBufferCount() const;
    int GetActiveStreamCount() const;
    
private:
    SM120ResourceManager() = default;
    ~SM120ResourceManager();
    
    struct BufferInfo {
        size_t size;
        std::string name;
        std::chrono::time_point<std::chrono::steady_clock> allocated_time;
    };
    
    struct StreamInfo {
        std::string name;
        std::chrono::time_point<std::chrono::steady_clock> created_time;
    };
    
    mutable std::mutex mutex_;
    std::unordered_map<void*, BufferInfo> buffers_;
    std::unordered_map<cudaStream_t, StreamInfo> streams_;
};

// GPU capability checker
class SM120CapabilityChecker {
public:
    static SM120CapabilityChecker& Instance();
    
    bool IsRTX50SeriesGPU() const;
    bool SupportsSM120() const;
    bool SupportsTensorCores() const;
    bool SupportsCooperativeGroups() const;
    bool SupportsBFloat16() const;
    bool SupportsFP8() const;
    
    int GetComputeCapabilityMajor() const;
    int GetComputeCapabilityMinor() const;
    size_t GetSharedMemoryPerBlock() const;
    int GetMaxThreadsPerBlock() const;
    int GetWarpSize() const;
    
    std::string GetGPUName() const;
    std::string GetDriverVersion() const;
    std::string GetCUDAVersion() const;
    
    void PrintCapabilities() const;
    
private:
    SM120CapabilityChecker();
    
    cudaDeviceProp device_prop_;
    bool capabilities_checked_;
    
    void CheckCapabilities();
};

// Fallback operation templates
template<typename T>
cudaError_t FallbackMatMul(const T* A, const T* B, T* C,
                          int M, int N, int K, 
                          cudaStream_t stream = nullptr);

template<typename T>
cudaError_t FallbackConv2D(const T* input, const T* filter, T* output,
                          int batch_size, int input_height, int input_width,
                          int input_channels, int output_channels,
                          int filter_height, int filter_width,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          cudaStream_t stream = nullptr);

// Utility functions
inline void SM120LogKernelLaunch(const std::string& kernel_name,
                                dim3 grid, dim3 block,
                                size_t shared_mem = 0,
                                cudaStream_t stream = nullptr) {
    if (SM120PerformanceMonitor::Instance().IsProfilingEnabled()) {
        SM120ErrorHandler::Instance().LogInfo(
            "Launching " + kernel_name + 
            " with grid(" + std::to_string(grid.x) + "," + std::to_string(grid.y) + "," + std::to_string(grid.z) + ")" +
            " block(" + std::to_string(block.x) + "," + std::to_string(block.y) + "," + std::to_string(block.z) + ")" +
            " shared_mem=" + std::to_string(shared_mem));
    }
}

inline bool SM120VerifyInputTensors(const void* input, const std::vector<int>& shape,
                                   const std::string& tensor_name) {
    if (!input) {
        SM120ErrorHandler::Instance().LogError("Null input tensor: " + tensor_name);
        return false;
    }
    
    if (shape.empty()) {
        SM120ErrorHandler::Instance().LogError("Empty shape for tensor: " + tensor_name);
        return false;
    }
    
    for (int dim : shape) {
        if (dim <= 0) {
            SM120ErrorHandler::Instance().LogError("Invalid dimension in tensor: " + tensor_name);
            return false;
        }
    }
    
    return true;
}

inline size_t SM120CalculateElements(const std::vector<int>& shape) {
    size_t elements = 1;
    for (int dim : shape) {
        elements *= dim;
    }
    return elements;
}

#endif // SM120_ERROR_HANDLING_H_
