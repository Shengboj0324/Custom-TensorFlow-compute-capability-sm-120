// SM120 Error Handling and Performance Monitoring Implementation
// Production-grade error handling with automatic fallback and diagnostics
// Copyright 2024 - TensorFlow SM120 Optimization Project

#include "sm120_error_handling.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

// Timer implementation
SM120Timer::SM120Timer(const std::string& name) 
    : name_(name), stopped_(false) {
    start_time_ = std::chrono::high_resolution_clock::now();
}

SM120Timer::~SM120Timer() {
    if (!stopped_) {
        Stop();
    }
}

void SM120Timer::Stop() {
    if (!stopped_) {
        end_time_ = std::chrono::high_resolution_clock::now();
        stopped_ = true;
        
        // Record metrics
        SM120KernelMetrics metrics;
        metrics.execution_time_ms = GetElapsedMs();
        SM120PerformanceMonitor::Instance().RecordMetrics(name_, metrics);
    }
}

float SM120Timer::GetElapsedMs() const {
    auto end = stopped_ ? end_time_ : std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_);
    return duration.count() / 1000.0f;
}

// Error Handler implementation
SM120ErrorHandler& SM120ErrorHandler::Instance() {
    static SM120ErrorHandler instance;
    return instance;
}

void SM120ErrorHandler::HandleCudaError(cudaError_t error, const char* call,
                                       const char* file, int line) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::stringstream ss;
    ss << "CUDA Error in " << file << ":" << line << " - " << call 
       << " failed with error: " << cudaGetErrorString(error) 
       << " (" << static_cast<int>(error) << ")";
    
    AddToLog("ERROR", ss.str());
    error_count_++;
    
    // Get additional device context
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    ss.str("");
    ss << "Device context: " << prop.name 
       << ", Free memory: " << (free_mem / 1024 / 1024) << "MB"
       << ", Total memory: " << (total_mem / 1024 / 1024) << "MB";
    AddToLog("INFO", ss.str());
    
    // Apply error recovery strategy
    switch (error_strategy_) {
        case SM120ErrorStrategy::FALLBACK_TO_STANDARD:
            if (fallback_enabled_) {
                AddToLog("INFO", "Falling back to standard TensorFlow operations");
                return;
            }
            break;
        case SM120ErrorStrategy::RETRY_WITH_SMALLER_TILE:
            AddToLog("INFO", "Retrying with smaller tile sizes");
            break;
        case SM120ErrorStrategy::SWITCH_TO_FP32:
            AddToLog("INFO", "Switching to FP32 precision");
            break;
        case SM120ErrorStrategy::ABORT_OPERATION:
            AddToLog("FATAL", "Aborting operation due to critical error");
            throw std::runtime_error(ss.str());
    }
}

void SM120ErrorHandler::HandleAlignmentError(const void* ptr, size_t alignment,
                                           const char* file, int line) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::stringstream ss;
    ss << "Memory alignment error in " << file << ":" << line
       << " - Pointer " << ptr << " is not aligned to " << alignment << " bytes"
       << " (actual alignment: " << (reinterpret_cast<uintptr_t>(ptr) % alignment) << ")";
    
    AddToLog("ERROR", ss.str());
    error_count_++;
}

void SM120ErrorHandler::HandleShapeError(const std::string& operation,
                                       const std::vector<int>& expected_shape,
                                       const std::vector<int>& actual_shape) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::stringstream ss;
    ss << "Shape mismatch in operation " << operation << " - Expected: [";
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        ss << expected_shape[i];
        if (i < expected_shape.size() - 1) ss << ", ";
    }
    ss << "], Actual: [";
    for (size_t i = 0; i < actual_shape.size(); ++i) {
        ss << actual_shape[i];
        if (i < actual_shape.size() - 1) ss << ", ";
    }
    ss << "]";
    
    AddToLog("ERROR", ss.str());
    error_count_++;
}

void SM120ErrorHandler::SetErrorStrategy(SM120ErrorStrategy strategy) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_strategy_ = strategy;
}

SM120ErrorStrategy SM120ErrorHandler::GetErrorStrategy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_strategy_;
}

void SM120ErrorHandler::EnableFallback(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    fallback_enabled_ = enable;
}

bool SM120ErrorHandler::IsFallbackEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fallback_enabled_;
}

void SM120ErrorHandler::LogError(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    AddToLog("ERROR", message);
    error_count_++;
}

void SM120ErrorHandler::LogWarning(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    AddToLog("WARNING", message);
    warning_count_++;
}

void SM120ErrorHandler::LogInfo(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    AddToLog("INFO", message);
}

int SM120ErrorHandler::GetErrorCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_count_;
}

int SM120ErrorHandler::GetWarningCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return warning_count_;
}

std::vector<std::string> SM120ErrorHandler::GetRecentErrors(int count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> recent;
    
    auto start_it = error_log_.end() - std::min(count, static_cast<int>(error_log_.size()));
    recent.assign(start_it, error_log_.end());
    
    return recent;
}

void SM120ErrorHandler::AddToLog(const std::string& level, const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] "
       << "[" << level << "] " << message;
    
    error_log_.push_back(ss.str());
    
    // Keep only last 1000 entries
    if (error_log_.size() > 1000) {
        error_log_.erase(error_log_.begin());
    }
    
    // Output to console if enabled
    std::cout << ss.str() << std::endl;
}

// Performance Monitor implementation
SM120PerformanceMonitor& SM120PerformanceMonitor::Instance() {
    static SM120PerformanceMonitor instance;
    return instance;
}

std::unique_ptr<SM120Timer> SM120PerformanceMonitor::StartTimer(const std::string& kernel_name) {
    if (!profiling_enabled_) {
        return nullptr;
    }
    return std::make_unique<SM120Timer>(kernel_name);
}

void SM120PerformanceMonitor::RecordMetrics(const std::string& kernel_name, 
                                          const SM120KernelMetrics& metrics) {
    if (!profiling_enabled_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_history_[kernel_name].push_back(metrics);
    
    // Keep only last 100 measurements per kernel
    if (metrics_history_[kernel_name].size() > 100) {
        metrics_history_[kernel_name].erase(metrics_history_[kernel_name].begin());
    }
}

SM120KernelMetrics SM120PerformanceMonitor::GetAverageMetrics(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = metrics_history_.find(kernel_name);
    if (it == metrics_history_.end() || it->second.empty()) {
        return SM120KernelMetrics{};
    }
    
    return CalculateAverageMetrics(it->second);
}

std::unordered_map<std::string, SM120KernelMetrics> SM120PerformanceMonitor::GetAllMetrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, SM120KernelMetrics> result;
    for (const auto& [kernel_name, history] : metrics_history_) {
        if (!history.empty()) {
            result[kernel_name] = CalculateAverageMetrics(history);
        }
    }
    
    return result;
}

void SM120PerformanceMonitor::SetTuningHints(const std::string& kernel_name, 
                                           const SM120TuningHints& hints) {
    std::lock_guard<std::mutex> lock(mutex_);
    tuning_hints_[kernel_name] = hints;
}

SM120TuningHints SM120PerformanceMonitor::GetTuningHints(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = tuning_hints_.find(kernel_name);
    if (it != tuning_hints_.end()) {
        return it->second;
    }
    
    return SM120TuningHints{};
}

void SM120PerformanceMonitor::EnableProfiling(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    profiling_enabled_ = enable;
}

bool SM120PerformanceMonitor::IsProfilingEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiling_enabled_;
}

void SM120PerformanceMonitor::ResetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_history_.clear();
    tuning_hints_.clear();
}

void SM120PerformanceMonitor::PrintSummary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "\n=== SM120 Performance Summary ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    for (const auto& [kernel_name, history] : metrics_history_) {
        if (history.empty()) continue;
        
        auto avg_metrics = CalculateAverageMetrics(history);
        
        std::cout << "\nKernel: " << kernel_name << std::endl;
        std::cout << "  Samples: " << history.size() << std::endl;
        std::cout << "  Avg Execution Time: " << avg_metrics.execution_time_ms << " ms" << std::endl;
        std::cout << "  Avg Memory Bandwidth: " << avg_metrics.memory_bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "  Avg Occupancy: " << avg_metrics.occupancy_percent << "%" << std::endl;
        std::cout << "  Blocks Launched: " << avg_metrics.blocks_launched << std::endl;
        std::cout << "  Threads per Block: " << avg_metrics.threads_per_block << std::endl;
    }
    
    std::cout << "\n=== End Summary ===" << std::endl;
}

void SM120PerformanceMonitor::UpdateTuningBasedOnPerformance() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& [kernel_name, history] : metrics_history_) {
        if (history.size() < 10) continue; // Need enough samples
        
        auto avg_metrics = CalculateAverageMetrics(history);
        SM120TuningHints hints = GetTuningHints(kernel_name);
        
        // Adaptive tuning based on performance
        if (avg_metrics.occupancy_percent < 50.0f) {
            // Low occupancy - try smaller block size
            hints.optimal_block_size = std::max(64, hints.optimal_block_size / 2);
        } else if (avg_metrics.occupancy_percent > 90.0f && avg_metrics.execution_time_ms > 1.0f) {
            // High occupancy but slow - try larger block size
            hints.optimal_block_size = std::min(1024, hints.optimal_block_size * 2);
        }
        
        // Adjust tile size based on memory bandwidth
        if (avg_metrics.memory_bandwidth_gb_s < 500.0f) {
            // Low bandwidth - try smaller tiles for better cache utilization
            hints.preferred_tile_size = std::max(8, hints.preferred_tile_size / 2);
        }
        
        tuning_hints_[kernel_name] = hints;
    }
}

bool SM120PerformanceMonitor::ShouldUseTensorCores(const std::string& kernel_name) const {
    auto hints = GetTuningHints(kernel_name);
    auto metrics = GetAverageMetrics(kernel_name);
    
    // Use tensor cores if enabled in hints and performance is good
    return hints.use_tensor_cores && metrics.arithmetic_intensity > 1.0f;
}

int SM120PerformanceMonitor::GetOptimalTileSize(const std::string& kernel_name) const {
    auto hints = GetTuningHints(kernel_name);
    return hints.preferred_tile_size;
}

SM120KernelMetrics SM120PerformanceMonitor::CalculateAverageMetrics(
    const std::vector<SM120KernelMetrics>& history) const {
    
    SM120KernelMetrics avg;
    if (history.empty()) return avg;
    
    for (const auto& metrics : history) {
        avg.execution_time_ms += metrics.execution_time_ms;
        avg.memory_bandwidth_gb_s += metrics.memory_bandwidth_gb_s;
        avg.arithmetic_intensity += metrics.arithmetic_intensity;
        avg.occupancy_percent += metrics.occupancy_percent;
        avg.blocks_launched += metrics.blocks_launched;
        avg.threads_per_block += metrics.threads_per_block;
        avg.shared_memory_bytes += metrics.shared_memory_bytes;
        avg.register_count += metrics.register_count;
    }
    
    size_t count = history.size();
    avg.execution_time_ms /= count;
    avg.memory_bandwidth_gb_s /= count;
    avg.arithmetic_intensity /= count;
    avg.occupancy_percent /= count;
    avg.blocks_launched /= count;
    avg.threads_per_block /= count;
    avg.shared_memory_bytes /= count;
    avg.register_count /= count;
    
    return avg;
}

// Resource Manager implementation
SM120ResourceManager& SM120ResourceManager::Instance() {
    static SM120ResourceManager instance;
    return instance;
}

SM120ResourceManager::~SM120ResourceManager() {
    CleanupAll();
}

void SM120ResourceManager::RegisterBuffer(void* ptr, size_t size, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    BufferInfo info;
    info.size = size;
    info.name = name;
    info.allocated_time = std::chrono::steady_clock::now();
    
    buffers_[ptr] = info;
}

void SM120ResourceManager::UnregisterBuffer(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_.erase(ptr);
}

void SM120ResourceManager::CleanupAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all registered buffers
    for (auto& [ptr, info] : buffers_) {
        SM120ErrorHandler::Instance().LogWarning("Cleaning up leaked buffer: " + info.name);
        cudaFree(ptr);
    }
    buffers_.clear();
    
    // Destroy all registered streams
    for (auto& [stream, info] : streams_) {
        SM120ErrorHandler::Instance().LogWarning("Cleaning up stream: " + info.name);
        cudaStreamDestroy(stream);
    }
    streams_.clear();
}

void SM120ResourceManager::RegisterStream(cudaStream_t stream, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    StreamInfo info;
    info.name = name;
    info.created_time = std::chrono::steady_clock::now();
    
    streams_[stream] = info;
}

void SM120ResourceManager::UnregisterStream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    streams_.erase(stream);
}

void SM120ResourceManager::SynchronizeAllStreams() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [stream, info] : streams_) {
        SM120_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

size_t SM120ResourceManager::GetTotalAllocatedMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t total = 0;
    for (const auto& [ptr, info] : buffers_) {
        total += info.size;
    }
    
    return total;
}

int SM120ResourceManager::GetActiveBufferCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffers_.size();
}

int SM120ResourceManager::GetActiveStreamCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return streams_.size();
}

// Capability Checker implementation
SM120CapabilityChecker& SM120CapabilityChecker::Instance() {
    static SM120CapabilityChecker instance;
    return instance;
}

SM120CapabilityChecker::SM120CapabilityChecker() : capabilities_checked_(false) {
    CheckCapabilities();
}

void SM120CapabilityChecker::CheckCapabilities() {
    if (capabilities_checked_) return;
    
    int device;
    SM120_CUDA_CHECK(cudaGetDevice(&device));
    SM120_CUDA_CHECK(cudaGetDeviceProperties(&device_prop_, device));
    
    capabilities_checked_ = true;
}

bool SM120CapabilityChecker::IsRTX50SeriesGPU() const {
    // Check if this is a Blackwell architecture GPU (compute capability 12.x)
    return GetComputeCapabilityMajor() == 12;
}

bool SM120CapabilityChecker::SupportsSM120() const {
    return GetComputeCapabilityMajor() == 12 && GetComputeCapabilityMinor() == 0;
}

bool SM120CapabilityChecker::SupportsTensorCores() const {
    // Tensor cores are supported from compute capability 7.0+
    return GetComputeCapabilityMajor() >= 7;
}

bool SM120CapabilityChecker::SupportsCooperativeGroups() const {
    // Cooperative groups supported from compute capability 6.0+
    return GetComputeCapabilityMajor() >= 6;
}

bool SM120CapabilityChecker::SupportsBFloat16() const {
    // BFloat16 supported from compute capability 8.0+
    return GetComputeCapabilityMajor() >= 8;
}

bool SM120CapabilityChecker::SupportsFP8() const {
    // FP8 supported from compute capability 12.0+ (Blackwell)
    return GetComputeCapabilityMajor() >= 12;
}

int SM120CapabilityChecker::GetComputeCapabilityMajor() const {
    return device_prop_.major;
}

int SM120CapabilityChecker::GetComputeCapabilityMinor() const {
    return device_prop_.minor;
}

size_t SM120CapabilityChecker::GetSharedMemoryPerBlock() const {
    return device_prop_.sharedMemPerBlock;
}

int SM120CapabilityChecker::GetMaxThreadsPerBlock() const {
    return device_prop_.maxThreadsPerBlock;
}

int SM120CapabilityChecker::GetWarpSize() const {
    return device_prop_.warpSize;
}

std::string SM120CapabilityChecker::GetGPUName() const {
    return std::string(device_prop_.name);
}

std::string SM120CapabilityChecker::GetDriverVersion() const {
    int driver_version;
    cudaDriverGetVersion(&driver_version);
    
    int major = driver_version / 1000;
    int minor = (driver_version % 1000) / 10;
    
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string SM120CapabilityChecker::GetCUDAVersion() const {
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    
    int major = runtime_version / 1000;
    int minor = (runtime_version % 1000) / 10;
    
    return std::to_string(major) + "." + std::to_string(minor);
}

void SM120CapabilityChecker::PrintCapabilities() const {
    std::cout << "\n=== SM120 GPU Capabilities ===" << std::endl;
    std::cout << "GPU Name: " << GetGPUName() << std::endl;
    std::cout << "Compute Capability: " << GetComputeCapabilityMajor() << "." << GetComputeCapabilityMinor() << std::endl;
    std::cout << "RTX 50-series: " << (IsRTX50SeriesGPU() ? "Yes" : "No") << std::endl;
    std::cout << "SM120 Support: " << (SupportsSM120() ? "Yes" : "No") << std::endl;
    std::cout << "Tensor Cores: " << (SupportsTensorCores() ? "Yes" : "No") << std::endl;
    std::cout << "Cooperative Groups: " << (SupportsCooperativeGroups() ? "Yes" : "No") << std::endl;
    std::cout << "BFloat16: " << (SupportsBFloat16() ? "Yes" : "No") << std::endl;
    std::cout << "FP8: " << (SupportsFP8() ? "Yes" : "No") << std::endl;
    std::cout << "Shared Memory per Block: " << GetSharedMemoryPerBlock() << " bytes" << std::endl;
    std::cout << "Max Threads per Block: " << GetMaxThreadsPerBlock() << std::endl;
    std::cout << "Warp Size: " << GetWarpSize() << std::endl;
    std::cout << "CUDA Driver Version: " << GetDriverVersion() << std::endl;
    std::cout << "CUDA Runtime Version: " << GetCUDAVersion() << std::endl;
    std::cout << "=== End Capabilities ===" << std::endl;
}

// Fallback operations
template<typename T>
cudaError_t FallbackMatMul(const T* A, const T* B, T* C,
                          int M, int N, int K, 
                          cudaStream_t stream) {
    // Use cuBLAS for fallback
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasStatus_t status;
    if constexpr (std::is_same_v<T, float>) {
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K, &alpha, B, N, A, K, &beta, C, N);
    } else if constexpr (std::is_same_v<T, half>) {
        status = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K, 
                           reinterpret_cast<const __half*>(&alpha),
                           reinterpret_cast<const __half*>(B), N,
                           reinterpret_cast<const __half*>(A), K,
                           reinterpret_cast<const __half*>(&beta),
                           reinterpret_cast<__half*>(C), N);
    }
    
    cublasDestroy(handle);
    
    return (status == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// Explicit template instantiations
template cudaError_t FallbackMatMul<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template cudaError_t FallbackMatMul<half>(const half*, const half*, half*, int, int, int, cudaStream_t);
