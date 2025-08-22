/*
 * Python Bindings for TensorFlow SM120 Operations
 * 
 * This file provides Python bindings using pybind11 for the SM120 optimized
 * TensorFlow operations, enabling seamless integration with Python workflows.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/framework/python_api.h"

#if GOOGLE_CUDA
#include "src/cuda_kernels/sm120_kernel_launcher_fixed.h"
#endif

namespace py = pybind11;

namespace tensorflow {
namespace sm120_python {

// Python wrapper for SM120 capabilities detection
py::dict get_sm120_capabilities() {
    py::dict result;
    
#if GOOGLE_CUDA
    if (sm120_kernels::IsSM120Supported()) {
        auto caps = sm120_kernels::GetSM120AdvancedCapabilities();
        
        result["available"] = true;
        result["major_version"] = caps.major_version;
        result["minor_version"] = caps.minor_version;
        result["multiprocessor_count"] = caps.multiprocessor_count;
        result["shared_memory_per_sm"] = caps.shared_memory_per_multiprocessor;
        result["l2_cache_size"] = caps.l2_cache_size;
        result["supports_tensor_cores_5th_gen"] = caps.supports_tensor_cores_5th_gen;
        result["supports_fp8_arithmetic"] = caps.supports_fp8_arithmetic;
        result["peak_fp32_performance_tflops"] = caps.peak_fp32_performance_tflops;
        result["peak_fp16_performance_tflops"] = caps.peak_fp16_performance_tflops;
        result["peak_memory_bandwidth_gbps"] = caps.peak_memory_bandwidth_gbps;
    } else {
        result["available"] = false;
        result["error"] = "No SM120 compatible GPU found";
    }
#else
    result["available"] = false;
    result["error"] = "CUDA support not compiled";
#endif
    
    return result;
}

// Python wrapper for memory bandwidth benchmarking
float benchmark_memory_bandwidth(int size_mb = 1000) {
#if GOOGLE_CUDA
    return sm120_kernels::BenchmarkSM120MemoryBandwidth(size_mb * 1024 * 1024);
#else
    return 0.0f;
#endif
}

// Python wrapper for compute throughput benchmarking
float benchmark_compute_throughput(const std::string& operation_type, 
                                  int problem_size,
                                  int precision_mode = 0) {
#if GOOGLE_CUDA
    auto precision = static_cast<sm120_kernels::PrecisionMode>(precision_mode);
    return sm120_kernels::BenchmarkSM120ComputeThroughput(
        operation_type, problem_size, precision);
#else
    return 0.0f;
#endif
}

// Python wrapper for optimal kernel configuration
py::dict calculate_optimal_config(int problem_size, int element_size, 
                                 const std::string& kernel_name) {
    py::dict result;
    
#if GOOGLE_CUDA
    auto config = sm120_kernels::CalculateOptimalKernelConfig(
        problem_size, element_size, kernel_name.c_str());
    
    result["grid_x"] = config.grid_size.x;
    result["grid_y"] = config.grid_size.y;
    result["grid_z"] = config.grid_size.z;
    result["block_x"] = config.block_size.x;
    result["block_y"] = config.block_size.y;
    result["block_z"] = config.block_size.z;
    result["shared_memory_size"] = config.shared_memory_size;
    result["registers_per_thread"] = config.registers_per_thread;
    result["expected_occupancy"] = config.expected_occupancy;
#else
    result["error"] = "CUDA support not available";
#endif
    
    return result;
}

} // namespace sm120_python
} // namespace tensorflow

// Python module definition
PYBIND11_MODULE(_sm120_ops, m) {
    m.doc() = "TensorFlow SM120 Operations - Python Bindings";
    
    // Module information
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "TensorFlow SM120 Project";
    
    // Capability detection
    m.def("get_sm120_capabilities", &tensorflow::sm120_python::get_sm120_capabilities,
          "Get detailed SM120 GPU capabilities");
    
    // Benchmarking functions
    m.def("benchmark_memory_bandwidth", &tensorflow::sm120_python::benchmark_memory_bandwidth,
          "Benchmark GPU memory bandwidth",
          py::arg("size_mb") = 1000);
    
    m.def("benchmark_compute_throughput", &tensorflow::sm120_python::benchmark_compute_throughput,
          "Benchmark compute throughput for specific operations",
          py::arg("operation_type"), py::arg("problem_size"), py::arg("precision_mode") = 0);
    
    // Configuration utilities
    m.def("calculate_optimal_config", &tensorflow::sm120_python::calculate_optimal_config,
          "Calculate optimal kernel configuration for given problem size",
          py::arg("problem_size"), py::arg("element_size"), py::arg("kernel_name"));
    
    // Constants
    m.attr("SM120_WARP_SIZE") = 32;
    m.attr("SM120_MAX_THREADS_PER_BLOCK") = 1024;
    m.attr("SM120_SHARED_MEMORY_SIZE") = 163840;
    m.attr("SM120_L2_CACHE_SIZE") = 114688 * 1024;
    
    // Enums
    py::enum_<tensorflow::sm120_kernels::PrecisionMode>(m, "PrecisionMode")
        .value("FP32", tensorflow::sm120_kernels::PrecisionMode::FP32)
        .value("FP16", tensorflow::sm120_kernels::PrecisionMode::FP16)
        .value("BF16", tensorflow::sm120_kernels::PrecisionMode::BF16)
        .value("INT8", tensorflow::sm120_kernels::PrecisionMode::INT8)
        .value("FP8_E4M3", tensorflow::sm120_kernels::PrecisionMode::FP8_E4M3)
        .value("FP8_E5M2", tensorflow::sm120_kernels::PrecisionMode::FP8_E5M2);
    
    py::enum_<tensorflow::sm120_kernels::OptimizationLevel>(m, "OptimizationLevel")
        .value("BASIC", tensorflow::sm120_kernels::OptimizationLevel::BASIC)
        .value("ADVANCED", tensorflow::sm120_kernels::OptimizationLevel::ADVANCED)
        .value("AGGRESSIVE", tensorflow::sm120_kernels::OptimizationLevel::AGGRESSIVE);
    
    py::enum_<tensorflow::sm120_kernels::ActivationType>(m, "ActivationType")
        .value("RELU", tensorflow::sm120_kernels::ActivationType::RELU)
        .value("LEAKY_RELU", tensorflow::sm120_kernels::ActivationType::LEAKY_RELU)
        .value("GELU", tensorflow::sm120_kernels::ActivationType::GELU)
        .value("SWISH", tensorflow::sm120_kernels::ActivationType::SWISH)
        .value("MISH", tensorflow::sm120_kernels::ActivationType::MISH)
        .value("TANH", tensorflow::sm120_kernels::ActivationType::TANH)
        .value("SIGMOID", tensorflow::sm120_kernels::ActivationType::SIGMOID);
}
