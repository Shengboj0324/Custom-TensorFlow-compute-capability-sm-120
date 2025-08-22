/*
 * Header for Python Bindings of SM120 Operations
 * 
 * This header declares the interface for Python bindings of TensorFlow
 * SM120 optimized operations using pybind11.
 */

#ifndef TENSORFLOW_SM120_PYTHON_OPS_H_
#define TENSORFLOW_SM120_PYTHON_OPS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace tensorflow {
namespace sm120_python {

// Capability detection and system information
pybind11::dict get_sm120_capabilities();

// Performance benchmarking
float benchmark_memory_bandwidth(int size_mb = 1000);
float benchmark_compute_throughput(const std::string& operation_type, 
                                  int problem_size, int precision_mode = 0);

// Configuration utilities
pybind11::dict calculate_optimal_config(int problem_size, int element_size, 
                                       const std::string& kernel_name);

// Utility functions
bool is_sm120_supported();
std::vector<int> get_supported_architectures();
std::string get_cuda_version();
std::string get_driver_version();

} // namespace sm120_python
} // namespace tensorflow

#endif // TENSORFLOW_SM120_PYTHON_OPS_H_
