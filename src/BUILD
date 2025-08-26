# BUILD file for TensorFlow sm_120 optimized kernels and operations
# This file defines the build targets for RTX 50-series GPU optimizations

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library", "tf_kernel_library")
load("//tensorflow:tensorflow.default.bzl", "tf_cuda_cc_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

# CUDA kernel library for sm_120 optimizations
tf_cuda_library(
    name = "sm120_optimized_kernels",
    srcs = if_cuda([
        "cuda_kernels/sm120_optimized_kernels_fixed.cu",
    ]),
    hdrs = if_cuda([
        "cuda_kernels/sm120_kernel_launcher_fixed.h",
    ]),
    copts = if_cuda([
        "-DGOOGLE_CUDA=1",
        "-DEIGEN_USE_GPU",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_89,code=sm_89",   # Highest supported target
        "-gencode=arch=compute_89,code=sm_89",    # RTX 40-series fallback
        "-gencode=arch=compute_86,code=sm_86",    # RTX 30-series fallback
        "-use_fast_math",
        "-O3",
        "--maxrregcount=128",
        "-Wno-deprecated-gpu-targets",
        "-DCUDA_API_PER_THREAD_DEFAULT_STREAM",
        "-DTENSORFLOW_USE_ROCM=0",
    ]),
    cuda_deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
        "@local_config_cuda//cuda:cusolver",
        "@local_config_cuda//cuda:cusparse",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util:gpu_kernel_helper",
        "//tensorflow/core/common_runtime/gpu:gpu_event_mgr",
        "//third_party/eigen3",
        "@cub_archive//:cub",
    ],
    alwayslink = 1,
)

# Kernel implementation library
tf_cuda_library(
    name = "sm120_kernel_implementations",
    srcs = if_cuda([
        "tensorflow_ops/sm120_kernel_implementations.cu",
    ]),
    hdrs = if_cuda([
        "cuda_kernels/sm120_kernel_launcher_fixed.h",
    ]),
    copts = if_cuda([
        "-DGOOGLE_CUDA=1",
        "-DEIGEN_USE_GPU",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_86,code=sm_86",
        "-use_fast_math",
        "-O3",
        "--maxrregcount=128",
        "-Xptxas=-v",  # Verbose PTX assembly
        "-lineinfo",   # Debug line info
    ]),
    cuda_deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cudnn",
    ],
    deps = [
        ":sm120_optimized_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util:gpu_kernel_helper",
        "//third_party/eigen3",
        "@cub_archive//:cub",
    ],
    alwayslink = 1,
)

# TensorFlow operations library
tf_kernel_library(
    name = "sm120_ops",
    srcs = [
        "tensorflow_ops/sm120_ops_fixed.cc",
    ],
    hdrs = if_cuda([
        "cuda_kernels/sm120_kernel_launcher_fixed.h",
    ]),
    gpu_srcs = if_cuda([
        "tensorflow_ops/sm120_ops_fixed.cc",
    ]),
    deps = [
        ":sm120_kernel_implementations",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util:gpu_kernel_helper",
        "//tensorflow/core/common_runtime/gpu:gpu_event_mgr",
        "//third_party/eigen3",
    ] + if_cuda([
        ":sm120_optimized_kernels",
    ]),
    alwayslink = 1,
)

# Python bindings for sm_120 operations
tf_cuda_library(
    name = "sm120_python_ops",
    srcs = if_cuda([
        "python_bindings/sm120_python_ops.cc",
    ]),
    hdrs = if_cuda([
        "python_bindings/sm120_python_ops.h",
    ]),
    copts = if_cuda([
        "-DGOOGLE_CUDA=1",
        "-DEIGEN_USE_GPU",
    ]),
    deps = [
        ":sm120_ops",
        "//tensorflow/core:framework",
        "//tensorflow/python:pywrap_tensorflow_internal",
        "//third_party/python_runtime:headers",
    ],
    alwayslink = 1,
)

# Test suite for sm_120 kernels
tf_cuda_cc_test(
    name = "sm120_kernels_test",
    size = "large",
    srcs = if_cuda([
        "tests/sm120_kernels_test.cc",
    ]),
    tags = [
        "gpu",
        "manual",  # Requires RTX 50-series GPU
        "notap",
    ],
    deps = [
        ":sm120_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime/gpu:gpu_init",
        "@com_google_googletest//:gtest",
    ],
)

# Benchmark suite
tf_cuda_cc_test(
    name = "sm120_benchmark",
    size = "large", 
    srcs = if_cuda([
        "benchmarks/sm120_benchmark.cc",
    ]),
    tags = [
        "gpu",
        "benchmark",
        "manual",
    ],
    deps = [
        ":sm120_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/util:command_line_flags",
        "@com_google_benchmark//:benchmark",
    ],
)

# Configuration check for sm_120 support
genrule(
    name = "check_sm120_support",
    outs = ["sm120_support_check.txt"],
    cmd = select({
        "@local_config_cuda//cuda:using_nvcc": """
            if nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | grep -q "12.0"; then
                echo "RTX 50-series GPU with sm_120 support detected" > $@
            else
                echo "Warning: No RTX 50-series GPU detected. Falling back to compatibility mode." > $@
            fi
        """,
        "//conditions:default": """
            echo "CUDA not available. sm_120 optimizations disabled." > $@
        """,
    }),
)

# Documentation generation
genrule(
    name = "generate_docs",
    srcs = [
        "cuda_kernels/sm120_kernel_launcher_fixed.h",
        "tensorflow_ops/sm120_ops_fixed.cc",
    ],
    outs = ["sm120_api_docs.html"],
    cmd = """
        echo "<html><head><title>SM120 API Documentation</title></head><body>" > $@
        echo "<h1>TensorFlow sm_120 Optimizations API</h1>" >> $@
        echo "<h2>CUDA Kernels</h2>" >> $@
        grep -E "^(template|cudaError_t|class|struct)" $(location cuda_kernels/sm120_kernel_launcher_fixed.h) | \
            sed 's/</\&lt;/g' | sed 's/>/\&gt;/g' | sed 's/^/<pre>/' | sed 's/$$/</pre>/' >> $@
        echo "<h2>TensorFlow Operations</h2>" >> $@
        grep -E "^(REGISTER_OP|template|class)" $(location tensorflow_ops/sm120_ops_fixed.cc) | \
            sed 's/</\&lt;/g' | sed 's/>/\&gt;/g' | sed 's/^/<pre>/' | sed 's/$$/</pre>/' >> $@
        echo "</body></html>" >> $@
    """,
)

# Filegroup for all sm_120 sources
filegroup(
    name = "sm120_all_sources",
    srcs = glob([
        "**/*.cu",
        "**/*.cc", 
        "**/*.h",
        "**/*.hpp",
    ]),
)

# Alias for easy reference
alias(
    name = "sm120",
    actual = ":sm120_ops",
)

# Export targets for external use
exports_files([
    "cuda_kernels/sm120_kernel_launcher_fixed.h",
    "tensorflow_ops/sm120_ops_fixed.cc",
])
