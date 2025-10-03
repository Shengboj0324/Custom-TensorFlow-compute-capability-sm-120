# CMakeLists.txt for TensorFlow SM120 Optimizations
# Advanced build system for RTX 50-series GPU support

cmake_minimum_required(VERSION 3.18)
project(TensorFlow_SM120 LANGUAGES CXX CUDA)

# Project information
set(PROJECT_VERSION "1.0.0")
set(PROJECT_DESCRIPTION "TensorFlow optimizations for RTX 50-series GPUs (sm_120)")

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /O2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# Find required packages
find_package(CUDA 12.8 REQUIRED)
find_package(PkgConfig REQUIRED)

# CUDA architecture settings for sm_120
set(CMAKE_CUDA_ARCHITECTURES "89;86;80;75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -O3 --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-deprecated-declarations")  # Global deprecation suppression
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-error=deprecated-declarations")  # Don't treat as error
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# Global C++ flags to suppress TensorFlow deprecation warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

# Print configuration for debugging
message(STATUS "SM120 Build Configuration:")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "  CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "  Deprecation warnings suppressed: YES")

# Advanced CUDA compiler flags for sm_120 support
# Note: sm_120 now supported in CUDA 12.8
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_120,code=sm_120")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=128")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")

# Check for sm_120 GPU support
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_COMPUTE_CAPS
    ERROR_QUIET
)

string(FIND "${GPU_COMPUTE_CAPS}" "12.0" SM120_FOUND)
if(SM120_FOUND GREATER -1)
    message(STATUS "RTX 50-series GPU with sm_120 support detected")
    set(HAVE_SM120_GPU TRUE)
else()
    message(WARNING "No RTX 50-series GPU detected. Building with compatibility mode.")
    set(HAVE_SM120_GPU FALSE)
endif()

# Initialize build options
set(BUILD_TENSORFLOW_OPS OFF)

# Find TensorFlow
find_path(TENSORFLOW_INCLUDE_DIR
    NAMES tensorflow/core/framework/op.h
    PATHS
        /usr/local/include
        /usr/include
        ${CMAKE_INSTALL_PREFIX}/include
        # Add Python site-packages paths for CI environments
        $ENV{VIRTUAL_ENV}/lib/python3.11/site-packages/tensorflow/include
        /opt/hostedtoolcache/Python/3.11.*/x64/lib/python3.11/site-packages/tensorflow/include
    PATH_SUFFIXES tensorflow
)

if(NOT TENSORFLOW_INCLUDE_DIR)
    # Try to find TensorFlow via Python in CI environments
    execute_process(
        COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
        OUTPUT_VARIABLE TF_INCLUDE_FROM_PYTHON
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(TF_INCLUDE_FROM_PYTHON)
        set(TENSORFLOW_INCLUDE_DIR ${TF_INCLUDE_FROM_PYTHON})
        message(STATUS "Found TensorFlow headers via Python: ${TENSORFLOW_INCLUDE_DIR}")
    else()
        message(WARNING "TensorFlow headers not found. Only pure CUDA layers will be built.")
        set(BUILD_TENSORFLOW_OPS OFF)
    endif()
else()
    set(BUILD_TENSORFLOW_OPS ON)
endif()

if(BUILD_TENSORFLOW_OPS)
    message(STATUS "TensorFlow include directory: ${TENSORFLOW_INCLUDE_DIR}")
    message(STATUS "Building TensorFlow operations: YES")
else()
    message(STATUS "Building TensorFlow operations: NO (TensorFlow headers not found)")
endif()

# Find TensorFlow libraries
execute_process(
    COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
    OUTPUT_VARIABLE TF_LIB_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

find_library(TENSORFLOW_FRAMEWORK_LIB
    NAMES tensorflow_framework libtensorflow_framework.so.2 libtensorflow_framework
    PATHS
        ${TF_LIB_PATH}
        /usr/local/lib
        /usr/lib
        ${CMAKE_INSTALL_PREFIX}/lib
        /opt/hostedtoolcache/Python/*/x64/lib/python*/site-packages/tensorflow
)

# CUDA libraries
find_library(CUDA_CUDART_LIBRARY cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CUBLAS_LIBRARY cublas ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Find cuDNN library (including pip-installed locations)
find_library(CUDA_CUDNN_LIBRARY
    NAMES cudnn libcudnn
    PATHS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDNN_PIP_PATH}/lib
        /opt/hostedtoolcache/Python/*/x64/lib/python*/site-packages/nvidia/cudnn/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

find_library(CUDA_CUFFT_LIBRARY cufft ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CURAND_LIBRARY curand ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CUSOLVER_LIBRARY cusolver ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Validate critical libraries
if(NOT CUDA_CUDNN_LIBRARY)
    message(STATUS "cuDNN pip path: ${CUDNN_PIP_PATH}")
    message(WARNING "cuDNN library not found. Using stub library for compatibility.")
    # Create a stub library path for linking
    set(CUDA_CUDNN_LIBRARY "")
endif()

if(NOT TENSORFLOW_FRAMEWORK_LIB)
    message(STATUS "TensorFlow lib path: ${TF_LIB_PATH}")
    message(WARNING "TensorFlow framework library not found. Using Python TensorFlow installation.")
    # For pip-installed TensorFlow, we don't need the framework library
    set(TENSORFLOW_FRAMEWORK_LIB "")
endif()
find_library(CUDA_CUSPARSE_LIBRARY cusparse ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Check for cuDNN (including pip-installed locations)
execute_process(
    COMMAND python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])"
    OUTPUT_VARIABLE CUDNN_PIP_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

find_path(CUDNN_INCLUDE_DIR cudnn.h
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/include
          /usr/include
          /usr/local/include
          ${CUDNN_PIP_PATH}/include
          /opt/hostedtoolcache/Python/*/x64/lib/python*/site-packages/nvidia/cudnn/include)

if(NOT CUDNN_INCLUDE_DIR)
    message(STATUS "cuDNN pip path: ${CUDNN_PIP_PATH}")
    message(FATAL_ERROR "cuDNN headers not found. Please install cuDNN 9.x for CUDA 12.4+")
else()
    message(STATUS "Found cuDNN headers: ${CUDNN_INCLUDE_DIR}")
endif()

# Include directories - TensorFlow includes only if available
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Add TensorFlow includes only for TensorFlow ops
if(BUILD_TENSORFLOW_OPS AND TENSORFLOW_INCLUDE_DIR)
    include_directories(${TENSORFLOW_INCLUDE_DIR})
endif()

# Preprocessor definitions
add_definitions(-DGOOGLE_CUDA=1)
add_definitions(-DEIGEN_USE_GPU)
add_definitions(-DTENSORFLOW_USE_ROCM=0)

if(HAVE_SM120_GPU)
    add_definitions(-DHAVE_SM120_GPU=1)
endif()

# ============================================================================
# NEW LAYERED ARCHITECTURE SOURCE FILES
# ============================================================================

# Pure CUDA kernel layer (no external dependencies)
set(SM120_PURE_CUDA_SOURCES
    src/cuda_kernels/sm120_pure_kernels.cu
)

# C interface layer (calls pure CUDA kernels)
set(SM120_C_INTERFACE_SOURCES
    src/cuda_kernels/sm120_c_interface.cu
)

# TensorFlow wrapper layer (calls C interface)
set(SM120_TENSORFLOW_SOURCES
    src/tensorflow_ops/sm120_tensorflow_ops.cc
    src/tensorflow_ops/sm120_ops.cc
    src/tensorflow_ops/sm120_complete_ops.cc
    src/tensorflow_ops/sm120_gradients.cc
)

# Header files
set(SM120_HEADERS
    src/cuda_kernels/sm120_c_interface.h
)

# ============================================================================
# CREATE LAYERED LIBRARIES
# ============================================================================

# Step 1: Create pure CUDA kernels library (no external dependencies)
add_library(sm120_pure_kernels SHARED ${SM120_PURE_CUDA_SOURCES})

# Step 2: Create C interface library (depends on pure kernels)
add_library(sm120_c_interface SHARED ${SM120_C_INTERFACE_SOURCES})

# Step 3: Create TensorFlow ops library (depends on C interface) - conditional
if(BUILD_TENSORFLOW_OPS)
    add_library(sm120_tensorflow_ops SHARED ${SM120_TENSORFLOW_SOURCES})
endif()

# Link CUDA libraries (only if found)
set(CUDA_LIBS_TO_LINK
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUBLAS_LIBRARY}
    ${CUDA_CUFFT_LIBRARY}
    ${CUDA_CURAND_LIBRARY}
    ${CUDA_CUSOLVER_LIBRARY}
    ${CUDA_CUSPARSE_LIBRARY}
)

# Add cuDNN if found
if(CUDA_CUDNN_LIBRARY)
    list(APPEND CUDA_LIBS_TO_LINK ${CUDA_CUDNN_LIBRARY})
endif()

# ============================================================================
# CONFIGURE LIBRARY PROPERTIES AND LINKING
# ============================================================================

# Configure pure CUDA kernels library
set_target_properties(sm120_pure_kernels PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(sm120_pure_kernels
    ${CUDA_CUDART_LIBRARY}
)

# Configure C interface library
set_target_properties(sm120_c_interface PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(sm120_c_interface
    sm120_pure_kernels
    ${CUDA_CUDART_LIBRARY}
)

# Configure TensorFlow ops library - conditional
if(BUILD_TENSORFLOW_OPS)
    set_target_properties(sm120_tensorflow_ops PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(sm120_tensorflow_ops
        sm120_c_interface
        ${CUDA_CUDART_LIBRARY}
    )
endif()

# ============================================================================
# COMPILER FLAGS FOR EACH LAYER
# ============================================================================

# Pure CUDA kernels - minimal flags, no TensorFlow dependencies
target_compile_options(sm120_pure_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
        -Xcompiler=-fPIC
    >
)

# C interface - minimal flags
target_compile_options(sm120_c_interface PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
        -Xcompiler=-fPIC
    >
)

# TensorFlow ops - comprehensive flags to suppress TensorFlow warnings
if(BUILD_TENSORFLOW_OPS)
    target_compile_options(sm120_tensorflow_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-deprecated-declarations
            -Wno-error=deprecated-declarations
            -Wno-error
        >
    )
endif()

if(BUILD_TENSORFLOW_OPS)
    set_target_properties(sm120_tensorflow_ops PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME "_sm120_ops"
    )
endif()

# Python extension module - conditional
if(BUILD_TENSORFLOW_OPS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    add_library(sm120_python_module SHARED
        src/python_bindings/sm120_python_ops.cc
    )

    target_include_directories(sm120_python_module PRIVATE
        ${Python3_INCLUDE_DIRS}
    )

    target_link_libraries(sm120_python_module
        sm120_tensorflow_ops
        ${Python3_LIBRARIES}
    )

    set_target_properties(sm120_python_module PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_sm120_ops"
        SUFFIX ".so"
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Test executables
option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    find_package(GTest QUIET)
    
    if(GTest_FOUND)
        add_executable(sm120_tests
            tests/sm120_kernels_test.cc
            tests/sm120_ops_test.cc
        )
        
        target_link_libraries(sm120_tests
            sm120_tensorflow_ops
            GTest::gtest
            GTest::gtest_main
        )
        
        enable_testing()
        add_test(NAME SM120Tests COMMAND sm120_tests)
    else()
        message(WARNING "Google Test not found. Tests will not be built.")
    endif()
endif()

# Benchmark executables
option(BUILD_BENCHMARKS "Build benchmark executables" ON)

if(BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    
    if(benchmark_FOUND)
        add_executable(sm120_benchmarks
            benchmarks/sm120_benchmark.cc
            benchmarks/matmul_benchmark.cc
            benchmarks/conv2d_benchmark.cc
            benchmarks/attention_benchmark.cc
        )
        
        target_link_libraries(sm120_benchmarks
            sm120_tensorflow_ops
            benchmark::benchmark
            benchmark::benchmark_main
        )
    else()
        message(WARNING "Google Benchmark not found. Benchmarks will not be built.")
    endif()
endif()

# Documentation generation
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# ============================================================================
# INSTALLATION
# ============================================================================

# Install CUDA layers (always available)
install(TARGETS sm120_pure_kernels sm120_c_interface
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install TensorFlow ops if built
if(BUILD_TENSORFLOW_OPS)
    install(TARGETS sm120_tensorflow_ops
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(FILES ${SM120_HEADERS}
    DESTINATION include/tensorflow_sm120
)

install(FILES python/sm120_ops.py
    DESTINATION lib/python3/site-packages/tensorflow_sm120
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorFlowSM120Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorFlowSM120Config.cmake"
    INSTALL_DESTINATION lib/cmake/TensorFlowSM120
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TensorFlowSM120ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TensorFlowSM120Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorFlowSM120ConfigVersion.cmake"
    DESTINATION lib/cmake/TensorFlowSM120
)

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "tensorflow-sm120")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "TensorFlow SM120 Project")
set(CPACK_PACKAGE_CONTACT "tensorflow-sm120@example.com")

set(CPACK_GENERATOR "DEB;RPM;TGZ")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libgcc1, libstdc++6, cuda-runtime-12-8, libcudnn9")
set(CPACK_RPM_PACKAGE_REQUIRES "glibc, gcc, libstdc++, cuda-runtime-12-8, libcudnn9")

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "TensorFlow SM120 Configuration Summary:")
message(STATUS "======================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorFlow include: ${TENSORFLOW_INCLUDE_DIR}")
message(STATUS "SM120 GPU detected: ${HAVE_SM120_GPU}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "Build documentation: ${DOXYGEN_FOUND}")
message(STATUS "")

# Custom targets for development
add_custom_target(format
    COMMAND find src tests benchmarks -name "*.cc" -o -name "*.cu" -o -name "*.h" | xargs clang-format -i
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Formatting source code"
)

add_custom_target(lint
    COMMAND find src tests benchmarks -name "*.cc" -o -name "*.cu" -o -name "*.h" | xargs clang-tidy
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running static analysis"
)

add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
    COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Deep clean build directory"
)
