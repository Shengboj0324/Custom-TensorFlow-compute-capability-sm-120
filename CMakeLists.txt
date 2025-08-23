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
find_package(CUDA 12.4 REQUIRED)
find_package(PkgConfig REQUIRED)

# CUDA architecture settings for sm_120
set(CMAKE_CUDA_ARCHITECTURES "120;89;86;80;75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -O3 --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# Advanced CUDA compiler flags for sm_120
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_120,code=sm_120")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
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

# Find TensorFlow
find_path(TENSORFLOW_INCLUDE_DIR
    NAMES tensorflow/core/framework/op.h
    PATHS
        /usr/local/include
        /usr/include
        ${CMAKE_INSTALL_PREFIX}/include
    PATH_SUFFIXES tensorflow
)

if(NOT TENSORFLOW_INCLUDE_DIR)
    message(FATAL_ERROR "TensorFlow headers not found. Please install TensorFlow development headers.")
endif()

message(STATUS "TensorFlow include directory: ${TENSORFLOW_INCLUDE_DIR}")

# Find TensorFlow libraries
find_library(TENSORFLOW_FRAMEWORK_LIB
    NAMES tensorflow_framework
    PATHS
        /usr/local/lib
        /usr/lib
        ${CMAKE_INSTALL_PREFIX}/lib
)

# CUDA libraries
find_library(CUDA_CUDART_LIBRARY cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CUBLAS_LIBRARY cublas ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CUDNN_LIBRARY cudnn ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CUFFT_LIBRARY cufft ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CURAND_LIBRARY curand ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CUSOLVER_LIBRARY cusolver ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CUSPARSE_LIBRARY cusparse ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Check for cuDNN
find_path(CUDNN_INCLUDE_DIR cudnn.h
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/include
          /usr/include
          /usr/local/include)

if(NOT CUDNN_INCLUDE_DIR)
    message(FATAL_ERROR "cuDNN headers not found. Please install cuDNN 9.x for CUDA 12.4+")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TENSORFLOW_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Preprocessor definitions
add_definitions(-DGOOGLE_CUDA=1)
add_definitions(-DEIGEN_USE_GPU)
add_definitions(-DTENSORFLOW_USE_ROCM=0)

if(HAVE_SM120_GPU)
    add_definitions(-DHAVE_SM120_GPU=1)
endif()

# Source files
set(SM120_CUDA_SOURCES
    src/cuda_kernels/sm120_optimized_kernels_fixed.cu
    src/tensorflow_ops/sm120_kernel_implementations.cu
)

set(SM120_CXX_SOURCES
    src/tensorflow_ops/sm120_ops_fixed.cc
)

set(SM120_HEADERS
    src/cuda_kernels/sm120_kernel_launcher_fixed.h
)

# Create CUDA library for kernels
add_library(sm120_cuda_kernels SHARED ${SM120_CUDA_SOURCES})

target_link_libraries(sm120_cuda_kernels
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUBLAS_LIBRARY}
    ${CUDA_CUDNN_LIBRARY}
    ${CUDA_CUFFT_LIBRARY}
    ${CUDA_CURAND_LIBRARY}
    ${CUDA_CUSOLVER_LIBRARY}
    ${CUDA_CUSPARSE_LIBRARY}
)

set_target_properties(sm120_cuda_kernels PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create TensorFlow operations library
add_library(sm120_tensorflow_ops SHARED ${SM120_CXX_SOURCES})

target_link_libraries(sm120_tensorflow_ops
    sm120_cuda_kernels
    ${TENSORFLOW_FRAMEWORK_LIB}
)

set_target_properties(sm120_tensorflow_ops PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "_sm120_ops"
)

# Python extension module
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

# Installation
install(TARGETS sm120_cuda_kernels sm120_tensorflow_ops
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

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
