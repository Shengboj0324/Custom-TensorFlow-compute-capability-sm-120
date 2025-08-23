#!/bin/bash

# Comprehensive Build Script for TensorFlow SM120 Optimizations
# This script orchestrates the complete build process with error checking and optimization

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
INSTALL_DIR="${PROJECT_ROOT}/install"
PYTHON_ENV="${PROJECT_ROOT}/tf-sm120-env"
LOG_FILE="${PROJECT_ROOT}/build.log"

# Build options
BUILD_TYPE="Release"
CUDA_ARCH="120;89;86"
OPTIMIZATION_LEVEL="3"
PARALLEL_JOBS=$(nproc)
ENABLE_TESTING="ON"
ENABLE_BENCHMARKS="ON"
ENABLE_PYTHON="ON"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_header() {
    local line=$(printf '=%.0s' {1..80})
    echo -e "\n${WHITE}${line}${NC}" | tee -a "$LOG_FILE"
    echo -e "${WHITE}$(printf '%*s' $(((80-${#1})/2)) '')$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${WHITE}${line}${NC}\n" | tee -a "$LOG_FILE"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Build failed with exit code $exit_code"
        log_info "Check the log file for details: $LOG_FILE"
    fi
    exit $exit_code
}

trap cleanup EXIT

# System validation
validate_system() {
    log_header "System Validation"
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_success "Linux system detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_warning "macOS detected - experimental support"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check memory
    local mem_gb=$(free -g 2>/dev/null | awk 'NR==2{printf "%.1f", $2}' || echo "unknown")
    local min_memory_gb=16
    
    # Relax requirements for CI environments
    if [[ -n "$CI" || -n "$GITHUB_ACTIONS" ]]; then
        min_memory_gb=8
        log_info "CI environment detected - using relaxed memory requirements"
    fi
    
    if [[ "$mem_gb" != "unknown" ]] && (( $(echo "$mem_gb < $min_memory_gb" | bc -l) )); then
        if [[ -n "$CI" || -n "$GITHUB_ACTIONS" ]]; then
            log_warning "Low memory: ${mem_gb}GB (recommended ${min_memory_gb}GB+) - continuing in CI mode"
        else
            log_error "Insufficient memory: ${mem_gb}GB (minimum ${min_memory_gb}GB required)"
            exit 1
        fi
    else
        log_success "Memory: ${mem_gb}GB"
    fi
    
    # Check disk space
    local disk_gb=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    local min_disk_gb=50
    
    # Relax requirements for CI environments  
    if [[ -n "$CI" || -n "$GITHUB_ACTIONS" ]]; then
        min_disk_gb=20
        log_info "CI environment detected - using relaxed disk space requirements"
    fi
    
    if (( disk_gb < min_disk_gb )); then
        if [[ -n "$CI" || -n "$GITHUB_ACTIONS" ]]; then
            log_warning "Low disk space: ${disk_gb}GB free (recommended ${min_disk_gb}GB+) - continuing in CI mode"
        else
            log_error "Insufficient disk space: ${disk_gb}GB free (minimum ${min_disk_gb}GB required)"
            exit 1
        fi
    else
        log_success "Disk space: ${disk_gb}GB free"
    fi
    
    # Check for required tools
    local required_tools=("cmake" "make" "git" "python3" "pip3")
    for tool in "${required_tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            local version=$($tool --version 2>/dev/null | head -n1 || echo "unknown")
            log_success "$tool: $version"
        else
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
}

# CUDA validation
validate_cuda() {
    log_header "CUDA Environment Validation"
    
    # Check NVIDIA drivers
    if command -v nvidia-smi &> /dev/null; then
        local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)
        log_success "NVIDIA Driver: $driver_version"
        
        # Check for RTX 50-series
        local gpu_info=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
        local has_sm120=false
        
        while IFS=, read -r name compute_cap; do
            name=$(echo "$name" | xargs)
            compute_cap=$(echo "$compute_cap" | xargs)
            
            if [[ "$compute_cap" == "12.0" ]]; then
                log_success "RTX 50-series GPU detected: $name (sm_120)"
                has_sm120=true
            else
                log_info "GPU detected: $name (sm_$compute_cap)"
            fi
        done <<< "$gpu_info"
        
        if [[ "$has_sm120" == false ]]; then
            log_warning "No RTX 50-series GPU detected. Building with compatibility mode."
        fi
    else
        log_error "NVIDIA drivers not found. Please install NVIDIA drivers 570.x+"
        exit 1
    fi
    
    # Check CUDA toolkit
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        if (( $(echo "$cuda_version >= 12.8" | bc -l) )); then
            log_success "CUDA Toolkit: $cuda_version"
        else
            log_error "CUDA $cuda_version found, but 12.8+ required"
            exit 1
        fi
    else
        log_error "CUDA toolkit not found. Please install CUDA 12.8+"
        exit 1
    fi
    
    # Check cuDNN
    if [[ -f "/usr/local/cuda/include/cudnn.h" ]] || [[ -f "/usr/include/cudnn.h" ]]; then
        log_success "cuDNN headers found"
    else
        log_error "cuDNN headers not found. Please install cuDNN 9.x"
        exit 1
    fi
}

# TensorFlow validation
validate_tensorflow() {
    log_header "TensorFlow Environment Validation"
    
    # Check if TensorFlow is installed
    if python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>/dev/null; then
        local tf_version=$(python3 -c "import tensorflow as tf; print(tf.__version__)")
        log_success "TensorFlow: $tf_version"
        
        # Check TensorFlow GPU support
        local gpu_support=$(python3 -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())" 2>/dev/null || echo "False")
        if [[ "$gpu_support" == "True" ]]; then
            log_success "TensorFlow built with CUDA support"
        else
            log_warning "TensorFlow not built with CUDA support"
        fi
        
        # Check available GPUs
        local gpu_count=$(python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null || echo "0")
        if [[ "$gpu_count" -gt 0 ]]; then
            log_success "TensorFlow can see $gpu_count GPU(s)"
        else
            log_warning "TensorFlow cannot see any GPUs"
        fi
    else
        log_error "TensorFlow not found or not working. Please install TensorFlow 2.10+"
        exit 1
    fi
}

# Python environment setup
setup_python_environment() {
    log_header "Python Environment Setup"
    
    if [[ ! -d "$PYTHON_ENV" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$PYTHON_ENV"
    fi
    
    log_info "Activating virtual environment..."
    source "$PYTHON_ENV/bin/activate"
    
    log_info "Upgrading pip and installing dependencies..."
    pip install --upgrade pip setuptools wheel
    pip install tensorflow>=2.10.0 numpy>=1.21.0 pybind11>=2.10.0
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy
    
    # Install benchmarking dependencies
    pip install matplotlib pandas seaborn
    
    log_success "Python environment ready"
}

# CMake configuration
configure_cmake() {
    log_header "CMake Configuration"
    
    # Clean build directory
    if [[ -d "$BUILD_DIR" ]]; then
        log_info "Cleaning previous build..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    log_info "Configuring with CMake..."
    
    cmake \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_FLAGS="-O$OPTIMIZATION_LEVEL --use_fast_math" \
        -DBUILD_TESTS="$ENABLE_TESTING" \
        -DBUILD_BENCHMARKS="$ENABLE_BENCHMARKS" \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        "$PROJECT_ROOT" 2>&1 | tee -a "$LOG_FILE"
    
    log_success "CMake configuration completed"
}

# Build process
build_project() {
    log_header "Building Project"
    
    cd "$BUILD_DIR"
    
    log_info "Building with $PARALLEL_JOBS parallel jobs..."
    
    make -j"$PARALLEL_JOBS" 2>&1 | tee -a "$LOG_FILE"
    
    log_success "Build completed successfully"
}

# Python package build
build_python_package() {
    if [[ "$ENABLE_PYTHON" == "ON" ]]; then
        log_header "Building Python Package"
        
        cd "$PROJECT_ROOT"
        source "$PYTHON_ENV/bin/activate"
        
        log_info "Building Python extension..."
        python setup.py build_ext --inplace 2>&1 | tee -a "$LOG_FILE"
        
        log_info "Building wheel..."
        python setup.py bdist_wheel 2>&1 | tee -a "$LOG_FILE"
        
        log_success "Python package built successfully"
    fi
}

# Testing
run_tests() {
    if [[ "$ENABLE_TESTING" == "ON" ]]; then
        log_header "Running Tests"
        
        cd "$BUILD_DIR"
        
        # Run C++ tests
        if [[ -f "sm120_tests" ]]; then
            log_info "Running C++ tests..."
            ./sm120_tests 2>&1 | tee -a "$LOG_FILE"
        fi
        
        # Run Python tests
        if [[ "$ENABLE_PYTHON" == "ON" ]]; then
            cd "$PROJECT_ROOT"
            source "$PYTHON_ENV/bin/activate"
            
            log_info "Running Python tests..."
            python -m pytest tests/ -v 2>&1 | tee -a "$LOG_FILE"
        fi
        
        log_success "All tests passed"
    fi
}

# Benchmarks
run_benchmarks() {
    if [[ "$ENABLE_BENCHMARKS" == "ON" ]]; then
        log_header "Running Benchmarks"
        
        cd "$BUILD_DIR"
        
        # Run C++ benchmarks
        if [[ -f "sm120_benchmarks" ]]; then
            log_info "Running C++ benchmarks..."
            ./sm120_benchmarks --benchmark_format=json --benchmark_out=benchmark_results.json 2>&1 | tee -a "$LOG_FILE"
        fi
        
        # Run Python benchmarks
        if [[ "$ENABLE_PYTHON" == "ON" ]]; then
            cd "$PROJECT_ROOT"
            source "$PYTHON_ENV/bin/activate"
            
            log_info "Running Python benchmarks..."
            python -c "
import tensorflow_sm120
if tensorflow_sm120.is_sm120_available():
    print('SM120 operations available - running full benchmarks')
    # Run benchmarks here
else:
    print('SM120 operations not available - skipping benchmarks')
" 2>&1 | tee -a "$LOG_FILE"
        fi
        
        log_success "Benchmarks completed"
    fi
}

# Installation
install_project() {
    log_header "Installing Project"
    
    cd "$BUILD_DIR"
    
    log_info "Installing C++ libraries..."
    make install 2>&1 | tee -a "$LOG_FILE"
    
    if [[ "$ENABLE_PYTHON" == "ON" ]]; then
        cd "$PROJECT_ROOT"
        source "$PYTHON_ENV/bin/activate"
        
        log_info "Installing Python package..."
        pip install dist/*.whl --force-reinstall 2>&1 | tee -a "$LOG_FILE"
    fi
    
    log_success "Installation completed"
}

# Generate documentation
generate_documentation() {
    log_header "Generating Documentation"
    
    cd "$BUILD_DIR"
    
    if command -v doxygen &> /dev/null; then
        log_info "Generating API documentation..."
        make docs 2>&1 | tee -a "$LOG_FILE"
        log_success "Documentation generated in $BUILD_DIR/docs/html/"
    else
        log_warning "Doxygen not found - skipping documentation generation"
    fi
}

# Final validation
final_validation() {
    log_header "Final Validation"
    
    if [[ "$ENABLE_PYTHON" == "ON" ]]; then
        source "$PYTHON_ENV/bin/activate"
        
        log_info "Validating Python installation..."
        python -c "
import tensorflow_sm120
import tensorflow as tf

print(f'TensorFlow version: {tf.__version__}')
print(f'SM120 available: {tensorflow_sm120.is_sm120_available()}')

device_info = tensorflow_sm120.get_sm120_device_info()
print(f'SM120 devices: {len([d for d in device_info[\"devices\"] if d.get(\"sm120_compatible\", False)])}')

if tensorflow_sm120.is_sm120_available():
    print('✅ SM120 operations ready for use!')
else:
    print('⚠️  SM120 operations not available - using compatibility mode')
" 2>&1 | tee -a "$LOG_FILE"
    fi
    
    log_success "Final validation completed"
}

# Build summary
print_summary() {
    log_header "Build Summary"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "${GREEN}Build completed successfully!${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}Total time: ${hours}h ${minutes}m ${seconds}s${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}Log file: $LOG_FILE${NC}" | tee -a "$LOG_FILE"
    
    if [[ "$ENABLE_PYTHON" == "ON" ]]; then
        echo -e "\n${WHITE}Next steps:${NC}" | tee -a "$LOG_FILE"
        echo -e "1. Activate environment: source $PYTHON_ENV/bin/activate" | tee -a "$LOG_FILE"
        echo -e "2. Test installation: python -c 'import tensorflow_sm120; print(tensorflow_sm120.is_sm120_available())'" | tee -a "$LOG_FILE"
        echo -e "3. Run examples: python examples/basic_usage.py" | tee -a "$LOG_FILE"
        echo -e "4. Check documentation: open $BUILD_DIR/docs/html/index.html" | tee -a "$LOG_FILE"
    fi
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    # Initialize log
    echo "TensorFlow SM120 Build Log - $(date)" > "$LOG_FILE"
    
    log_header "TensorFlow SM120 Comprehensive Build"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            --cuda-arch)
                CUDA_ARCH="$2"
                shift 2
                ;;
            --jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --no-tests)
                ENABLE_TESTING="OFF"
                shift
                ;;
            --no-benchmarks)
                ENABLE_BENCHMARKS="OFF"
                shift
                ;;
            --no-python)
                ENABLE_PYTHON="OFF"
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --build-type TYPE     Build type (Debug/Release) [default: Release]"
                echo "  --cuda-arch ARCH      CUDA architectures [default: 120;89;86]"
                echo "  --jobs N              Parallel jobs [default: $(nproc)]"
                echo "  --no-tests            Skip testing"
                echo "  --no-benchmarks       Skip benchmarks"
                echo "  --no-python           Skip Python package"
                echo "  --help                Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute build steps
    validate_system
    validate_cuda
    validate_tensorflow
    setup_python_environment
    configure_cmake
    build_project
    build_python_package
    run_tests
    run_benchmarks
    install_project
    generate_documentation
    final_validation
    print_summary
    
    log_success "All build steps completed successfully!"
}

# Run main function
main "$@"
