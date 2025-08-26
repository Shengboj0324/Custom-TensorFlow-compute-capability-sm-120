#!/bin/bash

# Custom TensorFlow sm_120 Build Script
# Builds TensorFlow with native support for RTX 50-series GPUs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TENSORFLOW_VERSION="r2.20"
BUILD_DIR="$(pwd)/build"
PATCHES_DIR="$(pwd)/patches"
JOBS=$(nproc)

# Global variables (set during prerequisite checks)
CUDA_VERSION=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        log_error "CUDA not found. Please run setup-environment.sh first."
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    if ! (( $(echo "$CUDA_VERSION >= 12.4" | bc -l) )); then
        log_error "CUDA $CUDA_VERSION found, but 12.4+ required"
        exit 1
    fi
    log_success "CUDA $CUDA_VERSION found"
    
    # Check GCC
    if ! command -v gcc &> /dev/null; then
        log_error "GCC not found. Please install build-essential."
        exit 1
    fi

    local gcc_version=$(gcc --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -n1)
    if [[ $(echo "$gcc_version < 9.0" | bc -l) -eq 1 ]]; then
        log_error "GCC $gcc_version found, but version 9.0+ required"
        exit 1
    fi
    log_success "GCC $gcc_version found"
    
    # Check Bazel
    if ! command -v bazel &> /dev/null; then
        log_error "Bazel not found. Please run setup-environment.sh first."
        exit 1
    fi
    log_success "Bazel found"
    
    # Check Python environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_warning "No virtual environment detected. Attempting to activate tf-build-env..."
        
        # Try multiple possible locations for the virtual environment
        venv_paths=(
            "${BUILD_DIR}/../tf-build-env/bin/activate"
            "./tf-build-env/bin/activate"
            "../tf-build-env/bin/activate"
            "$HOME/tf-build-env/bin/activate"
        )
        
        venv_activated=false
        for venv_path in "${venv_paths[@]}"; do
            if [[ -f "$venv_path" ]]; then
                log_info "Found virtual environment at: $venv_path"
                source "$venv_path" && {
                    venv_activated=true
                    break
                }
            fi
        done
        
        if [[ "$venv_activated" == "false" ]]; then
            # In CI environments, try to use the system Python if available
            if [[ -n "$CI" || -n "$GITHUB_ACTIONS" ]]; then
                log_warning "Virtual environment not found in CI - using system Python"
                # Verify Python is available
                if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
                    log_error "No Python interpreter found"
                    exit 1
                fi
            else
                log_error "Failed to activate virtual environment. Please run setup-environment.sh first."
                exit 1
            fi
        fi
    fi
    
    # Log the Python environment being used
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_success "Python environment: $VIRTUAL_ENV"
    else
        log_success "Python environment: system Python ($(which python3 || which python))"
    fi
}

# Clone TensorFlow repository
clone_tensorflow() {
    log_info "Cloning TensorFlow repository..."
    
    if [[ -d "tensorflow" ]]; then
        log_info "TensorFlow directory exists. Updating..."
        cd tensorflow
        git fetch origin
        git checkout $TENSORFLOW_VERSION
        git pull origin $TENSORFLOW_VERSION
        cd ..
    else
        git clone --depth 1 --branch $TENSORFLOW_VERSION https://github.com/tensorflow/tensorflow.git
    fi
    
    log_success "TensorFlow $TENSORFLOW_VERSION ready"
}

# Apply patches for known issues
apply_patches() {
    log_info "Applying patches for known build issues..."
    
    cd tensorflow
    
    # Apply matrix naming fix
    if [[ -f "$PATCHES_DIR/fix-matrix-naming.patch" ]]; then
        log_info "Applying matrix naming fix..."
        git apply "$PATCHES_DIR/fix-matrix-naming.patch" || log_warning "Matrix naming patch already applied or failed"
    fi
    
    # Apply template error fix
    if [[ -f "$PATCHES_DIR/fix-template-errors.patch" ]]; then
        log_info "Applying template error fix..."
        git apply "$PATCHES_DIR/fix-template-errors.patch" || log_warning "Template error patch already applied or failed"
    fi
    
    # Apply C23 extensions fix
    if [[ -f "$PATCHES_DIR/fix-c23-extensions.patch" ]]; then
        log_info "Applying C23 extensions fix..."
        git apply "$PATCHES_DIR/fix-c23-extensions.patch" || log_warning "C23 extensions patch already applied or failed"
    fi
    
    cd ..
    log_success "Patches applied"
}

# Configure TensorFlow build
configure_tensorflow() {
    log_info "Configuring TensorFlow build..."
    
    cd tensorflow
    
    # Set environment variables for configuration
    export PYTHON_BIN_PATH=$(which python)
    export PYTHON_LIB_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
    export TF_ENABLE_XLA=1
    export TF_NEED_OPENCL_SYCL=0
    export TF_NEED_ROCM=0
    export TF_NEED_CUDA=1
    export TF_NEED_TENSORRT=0
    export TF_CUDA_VERSION=$CUDA_VERSION
    export TF_CUDNN_VERSION=9
    export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.4
    export CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
    export TF_CUDA_COMPUTE_CAPABILITIES="12.0"
    export GCC_HOST_COMPILER_PATH=$(which gcc)
    export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
    export TF_SET_ANDROID_WORKSPACE=0
    
    # Run configure script non-interactively
    log_info "Running configure script..."
    python configure.py
    
    cd ..
    log_success "TensorFlow configured for sm_120 build"
}

# Build TensorFlow C++ library
build_cpp_library() {
    log_info "Building TensorFlow C++ library..."
    log_info "This may take 2-4 hours depending on your system..."
    
    cd tensorflow
    
    # Build with optimizations for sm_120
    bazel build \
        --config=opt \
        --config=cuda \
        --copt=-Wno-error=c23-extensions \
        --copt=-Wno-error=unused-command-line-argument \
        --verbose_failures \
        --jobs=$JOBS \
        //tensorflow:libtensorflow.so \
        //tensorflow:libtensorflow_cc.so
    
    cd ..
    log_success "C++ library built successfully"
}

# Build Python wheel
build_python_wheel() {
    log_info "Building Python wheel..."
    
    cd tensorflow
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    
    # Build pip package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "$BUILD_DIR"
    
    cd ..
    log_success "Python wheel built successfully"
}

# Create wheel with proper naming
create_named_wheel() {
    log_info "Creating properly named wheel..."
    
    cd "$BUILD_DIR"
    
    # Find the generated wheel
    local wheel_file=$(ls tensorflow-*.whl | head -n1)
    if [[ -z "$wheel_file" ]]; then
        log_error "No wheel file found in $BUILD_DIR"
        exit 1
    fi
    
    # Extract wheel info
    local wheel_name=$(echo "$wheel_file" | sed 's/\.whl$//')
    local python_version=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    local platform=$(python -c "import sysconfig; print(sysconfig.get_platform().replace('-', '_'))")
    
    # Create new wheel name with sm_120 identifier
    local new_wheel_name="${wheel_name}_sm120-${python_version}-${python_version}-${platform}.whl"
    
    # Rename wheel
    mv "$wheel_file" "$new_wheel_name"
    
    cd ..
    log_success "Wheel renamed to: $new_wheel_name"
}

# Verify build
verify_build() {
    log_info "Verifying build..."
    
    # Check if library files exist
    if [[ -f "tensorflow/bazel-bin/tensorflow/libtensorflow.so" ]]; then
        log_success "libtensorflow.so built successfully"
    else
        log_error "libtensorflow.so not found"
        exit 1
    fi
    
    # Check if wheel exists
    local wheel_count=$(ls "$BUILD_DIR"/*.whl 2>/dev/null | wc -l)
    if [[ $wheel_count -gt 0 ]]; then
        log_success "Python wheel built successfully"
        log_info "Wheel location: $BUILD_DIR"
        ls -la "$BUILD_DIR"/*.whl
    else
        log_error "No wheel file found"
        exit 1
    fi
}

# Generate build report
generate_report() {
    log_info "Generating build report..."
    
    local report_file="$BUILD_DIR/build-report.txt"
    
    cat > "$report_file" << EOF
TensorFlow sm_120 Build Report
=============================

Build Date: $(date)
TensorFlow Version: $TENSORFLOW_VERSION
Target Architecture: sm_120 (RTX 50-series)

System Information:
- OS: $(lsb_release -d | cut -f2)
- Kernel: $(uname -r)
- CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
- Memory: $(free -h | grep "Mem:" | awk '{print $2}')

Build Environment:
- CUDA Version: $(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
- GCC Version: $(gcc --version | head -n1)
- Bazel Version: $(bazel version | grep "Build label" | cut -d' ' -f3)
- Python Version: $(python --version)

GPU Information:
$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv)

Build Configuration:
- XLA: Enabled
- CUDA Compute Capabilities: 12.0
- Optimization Level: opt
- Jobs: $JOBS

Build Artifacts:
$(ls -la "$BUILD_DIR")

Installation Command:
pip install $BUILD_DIR/tensorflow-*sm120*.whl

EOF
    
    log_success "Build report saved to: $report_file"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    
    cd tensorflow 2>/dev/null || true
    bazel clean --expunge 2>/dev/null || true
    cd .. 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main build function
main() {
    local start_time=$(date +%s)
    
    log_info "Starting TensorFlow sm_120 build process..."
    log_info "This process may take 2-4 hours depending on your system"
    
    # Set up error handling
    trap cleanup EXIT
    
    check_prerequisites
    clone_tensorflow
    apply_patches
    configure_tensorflow
    build_cpp_library
    build_python_wheel
    create_named_wheel
    verify_build
    generate_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    log_success "Build completed successfully in ${hours}h ${minutes}m"
    log_info "Installation command:"
    log_info "  pip install $BUILD_DIR/tensorflow-*sm120*.whl"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Install the wheel: pip install $BUILD_DIR/tensorflow-*sm120*.whl"
    log_info "  2. Test installation: python scripts/validate-installation.py"
    log_info "  3. Run benchmarks: python examples/benchmark.py"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "TensorFlow sm_120 Build Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --clean        Clean build directory before building"
        echo "  --jobs N       Number of parallel jobs (default: $JOBS)"
        echo ""
        exit 0
        ;;
    --clean)
        log_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR" tensorflow
        ;;
    --jobs)
        JOBS="$2"
        shift
        ;;
esac

# Run main function
main "$@"
