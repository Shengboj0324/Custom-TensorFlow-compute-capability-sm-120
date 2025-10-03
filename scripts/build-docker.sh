#!/bin/bash

# Docker-based TensorFlow sm_120 Build Script
# Provides a consistent build environment using Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="tensorflow-sm120-builder"
DOCKER_TAG="latest"
BUILD_DIR="$(pwd)/build"
CONTAINER_NAME="tf-sm120-build-$(date +%s)"

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

# Check Docker installation
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    log_success "Docker is available"
}

# Check NVIDIA Docker runtime
check_nvidia_docker() {
    # Skip GPU runtime check in CI environments
    if [[ -n "$CI" || -n "$GITHUB_ACTIONS" ]]; then
        log_warning "Skipping NVIDIA Docker runtime check in CI environment"
        return 0
    fi

    if ! docker run --rm --gpus all nvidia/cuda:12.4.0-devel-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_error "NVIDIA Docker runtime is not properly configured."
        log_info "Please install nvidia-container-toolkit:"
        log_info "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi

    log_success "NVIDIA Docker runtime is available"
}

# Build Docker image
build_docker_image() {
    log_info "Building Docker image for TensorFlow sm_120 build..."
    
    # Create temporary Dockerfile
    cat > Dockerfile.tf-sm120 << 'EOF'
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    pkg-config \
    zip \
    unzip \
    wget \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Install LLVM 22
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-22 main" && \
    apt-get update && \
    apt-get install -y \
    clang-22 \
    llvm-22 \
    llvm-22-dev \
    llvm-22-tools && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-22 100 && \
    rm -rf /var/lib/apt/lists/*

# Install cuDNN
RUN apt-get update && \
    apt-get install -y libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Install Bazel via Bazelisk
RUN wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 && \
    chmod +x /usr/local/bin/bazel

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install numpy packaging requests

# Set working directory
WORKDIR /workspace

# Copy scripts and patches
COPY scripts/ /workspace/scripts/
COPY patches/ /workspace/patches/

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Create build script
RUN cat > /workspace/build-in-container.sh << 'SCRIPT'
#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
TENSORFLOW_VERSION="r2.20"
BUILD_DIR="/workspace/build"
PATCHES_DIR="/workspace/patches"

log_info "Starting containerized TensorFlow sm_120 build..."

# Clone TensorFlow
if [[ ! -d "tensorflow" ]]; then
    log_info "Cloning TensorFlow..."
    git clone --depth 1 --branch $TENSORFLOW_VERSION https://github.com/tensorflow/tensorflow.git
fi

cd tensorflow

# Apply patches if they exist
if [[ -d "$PATCHES_DIR" ]]; then
    log_info "Applying patches..."
    for patch in "$PATCHES_DIR"/*.patch; do
        if [[ -f "$patch" ]]; then
            log_info "Applying $(basename "$patch")..."
            git apply "$patch" || log_warning "Patch $(basename "$patch") failed or already applied"
        fi
    done
fi

# Configure build
log_info "Configuring TensorFlow build..."
export PYTHON_BIN_PATH=$(which python3)
export TF_ENABLE_XLA=1
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_CUDA_VERSION=12.8
export TF_CUDNN_VERSION=9.8
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
export TF_CUDA_COMPUTE_CAPABILITIES="12.0"
export GCC_HOST_COMPILER_PATH=$(which clang)
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

python3 configure.py

# Build TensorFlow
log_info "Building TensorFlow (this will take 2-4 hours)..."
bazel build \
    --config=opt \
    --config=cuda \
    --copt=-Wno-error=c23-extensions \
    --copt=-Wno-error=unused-command-line-argument \
    --verbose_failures \
    --jobs=$(nproc) \
    //tensorflow:libtensorflow.so

# Build Python wheel
log_info "Building Python wheel..."
mkdir -p "$BUILD_DIR"
./bazel-bin/tensorflow/tools/pip_package/build_pip_package "$BUILD_DIR"

# Rename wheel to indicate sm_120 support
cd "$BUILD_DIR"
wheel_file=$(ls tensorflow-*.whl | head -n1)
if [[ -n "$wheel_file" ]]; then
    new_name=$(echo "$wheel_file" | sed 's/\.whl$/_sm120.whl/')
    mv "$wheel_file" "$new_name"
    log_success "Build completed: $new_name"
else
    log_error "No wheel file found"
    exit 1
fi

log_success "TensorFlow sm_120 build completed successfully!"
SCRIPT

RUN chmod +x /workspace/build-in-container.sh

# Default command
CMD ["/workspace/build-in-container.sh"]
EOF

    # Build the image
    docker build -t "$DOCKER_IMAGE:$DOCKER_TAG" -f Dockerfile.tf-sm120 .
    
    # Clean up temporary Dockerfile
    rm Dockerfile.tf-sm120
    
    log_success "Docker image built: $DOCKER_IMAGE:$DOCKER_TAG"
}

# Run build in container
run_build() {
    log_info "Starting containerized build..."
    log_info "Container name: $CONTAINER_NAME"
    
    # Create build directory on host
    mkdir -p "$BUILD_DIR"
    
    # Run the build container (with or without GPU based on environment)
    if [[ -n "$CI" || -n "$GITHUB_ACTIONS" ]]; then
        # CI environment - no GPU support
        docker run \
            --name "$CONTAINER_NAME" \
            --rm \
            -v "$BUILD_DIR:/workspace/build" \
            -v "$(pwd)/patches:/workspace/patches:ro" \
            "$DOCKER_IMAGE:$DOCKER_TAG"
    else
        # Local environment - with GPU support
        docker run \
            --name "$CONTAINER_NAME" \
            --gpus all \
            --rm \
            -v "$BUILD_DIR:/workspace/build" \
            -v "$(pwd)/patches:/workspace/patches:ro" \
            -e "NVIDIA_VISIBLE_DEVICES=all" \
            -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility" \
            "$DOCKER_IMAGE:$DOCKER_TAG"
    fi
    
    log_success "Build completed"
}

# Copy artifacts from container
copy_artifacts() {
    log_info "Copying build artifacts..."
    
    # Check if build directory has artifacts
    if ls "$BUILD_DIR"/*.whl &> /dev/null; then
        log_success "Build artifacts found in $BUILD_DIR"
        ls -la "$BUILD_DIR"
    else
        log_error "No build artifacts found"
        exit 1
    fi
}

# Generate installation instructions
generate_instructions() {
    local wheel_file=$(ls "$BUILD_DIR"/*sm120*.whl | head -n1)
    
    if [[ -n "$wheel_file" ]]; then
        cat > "$BUILD_DIR/INSTALL.md" << EOF
# TensorFlow sm_120 Installation Instructions

## Built Wheel
\`$(basename "$wheel_file")\`

## Installation

1. **Create a virtual environment** (recommended):
   \`\`\`bash
   python -m venv tf-sm120-env
   source tf-sm120-env/bin/activate  # On Windows: tf-sm120-env\\Scripts\\activate
   \`\`\`

2. **Install the wheel**:
   \`\`\`bash
   pip install $(basename "$wheel_file")
   \`\`\`

3. **Verify installation**:
   \`\`\`python
   import tensorflow as tf
   print(f"TensorFlow version: {tf.__version__}")
   print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
   
   # Check compute capability
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           details = tf.config.experimental.get_device_details(gpu)
           print(f"GPU: {gpu.name}")
           print(f"Compute capability: {details.get('compute_capability', 'Unknown')}")
   \`\`\`

## System Requirements
- NVIDIA RTX 50-series GPU (5080/5090)
- NVIDIA drivers 570.x or newer
- CUDA 12.4+ runtime libraries
- cuDNN 9.x runtime libraries

## Troubleshooting
- If you get CUDA errors, ensure your NVIDIA drivers and CUDA runtime are properly installed
- If you get import errors, make sure you're using the correct Python version (3.9-3.13)
- For performance issues, verify that TensorFlow is actually using the GPU with \`tf.config.list_physical_devices('GPU')\`

Built on: $(date)
Build environment: Docker container with CUDA 12.4, cuDNN 9.x, LLVM 22
EOF
        
        log_success "Installation instructions saved to $BUILD_DIR/INSTALL.md"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Remove any stopped containers
    if docker ps -a --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # Remove temporary files
    rm -f Dockerfile.tf-sm120
}

# Show help
show_help() {
    cat << EOF
TensorFlow sm_120 Docker Build Script

Usage: $0 [options]

Options:
  --help, -h           Show this help message
  --build-image        Build the Docker image only
  --run-build          Run the build (assumes image exists)
  --clean              Remove Docker image and build artifacts
  --image-name NAME    Custom Docker image name (default: $DOCKER_IMAGE)
  --tag TAG            Custom Docker tag (default: $DOCKER_TAG)

Examples:
  $0                   # Full build process (build image + run build)
  $0 --build-image     # Build Docker image only
  $0 --run-build       # Run build using existing image
  $0 --clean           # Clean up everything

The build process will:
1. Create a Docker image with all required dependencies
2. Build TensorFlow with sm_120 support inside the container
3. Copy the resulting wheel to the local build/ directory

Requirements:
- Docker with NVIDIA runtime support
- NVIDIA RTX 50-series GPU
- At least 32GB RAM and 100GB free disk space
EOF
}

# Main function
main() {
    local build_image=true
    local run_build=true
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --build-image)
                build_image=true
                run_build=false
                ;;
            --run-build)
                build_image=false
                run_build=true
                ;;
            --clean)
                log_info "Cleaning up Docker image and build artifacts..."
                docker rmi "$DOCKER_IMAGE:$DOCKER_TAG" 2>/dev/null || true
                rm -rf "$BUILD_DIR"
                log_success "Cleanup completed"
                exit 0
                ;;
            --image-name)
                DOCKER_IMAGE="$2"
                shift
                ;;
            --tag)
                DOCKER_TAG="$2"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
    
    # Set up error handling
    trap cleanup EXIT
    
    log_info "Starting Docker-based TensorFlow sm_120 build..."
    
    check_docker
    check_nvidia_docker
    
    if [[ "$build_image" == true ]]; then
        build_docker_image
    fi
    
    if [[ "$run_build" == true ]]; then
        run_build
        copy_artifacts
        generate_instructions
    fi
    
    log_success "Docker build process completed!"
    log_info "Build artifacts are available in: $BUILD_DIR"
    log_info "See $BUILD_DIR/INSTALL.md for installation instructions"
}

# Run main function
main "$@"
