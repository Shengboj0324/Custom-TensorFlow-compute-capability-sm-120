#!/bin/bash

# Custom TensorFlow sm_120 Build Environment Setup
# This script sets up the required environment for building TensorFlow with RTX 50-series support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            OS="ubuntu"
        elif command -v yum &> /dev/null; then
            OS="centos"
        else
            log_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_warning "macOS support is experimental"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    log_info "Detected OS: $OS"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check memory (minimum 16GB, recommended 32GB)
    local mem_gb=$(free -g | awk 'NR==2{printf "%.1f", $2}')
    if (( $(echo "$mem_gb < 16" | bc -l) )); then
        log_error "Insufficient memory: ${mem_gb}GB (minimum 16GB required)"
        exit 1
    elif (( $(echo "$mem_gb < 32" | bc -l) )); then
        log_warning "Memory: ${mem_gb}GB (32GB recommended for optimal build performance)"
    else
        log_success "Memory: ${mem_gb}GB"
    fi
    
    # Check disk space (minimum 100GB free)
    local disk_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if (( disk_gb < 100 )); then
        log_error "Insufficient disk space: ${disk_gb}GB free (minimum 100GB required)"
        exit 1
    else
        log_success "Disk space: ${disk_gb}GB free"
    fi
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
        log_info "NVIDIA GPU detected:"
        echo "$gpu_info" | while IFS=, read -r name compute_cap; do
            name=$(echo "$name" | xargs)
            compute_cap=$(echo "$compute_cap" | xargs)
            if [[ "$compute_cap" == "12.0" ]]; then
                log_success "  - $name (Compute Capability: $compute_cap) ✓"
            else
                log_warning "  - $name (Compute Capability: $compute_cap) - Not RTX 50-series"
            fi
        done
    else
        log_warning "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $OS in
        ubuntu)
            sudo apt-get update
            sudo apt-get install -y \
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
                lsb-release
            ;;
        centos)
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                curl \
                git \
                python3 \
                python3-devel \
                python3-pip \
                pkgconfig \
                zip \
                unzip \
                wget \
                which
            ;;
    esac
    
    log_success "System dependencies installed"
}

# Install CUDA Toolkit
install_cuda() {
    log_info "Checking CUDA installation..."
    
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        if [[ "$cuda_version" =~ ^1[2-9]\.[0-9]+$ ]] && (( $(echo "$cuda_version >= 12.4" | bc -l) )); then
            log_success "CUDA $cuda_version is already installed"
            return 0
        else
            log_warning "CUDA $cuda_version found, but version 12.4+ is required"
        fi
    fi
    
    log_info "Installing CUDA Toolkit 12.4..."
    
    case $OS in
        ubuntu)
            # Add NVIDIA package repository
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
            sudo dpkg -i cuda-keyring_1.0-1_all.deb
            sudo apt-get update
            
            # Install CUDA toolkit
            sudo apt-get install -y cuda-toolkit-12-4
            
            # Add to PATH
            echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
            ;;
        centos)
            # Add NVIDIA repository
            sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
            sudo yum clean all
            sudo yum install -y cuda-toolkit-12-4
            
            # Add to PATH
            echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
            ;;
    esac
    
    log_success "CUDA Toolkit installed"
}

# Install cuDNN
install_cudnn() {
    log_info "Installing cuDNN..."
    
    case $OS in
        ubuntu)
            sudo apt-get install -y libcudnn9-dev-cuda-12
            ;;
        centos)
            log_warning "cuDNN installation on CentOS requires manual download from NVIDIA"
            log_info "Please download cuDNN 9.x for CUDA 12.x from:"
            log_info "https://developer.nvidia.com/cudnn"
            ;;
    esac
    
    log_success "cuDNN installation completed"
}

# GCC is already available in most distributions, no additional installation needed
verify_gcc() {
    log_info "Verifying GCC installation..."

    if ! command -v gcc &> /dev/null; then
        log_error "GCC not found. Installing build-essential..."
        case $OS in
            ubuntu)
                sudo apt-get update
                sudo apt-get install -y build-essential
                ;;
            centos)
                sudo yum groupinstall -y "Development Tools"
                ;;
        esac
    fi

    local gcc_version=$(gcc --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -n1)
    if [[ $(echo "$gcc_version < 9.0" | bc -l) -eq 1 ]]; then
        log_warning "GCC $gcc_version found, but version 9.0+ recommended"
    else
        log_success "GCC $gcc_version found"
    fi
}

# Install Bazel
install_bazel() {
    log_info "Installing Bazel..."
    
    # Install Bazelisk (automatically manages Bazel versions)
    local bazelisk_version="v1.19.0"
    local bazelisk_url="https://github.com/bazelbuild/bazelisk/releases/download/${bazelisk_version}/bazelisk-linux-amd64"
    
    sudo wget -O /usr/local/bin/bazel "$bazelisk_url"
    sudo chmod +x /usr/local/bin/bazel
    
    log_success "Bazel (via Bazelisk) installed"
}

# Set up Python environment
setup_python() {
    log_info "Setting up Python environment..."
    
    # Check Python version
    local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if ! [[ "$python_version" =~ ^3\.(9|1[0-3])$ ]]; then
        log_error "Python $python_version is not supported. Requires Python 3.9-3.13"
        exit 1
    fi
    
    log_success "Python $python_version is supported"
    
    # Create virtual environment
    python3 -m venv tf-build-env
    source tf-build-env/bin/activate
    
    # Upgrade pip and install build dependencies
    pip install --upgrade pip setuptools wheel
    pip install numpy packaging requests
    
    log_success "Python build environment created"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Source environment
    source ~/.bashrc
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        log_success "CUDA $cuda_version verified"
    else
        log_error "CUDA verification failed"
    fi
    
    # Check GCC
    if command -v gcc &> /dev/null; then
        local gcc_version=$(gcc --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -n1)
        log_success "GCC $gcc_version verified"
    else
        log_error "GCC verification failed"
    fi
    
    # Check Bazel
    if command -v bazel &> /dev/null; then
        local bazel_version=$(bazel version | grep "Build label" | cut -d' ' -f3)
        log_success "Bazel $bazel_version verified"
    else
        log_error "Bazel verification failed"
    fi
    
    log_success "Environment setup completed successfully!"
}

# Main function
main() {
    log_info "Starting TensorFlow sm_120 build environment setup..."
    
    check_root
    detect_os
    check_requirements
    install_system_deps
    install_cuda
    install_cudnn
    verify_gcc
    install_bazel
    setup_python
    verify_installation
    
    log_success "Setup completed! Please run 'source ~/.bashrc' to update your environment."
    log_info "Next steps:"
    log_info "  1. Activate Python environment: source tf-build-env/bin/activate"
    log_info "  2. Run the build script: ./scripts/build-tensorflow.sh"
}

# Run main function
main "$@"
