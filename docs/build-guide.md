# Complete Build Guide for TensorFlow sm_120

This comprehensive guide walks you through building TensorFlow with native support for RTX 50-series GPUs (compute capability 12.0).

## 📋 Prerequisites

### System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU** | RTX 5080/5090 | Compute capability 12.0 (sm_120) |
| **RAM** | 32GB+ recommended | 16GB minimum for build |
| **Storage** | 100GB+ free space | Build artifacts are large |
| **OS** | Ubuntu 22.04+ / CentOS 9+ | Windows via WSL2 supported |

### Software Dependencies

| Software | Version | Installation |
|----------|---------|-------------|
| **CUDA Toolkit** | 12.8+ | `sudo apt install cuda-toolkit-12-8` |
| **cuDNN** | 9.8+ | `sudo apt install libcudnn9-dev-cuda-12` |
| **LLVM** | 22.x | `sudo apt install clang-22 llvm-22-dev` |
| **Bazel** | Latest | Via Bazelisk (automatic) |
| **Python** | 3.9-3.13 | `python3 --version` |
| **Git** | Latest | `sudo apt install git` |

## 🚀 Quick Start (Recommended)

### Option 1: Docker Build

The Docker approach provides a consistent, isolated build environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/Custom-TensorFlow-compute-capability-sm-120.git
cd Custom-TensorFlow-compute-capability-sm-120

# Build using Docker (recommended)
./scripts/build-docker.sh

# Install the resulting wheel
pip install ./build/tensorflow-*sm120*.whl
```

### Option 2: Native Build

For advanced users who prefer building directly on their system:

```bash
# Set up the build environment
./scripts/setup-environment.sh

# Build TensorFlow
./scripts/build-tensorflow.sh

# Install the wheel
pip install ./build/tensorflow-*sm120*.whl
```

## 🔧 Detailed Build Process

### Step 1: Environment Preparation

#### Ubuntu/Debian Systems

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt install -y \
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
    wget
```

#### Install CUDA 12.8

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA toolkit
sudo apt install -y cuda-toolkit-12-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Install cuDNN 9.x

```bash
# Install cuDNN development package
sudo apt install -y libcudnn9-dev-cuda-12

# Verify installation
ls /usr/lib/x86_64-linux-gnu/libcudnn*
```

#### Install LLVM 22

```bash
# Add LLVM repository
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-22 main"
sudo apt update

# Install LLVM 22
sudo apt install -y \
    clang-22 \
    llvm-22 \
    llvm-22-dev \
    llvm-22-tools

# Set as default
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-22 100
```

#### Install Bazel

```bash
# Install Bazelisk (manages Bazel versions automatically)
sudo wget -O /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```

### Step 2: Python Environment Setup

```bash
# Create virtual environment
python3 -m venv tf-sm120-build
source tf-sm120-build/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install numpy packaging requests six mock keras_preprocessing
```

### Step 3: Clone and Configure TensorFlow

```bash
# Clone TensorFlow repository
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

# Checkout stable release (recommended)
git checkout r2.20  # or latest stable version

# Apply sm_120 patches
cd ..
./patches/apply-patches.sh tensorflow

cd tensorflow
```

### Step 4: Configure Build

Run the interactive configuration script:

```bash
./configure
```

**Configuration Options:**

- **Python interpreter**: Use the one from your virtual environment
- **Python library path**: Usually auto-detected
- **CUDA support**: YES
- **CUDA toolkit path**: `/usr/local/cuda-12.8`
- **cuDNN version**: `9`
- **cuDNN path**: `/usr/lib/x86_64-linux-gnu`
- **Compute capabilities**: `12.0` (critical for sm_120 support)
- **Clang compiler**: `/usr/bin/clang-22`
- **Optimization flags**: `-march=native`
- **XLA support**: YES (recommended)
- **TensorRT support**: NO (unless you have TensorRT)

**Non-interactive Configuration (Advanced):**

```bash
export PYTHON_BIN_PATH=$(which python3)
export TF_ENABLE_XLA=1
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_CUDA_VERSION=12.8
export TF_CUDNN_VERSION=9
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.8
export CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
export TF_CUDA_COMPUTE_CAPABILITIES=12.0
export GCC_HOST_COMPILER_PATH=/usr/bin/clang-22
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

python3 configure.py
```

### Step 5: Build TensorFlow

#### Build C++ Library

```bash
# Build the main library (this takes 2-4 hours)
bazel build \
    --config=opt \
    --config=cuda \
    --config=v2 \
    --copt=-Wno-error=c23-extensions \
    --copt=-Wno-error=unused-command-line-argument \
    --copt=-march=native \
    --copt=-O3 \
    --verbose_failures \
    --jobs=$(nproc) \
    //tensorflow:libtensorflow.so
```

#### Build Python Wheel

```bash
# Build the pip package
bazel build \
    --config=opt \
    --config=cuda \
    --config=v2 \
    --copt=-Wno-error=c23-extensions \
    --verbose_failures \
    --jobs=$(nproc) \
    //tensorflow/tools/pip_package:build_pip_package

# Create the wheel
mkdir -p ../build
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ../build
```

### Step 6: Install and Test

```bash
# Install the wheel
cd ../build
pip install tensorflow-*.whl

# Basic test
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    details = tf.config.experimental.get_device_details(gpus[0])
    print(f'Compute capability: {details.get(\"compute_capability\", \"Unknown\")}')
"
```

## 🐛 Troubleshooting

### Common Build Errors

#### 1. C23 Extensions Error

**Error:**
```
error: use of C23 extension [-Werror,-Wc23-extensions]
```

**Solution:**
Ensure you're using the `--copt=-Wno-error=c23-extensions` flag and that the C23 extensions patch is applied.

#### 2. Matrix Function Naming Conflict

**Error:**
```
error: 'set_matrix3x3' conflicts with previous declaration
```

**Solution:**
Apply the matrix naming patch:
```bash
cd tensorflow
git apply ../patches/fix-matrix-naming.patch
```

#### 3. Template Instantiation Error

**Error:**
```
error: template argument deduction/substitution failed
```

**Solution:**
Apply the template error patch:
```bash
cd tensorflow
git apply ../patches/fix-template-errors.patch
```

#### 4. CUDA Not Found

**Error:**
```
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
```

**Solution:**
1. Verify GPU is detected: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Ensure drivers are up to date: `sudo apt install nvidia-driver-570-open`

#### 5. Out of Memory During Build

**Error:**
```
ERROR: Not enough memory to complete build
```

**Solution:**
1. Reduce parallel jobs: `--jobs=4` instead of `--jobs=$(nproc)`
2. Add swap space:
   ```bash
   sudo fallocate -l 32G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```
3. Use `--ram_utilization_factor=50` to limit memory usage

### Build Optimization Tips

#### For Faster Builds

1. **Use ccache:**
   ```bash
   sudo apt install ccache
   export CC="ccache clang-22"
   export CXX="ccache clang++-22"
   ```

2. **Enable build caching:**
   ```bash
   echo "build --disk_cache=/tmp/bazel-cache" >> .bazelrc
   ```

3. **Use faster linker:**
   ```bash
   echo "build --linkopt=-fuse-ld=lld" >> .bazelrc
   ```

#### For Smaller Builds

1. **Disable unnecessary features:**
   ```bash
   export TF_NEED_AWS=0
   export TF_NEED_GCP=0
   export TF_NEED_HDFS=0
   export TF_NEED_KAFKA=0
   ```

2. **Build only what you need:**
   ```bash
   # Only build Python package, not C++ library
   bazel build //tensorflow/tools/pip_package:build_pip_package
   ```

## 🔧 Advanced Configuration

### Custom Compute Capabilities

To support multiple GPU architectures:

```bash
export TF_CUDA_COMPUTE_CAPABILITIES="8.9,9.0,12.0"
```

### Optimization Levels

For maximum performance:

```bash
export CC_OPT_FLAGS="-march=native -mtune=native -O3 -DNDEBUG"
```

For debugging:

```bash
export CC_OPT_FLAGS="-g -O0"
```

### Memory Optimization

For systems with limited memory:

```bash
bazel build \
    --config=opt \
    --config=cuda \
    --ram_utilization_factor=50 \
    --local_ram_resources=16384 \
    --jobs=4 \
    //tensorflow:libtensorflow.so
```

## 📊 Verification

### Performance Testing

```bash
# Run comprehensive tests
python3 scripts/validate-installation.py

# Run performance benchmarks
python3 examples/benchmark.py

# Test specific operations
python3 examples/basic-gpu-test.py
```

### Expected Performance

On RTX 5090 with sm_120 optimizations:

- **Matrix Multiplication (4096x4096)**: ~800+ GFLOPS
- **Convolution (ResNet-50 layer)**: ~15-25% faster than sm_89
- **Mixed Precision**: ~2x speedup with FP16
- **Memory Bandwidth**: ~1000+ GB/s

## 🚀 Next Steps

1. **Install your custom wheel:**
   ```bash
   pip install ./build/tensorflow-*sm120*.whl
   ```

2. **Verify sm_120 support:**
   ```bash
   python3 -c "
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       details = tf.config.experimental.get_device_details(gpus[0])
       cc = details.get('compute_capability')
       if cc == (12, 0):
           print('✅ sm_120 support confirmed!')
       else:
           print(f'ℹ️  Compute capability: {cc}')
   "
   ```

3. **Run your models:**
   Your existing TensorFlow code should work without changes and automatically benefit from sm_120 optimizations.

4. **Monitor performance:**
   Use `nvidia-smi` to monitor GPU utilization during training.

## 📚 Additional Resources

- [TensorFlow Build from Source Guide](https://www.tensorflow.org/install/source)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Bazel Documentation](https://bazel.build/docs)
- [Project Issues](https://github.com/yourusername/Custom-TensorFlow-compute-capability-sm-120/issues)

---

**Need Help?** Check the [troubleshooting guide](troubleshooting.md) or open an issue on GitHub.
