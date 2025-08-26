# Docker Build Instructions for TensorFlow SM120

## Overview

This directory contains Docker configurations for building TensorFlow with SM120 optimizations in containerized environments.

## Available Dockerfiles

- `Dockerfile.ubuntu` - Ubuntu 22.04 based build (recommended)
- `Dockerfile.centos` - CentOS 8 based build (experimental)

## Building Docker Images

### Basic Build
```bash
# Ubuntu-based build
docker build -f docker/Dockerfile.ubuntu -t tensorflow-sm120:ubuntu .

# CentOS-based build  
docker build -f docker/Dockerfile.centos -t tensorflow-sm120:centos .
```

### Build with Custom Arguments
```bash
docker build -f docker/Dockerfile.ubuntu \
  --build-arg CUDA_ARCH=sm_120 \
  --build-arg BUILD_VERSION=1.0.0 \
  -t tensorflow-sm120:ubuntu-sm120 .
```

## Running Docker Containers

### CPU-Only Mode (CI/Testing)
```bash
# Basic functionality test (no GPU required)
docker run --rm tensorflow-sm120:ubuntu \
  python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### GPU Mode (Requires NVIDIA Docker Runtime)
```bash
# With GPU support (requires nvidia-docker2 or Docker 19.03+ with nvidia-container-toolkit)
docker run --rm --gpus all tensorflow-sm120:ubuntu \
  python -c "import tensorflow as tf; print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

## GPU Runtime Requirements

### For GPU-enabled containers, you need:

1. **NVIDIA GPU** with compute capability 12.0+ (RTX 50-series)
2. **NVIDIA Driver** 560+ installed on host
3. **NVIDIA Container Toolkit** installed:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Runtime:
```bash
# Test NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi
```

## CI/CD Usage

In CI environments (GitHub Actions, GitLab CI, etc.) where GPU hardware is not available:

- ✅ **Docker builds work** - Images build successfully without GPU
- ✅ **CPU testing works** - Basic functionality can be tested
- ❌ **GPU testing fails** - `--gpus all` flag will fail without NVIDIA runtime

The build system automatically detects CI environments and adjusts accordingly.

## Troubleshooting

### Common Issues:

1. **"could not select device driver with capabilities: [[gpu]]"**
   - **Cause**: NVIDIA Docker runtime not installed or configured
   - **Fix**: Install nvidia-container-toolkit or run without `--gpus` flag

2. **"CUDA driver version is insufficient"**
   - **Cause**: Host NVIDIA driver too old
   - **Fix**: Update to driver 560+ for RTX 50-series support

3. **"No space left on device"**
   - **Cause**: Docker build requires significant disk space
   - **Fix**: Clean Docker cache with `docker system prune -af`

### Debug Commands:
```bash
# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi

# Check container without GPU
docker run --rm tensorflow-sm120:ubuntu nvidia-smi

# Check build logs
docker build --no-cache -f docker/Dockerfile.ubuntu -t debug-build .
```

## Performance Notes

- **Build time**: 30-60 minutes depending on system
- **Image size**: ~8-12GB (includes CUDA toolkit and TensorFlow)
- **Memory usage**: 4-8GB during build
- **Disk space**: 20-30GB required for build process

## Security Considerations

- Images run as non-root user by default
- No sensitive data included in images
- Build arguments are not cached in final image
- Minimal attack surface with only required packages
