# Troubleshooting Guide for TensorFlow sm_120

This guide covers common issues encountered when building and using TensorFlow with RTX 50-series GPU support.

## 🔍 Quick Diagnostics

### System Check Script

Run this diagnostic script to check your environment:

```bash
#!/bin/bash
echo "=== TensorFlow sm_120 System Diagnostics ==="

# Check NVIDIA GPU
echo "1. GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
else
    echo "   ❌ nvidia-smi not found"
fi

# Check CUDA
echo -e "\n2. CUDA Installation:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "   ❌ nvcc not found"
fi

# Check cuDNN
echo -e "\n3. cuDNN Installation:"
if ls /usr/lib/x86_64-linux-gnu/libcudnn* &> /dev/null; then
    echo "   ✅ cuDNN libraries found"
    ls /usr/lib/x86_64-linux-gnu/libcudnn*.so* | head -3
else
    echo "   ❌ cuDNN libraries not found"
fi

# Check LLVM
echo -e "\n4. LLVM/Clang:"
if command -v clang &> /dev/null; then
    clang --version | head -1
else
    echo "   ❌ clang not found"
fi

# Check Bazel
echo -e "\n5. Bazel:"
if command -v bazel &> /dev/null; then
    bazel version | grep "Build label"
else
    echo "   ❌ bazel not found"
fi

# Check Python
echo -e "\n6. Python Environment:"
python3 --version
pip list | grep -E "(tensorflow|numpy)" || echo "   No TensorFlow/NumPy found"

# Check TensorFlow (if installed)
echo -e "\n7. TensorFlow Status:"
python3 -c "
try:
    import tensorflow as tf
    print(f'   ✅ TensorFlow {tf.__version__} imported')
    print(f'   CUDA built: {tf.test.is_built_with_cuda()}')
    gpus = tf.config.list_physical_devices('GPU')
    print(f'   GPU devices: {len(gpus)}')
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        cc = details.get('compute_capability', 'Unknown')
        if cc == (12, 0):
            print(f'   ✅ sm_120 support confirmed!')
        else:
            print(f'   Compute capability: {cc}')
except ImportError:
    print('   ❌ TensorFlow not installed')
except Exception as e:
    print(f'   ❌ TensorFlow error: {e}')
"
```

Save this as `diagnose.sh`, make it executable (`chmod +x diagnose.sh`), and run it.

## 🚨 Build Issues

### Issue 1: C23 Extensions Error

**Symptoms:**
```
error: use of C23 extension [-Werror,-Wc23-extensions]
error: '_BitInt' in C++; did you mean 'BitInt'? [-Werror,-Wc23-extensions]
```

**Cause:** LLVM 22 generates C23 extension warnings that are treated as errors.

**Solutions:**

1. **Apply the C23 extensions patch:**
   ```bash
   cd tensorflow
   git apply ../patches/fix-c23-extensions.patch
   ```

2. **Manual fix - Add compiler flags:**
   ```bash
   bazel build \
       --copt=-Wno-error=c23-extensions \
       --copt=-Wno-c23-extensions \
       --copt=-Wno-error=unused-command-line-argument \
       //tensorflow:libtensorflow.so
   ```

3. **Update .bazelrc file:**
   ```bash
   echo "build --copt=-Wno-error=c23-extensions" >> .bazelrc
   echo "build --copt=-Wno-c23-extensions" >> .bazelrc
   ```

### Issue 2: Matrix Function Naming Conflict

**Symptoms:**
```
error: 'set_matrix3x3' conflicts with previous declaration
error: redefinition of 'set_matrix3x3'
```

**Cause:** Function naming collision in Triton support code.

**Solutions:**

1. **Apply the matrix naming patch:**
   ```bash
   cd tensorflow
   git apply ../patches/fix-matrix-naming.patch
   ```

2. **Manual fix:**
   Edit `third_party/xla/xla/service/gpu/fusions/triton/triton_support.cc`:
   ```cpp
   // Change this:
   bool set_matrix3x3(const Matrix& matrix) {
   
   // To this:
   bool set_matrix_3x3(const Matrix& matrix) {
   ```

### Issue 3: Template Instantiation Errors

**Symptoms:**
```
error: template argument deduction/substitution failed
error: no matching function for template argument deduction
```

**Cause:** Template compatibility issues with LLVM 22.

**Solutions:**

1. **Apply the template errors patch:**
   ```bash
   cd tensorflow
   git apply ../patches/fix-template-errors.patch
   ```

2. **Check LLVM version:**
   ```bash
   clang --version
   # Should show version 22.x.x
   ```

### Issue 4: Out of Memory During Build

**Symptoms:**
```
ERROR: Not enough memory to complete build
c++: fatal error: Killed signal terminated program cc1plus
```

**Solutions:**

1. **Reduce parallel jobs:**
   ```bash
   bazel build --jobs=4 //tensorflow:libtensorflow.so
   ```

2. **Add swap space:**
   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Limit memory usage:**
   ```bash
   bazel build \
       --ram_utilization_factor=50 \
       --local_ram_resources=8192 \
       --jobs=2 \
       //tensorflow:libtensorflow.so
   ```

4. **Clean build cache:**
   ```bash
   bazel clean --expunge
   ```

### Issue 5: CUDA Not Found During Configuration

**Symptoms:**
```
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
Cannot find cuda toolkit path
```

**Solutions:**

1. **Verify GPU detection:**
   ```bash
   nvidia-smi
   lspci | grep -i nvidia
   ```

2. **Check CUDA installation:**
   ```bash
   ls /usr/local/cuda*/bin/nvcc
   nvcc --version
   ```

3. **Set CUDA paths manually:**
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

4. **Reinstall CUDA:**
   ```bash
   sudo apt purge 'cuda*' 'nvidia*'
   sudo apt autoremove
   sudo apt install cuda-toolkit-12-8
   sudo reboot
   ```

### Issue 6: cuDNN Version Mismatch

**Symptoms:**
```
Could not find cuDNN
cuDNN version mismatch
```

**Solutions:**

1. **Check cuDNN installation:**
   ```bash
   ls /usr/lib/x86_64-linux-gnu/libcudnn*
   cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR
   ```

2. **Install correct cuDNN version:**
   ```bash
   sudo apt install libcudnn9-dev-cuda-12
   ```

3. **Manual cuDNN installation:**
   ```bash
   # Download from NVIDIA Developer portal
   tar -xvf cudnn-linux-x86_64-9.x.x.x_cuda12-archive.tar.xz
   sudo cp cudnn-*/include/cudnn*.h /usr/local/cuda/include
   sudo cp cudnn-*/lib/libcudnn* /usr/local/cuda/lib64
   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
   ```

### Issue 7: Bazel Build Timeouts

**Symptoms:**
```
ERROR: Timeout exceeded
Build timed out after 3600 seconds
```

**Solutions:**

1. **Increase timeout:**
   ```bash
   bazel build --remote_timeout=7200 //tensorflow:libtensorflow.so
   ```

2. **Use local build only:**
   ```bash
   bazel build --strategy=CppCompile=local //tensorflow:libtensorflow.so
   ```

3. **Disable remote caching:**
   ```bash
   bazel build --noremote_cache //tensorflow:libtensorflow.so
   ```

## 🔧 Runtime Issues

### Issue 1: TensorFlow Cannot Find GPU

**Symptoms:**
```python
tf.config.list_physical_devices('GPU')  # Returns empty list
```

**Solutions:**

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   sudo apt install nvidia-driver-570-open
   ```

2. **Verify CUDA runtime:**
   ```bash
   python3 -c "
   import tensorflow as tf
   print('CUDA built:', tf.test.is_built_with_cuda())
   print('GPU support:', tf.test.is_built_with_gpu_support())
   "
   ```

3. **Set GPU memory growth:**
   ```python
   import tensorflow as tf
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

### Issue 2: GPU Memory Errors

**Symptoms:**
```
ResourceExhaustedError: Out of memory
CUDA_ERROR_OUT_OF_MEMORY
```

**Solutions:**

1. **Enable memory growth:**
   ```python
   import tensorflow as tf
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
   ```

2. **Set memory limit:**
   ```python
   import tensorflow as tf
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       tf.config.experimental.set_virtual_device_configuration(
           gpus[0],
           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
       )
   ```

3. **Use mixed precision:**
   ```python
   import tensorflow as tf
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

### Issue 3: Compute Capability Not Detected

**Symptoms:**
```python
# Compute capability shows as something other than (12, 0)
details = tf.config.experimental.get_device_details(gpu)
print(details['compute_capability'])  # Not (12, 0)
```

**Solutions:**

1. **Verify sm_120 patch was applied:**
   ```bash
   cd tensorflow
   git log --oneline | grep -i sm120
   ```

2. **Check TensorFlow build configuration:**
   ```python
   import tensorflow as tf
   print(tf.sysconfig.get_build_info())
   ```

3. **Rebuild with explicit sm_120 support:**
   ```bash
   export TF_CUDA_COMPUTE_CAPABILITIES=12.0
   bazel clean --expunge
   ./configure
   bazel build //tensorflow:libtensorflow.so
   ```

### Issue 4: Performance Not Improved

**Symptoms:**
- No performance improvement over previous GPU
- Slower than expected performance

**Solutions:**

1. **Verify sm_120 optimizations are active:**
   ```python
   import tensorflow as tf
   with tf.device('/GPU:0'):
       # Run a computation and check GPU utilization
       a = tf.random.normal([4096, 4096])
       b = tf.random.normal([4096, 4096])
       c = tf.matmul(a, b)
       print(c.numpy().mean())
   ```

2. **Enable XLA optimization:**
   ```python
   import tensorflow as tf
   tf.config.optimizer.set_jit(True)
   ```

3. **Use mixed precision:**
   ```python
   import tensorflow as tf
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

4. **Check GPU utilization:**
   ```bash
   # In another terminal while running TensorFlow
   watch -n 1 nvidia-smi
   ```

## 🐳 Docker Issues

### Issue 1: Docker NVIDIA Runtime Not Available

**Symptoms:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Solutions:**

1. **Install nvidia-container-toolkit:**
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Configure Docker daemon:**
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

### Issue 2: Docker Build Fails

**Symptoms:**
```
Docker build fails with various errors
```

**Solutions:**

1. **Increase Docker resources:**
   - Memory: 16GB+
   - Swap: 8GB+
   - Disk: 100GB+

2. **Use multi-stage build:**
   ```bash
   # Build in stages to save memory
   docker build --target builder -t tf-builder .
   docker build --target runtime -t tf-runtime .
   ```

3. **Clean Docker cache:**
   ```bash
   docker system prune -af
   docker builder prune -af
   ```

## 🔍 Advanced Debugging

### Enable Verbose Logging

```bash
export TF_CPP_MIN_LOG_LEVEL=0
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
```

### CUDA Debugging

```bash
# Check CUDA samples
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

# Check compute capability
cd /usr/local/cuda/samples/1_Utilities/bandwidthTest
sudo make
./bandwidthTest
```

### Build Debugging

```bash
# Verbose Bazel output
bazel build --verbose_failures --subcommands --explain=build.log //tensorflow:libtensorflow.so

# Check build log
less build.log
```

### Memory Debugging

```bash
# Check system memory
free -h
cat /proc/meminfo | grep -E "(MemTotal|MemAvailable|SwapTotal)"

# Monitor during build
watch -n 1 'free -h && echo "---" && ps aux --sort=-%mem | head -10'
```

## 📞 Getting Help

### Before Asking for Help

1. **Run the diagnostic script** (provided at the top)
2. **Check the build logs** for specific error messages
3. **Search existing issues** on GitHub
4. **Try the Docker build** if native build fails

### Information to Include

When reporting issues, please include:

1. **System information:**
   ```bash
   uname -a
   lsb_release -a
   nvidia-smi
   nvcc --version
   clang --version
   ```

2. **Build configuration:**
   ```bash
   cat .tf_configure.bazelrc  # If it exists
   echo $TF_CUDA_COMPUTE_CAPABILITIES
   ```

3. **Error messages:** Full error output, not just the summary

4. **Steps to reproduce:** Exact commands that led to the issue

### Common Support Channels

- **GitHub Issues:** For bugs and feature requests
- **GitHub Discussions:** For questions and general help
- **Stack Overflow:** Tag with `tensorflow` and `cuda`

## 📚 Additional Resources

- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Bazel Troubleshooting](https://bazel.build/docs/user-manual#troubleshooting)
- [Docker NVIDIA Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

**Still having issues?** Open a [GitHub issue](https://github.com/yourusername/Custom-TensorFlow-compute-capability-sm-120/issues) with detailed information about your problem.
