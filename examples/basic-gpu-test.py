#!/usr/bin/env python3
"""
Basic GPU Test for TensorFlow sm_120

This script performs basic GPU functionality tests to verify that TensorFlow
is properly utilizing RTX 50-series GPUs with compute capability 12.0.
"""

import os
import sys
import time
import numpy as np
from typing import List

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not found. Please install the custom sm_120 wheel.")
    sys.exit(1)

# Suppress TensorFlow warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"{title}")
    print(f"{'-'*40}")


def check_tensorflow_info() -> None:
    """Display TensorFlow version and build information."""
    print_section("TensorFlow Information")

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"TensorFlow Location: {tf.__file__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")

    # Check for XLA support
    try:
        print(f"XLA enabled: {tf.config.optimizer.get_jit() is not None}")
    except Exception:
        print("XLA status: Unknown")


def check_gpu_devices() -> List[tf.config.PhysicalDevice]:
    """Check and display available GPU devices."""
    print_section("GPU Device Information")

    # List all physical devices
    all_devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices("GPU")

    print(f"Total physical devices: {len(all_devices)}")
    print(f"GPU devices found: {len(gpu_devices)}")

    if not gpu_devices:
        print("⚠️  No GPU devices found!")
        return []

    # Display detailed information for each GPU
    for i, gpu in enumerate(gpu_devices):
        print(f"\nGPU {i}:")
        print(f"  Name: {gpu.name}")
        print(f"  Device Type: {gpu.device_type}")

        # Get device details
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print("  Device Details:")
            for key, value in details.items():
                if key == "compute_capability":
                    if isinstance(value, tuple):
                        cc_str = f"{value[0]}.{value[1]}"
                        if value == (12, 0):
                            print(f"    {key}: {cc_str} ✅ (sm_120 - RTX 50-series)")
                        else:
                            print(f"    {key}: {cc_str}")
                    else:
                        print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value}")
        except Exception as e:
            print(f"  Could not get device details: {e}")

    return gpu_devices


def test_basic_operations(gpu_devices: List[tf.config.PhysicalDevice]) -> bool:
    """Test basic TensorFlow operations on GPU."""
    print_section("Basic GPU Operations Test")

    if not gpu_devices:
        print("❌ No GPU devices available for testing")
        return False

    try:
        # Test on first GPU
        with tf.device("/GPU:0"):
            print("Testing matrix operations on GPU...")

            # Create test matrices
            size = 1000
            a = tf.random.normal([size, size], dtype=tf.float32)
            b = tf.random.normal([size, size], dtype=tf.float32)

            print(f"  Matrix size: {size}x{size}")
            print(f"  Data type: {a.dtype}")
            print(f"  Device: {a.device}")

            # Perform matrix multiplication
            start_time = time.time()
            c = tf.matmul(a, b)
            result = c.numpy()  # Force computation
            end_time = time.time()

            print(
                f"  Matrix multiplication completed in {end_time - start_time:.4f} seconds"
            )
            print(f"  Result shape: {result.shape}")
            print(f"  Result mean: {np.mean(result):.6f}")
            print("✅ Basic matrix operations successful")

            return True

    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False


def test_tensor_operations(gpu_devices: List[tf.config.PhysicalDevice]) -> bool:
    """Test various tensor operations on GPU."""
    print_section("Advanced Tensor Operations Test")

    if not gpu_devices:
        print("❌ No GPU devices available for testing")
        return False

    try:
        with tf.device("/GPU:0"):
            print("Testing advanced tensor operations...")

            # Test 1: Convolution operation
            print("  1. Testing 2D convolution...")
            batch_size = 32
            height, width, channels = 224, 224, 3
            filters = 64
            kernel_size = 3

            # Create input tensor
            x = tf.random.normal([batch_size, height, width, channels])

            # Create convolution layer
            conv_layer = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
            )

            start_time = time.time()
            conv_output = conv_layer(x)
            _ = conv_output.numpy()  # Force computation
            end_time = time.time()

            print(f"     Input shape: {x.shape}")
            print(f"     Output shape: {conv_output.shape}")
            print(f"     Time: {end_time - start_time:.4f} seconds")

            # Test 2: Reduction operations
            print("  2. Testing reduction operations...")
            large_tensor = tf.random.normal([10000, 10000])

            start_time = time.time()
            sum_result = tf.reduce_sum(large_tensor)
            mean_result = tf.reduce_mean(large_tensor)
            max_result = tf.reduce_max(large_tensor)
            _ = sum_result.numpy(), mean_result.numpy(), max_result.numpy()
            end_time = time.time()

            print(f"     Tensor shape: {large_tensor.shape}")
            print(f"     Sum: {sum_result.numpy():.6f}")
            print(f"     Mean: {mean_result.numpy():.6f}")
            print(f"     Max: {max_result.numpy():.6f}")
            print(f"     Time: {end_time - start_time:.4f} seconds")

            # Test 3: Broadcasting operations
            print("  3. Testing broadcasting operations...")
            a = tf.random.normal([1000, 1, 1000])
            b = tf.random.normal([1, 1000, 1])

            start_time = time.time()
            broadcast_result = a + b  # Broadcasting
            _ = broadcast_result.numpy()
            end_time = time.time()

            print(f"     Shape A: {a.shape}")
            print(f"     Shape B: {b.shape}")
            print(f"     Result shape: {broadcast_result.shape}")
            print(f"     Time: {end_time - start_time:.4f} seconds")

            print("✅ Advanced tensor operations successful")
            return True

    except Exception as e:
        print(f"❌ Advanced tensor operations failed: {e}")
        return False


def test_mixed_precision() -> bool:
    """Test mixed precision operations (important for RTX 50-series)."""
    print_section("Mixed Precision Test")

    try:
        # Enable mixed precision
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

        print(f"Mixed precision policy: {policy.name}")
        print(f"Compute dtype: {policy.compute_dtype}")
        print(f"Variable dtype: {policy.variable_dtype}")

        with tf.device("/GPU:0"):
            # Create a simple model
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(1024, activation="relu", input_shape=(512,)),
                    tf.keras.layers.Dense(1024, activation="relu"),
                    tf.keras.layers.Dense(
                        10, activation="softmax", dtype="float32"
                    ),  # Keep output in float32
                ]
            )

            # Test forward pass
            batch_size = 256
            x = tf.random.normal([batch_size, 512], dtype=tf.float32)

            start_time = time.time()
            y = model(x)
            _ = y.numpy()
            end_time = time.time()

            print(f"  Input shape: {x.shape}, dtype: {x.dtype}")
            print(f"  Output shape: {y.shape}, dtype: {y.dtype}")
            print(f"  Forward pass time: {end_time - start_time:.4f} seconds")

            # Reset policy
            tf.keras.mixed_precision.set_global_policy("float32")

            print("✅ Mixed precision operations successful")
            return True

    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        # Reset policy on error
        tf.keras.mixed_precision.set_global_policy("float32")
        return False


def benchmark_performance(gpu_devices: List[tf.config.PhysicalDevice]) -> None:
    """Run performance benchmarks on GPU."""
    print_section("Performance Benchmark")

    if not gpu_devices:
        print("❌ No GPU devices available for benchmarking")
        return

    # Test different matrix sizes
    sizes = [512, 1024, 2048, 4096]

    print("Matrix multiplication benchmark:")
    print(f"{'Size':<8} {'GPU Time (ms)':<15} {'GFLOPS':<10}")
    print("-" * 35)

    for size in sizes:
        try:
            with tf.device("/GPU:0"):
                # Create random matrices
                a = tf.random.normal([size, size], dtype=tf.float32)
                b = tf.random.normal([size, size], dtype=tf.float32)

                # Warm up
                for _ in range(3):
                    _ = tf.matmul(a, b).numpy()

                # Benchmark
                times = []
                for _ in range(10):
                    start = time.time()
                    result = tf.matmul(a, b)
                    _ = result.numpy()  # Force computation
                    end = time.time()
                    times.append(end - start)

                avg_time = np.mean(times) * 1000  # Convert to ms

                # Calculate GFLOPS (2 * size^3 operations)
                ops = 2 * size**3
                gflops = ops / (avg_time / 1000) / 1e9

                print(f"{size:<8} {avg_time:<15.2f} {gflops:<10.1f}")

        except Exception as e:
            print(f"{size:<8} Error: {e}")


def run_memory_test(gpu_devices: List[tf.config.PhysicalDevice]) -> bool:
    """Test GPU memory allocation and usage."""
    print_section("GPU Memory Test")

    if not gpu_devices:
        print("❌ No GPU devices available for memory testing")
        return False

    try:
        # Get memory info before allocation
        gpu_details = tf.config.experimental.get_device_details(gpu_devices[0])
        total_memory = gpu_details.get("memory_limit", "Unknown")
        print(f"Total GPU memory: {total_memory}")

        with tf.device("/GPU:0"):
            print("Testing memory allocation...")

            # Allocate increasingly large tensors
            tensors = []
            sizes_mb = [100, 500, 1000, 2000]  # MB

            for size_mb in sizes_mb:
                try:
                    # Calculate tensor size (float32 = 4 bytes)
                    elements = (size_mb * 1024 * 1024) // 4
                    side = int(np.sqrt(elements))

                    print(f"  Allocating ~{size_mb}MB tensor ({side}x{side})...")
                    tensor = tf.random.normal([side, side], dtype=tf.float32)

                    # Perform operation to ensure allocation
                    result = tf.reduce_sum(tensor)
                    _ = result.numpy()

                    tensors.append(tensor)
                    print(f"    ✅ Successfully allocated {size_mb}MB")

                except tf.errors.ResourceExhaustedError as e:
                    print(f"    ❌ Memory exhausted at {size_mb}MB: {e}")
                    break
                except Exception as e:
                    print(f"    ❌ Error allocating {size_mb}MB: {e}")
                    break

            # Clean up
            del tensors

            print("✅ Memory test completed")
            return True

    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def main():
    """Main test function."""
    print_header("TensorFlow sm_120 GPU Test Suite")

    print("Testing TensorFlow installation with RTX 50-series GPU support...")

    # Check TensorFlow info
    check_tensorflow_info()

    # Check GPU devices
    gpu_devices = check_gpu_devices()

    # Run tests
    test_results = []

    # Basic operations test
    test_results.append(("Basic Operations", test_basic_operations(gpu_devices)))

    # Advanced tensor operations
    test_results.append(("Advanced Operations", test_tensor_operations(gpu_devices)))

    # Mixed precision test
    test_results.append(("Mixed Precision", test_mixed_precision()))

    # Memory test
    test_results.append(("Memory Test", run_memory_test(gpu_devices)))

    # Performance benchmark
    if gpu_devices:
        benchmark_performance(gpu_devices)

    # Summary
    print_section("Test Summary")

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:<20}: {status}")
        if result:
            passed_tests += 1

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print(
            "\n🎉 All tests passed! Your TensorFlow sm_120 installation is working correctly."
        )

        if gpu_devices:
            # Check if we have sm_120 GPU
            has_sm120 = False
            for gpu in gpu_devices:
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    cc = details.get("compute_capability")
                    if cc == (12, 0):
                        has_sm120 = True
                        break
                except Exception:
                    pass

            if has_sm120:
                print("🚀 RTX 50-series GPU (sm_120) detected and working optimally!")
            else:
                print(
                    "ℹ️  GPU detected but not RTX 50-series. Build will work on other GPUs too."
                )

        return 0
    else:
        print(
            f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check your installation."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
