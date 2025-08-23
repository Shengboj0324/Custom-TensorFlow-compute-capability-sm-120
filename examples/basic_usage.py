#!/usr/bin/env python3
"""
Basic Usage Examples for TensorFlow SM120 Operations

This script demonstrates how to use the TensorFlow SM120 optimizations
for RTX 50-series GPUs in practical scenarios.
"""

import sys
import os
import time
import numpy as np

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

try:
    import tensorflow as tf
    import sm120_ops

    print(f"✓ TensorFlow {tf.__version__} loaded successfully")
    print(f"✓ SM120 operations loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Please ensure TensorFlow and SM120 operations are installed correctly")
    sys.exit(1)


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"{title}")
    print(f"{'-'*40}")


def check_sm120_availability():
    """Check if SM120 operations are available."""
    print_header("SM120 Availability Check")

    # Check if SM120 operations are available
    available = sm120_ops.is_sm120_available()
    print(f"SM120 operations available: {available}")

    # Get detailed device information
    device_info = sm120_ops.get_sm120_device_info()
    print(f"Library loaded: {device_info['library_loaded']}")
    print(
        f"Compatible devices found: {len([d for d in device_info['devices'] if d.get('sm120_compatible', False)])}"
    )

    # Display device details
    for i, device in enumerate(device_info["devices"]):
        print(f"\nGPU {i}:")
        print(f"  Name: {device.get('name', 'Unknown')}")
        if "compute_capability" in device:
            cc = device["compute_capability"]
            if isinstance(cc, tuple):
                cc_str = f"{cc[0]}.{cc[1]}"
                if cc == (12, 0):
                    print(f"  Compute Capability: {cc_str} ✅ (RTX 50-series)")
                else:
                    print(f"  Compute Capability: {cc_str}")
            else:
                print(f"  Compute Capability: {cc}")

        if "error" in device:
            print(f"  Error: {device['error']}")

    return available


def demonstrate_advanced_matmul():
    """Demonstrate advanced matrix multiplication."""
    print_header("Advanced Matrix Multiplication Demo")

    # Test different matrix sizes and data types
    test_configs = [
        {"size": (512, 512), "dtype": tf.float32, "name": "FP32 Small"},
        {"size": (1024, 1024), "dtype": tf.float32, "name": "FP32 Medium"},
        {"size": (2048, 2048), "dtype": tf.float16, "name": "FP16 Large"},
    ]

    for config in test_configs:
        print_section(f"Testing {config['name']} - {config['size']}")

        size = config["size"]
        dtype = config["dtype"]

        # Create test matrices
        a = tf.random.normal(size, dtype=dtype)
        b = tf.random.normal([size[1], size[0]], dtype=dtype)

        print(f"Matrix A shape: {a.shape}, dtype: {a.dtype}")
        print(f"Matrix B shape: {b.shape}, dtype: {b.dtype}")

        # Standard TensorFlow matmul
        start_time = time.time()
        result_standard = tf.matmul(a, b)
        standard_time = time.time() - start_time

        # SM120 optimized matmul
        start_time = time.time()
        result_sm120 = sm120_ops.advanced_matmul(a, b, optimization_level=2)
        sm120_time = time.time() - start_time

        # Verify correctness
        max_diff = tf.reduce_max(tf.abs(result_standard - result_sm120))

        print(f"Standard TensorFlow: {standard_time*1000:.2f}ms")
        print(f"SM120 Optimized:     {sm120_time*1000:.2f}ms")
        print(f"Speedup:             {standard_time/sm120_time:.2f}x")
        print(f"Max difference:      {max_diff.numpy():.2e}")

        if max_diff < 1e-3:
            print("✅ Results match within tolerance")
        else:
            print("⚠️  Results differ more than expected")


def demonstrate_advanced_conv2d():
    """Demonstrate advanced 2D convolution."""
    print_header("Advanced 2D Convolution Demo")

    # Test different convolution configurations
    test_configs = [
        {
            "name": "Small Conv",
            "input_shape": (8, 32, 32, 16),
            "filter_shape": (3, 3, 16, 32),
            "strides": 1,
            "padding": "SAME",
        },
        {
            "name": "Large Conv",
            "input_shape": (4, 128, 128, 64),
            "filter_shape": (3, 3, 64, 128),
            "strides": 2,
            "padding": "SAME",
        },
    ]

    for config in test_configs:
        print_section(f"Testing {config['name']}")

        dtype = tf.float32

        # Create test data
        input_tensor = tf.random.normal(config["input_shape"], dtype=dtype)
        filter_tensor = tf.random.normal(config["filter_shape"], dtype=dtype)

        print(f"Input shape:  {input_tensor.shape}")
        print(f"Filter shape: {filter_tensor.shape}")
        print(f"Strides:      {config['strides']}")
        print(f"Padding:      {config['padding']}")

        # Standard TensorFlow conv2d
        start_time = time.time()
        result_standard = tf.nn.conv2d(
            input_tensor,
            filter_tensor,
            strides=config["strides"],
            padding=config["padding"],
        )
        standard_time = time.time() - start_time

        # SM120 optimized conv2d
        start_time = time.time()
        result_sm120 = sm120_ops.advanced_conv2d(
            input_tensor,
            filter_tensor,
            strides=config["strides"],
            padding=config["padding"],
            optimization_level=2,
        )
        sm120_time = time.time() - start_time

        # Verify correctness
        max_diff = tf.reduce_max(tf.abs(result_standard - result_sm120))

        print(f"Output shape:        {result_standard.shape}")
        print(f"Standard TensorFlow: {standard_time*1000:.2f}ms")
        print(f"SM120 Optimized:     {sm120_time*1000:.2f}ms")
        print(f"Speedup:             {standard_time/sm120_time:.2f}x")
        print(f"Max difference:      {max_diff.numpy():.2e}")

        if max_diff < 1e-3:
            print("✅ Results match within tolerance")
        else:
            print("⚠️  Results differ more than expected")


def demonstrate_flash_attention():
    """Demonstrate Flash Attention implementation."""
    print_header("Flash Attention Demo")

    # Test different attention configurations
    test_configs = [
        {
            "name": "Small Attention",
            "batch_size": 2,
            "num_heads": 8,
            "seq_len": 64,
            "head_dim": 64,
        },
        {
            "name": "Large Attention",
            "batch_size": 1,
            "num_heads": 16,
            "seq_len": 256,
            "head_dim": 64,
        },
    ]

    for config in test_configs:
        print_section(f"Testing {config['name']}")

        batch_size = config["batch_size"]
        num_heads = config["num_heads"]
        seq_len = config["seq_len"]
        head_dim = config["head_dim"]

        dtype = tf.float32

        # Create test data
        queries = tf.random.normal(
            [batch_size, num_heads, seq_len, head_dim], dtype=dtype
        )
        keys = tf.random.normal([batch_size, num_heads, seq_len, head_dim], dtype=dtype)
        values = tf.random.normal(
            [batch_size, num_heads, seq_len, head_dim], dtype=dtype
        )

        scale = 1.0 / np.sqrt(float(head_dim))

        print(f"Batch size:  {batch_size}")
        print(f"Num heads:   {num_heads}")
        print(f"Sequence:    {seq_len}")
        print(f"Head dim:    {head_dim}")
        print(f"Scale:       {scale:.4f}")

        # Standard attention implementation
        start_time = time.time()
        scores = tf.matmul(queries, keys, transpose_b=True) * scale
        attention_weights = tf.nn.softmax(scores, axis=-1)
        result_standard = tf.matmul(attention_weights, values)
        standard_time = time.time() - start_time

        # SM120 Flash Attention
        start_time = time.time()
        result_sm120, weights_sm120 = sm120_ops.flash_attention(
            queries, keys, values, scale=scale
        )
        sm120_time = time.time() - start_time

        # Verify correctness
        max_diff = tf.reduce_max(tf.abs(result_standard - result_sm120))

        print(f"Output shape:        {result_standard.shape}")
        print(f"Standard Attention:  {standard_time*1000:.2f}ms")
        print(f"Flash Attention:     {sm120_time*1000:.2f}ms")
        print(f"Speedup:             {standard_time/sm120_time:.2f}x")
        print(f"Max difference:      {max_diff.numpy():.2e}")

        if max_diff < 1e-2:  # Flash attention may have slightly different numerics
            print("✅ Results match within tolerance")
        else:
            print("⚠️  Results differ more than expected")


def demonstrate_optimized_dense_layer():
    """Demonstrate optimized dense layer."""
    print_header("Optimized Dense Layer Demo")

    print_section("Building Simple Neural Network")

    # Create a simple model using optimized operations
    class OptimizedModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # Note: This would use sm120_ops.optimized_dense in a real implementation
            self.dense1 = tf.keras.layers.Dense(512, activation="relu")
            self.dense2 = tf.keras.layers.Dense(256, activation="relu")
            self.dense3 = tf.keras.layers.Dense(10, activation="softmax")

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dense2(x)
            return self.dense3(x)

    # Create test data
    batch_size = 64
    input_dim = 784

    x = tf.random.normal([batch_size, input_dim])

    print(f"Input shape: {x.shape}")

    # Create and test model
    model = OptimizedModel()

    start_time = time.time()
    output = model(x)
    inference_time = time.time() - start_time

    print(f"Output shape: {output.shape}")
    print(f"Inference time: {inference_time*1000:.2f}ms")
    print(f"Throughput: {batch_size/inference_time:.1f} samples/sec")

    # Verify output properties
    output_sum = tf.reduce_sum(output, axis=-1)
    print(f"Output sum (should be ~1.0): {tf.reduce_mean(output_sum):.4f}")

    if tf.reduce_all(tf.abs(output_sum - 1.0) < 1e-5):
        print("✅ Softmax output is properly normalized")
    else:
        print("⚠️  Softmax output normalization issue")


def run_performance_benchmark():
    """Run a comprehensive performance benchmark."""
    print_header("Performance Benchmark Suite")

    if not sm120_ops.is_sm120_available():
        print("⚠️  SM120 operations not available - skipping detailed benchmarks")
        return

    print_section("Matrix Multiplication Scaling")

    sizes = [256, 512, 1024, 2048]
    results = []

    for size in sizes:
        print(f"Testing {size}x{size} matrices...")

        a = tf.random.normal([size, size], dtype=tf.float16)
        b = tf.random.normal([size, size], dtype=tf.float16)

        # Benchmark SM120 implementation
        benchmark_results = sm120_ops.benchmark_operation(
            sm120_ops.advanced_matmul, a, b, num_iterations=10, warmup_iterations=3
        )

        ops = 2 * size**3  # Matrix multiplication operations
        gflops = ops / benchmark_results["mean_time"] / 1e9

        results.append(
            {
                "size": size,
                "time_ms": benchmark_results["mean_time"] * 1000,
                "gflops": gflops,
            }
        )

        print(f"  Time: {benchmark_results['mean_time']*1000:.2f}ms")
        print(f"  GFLOPS: {gflops:.1f}")
        print(f"  Std dev: {benchmark_results['std_time']*1000:.2f}ms")

    print("\nPerformance Summary:")
    print("Size      Time (ms)    GFLOPS")
    print("-" * 30)
    for result in results:
        print(
            f"{result['size']:4d}      {result['time_ms']:8.2f}    {result['gflops']:6.1f}"
        )


def main():
    """Main demonstration function."""
    print_header("TensorFlow SM120 Operations - Basic Usage Demo")

    # Check availability first
    sm120_available = check_sm120_availability()

    if not sm120_available:
        print("\n⚠️  SM120 operations not available.")
        print("The demo will continue using fallback implementations,")
        print("but you won't see the full performance benefits.")
        print("For optimal performance, ensure you have:")
        print("  - RTX 50-series GPU (RTX 5080/5090)")
        print("  - NVIDIA drivers 570.x+")
        print("  - CUDA 12.8+")
        print("  - Properly compiled SM120 operations")

    # Run demonstrations
    try:
        demonstrate_advanced_matmul()
        demonstrate_advanced_conv2d()
        demonstrate_flash_attention()
        demonstrate_optimized_dense_layer()

        if sm120_available:
            run_performance_benchmark()

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print_header("Demo Completed Successfully!")

    if sm120_available:
        print("🚀 Your system is ready for high-performance SM120 operations!")
    else:
        print(
            "ℹ️  Install SM120 optimizations for maximum performance on RTX 50-series GPUs."
        )

    print("\nNext steps:")
    print("1. Integrate SM120 operations into your models")
    print("2. Run the full test suite: python -m pytest tests/")
    print("3. Check out advanced examples in the examples/ directory")
    print("4. Read the documentation for detailed API reference")

    return 0


if __name__ == "__main__":
    sys.exit(main())
