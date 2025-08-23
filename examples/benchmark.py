#!/usr/bin/env python3
"""
TensorFlow sm_120 Performance Benchmark

Comprehensive benchmarking suite for TensorFlow with RTX 50-series GPU support.
Tests various operations and compares performance with different configurations.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not found. Please install the custom sm_120 wheel.")
    sys.exit(1)

# Suppress TensorFlow warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class BenchmarkSuite:
    """Comprehensive benchmark suite for TensorFlow GPU performance."""

    def __init__(
        self, device: str = "/GPU:0", warmup_runs: int = 3, benchmark_runs: int = 10
    ):
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for the benchmark report."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "tensorflow_version": tf.__version__,
            "tensorflow_location": tf.__file__,
            "cuda_built": tf.test.is_built_with_cuda(),
            "gpu_support": tf.test.is_built_with_gpu_support(),
            "device": self.device,
        }

        # GPU information
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_info = []
            for i, gpu in enumerate(gpus):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    gpu_info.append({"index": i, "name": gpu.name, "details": details})
                except Exception as e:
                    gpu_info.append({"index": i, "name": gpu.name, "error": str(e)})
            info["gpus"] = gpu_info

        return info

    def _time_operation(self, operation_fn, *args, **kwargs) -> Tuple[float, Any]:
        """Time a TensorFlow operation with warmup."""
        # Warmup runs
        for _ in range(self.warmup_runs):
            _ = operation_fn(*args, **kwargs)

        # Benchmark runs
        times = []
        result = None

        for _ in range(self.benchmark_runs):
            start_time = time.time()
            result = operation_fn(*args, **kwargs)
            # Force computation
            if hasattr(result, "numpy"):
                _ = result.numpy()
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        return avg_time, result

    def benchmark_matrix_multiplication(self, sizes: List[int]) -> Dict[str, Any]:
        """Benchmark matrix multiplication for different sizes."""
        print("🔄 Benchmarking matrix multiplication...")

        results = {}

        with tf.device(self.device):
            for size in sizes:
                print(f"  Testing {size}x{size} matrices...")

                # Create test matrices
                a = tf.random.normal([size, size], dtype=tf.float32)
                b = tf.random.normal([size, size], dtype=tf.float32)

                # Benchmark
                avg_time, _ = self._time_operation(tf.matmul, a, b)

                # Calculate metrics
                ops = 2 * size**3  # Matrix multiplication operations
                gflops = ops / avg_time / 1e9

                results[f"matmul_{size}x{size}"] = {
                    "size": size,
                    "time_seconds": avg_time,
                    "time_ms": avg_time * 1000,
                    "operations": ops,
                    "gflops": gflops,
                }

                print(f"    Time: {avg_time*1000:.2f}ms, GFLOPS: {gflops:.1f}")

        return results

    def benchmark_convolution(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark 2D convolution operations."""
        print("🔄 Benchmarking convolution operations...")

        results = {}

        with tf.device(self.device):
            for i, config in enumerate(configs):
                batch_size = config["batch_size"]
                input_shape = config["input_shape"]  # [H, W, C]
                filters = config["filters"]
                kernel_size = config["kernel_size"]

                print(
                    f"  Config {i+1}: {batch_size}x{input_shape[0]}x{input_shape[1]}x{input_shape[2]} -> {filters} filters"
                )

                # Create input tensor
                x = tf.random.normal([batch_size] + input_shape, dtype=tf.float32)

                # Create convolution layer
                conv_layer = tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    activation="relu",
                )

                # Benchmark
                avg_time, output = self._time_operation(conv_layer, x)

                # Calculate metrics
                output_elements = np.prod(output.shape)
                throughput = (
                    output_elements / avg_time / 1e6
                )  # Million elements per second

                config_name = f"conv2d_b{batch_size}_h{input_shape[0]}_w{input_shape[1]}_c{input_shape[2]}_f{filters}"
                results[config_name] = {
                    "config": config,
                    "input_shape": x.shape.as_list(),
                    "output_shape": output.shape.as_list(),
                    "time_seconds": avg_time,
                    "time_ms": avg_time * 1000,
                    "throughput_meps": throughput,  # Million elements per second
                }

                print(
                    f"    Time: {avg_time*1000:.2f}ms, Throughput: {throughput:.1f} MEPS"
                )

        return results

    def benchmark_mixed_precision(self, sizes: List[int]) -> Dict[str, Any]:
        """Benchmark mixed precision performance."""
        print("🔄 Benchmarking mixed precision...")

        results = {}

        # Test both float32 and mixed_float16
        for precision in ["float32", "mixed_float16"]:
            print(f"  Testing {precision} precision...")

            # Set precision policy
            policy = tf.keras.mixed_precision.Policy(precision)
            tf.keras.mixed_precision.set_global_policy(policy)

            precision_results = {}

            with tf.device(self.device):
                for size in sizes:
                    print(f"    Matrix size: {size}x{size}")

                    # Create model for testing
                    model = tf.keras.Sequential(
                        [
                            tf.keras.layers.Dense(
                                size, activation="relu", input_shape=(size,)
                            ),
                            tf.keras.layers.Dense(size, activation="relu"),
                            tf.keras.layers.Dense(
                                size // 4,
                                activation="softmax",
                                dtype=(
                                    "float32" if precision == "mixed_float16" else None
                                ),
                            ),
                        ]
                    )

                    # Create input data
                    x = tf.random.normal([32, size], dtype=tf.float32)

                    # Benchmark forward pass
                    avg_time, output = self._time_operation(model, x)

                    precision_results[f"dense_{size}"] = {
                        "size": size,
                        "time_seconds": avg_time,
                        "time_ms": avg_time * 1000,
                        "input_dtype": str(x.dtype),
                        "output_dtype": str(output.dtype),
                    }

                    print(f"      Time: {avg_time*1000:.2f}ms")

            results[precision] = precision_results

        # Reset to default policy
        tf.keras.mixed_precision.set_global_policy("float32")

        return results

    def benchmark_reduction_operations(
        self, shapes: List[Tuple[int, ...]]
    ) -> Dict[str, Any]:
        """Benchmark various reduction operations."""
        print("🔄 Benchmarking reduction operations...")

        results = {}
        operations = ["sum", "mean", "max", "min", "std"]

        with tf.device(self.device):
            for shape in shapes:
                print(f"  Testing tensor shape: {shape}")

                # Create test tensor
                x = tf.random.normal(shape, dtype=tf.float32)
                elements = np.prod(shape)

                shape_results = {}

                for op_name in operations:
                    if op_name == "sum":
                        op_fn = tf.reduce_sum
                    elif op_name == "mean":
                        op_fn = tf.reduce_mean
                    elif op_name == "max":
                        op_fn = tf.reduce_max
                    elif op_name == "min":
                        op_fn = tf.reduce_min
                    elif op_name == "std":
                        op_fn = lambda x: tf.math.reduce_std(x)

                    avg_time, _ = self._time_operation(op_fn, x)
                    throughput = (
                        elements / avg_time / 1e6
                    )  # Million elements per second

                    shape_results[op_name] = {
                        "time_seconds": avg_time,
                        "time_ms": avg_time * 1000,
                        "throughput_meps": throughput,
                    }

                shape_key = "x".join(map(str, shape))
                results[f"reduction_{shape_key}"] = {
                    "shape": shape,
                    "elements": elements,
                    "operations": shape_results,
                }

                print(
                    f"    Avg time per operation: {np.mean([r['time_ms'] for r in shape_results.values()]):.2f}ms"
                )

        return results

    def benchmark_memory_bandwidth(self, sizes: List[int]) -> Dict[str, Any]:
        """Benchmark memory bandwidth using simple operations."""
        print("🔄 Benchmarking memory bandwidth...")

        results = {}

        with tf.device(self.device):
            for size in sizes:
                print(f"  Testing memory transfer for {size}MB...")

                # Calculate tensor dimensions for approximately size MB
                elements = (size * 1024 * 1024) // 4  # 4 bytes per float32
                tensor_size = int(np.sqrt(elements))
                actual_mb = (tensor_size**2 * 4) / (1024 * 1024)

                # Create large tensor
                x = tf.random.normal([tensor_size, tensor_size], dtype=tf.float32)

                # Test memory copy (GPU -> GPU)
                avg_time, _ = self._time_operation(tf.identity, x)

                # Calculate bandwidth
                bandwidth_gbps = (actual_mb / 1024) / avg_time  # GB/s

                results[f"memory_{size}MB"] = {
                    "target_mb": size,
                    "actual_mb": actual_mb,
                    "tensor_shape": [tensor_size, tensor_size],
                    "time_seconds": avg_time,
                    "time_ms": avg_time * 1000,
                    "bandwidth_gbps": bandwidth_gbps,
                }

                print(f"    Bandwidth: {bandwidth_gbps:.1f} GB/s")

        return results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        print("🚀 Starting TensorFlow sm_120 Performance Benchmark")
        print(f"Device: {self.device}")
        print(f"Warmup runs: {self.warmup_runs}, Benchmark runs: {self.benchmark_runs}")
        print("=" * 60)

        # Matrix multiplication benchmark
        matmul_sizes = [512, 1024, 2048, 4096, 8192]
        self.results["matrix_multiplication"] = self.benchmark_matrix_multiplication(
            matmul_sizes
        )

        # Convolution benchmark
        conv_configs = [
            {
                "batch_size": 32,
                "input_shape": [224, 224, 3],
                "filters": 64,
                "kernel_size": 3,
            },
            {
                "batch_size": 16,
                "input_shape": [512, 512, 3],
                "filters": 32,
                "kernel_size": 5,
            },
            {
                "batch_size": 8,
                "input_shape": [1024, 1024, 1],
                "filters": 16,
                "kernel_size": 7,
            },
        ]
        self.results["convolution"] = self.benchmark_convolution(conv_configs)

        # Mixed precision benchmark
        mixed_precision_sizes = [512, 1024, 2048]
        self.results["mixed_precision"] = self.benchmark_mixed_precision(
            mixed_precision_sizes
        )

        # Reduction operations benchmark
        reduction_shapes = [(10000, 10000), (1000000,), (100, 100, 100, 100)]
        self.results["reduction_operations"] = self.benchmark_reduction_operations(
            reduction_shapes
        )

        # Memory bandwidth benchmark
        memory_sizes = [100, 500, 1000, 2000]
        self.results["memory_bandwidth"] = self.benchmark_memory_bandwidth(memory_sizes)

        # Compile final results
        final_results = {
            "system_info": self.system_info,
            "benchmark_config": {
                "device": self.device,
                "warmup_runs": self.warmup_runs,
                "benchmark_runs": self.benchmark_runs,
            },
            "results": self.results,
        }

        return final_results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)

        # System info
        system_info = results["system_info"]
        print(f"TensorFlow Version: {system_info['tensorflow_version']}")
        print(f"CUDA Support: {system_info['cuda_built']}")
        print(f"Device: {system_info['device']}")

        if "gpus" in system_info and system_info["gpus"]:
            gpu = system_info["gpus"][0]
            print(f"GPU: {gpu['name']}")
            if "details" in gpu:
                details = gpu["details"]
                cc = details.get("compute_capability", "Unknown")
                if isinstance(cc, tuple):
                    cc_str = f"{cc[0]}.{cc[1]}"
                    if cc == (12, 0):
                        print(
                            f"Compute Capability: {cc_str} ✅ (sm_120 - RTX 50-series)"
                        )
                    else:
                        print(f"Compute Capability: {cc_str}")

        print()

        # Matrix multiplication summary
        if "matrix_multiplication" in results["results"]:
            print("Matrix Multiplication Performance:")
            matmul_results = results["results"]["matrix_multiplication"]
            for key, result in matmul_results.items():
                size = result["size"]
                gflops = result["gflops"]
                print(f"  {size:>4}x{size:<4}: {gflops:>7.1f} GFLOPS")

        print()

        # Mixed precision comparison
        if "mixed_precision" in results["results"]:
            print("Mixed Precision Performance (1024x1024 dense layer):")
            mixed_results = results["results"]["mixed_precision"]
            if "float32" in mixed_results and "mixed_float16" in mixed_results:
                fp32_time = (
                    mixed_results["float32"].get("dense_1024", {}).get("time_ms", 0)
                )
                fp16_time = (
                    mixed_results["mixed_float16"]
                    .get("dense_1024", {})
                    .get("time_ms", 0)
                )
                if fp32_time > 0 and fp16_time > 0:
                    speedup = fp32_time / fp16_time
                    print(f"  FP32:  {fp32_time:>7.2f}ms")
                    print(f"  FP16:  {fp16_time:>7.2f}ms")
                    print(f"  Speedup: {speedup:>5.2f}x")

        print()

        # Memory bandwidth summary
        if "memory_bandwidth" in results["results"]:
            print("Memory Bandwidth:")
            mem_results = results["results"]["memory_bandwidth"]
            bandwidths = [result["bandwidth_gbps"] for result in mem_results.values()]
            if bandwidths:
                avg_bandwidth = np.mean(bandwidths)
                max_bandwidth = np.max(bandwidths)
                print(f"  Average: {avg_bandwidth:>6.1f} GB/s")
                print(f"  Peak:    {max_bandwidth:>6.1f} GB/s")

        print("\n" + "=" * 60)


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description="TensorFlow sm_120 Performance Benchmark"
    )
    parser.add_argument(
        "--device", default="/GPU:0", help="Device to benchmark (default: /GPU:0)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup runs (default: 3)"
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of benchmark runs (default: 10)"
    )
    parser.add_argument(
        "--output", help="Output file for detailed results (JSON format)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with fewer tests"
    )

    args = parser.parse_args()

    # Check if GPU is available
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("❌ No GPU devices found. Cannot run GPU benchmark.")
        sys.exit(1)

    # Create benchmark suite
    benchmark = BenchmarkSuite(
        device=args.device, warmup_runs=args.warmup, benchmark_runs=args.runs
    )

    # Adjust for quick benchmark
    if args.quick:
        print("🏃 Running quick benchmark...")
        benchmark.warmup_runs = 1
        benchmark.benchmark_runs = 3

    try:
        # Run benchmarks
        results = benchmark.run_comprehensive_benchmark()

        # Print summary
        benchmark.print_summary(results)

        # Save detailed results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Detailed results saved to: {args.output}")

        print("\n✅ Benchmark completed successfully!")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
