#!/usr/bin/env python3
"""
TensorFlow SM120 Benchmark Tool

This module provides benchmarking capabilities for SM120 optimized operations.
It can be used as a standalone script or imported as a module.
"""

import argparse
import time
import sys
import os
from typing import Dict, List, Tuple, Optional

import tensorflow as tf
import numpy as np

# Add the parent directory to the path to import sm120_ops
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try importing as installed package first
    try:
        import sm120_ops
        SM120_AVAILABLE = sm120_ops.is_sm120_available()
    except ImportError:
        # Fallback to relative import for development
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import sm120_ops
        SM120_AVAILABLE = sm120_ops.is_sm120_available()
except ImportError:
    print("Warning: SM120 operations not available. Using standard TensorFlow operations.")
    SM120_AVAILABLE = False


class SM120Benchmark:
    """Benchmark suite for SM120 operations."""
    
    def __init__(self, warmup_iterations: int = 5, benchmark_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = {}
        
    def benchmark_matmul(self, shapes: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """Benchmark matrix multiplication operations."""
        results = {}
        
        for M, N, K in shapes:
            print(f"Benchmarking MatMul {M}x{N}x{K}...")
            
            # Create test data
            A = tf.random.normal([M, K], dtype=tf.float32)
            B = tf.random.normal([K, N], dtype=tf.float32)
            
            # Warmup
            for _ in range(self.warmup_iterations):
                if SM120_AVAILABLE:
                    _ = sm120_ops.sm120_matmul(A, B)
                else:
                    _ = tf.matmul(A, B)
            
            # Benchmark
            start_time = time.time()
            for _ in range(self.benchmark_iterations):
                if SM120_AVAILABLE:
                    result = sm120_ops.sm120_matmul(A, B)
                else:
                    result = tf.matmul(A, B)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / self.benchmark_iterations * 1000  # ms
            results[f"matmul_{M}x{N}x{K}"] = avg_time
            print(f"  Average time: {avg_time:.3f} ms")
            
        return results
    
    def benchmark_conv2d(self, shapes: List[Tuple[int, int, int, int]]) -> Dict[str, float]:
        """Benchmark 2D convolution operations."""
        results = {}
        
        for batch, height, width, channels in shapes:
            print(f"Benchmarking Conv2D {batch}x{height}x{width}x{channels}...")
            
            # Create test data
            input_tensor = tf.random.normal([batch, height, width, channels], dtype=tf.float32)
            filters = tf.random.normal([3, 3, channels, 64], dtype=tf.float32)
            
            # Warmup
            for _ in range(self.warmup_iterations):
                if SM120_AVAILABLE:
                    _ = sm120_ops.sm120_conv2d(input_tensor, filters, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    _ = tf.nn.conv2d(input_tensor, filters, strides=[1, 1, 1, 1], padding='SAME')
            
            # Benchmark
            start_time = time.time()
            for _ in range(self.benchmark_iterations):
                if SM120_AVAILABLE:
                    result = sm120_ops.sm120_conv2d(input_tensor, filters, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    result = tf.nn.conv2d(input_tensor, filters, strides=[1, 1, 1, 1], padding='SAME')
            end_time = time.time()
            
            avg_time = (end_time - start_time) / self.benchmark_iterations * 1000  # ms
            results[f"conv2d_{batch}x{height}x{width}x{channels}"] = avg_time
            print(f"  Average time: {avg_time:.3f} ms")
            
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Run all available benchmarks."""
        print("Starting SM120 Benchmark Suite...")
        print(f"SM120 Operations Available: {SM120_AVAILABLE}")
        print(f"Warmup iterations: {self.warmup_iterations}")
        print(f"Benchmark iterations: {self.benchmark_iterations}")
        print("-" * 50)
        
        all_results = {}
        
        # Matrix multiplication benchmarks
        matmul_shapes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
        all_results["matmul"] = self.benchmark_matmul(matmul_shapes)
        
        # Convolution benchmarks
        conv_shapes = [(32, 224, 224, 3), (64, 112, 112, 64), (128, 56, 56, 128)]
        all_results["conv2d"] = self.benchmark_conv2d(conv_shapes)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        for category, category_results in results.items():
            print(f"\n{category.upper()} Results:")
            for operation, time_ms in category_results.items():
                print(f"  {operation}: {time_ms:.3f} ms")
        
        print(f"\nSM120 Optimizations: {'ENABLED' if SM120_AVAILABLE else 'DISABLED'}")
        print("=" * 60)


def main():
    """Main entry point for the benchmark tool."""
    parser = argparse.ArgumentParser(description="TensorFlow SM120 Benchmark Tool")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer iterations")
    
    args = parser.parse_args()
    
    if args.quick:
        warmup = 2
        iterations = 10
    else:
        warmup = args.warmup
        iterations = args.iterations
    
    # Configure TensorFlow
    tf.config.run_functions_eagerly(False)
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Run benchmarks
    benchmark = SM120Benchmark(warmup_iterations=warmup, benchmark_iterations=iterations)
    results = benchmark.run_all_benchmarks()
    benchmark.print_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
