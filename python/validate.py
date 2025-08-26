#!/usr/bin/env python3
"""
TensorFlow SM120 Validation Tool

This module provides validation capabilities for SM120 optimized operations.
It can be used as a standalone script or imported as a module.
"""

import argparse
import sys
import os
import subprocess
from typing import Dict, List, Tuple, Optional, Any

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
    print("Warning: SM120 operations not available.")
    SM120_AVAILABLE = False


class Colors:
    """Terminal color codes."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[1;37m"
    NC = "\033[0m"  # No Color


def log_info(message: str) -> None:
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def log_success(message: str) -> None:
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def log_warning(message: str) -> None:
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def log_error(message: str) -> None:
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def log_header(message: str) -> None:
    print(f"\n{Colors.WHITE}{'='*60}{Colors.NC}")
    print(f"{Colors.WHITE}{message:^60}{Colors.NC}")
    print(f"{Colors.WHITE}{'='*60}{Colors.NC}")


class SM120Validator:
    """Validation suite for SM120 operations."""

    def __init__(self):
        self.validation_results = {}

    def validate_system_requirements(self) -> bool:
        """Validate system requirements for SM120."""
        log_header("System Requirements Validation")

        success = True

        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 9):
            log_success(
                f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
            )
        else:
            log_error(
                f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro} (3.9+ required)"
            )
            success = False

        # Check TensorFlow version
        tf_version = tf.__version__
        tf_major, tf_minor = map(int, tf_version.split(".")[:2])
        if tf_major >= 2 and tf_minor >= 10:
            log_success(f"TensorFlow version: {tf_version}")
        else:
            log_error(f"TensorFlow version: {tf_version} (2.10+ required)")
            success = False

        # Check CUDA availability
        if tf.test.is_built_with_cuda():
            log_success("TensorFlow built with CUDA support")
        else:
            log_warning("TensorFlow not built with CUDA support")

        # Check GPU availability
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            log_success(f"GPU devices found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                log_info(f"  GPU {i}: {gpu.name}")
        else:
            log_warning("No GPU devices found")

        return success
    
    def validate_sm120_operations(self) -> bool:
        """Validate SM120 operations functionality."""
        log_header("SM120 Operations Validation")

        if not SM120_AVAILABLE:
            log_error("SM120 operations not available")
            return False

        success = True

        try:
            # Test basic SM120 functionality
            log_info("Testing SM120 availability check...")
            is_available = sm120_ops.is_sm120_available()
            if is_available:
                log_success("SM120 operations are available")
            else:
                log_warning("SM120 operations loaded but no compatible GPU detected")

            # Test device info
            log_info("Testing SM120 device information...")
            device_info = sm120_ops.get_sm120_device_info()
            log_success(
                f"Device info retrieved: {len(device_info.get('devices', []))} devices"
            )

            # Test basic operations if GPU is available
            if is_available:
                log_info("Testing basic SM120 matrix multiplication...")
                A = tf.random.normal([128, 256], dtype=tf.float32)
                B = tf.random.normal([256, 128], dtype=tf.float32)

                result = sm120_ops.sm120_matmul(A, B)
                expected_shape = [128, 128]

                if result.shape.as_list() == expected_shape:
                    log_success("SM120 matrix multiplication test passed")
                else:
                    log_error(
                        f"SM120 matrix multiplication shape mismatch: got {result.shape}, expected {expected_shape}"
                    )
                    success = False

        except Exception as e:
            log_error(f"SM120 operations validation failed: {e}")
            success = False

        return success
    
    def validate_performance(self) -> bool:
        """Validate SM120 performance improvements."""
        log_header("Performance Validation")

        if not SM120_AVAILABLE:
            log_warning("Skipping performance validation - SM120 not available")
            return True

        try:
            # Simple performance comparison
            log_info("Running performance comparison...")

            A = tf.random.normal([512, 512], dtype=tf.float32)
            B = tf.random.normal([512, 512], dtype=tf.float32)

            # Warmup
            for _ in range(5):
                _ = tf.matmul(A, B)
                if SM120_AVAILABLE:
                    _ = sm120_ops.sm120_matmul(A, B)

            # Time standard TensorFlow
            import time

            start = time.time()
            for _ in range(10):
                _ = tf.matmul(A, B)
            tf_time = time.time() - start

            # Time SM120
            start = time.time()
            for _ in range(10):
                _ = sm120_ops.sm120_matmul(A, B)
            sm120_time = time.time() - start

            log_info(f"Standard TensorFlow time: {tf_time:.4f}s")
            log_info(f"SM120 optimized time: {sm120_time:.4f}s")

            if sm120_time < tf_time:
                speedup = tf_time / sm120_time
                log_success(f"SM120 speedup: {speedup:.2f}x")
            else:
                log_warning(
                    "SM120 did not show performance improvement (may be due to small problem size)"
                )

            return True

        except Exception as e:
            log_error(f"Performance validation failed: {e}")
            return False
    
    def run_all_validations(self) -> bool:
        """Run all validation tests."""
        log_header("TensorFlow SM120 Validation Suite")

        results = []

        # System requirements
        results.append(self.validate_system_requirements())

        # SM120 operations
        results.append(self.validate_sm120_operations())

        # Performance validation
        results.append(self.validate_performance())

        # Summary
        log_header("Validation Summary")

        if all(results):
            log_success("All validations passed!")
            return True
        else:
            log_error("Some validations failed. Check the output above for details.")
            return False


def main():
    """Main entry point for the validation tool."""
    parser = argparse.ArgumentParser(description="TensorFlow SM120 Validation Tool")
    parser.add_argument("--quick", action="store_true", help="Run quick validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Configure TensorFlow
    if not args.verbose:
        tf.get_logger().setLevel("ERROR")

    tf.config.run_functions_eagerly(False)

    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Run validation
    validator = SM120Validator()
    success = validator.run_all_validations()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
