"""
Comprehensive test suite for TensorFlow SM120 operations.

This module contains extensive tests for all SM120 optimized operations,
including correctness tests, performance benchmarks, and edge case validation.
"""

import unittest
import numpy as np
import tensorflow as tf
import time
from typing import Tuple
import warnings

# Import SM120 operations
try:
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
    import sm120_ops

    SM120_AVAILABLE = sm120_ops.is_sm120_available()
except ImportError as e:
    warnings.warn(f"Could not import sm120_ops: {e}")
    SM120_AVAILABLE = False


class TestSM120Operations(unittest.TestCase):
    """Test suite for SM120 operations."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.sm120_available = SM120_AVAILABLE

        # Set up TensorFlow
        tf.config.run_functions_eagerly(False)

        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Test data shapes
        cls.small_shapes = [(32, 64), (64, 32), (128, 256)]
        cls.medium_shapes = [(512, 1024), (1024, 512), (2048, 1024)]
        cls.large_shapes = [(4096, 4096), (8192, 4096), (4096, 8192)]

        # Supported data types
        cls.dtypes = [tf.float32, tf.float16]
        if hasattr(tf, "bfloat16"):
            cls.dtypes.append(tf.bfloat16)

    def setUp(self):
        """Set up each test."""
        if not self.sm120_available:
            self.skipTest("SM120 operations not available")

        # Reset configuration
        config = sm120_ops.get_config()
        config.optimization_level = 1
        config.use_tensor_cores = True
        config.fallback_to_standard = True

    def create_test_matrices(
        self, shape: Tuple[int, int], dtype: tf.DType
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create test matrices with known properties."""
        m, k = shape
        n = k  # Square-ish matrices for testing

        # Create well-conditioned matrices
        a = tf.random.normal([m, k], dtype=dtype, stddev=0.1)
        b = tf.random.normal([k, n], dtype=dtype, stddev=0.1)

        return a, b

    def assert_tensors_close(
        self,
        actual: tf.Tensor,
        expected: tf.Tensor,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ):
        """Assert that two tensors are close within tolerance."""
        if actual.dtype != expected.dtype:
            # Convert to common type for comparison
            common_dtype = tf.float32
            actual = tf.cast(actual, common_dtype)
            expected = tf.cast(expected, common_dtype)

        diff = tf.abs(actual - expected)
        max_diff = tf.reduce_max(diff)

        self.assertLessEqual(
            max_diff.numpy(),
            atol + rtol * tf.reduce_max(tf.abs(expected)).numpy(),
            f"Tensors not close: max_diff={max_diff.numpy()}, "
            f"rtol={rtol}, atol={atol}",
        )


class TestAdvancedMatMul(TestSM120Operations):
    """Tests for advanced matrix multiplication."""

    def test_basic_matmul(self):
        """Test basic matrix multiplication correctness."""
        for dtype in self.dtypes:
            for shape in self.small_shapes:
                with self.subTest(dtype=dtype, shape=shape):
                    a, b = self.create_test_matrices(shape, dtype)

                    # SM120 result
                    result_sm120 = sm120_ops.advanced_matmul(a, b)

                    # Reference result
                    result_ref = tf.matmul(a, b)

                    # Compare results
                    self.assert_tensors_close(result_sm120, result_ref)

    def test_transpose_variants(self):
        """Test matrix multiplication with transpose options."""
        dtype = tf.float32
        shape = (128, 256)
        a, b = self.create_test_matrices(shape, dtype)

        # Test all transpose combinations
        transpose_options = [(False, False), (True, False), (False, True), (True, True)]

        for transpose_a, transpose_b in transpose_options:
            with self.subTest(transpose_a=transpose_a, transpose_b=transpose_b):
                # SM120 result
                result_sm120 = sm120_ops.advanced_matmul(
                    a, b, transpose_a=transpose_a, transpose_b=transpose_b
                )

                # Reference result
                result_ref = tf.matmul(
                    a, b, transpose_a=transpose_a, transpose_b=transpose_b
                )

                self.assert_tensors_close(result_sm120, result_ref)

    def test_optimization_levels(self):
        """Test different optimization levels."""
        dtype = tf.float32
        shape = (512, 512)
        a, b = self.create_test_matrices(shape, dtype)

        reference = tf.matmul(a, b)

        for opt_level in [0, 1, 2]:
            with self.subTest(optimization_level=opt_level):
                result = sm120_ops.advanced_matmul(a, b, optimization_level=opt_level)
                self.assert_tensors_close(result, reference)

    def test_tensor_core_options(self):
        """Test Tensor Core enable/disable options."""
        dtype = tf.float16  # Tensor Cores work best with half precision
        shape = (256, 256)
        a, b = self.create_test_matrices(shape, dtype)

        reference = tf.matmul(a, b)

        for use_tensor_cores in [True, False]:
            with self.subTest(use_tensor_cores=use_tensor_cores):
                result = sm120_ops.advanced_matmul(
                    a, b, use_tensor_cores=use_tensor_cores
                )
                self.assert_tensors_close(result, reference, rtol=1e-2, atol=1e-2)

    def test_large_matrices(self):
        """Test performance with large matrices."""
        dtype = tf.float16
        shape = (2048, 2048)
        a, b = self.create_test_matrices(shape, dtype)

        # Benchmark SM120 implementation
        start_time = time.time()
        result_sm120 = sm120_ops.advanced_matmul(a, b)
        sm120_time = time.time() - start_time

        # Benchmark reference implementation
        start_time = time.time()
        result_ref = tf.matmul(a, b)
        ref_time = time.time() - start_time

        # Verify correctness
        self.assert_tensors_close(result_sm120, result_ref, rtol=1e-2, atol=1e-2)

        # Log performance comparison
        print(
            f"Large matmul ({shape}): SM120={sm120_time:.4f}s, "
            f"Reference={ref_time:.4f}s, Speedup={ref_time/sm120_time:.2f}x"
        )

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Mismatched dimensions
        a = tf.random.normal([32, 64], dtype=tf.float32)
        b = tf.random.normal([32, 64], dtype=tf.float32)  # Wrong shape

        with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
            sm120_ops.advanced_matmul(a, b)

        # Different dtypes
        a = tf.random.normal([32, 64], dtype=tf.float32)
        b = tf.random.normal([64, 32], dtype=tf.float16)

        with self.assertRaises(ValueError):
            sm120_ops.advanced_matmul(a, b)


class TestAdvancedConv2D(TestSM120Operations):
    """Tests for advanced 2D convolution."""

    def create_conv_test_data(
        self,
        batch_size: int,
        height: int,
        width: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dtype: tf.DType,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create test data for convolution."""
        input_tensor = tf.random.normal(
            [batch_size, height, width, in_channels], dtype=dtype, stddev=0.1
        )
        filter_tensor = tf.random.normal(
            [kernel_size, kernel_size, in_channels, out_channels],
            dtype=dtype,
            stddev=0.1,
        )
        return input_tensor, filter_tensor

    def test_basic_conv2d(self):
        """Test basic 2D convolution correctness."""
        batch_size, height, width = 8, 32, 32
        in_channels, out_channels = 16, 32
        kernel_size = 3

        for dtype in [tf.float32, tf.float16]:
            with self.subTest(dtype=dtype):
                input_tensor, filter_tensor = self.create_conv_test_data(
                    batch_size,
                    height,
                    width,
                    in_channels,
                    out_channels,
                    kernel_size,
                    dtype,
                )

                # SM120 result
                result_sm120 = sm120_ops.advanced_conv2d(
                    input_tensor, filter_tensor, strides=1, padding="SAME"
                )

                # Reference result
                result_ref = tf.nn.conv2d(
                    input_tensor, filter_tensor, strides=1, padding="SAME"
                )

                self.assert_tensors_close(
                    result_sm120, result_ref, rtol=1e-2, atol=1e-2
                )

    def test_stride_variations(self):
        """Test convolution with different stride values."""
        batch_size, height, width = 4, 64, 64
        in_channels, out_channels = 8, 16
        kernel_size = 3
        dtype = tf.float32

        input_tensor, filter_tensor = self.create_conv_test_data(
            batch_size, height, width, in_channels, out_channels, kernel_size, dtype
        )

        for stride in [1, 2, 3]:
            with self.subTest(stride=stride):
                result_sm120 = sm120_ops.advanced_conv2d(
                    input_tensor, filter_tensor, strides=stride, padding="SAME"
                )

                result_ref = tf.nn.conv2d(
                    input_tensor, filter_tensor, strides=stride, padding="SAME"
                )

                self.assert_tensors_close(result_sm120, result_ref)

    def test_padding_modes(self):
        """Test different padding modes."""
        batch_size, height, width = 4, 32, 32
        in_channels, out_channels = 8, 16
        kernel_size = 5
        dtype = tf.float32

        input_tensor, filter_tensor = self.create_conv_test_data(
            batch_size, height, width, in_channels, out_channels, kernel_size, dtype
        )

        for padding in ["SAME", "VALID"]:
            with self.subTest(padding=padding):
                result_sm120 = sm120_ops.advanced_conv2d(
                    input_tensor, filter_tensor, strides=1, padding=padding
                )

                result_ref = tf.nn.conv2d(
                    input_tensor, filter_tensor, strides=1, padding=padding
                )

                self.assert_tensors_close(result_sm120, result_ref)

    def test_large_conv2d(self):
        """Test performance with large convolutions."""
        batch_size, height, width = 16, 224, 224
        in_channels, out_channels = 64, 128
        kernel_size = 3
        dtype = tf.float16

        input_tensor, filter_tensor = self.create_conv_test_data(
            batch_size, height, width, in_channels, out_channels, kernel_size, dtype
        )

        # Benchmark SM120 implementation
        start_time = time.time()
        result_sm120 = sm120_ops.advanced_conv2d(
            input_tensor, filter_tensor, strides=1, padding="SAME"
        )
        sm120_time = time.time() - start_time

        # Benchmark reference implementation
        start_time = time.time()
        result_ref = tf.nn.conv2d(
            input_tensor, filter_tensor, strides=1, padding="SAME"
        )
        ref_time = time.time() - start_time

        # Verify correctness
        self.assert_tensors_close(result_sm120, result_ref, rtol=1e-2, atol=1e-2)

        print(
            f"Large conv2d: SM120={sm120_time:.4f}s, "
            f"Reference={ref_time:.4f}s, Speedup={ref_time/sm120_time:.2f}x"
        )


class TestFlashAttention(TestSM120Operations):
    """Tests for Flash Attention implementation."""

    def create_attention_test_data(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: tf.DType,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Create test data for attention."""
        queries = tf.random.normal(
            [batch_size, num_heads, seq_len, head_dim], dtype=dtype, stddev=0.1
        )
        keys = tf.random.normal(
            [batch_size, num_heads, seq_len, head_dim], dtype=dtype, stddev=0.1
        )
        values = tf.random.normal(
            [batch_size, num_heads, seq_len, head_dim], dtype=dtype, stddev=0.1
        )
        return queries, keys, values

    def reference_attention(
        self, queries: tf.Tensor, keys: tf.Tensor, values: tf.Tensor, scale: float
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Reference implementation of scaled dot-product attention."""
        scores = tf.matmul(queries, keys, transpose_b=True) * scale
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_output = tf.matmul(attention_weights, values)
        return attention_output, attention_weights

    def test_basic_attention(self):
        """Test basic attention correctness."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 64

        for dtype in [tf.float32, tf.float16]:
            with self.subTest(dtype=dtype):
                queries, keys, values = self.create_attention_test_data(
                    batch_size, num_heads, seq_len, head_dim, dtype
                )

                scale = 1.0 / np.sqrt(float(head_dim))

                # SM120 Flash Attention
                output_sm120, weights_sm120 = sm120_ops.flash_attention(
                    queries, keys, values, scale=scale
                )

                # Reference attention
                output_ref, weights_ref = self.reference_attention(
                    queries, keys, values, scale
                )

                # Compare outputs (Flash Attention should be very close)
                self.assert_tensors_close(
                    output_sm120, output_ref, rtol=1e-2, atol=1e-3
                )

    def test_causal_attention(self):
        """Test causal attention masking."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 64
        dtype = tf.float32

        queries, keys, values = self.create_attention_test_data(
            batch_size, num_heads, seq_len, head_dim, dtype
        )

        scale = 1.0 / np.sqrt(float(head_dim))

        # SM120 Flash Attention with causal mask
        output_sm120, weights_sm120 = sm120_ops.flash_attention(
            queries, keys, values, scale=scale, causal_mask=True
        )

        # Reference causal attention
        scores = tf.matmul(queries, keys, transpose_b=True) * scale

        # Apply causal mask
        mask_value = tf.constant(-1e9, dtype=scores.dtype)
        causal_mask = tf.linalg.band_part(
            tf.ones([seq_len, seq_len], dtype=scores.dtype), -1, 0
        )
        causal_mask = tf.where(tf.equal(causal_mask, 0), mask_value, 0.0)
        scores += causal_mask

        attention_weights = tf.nn.softmax(scores, axis=-1)
        output_ref = tf.matmul(attention_weights, values)

        # Compare outputs
        self.assert_tensors_close(output_sm120, output_ref, rtol=1e-2, atol=1e-3)

    def test_long_sequence_attention(self):
        """Test attention with long sequences."""
        batch_size, num_heads, seq_len, head_dim = 1, 8, 512, 64
        dtype = tf.float16

        queries, keys, values = self.create_attention_test_data(
            batch_size, num_heads, seq_len, head_dim, dtype
        )

        scale = 1.0 / np.sqrt(float(head_dim))

        # Benchmark Flash Attention
        start_time = time.time()
        output_sm120, _ = sm120_ops.flash_attention(queries, keys, values, scale=scale)
        flash_time = time.time() - start_time

        # Benchmark standard attention
        start_time = time.time()
        output_ref, _ = self.reference_attention(queries, keys, values, scale)
        standard_time = time.time() - start_time

        # Verify correctness
        self.assert_tensors_close(output_sm120, output_ref, rtol=1e-2, atol=1e-3)

        print(
            f"Long sequence attention (seq_len={seq_len}): "
            f"Flash={flash_time:.4f}s, Standard={standard_time:.4f}s, "
            f"Speedup={standard_time/flash_time:.2f}x"
        )


class TestPerformanceBenchmarks(TestSM120Operations):
    """Performance benchmark tests."""

    def test_matmul_performance_scaling(self):
        """Test matrix multiplication performance scaling."""
        dtype = tf.float16
        sizes = [512, 1024, 2048, 4096]

        results = []

        for size in sizes:
            a = tf.random.normal([size, size], dtype=dtype)
            b = tf.random.normal([size, size], dtype=dtype)

            # Benchmark SM120
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

            print(
                f"MatMul {size}x{size}: {benchmark_results['mean_time']*1000:.2f}ms, "
                f"{gflops:.1f} GFLOPS"
            )

        # Verify performance scaling
        self.assertGreater(
            results[-1]["gflops"],
            results[0]["gflops"],
            "Performance should improve with larger matrices",
        )

    def test_conv2d_performance_scaling(self):
        """Test convolution performance scaling."""
        dtype = tf.float16
        configs = [
            (16, 64, 64, 32, 64, 3),  # Small
            (16, 128, 128, 64, 128, 3),  # Medium
            (8, 224, 224, 64, 128, 3),  # Large
        ]

        for i, (batch_size, height, width, in_ch, out_ch, kernel_size) in enumerate(
            configs
        ):
            input_tensor = tf.random.normal(
                [batch_size, height, width, in_ch], dtype=dtype
            )
            filter_tensor = tf.random.normal(
                [kernel_size, kernel_size, in_ch, out_ch], dtype=dtype
            )

            benchmark_results = sm120_ops.benchmark_operation(
                sm120_ops.advanced_conv2d,
                input_tensor,
                filter_tensor,
                1,
                "SAME",
                num_iterations=5,
                warmup_iterations=2,
            )

            print(f"Conv2D config {i+1}: {benchmark_results['mean_time']*1000:.2f}ms")


class TestConfigurationAndFallback(TestSM120Operations):
    """Test configuration management and fallback behavior."""

    def test_configuration_changes(self):
        """Test configuration parameter changes."""
        config = sm120_ops.get_config()

        # Test optimization level changes
        for level in [0, 1, 2]:
            config.set_optimization_level(level)
            self.assertEqual(config.optimization_level, level)

        # Test invalid optimization level
        with self.assertRaises(ValueError):
            config.set_optimization_level(3)

        # Test tensor core toggle
        config.enable_tensor_cores(False)
        self.assertFalse(config.use_tensor_cores)

        config.enable_tensor_cores(True)
        self.assertTrue(config.use_tensor_cores)

    def test_fallback_behavior(self):
        """Test fallback to standard operations."""
        config = sm120_ops.get_config()
        original_fallback = config.fallback_to_standard

        # Enable fallback
        config.fallback_to_standard = True

        # This should work even if SM120 operations fail
        a = tf.random.normal([32, 64], dtype=tf.float32)
        b = tf.random.normal([64, 32], dtype=tf.float32)

        result = sm120_ops.advanced_matmul(a, b)
        expected = tf.matmul(a, b)

        self.assert_tensors_close(result, expected)

        # Restore original setting
        config.fallback_to_standard = original_fallback


if __name__ == "__main__":
    # Configure test environment
    print("SM120 Operations Test Suite")
    print("=" * 50)

    # Check SM120 availability
    if SM120_AVAILABLE:
        device_info = sm120_ops.get_sm120_device_info()
        print(f"SM120 Available: {device_info['available']}")
        compatible_devices = [
            d for d in device_info["devices"] if d.get("sm120_compatible", False)
        ]
        print(f"Compatible devices: {len(compatible_devices)}")

        for device in device_info["devices"]:
            if device.get("sm120_compatible", False):
                print(f"  - {device['name']} (CC {device['compute_capability']})")
    else:
        print("SM120 operations not available - tests will be skipped")

    print("=" * 50)

    # Run tests
    unittest.main(verbosity=2)
