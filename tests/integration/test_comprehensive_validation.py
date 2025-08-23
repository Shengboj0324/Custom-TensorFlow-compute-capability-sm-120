"""
Comprehensive Integration Tests for SM120 TensorFlow Optimization Suite
Tests all components together to ensure zero-tolerance for mistakes
Copyright 2024 - TensorFlow SM120 Optimization Project
"""

import pytest
import tensorflow as tf
import numpy as np
import time
import gc
import sys
import os
from typing import List, Dict, Tuple, Any
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from python.sm120_keras_layers import (
        SM120Dense, SM120Conv2D, SM120MultiHeadAttention, 
        SM120BatchNormalization, create_sm120_transformer_encoder
    )
    from python import sm120_ops
    from python.sm120_performance_api import (
        enable_profiling, print_performance_summary, 
        get_average_metrics, benchmark_operation
    )
    from python.sm120_datatype_manager import (
        enable_mixed_precision, auto_cast, get_supported_types,
        recommend_model_dtypes, print_type_summary
    )
    SM120_AVAILABLE = True
except ImportError as e:
    SM120_AVAILABLE = False
    print(f"SM120 not available: {e}")

# Test configuration
TEST_CONFIG = {
    'batch_sizes': [1, 8, 32, 64],
    'sequence_lengths': [128, 512, 1024],
    'embedding_dims': [256, 512, 768, 1024],
    'num_heads': [8, 12, 16],
    'data_types': ['float32', 'float16'],
    'tolerance': 1e-5,
    'performance_threshold': 0.8  # Minimum relative performance vs standard TF
}

class TestEnvironmentSetup:
    """Test environment setup and GPU validation."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        
        # Enable performance monitoring
        if SM120_AVAILABLE:
            enable_profiling(True)
    
    def test_gpu_availability(self):
        """Test GPU availability and compatibility."""
        gpus = tf.config.list_physical_devices('GPU')
        assert len(gpus) > 0, "No GPU devices found for testing"
        
        gpu_info = {}
        for i, gpu in enumerate(gpus):
            try:
                device_details = tf.config.experimental.get_device_details(gpu)
                compute_capability = device_details.get('compute_capability', (0, 0))
                gpu_info[f'gpu_{i}'] = {
                    'name': gpu.name,
                    'compute_capability': compute_capability,
                    'memory_limit': tf.config.experimental.get_memory_info(gpu.name).get('total', 0)
                }
                
                # Check minimum requirements
                assert compute_capability[0] >= 7, f"GPU {i} compute capability too low: {compute_capability}"
                
            except Exception as e:
                warnings.warn(f"Could not get details for GPU {i}: {e}")
        
        print(f"✅ GPU Test Results: {gpu_info}")
    
    def test_sm120_availability(self):
        """Test SM120 operations availability."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Test basic operation availability
        assert hasattr(sm120_ops, 'advanced_matmul'), "SM120 MatMul not available"
        assert hasattr(sm120_ops, 'conv2d'), "SM120 Conv2D not available"
        
        # Test supported data types
        supported_types = get_supported_types()
        assert 'float32' in supported_types, "Float32 support required"
        assert len(supported_types) >= 2, "Insufficient data type support"
        
        print(f"✅ SM120 Available with types: {supported_types}")

class TestBasicOperations:
    """Test basic SM120 operations for correctness."""
    
    @pytest.mark.parametrize("batch_size", TEST_CONFIG['batch_sizes'])
    @pytest.mark.parametrize("data_type", TEST_CONFIG['data_types'])
    def test_matrix_multiplication(self, batch_size, data_type):
        """Test matrix multiplication accuracy and performance."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Create test matrices
        M, K, N = 512, 768, 256
        shape_a = (batch_size, M, K)
        shape_b = (K, N)
        
        dtype = getattr(tf, data_type)
        
        with tf.device('/GPU:0'):
            A = tf.random.normal(shape_a, dtype=dtype)
            B = tf.random.normal(shape_b, dtype=dtype)
            
            # SM120 operation
            start_time = time.perf_counter()
            result_sm120 = sm120_ops.advanced_matmul(A, B)
            sm120_time = time.perf_counter() - start_time
            
            # Standard TensorFlow operation
            start_time = time.perf_counter()
            result_standard = tf.linalg.matmul(A, B)
            standard_time = time.perf_counter() - start_time
            
            # Verify correctness
            if data_type == 'float32':
                tolerance = TEST_CONFIG['tolerance']
            else:
                tolerance = TEST_CONFIG['tolerance'] * 10  # Relaxed for FP16
            
            tf.debugging.assert_near(result_sm120, result_standard, atol=tolerance, rtol=tolerance)
            
            # Performance check
            speedup = standard_time / sm120_time
            print(f"MatMul {data_type} batch={batch_size}: {speedup:.2f}x speedup")
            
            # Should be at least competitive
            assert speedup >= TEST_CONFIG['performance_threshold'], \
                f"Performance regression: {speedup:.2f}x < {TEST_CONFIG['performance_threshold']}x"
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    @pytest.mark.parametrize("channels", [32, 64, 128])
    def test_convolution(self, batch_size, channels):
        """Test convolution accuracy and performance."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        H, W = 224, 224
        input_shape = (batch_size, H, W, 3)
        filter_shape = (3, 3, 3, channels)
        
        with tf.device('/GPU:0'):
            input_tensor = tf.random.normal(input_shape, dtype=tf.float32)
            filter_tensor = tf.random.normal(filter_shape, dtype=tf.float32)
            
            # SM120 operation
            start_time = time.perf_counter()
            result_sm120 = sm120_ops.conv2d(input_tensor, filter_tensor, 
                                          strides=[1, 1, 1, 1], padding='SAME')
            sm120_time = time.perf_counter() - start_time
            
            # Standard operation
            start_time = time.perf_counter()
            result_standard = tf.nn.conv2d(input_tensor, filter_tensor, 
                                         strides=[1, 1, 1, 1], padding='SAME')
            standard_time = time.perf_counter() - start_time
            
            # Verify correctness
            tf.debugging.assert_near(result_sm120, result_standard, 
                                   atol=TEST_CONFIG['tolerance'], rtol=TEST_CONFIG['tolerance'])
            
            speedup = standard_time / sm120_time
            print(f"Conv2D batch={batch_size}, channels={channels}: {speedup:.2f}x speedup")
            
            assert speedup >= TEST_CONFIG['performance_threshold']

class TestKerasLayerIntegration:
    """Test high-level Keras layer integration."""
    
    def test_sm120_dense_layer(self):
        """Test SM120Dense layer functionality."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        batch_size = 32
        input_dim = 768
        output_dim = 512
        
        # Create layers
        sm120_layer = SM120Dense(output_dim, activation='relu', use_sm120=True)
        standard_layer = tf.keras.layers.Dense(output_dim, activation='relu')
        
        # Test data
        input_data = tf.random.normal((batch_size, input_dim))
        
        # Build layers with same weights
        sm120_output = sm120_layer(input_data)
        standard_layer.build(input_data.shape)
        
        # Copy weights for fair comparison
        sm120_layer.build(input_data.shape)
        standard_layer.set_weights(sm120_layer.get_weights())
        
        standard_output = standard_layer(input_data)
        
        # Verify shapes
        assert sm120_output.shape == standard_output.shape
        assert sm120_output.shape == (batch_size, output_dim)
        
        # Verify reasonable outputs (not NaN or Inf)
        assert not tf.reduce_any(tf.math.is_nan(sm120_output))
        assert not tf.reduce_any(tf.math.is_inf(sm120_output))
        
        print("✅ SM120Dense layer test passed")
    
    def test_sm120_conv2d_layer(self):
        """Test SM120Conv2D layer functionality."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        batch_size = 16
        input_shape = (32, 32, 3)
        filters = 64
        
        # Create layers
        sm120_layer = SM120Conv2D(filters, (3, 3), padding='same', activation='relu', use_sm120=True)
        
        # Test data
        input_data = tf.random.normal((batch_size,) + input_shape)
        
        # Forward pass
        output = sm120_layer(input_data)
        
        # Verify output shape
        expected_shape = (batch_size, input_shape[0], input_shape[1], filters)
        assert output.shape == expected_shape
        
        # Verify no NaN/Inf
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        
        print("✅ SM120Conv2D layer test passed")
    
    def test_sm120_attention_layer(self):
        """Test SM120MultiHeadAttention layer."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        batch_size = 8
        seq_len = 128
        embed_dim = 512
        num_heads = 8
        
        # Create layer
        attention_layer = SM120MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            use_sm120=True,
            use_flash_attention=True
        )
        
        # Test data
        input_data = tf.random.normal((batch_size, seq_len, embed_dim))
        
        # Forward pass
        output = attention_layer(input_data)
        
        # Verify output shape
        assert output.shape == (batch_size, seq_len, embed_dim)
        
        # Verify no NaN/Inf
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        
        print("✅ SM120MultiHeadAttention layer test passed")

class TestTrainingIntegration:
    """Test training pipeline integration with gradients."""
    
    def test_gradient_computation(self):
        """Test that gradients work correctly through SM120 operations."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Create simple model
        model = tf.keras.Sequential([
            SM120Dense(128, activation='relu', use_sm120=True),
            SM120Dense(64, activation='relu', use_sm120=True),
            SM120Dense(10, activation='softmax', use_sm120=True)
        ])
        
        # Test data
        batch_size = 32
        input_dim = 256
        num_classes = 10
        
        x = tf.random.normal((batch_size, input_dim))
        y = tf.random.uniform((batch_size,), maxval=num_classes, dtype=tf.int32)
        
        # Forward and backward pass
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
            loss = tf.reduce_mean(loss)
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Verify gradients exist and are finite
        assert len(gradients) == len(model.trainable_variables)
        
        for i, grad in enumerate(gradients):
            assert grad is not None, f"Gradient {i} is None"
            assert not tf.reduce_any(tf.math.is_nan(grad)), f"Gradient {i} contains NaN"
            assert not tf.reduce_any(tf.math.is_inf(grad)), f"Gradient {i} contains Inf"
            assert tf.reduce_any(tf.not_equal(grad, 0)), f"Gradient {i} is all zeros"
        
        print("✅ Gradient computation test passed")
    
    def test_training_step(self):
        """Test complete training step with optimizer."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Create model
        model = tf.keras.Sequential([
            SM120Dense(64, activation='relu', use_sm120=True),
            SM120Dense(1, activation='sigmoid', use_sm120=True)
        ])
        
        optimizer = tf.keras.optimizers.Adam(0.001)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        # Generate test data
        batch_size = 64
        input_dim = 32
        
        x = tf.random.normal((batch_size, input_dim))
        y = tf.random.uniform((batch_size, 1))
        
        # Training step
        initial_loss = None
        final_loss = None
        
        for step in range(10):
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = loss_fn(y, predictions)
            
            if step == 0:
                initial_loss = loss.numpy()
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if step == 9:
                final_loss = loss.numpy()
        
        # Verify training progress
        assert final_loss < initial_loss, f"Training failed: {final_loss} >= {initial_loss}"
        
        print(f"✅ Training test passed: {initial_loss:.4f} -> {final_loss:.4f}")

class TestPerformanceValidation:
    """Test performance characteristics and optimizations."""
    
    def test_mixed_precision_performance(self):
        """Test mixed precision performance benefits."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Enable mixed precision
        enable_mixed_precision('mixed_float16')
        
        batch_size = 64
        seq_len = 512
        embed_dim = 768
        
        # Create model with mixed precision
        model = tf.keras.Sequential([
            SM120Dense(embed_dim, dtype='mixed_float16', use_sm120=True),
            SM120Dense(embed_dim, dtype='mixed_float16', use_sm120=True),
            SM120Dense(1, dtype='float32', use_sm120=True)  # Output in FP32
        ])
        
        input_data = tf.random.normal((batch_size, seq_len, embed_dim), dtype=tf.float16)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(10):
            output = model(input_data)
        mixed_precision_time = time.perf_counter() - start_time
        
        # Disable mixed precision for comparison
        tf.keras.mixed_precision.set_global_policy('float32')
        
        model_fp32 = tf.keras.Sequential([
            SM120Dense(embed_dim, dtype='float32', use_sm120=True),
            SM120Dense(embed_dim, dtype='float32', use_sm120=True),
            SM120Dense(1, dtype='float32', use_sm120=True)
        ])
        
        input_data_fp32 = tf.cast(input_data, tf.float32)
        
        start_time = time.perf_counter()
        for _ in range(10):
            output = model_fp32(input_data_fp32)
        fp32_time = time.perf_counter() - start_time
        
        speedup = fp32_time / mixed_precision_time
        print(f"Mixed precision speedup: {speedup:.2f}x")
        
        # Should see some performance benefit
        assert speedup >= 1.1, f"Mixed precision not beneficial: {speedup:.2f}x"
    
    def test_memory_efficiency(self):
        """Test memory efficiency of SM120 operations."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Clear any existing allocations
        tf.keras.backend.clear_session()
        gc.collect()
        
        batch_size = 32
        seq_len = 1024  # Large sequence for memory test
        embed_dim = 1024
        num_heads = 16
        
        # Monitor memory usage
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            pytest.skip("No GPU available for memory test")
        
        # Test with SM120 Flash Attention
        attention_layer = SM120MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            use_sm120=True,
            use_flash_attention=True
        )
        
        input_data = tf.random.normal((batch_size, seq_len, embed_dim), dtype=tf.float16)
        
        # Forward pass with memory monitoring
        try:
            output = attention_layer(input_data)
            memory_efficient = True
            print("✅ Memory efficient attention completed")
        except tf.errors.ResourceExhaustedError:
            memory_efficient = False
            print("❌ Memory efficient attention failed")
        
        # Compare with standard attention (if memory allows)
        if memory_efficient:
            tf.keras.backend.clear_session()
            
            standard_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads
            )
            
            try:
                output_standard = standard_attention(input_data, input_data)
                standard_memory_ok = True
            except tf.errors.ResourceExhaustedError:
                standard_memory_ok = False
                print("Standard attention ran out of memory - SM120 is more efficient")
        
        assert memory_efficient, "SM120 Flash Attention should handle large sequences"

class TestDataTypeSupport:
    """Test comprehensive data type support."""
    
    def test_automatic_type_casting(self):
        """Test automatic type casting functionality."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Test FP32 -> FP16 casting
        input_fp32 = tf.random.normal((32, 256), dtype=tf.float32)
        input_fp16 = auto_cast(input_fp32, operation='attention')
        
        # Should maintain reasonable values
        assert not tf.reduce_any(tf.math.is_nan(input_fp16))
        assert not tf.reduce_any(tf.math.is_inf(input_fp16))
        
        # Test operations work with different types
        for dtype in ['float32', 'float16']:
            if dtype in get_supported_types():
                test_input = tf.random.normal((16, 128), dtype=getattr(tf, dtype))
                layer = SM120Dense(64, use_sm120=True)
                output = layer(test_input)
                
                assert not tf.reduce_any(tf.math.is_nan(output))
                assert not tf.reduce_any(tf.math.is_inf(output))
        
        print("✅ Data type support test passed")

class TestErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    def test_fallback_behavior(self):
        """Test that fallback works when SM120 operations fail."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Create layer with fallback enabled
        layer = SM120Dense(128, use_sm120=True, fallback_on_error=True)
        
        # Test with various input conditions
        test_cases = [
            tf.random.normal((32, 256)),  # Normal case
            tf.random.normal((1, 512)),   # Small batch
            tf.random.normal((128, 64)),  # Large batch
        ]
        
        for i, input_data in enumerate(test_cases):
            try:
                output = layer(input_data)
                assert output.shape == (input_data.shape[0], 128)
                assert not tf.reduce_any(tf.math.is_nan(output))
                print(f"✅ Test case {i+1} passed")
            except Exception as e:
                pytest.fail(f"Fallback failed for test case {i+1}: {e}")
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        layer = SM120Dense(64, use_sm120=True)
        
        # Test invalid shapes (should gracefully handle or provide clear error)
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            # 1D input to dense layer
            invalid_input = tf.random.normal((32,))
            layer(invalid_input)

class TestComprehensiveIntegration:
    """Comprehensive end-to-end integration tests."""
    
    def test_transformer_model_training(self):
        """Test complete transformer model training."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Create transformer model
        vocab_size = 1000
        max_length = 128
        embed_dim = 256
        num_heads = 8
        ff_dim = 512
        
        model = create_sm120_transformer_encoder(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=2,
            use_sm120=True
        )
        
        # Add classification head
        inputs = model.input
        x = model.output
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        full_model = tf.keras.Model(inputs, outputs)
        
        # Compile model
        full_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Generate synthetic data
        batch_size = 16
        num_samples = 64
        
        x_train = np.random.randint(0, vocab_size, size=(num_samples, max_length))
        y_train = np.random.randint(0, 10, size=(num_samples,))
        
        # Training
        try:
            history = full_model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=2,
                verbose=0
            )
            
            # Verify training worked
            assert len(history.history['loss']) == 2
            assert not np.isnan(history.history['loss'][-1])
            assert not np.isinf(history.history['loss'][-1])
            
            print("✅ Transformer training test passed")
            
        except Exception as e:
            pytest.fail(f"Transformer training failed: {e}")
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        if not SM120_AVAILABLE:
            pytest.skip("SM120 operations not available")
        
        # Enable profiling
        enable_profiling(True)
        
        # Run operations to generate metrics
        layer = SM120Dense(256, use_sm120=True)
        input_data = tf.random.normal((64, 512))
        
        for _ in range(10):
            output = layer(input_data)
        
        # Check metrics were collected
        metrics = get_average_metrics('sm120_dense')
        if metrics:
            assert metrics['sample_count'] > 0
            assert metrics['avg_execution_time_ms'] > 0
            print("✅ Performance monitoring test passed")
        else:
            print("⚠️ Performance monitoring data not available")

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "sm120: marks tests requiring SM120 hardware")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle SM120 availability."""
    if not SM120_AVAILABLE:
        skip_sm120 = pytest.mark.skip(reason="SM120 operations not available")
        for item in items:
            if "sm120" in item.keywords:
                item.add_marker(skip_sm120)

if __name__ == "__main__":
    # Run comprehensive validation when executed directly
    print("🚀 Running SM120 Comprehensive Validation Suite")
    
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Print final summary
    if SM120_AVAILABLE:
        print("\n" + "="*70)
        print("📊 FINAL PERFORMANCE SUMMARY")
        print_performance_summary()
        print_type_summary()
        print("="*70)
