"""
TensorFlow sm_120 Operations Python API

This module provides Python bindings for TensorFlow operations optimized 
for RTX 50-series GPUs with compute capability 12.0.

Features:
- Advanced matrix multiplication with Tensor Core acceleration
- High-performance convolution operations
- Flash Attention for transformer models
- Memory-optimized operations
- Performance profiling and monitoring
"""

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader
import os
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings

# Load the sm_120 operations library
try:
    _sm120_ops_so = tf.load_op_library(
        resource_loader.get_path_to_datafile("_sm120_ops.so")
    )
    _SM120_AVAILABLE = True
except (tf.errors.NotFoundError, OSError) as e:
    warnings.warn(f"SM120 operations library not found: {e}. "
                 f"Falling back to standard TensorFlow operations.")
    _SM120_AVAILABLE = False

def is_sm120_available() -> bool:
    """Check if SM120 operations are available."""
    if not _SM120_AVAILABLE:
        return False
    
    try:
        # Check if we have a compatible GPU
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return False
        
        for gpu in gpus:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                compute_cap = details.get('compute_capability', (0, 0))
                if isinstance(compute_cap, tuple) and compute_cap >= (12, 0):
                    return True
            except Exception:
                continue
        
        return False
    except Exception:
        return False

def get_sm120_device_info() -> Dict[str, Any]:
    """Get detailed information about SM120 compatible devices."""
    info = {
        'available': False,
        'devices': [],
        'library_loaded': _SM120_AVAILABLE
    }
    
    if not _SM120_AVAILABLE:
        return info
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                compute_cap = details.get('compute_capability', (0, 0))
                
                device_info = {
                    'index': i,
                    'name': gpu.name,
                    'compute_capability': compute_cap,
                    'sm120_compatible': isinstance(compute_cap, tuple) and compute_cap >= (12, 0),
                    'details': details
                }
                
                info['devices'].append(device_info)
                
                if device_info['sm120_compatible']:
                    info['available'] = True
                    
            except Exception as e:
                info['devices'].append({
                    'index': i,
                    'name': gpu.name,
                    'error': str(e)
                })
    except Exception:
        pass
    
    return info

class SM120Config:
    """Configuration class for SM120 operations."""
    
    def __init__(self):
        self.optimization_level = 1  # 0=basic, 1=advanced, 2=aggressive
        self.use_tensor_cores = True
        self.enable_profiling = False
        self.memory_optimization = True
        self.fallback_to_standard = True
        
    def set_optimization_level(self, level: int) -> None:
        """Set optimization level (0=basic, 1=advanced, 2=aggressive)."""
        if level not in [0, 1, 2]:
            raise ValueError("Optimization level must be 0, 1, or 2")
        self.optimization_level = level
        
    def enable_tensor_cores(self, enable: bool = True) -> None:
        """Enable or disable Tensor Core usage."""
        self.use_tensor_cores = enable
        
    def enable_memory_optimization(self, enable: bool = True) -> None:
        """Enable or disable memory optimizations."""
        self.memory_optimization = enable

# Global configuration instance
_config = SM120Config()

def get_config() -> SM120Config:
    """Get the global SM120 configuration."""
    return _config

def advanced_matmul(
    a: tf.Tensor,
    b: tf.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    use_tensor_cores: Optional[bool] = None,
    optimization_level: Optional[int] = None,
    name: Optional[str] = None
) -> tf.Tensor:
    """
    Advanced matrix multiplication optimized for RTX 50-series GPUs.
    
    This operation leverages sm_120 specific optimizations including:
    - 5th generation Tensor Cores for mixed precision
    - Advanced memory coalescing patterns
    - Optimal shared memory utilization
    - Multi-level tiling strategies
    
    Args:
        a: A 2-D tensor of shape [M, K] (or [K, M] if transpose_a=True).
        b: A 2-D tensor of shape [K, N] (or [N, K] if transpose_b=True).
        transpose_a: If True, a is transposed before multiplication.
        transpose_b: If True, b is transposed before multiplication.
        use_tensor_cores: Whether to use Tensor Cores. Defaults to config setting.
        optimization_level: Optimization level (0-2). Defaults to config setting.
        name: Optional name for the operation.
        
    Returns:
        A tensor of shape [M, N] containing the matrix product.
        
    Raises:
        InvalidArgumentError: If tensor shapes are incompatible.
        UnimplementedError: If SM120 operations are not available.
    """
    with tf.name_scope(name or "sm120_advanced_matmul"):
        # Validate inputs
        if a.dtype != b.dtype:
            raise ValueError(f"Input tensors must have the same dtype, got {a.dtype} and {b.dtype}")
        
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("Input tensors must be 2-dimensional")
        
        # Use config defaults if not specified
        if use_tensor_cores is None:
            use_tensor_cores = _config.use_tensor_cores
        if optimization_level is None:
            optimization_level = _config.optimization_level
        
        # Check if SM120 operations are available
        if _SM120_AVAILABLE and is_sm120_available():
            try:
                return _sm120_ops_so.sm120_advanced_mat_mul(
                    a=a, b=b,
                    transpose_a=transpose_a,
                    transpose_b=transpose_b,
                    use_tensor_cores=use_tensor_cores,
                    optimization_level=optimization_level
                )
            except Exception as e:
                if _config.fallback_to_standard:
                    warnings.warn(f"SM120 matmul failed, falling back to standard: {e}")
                else:
                    raise
        
        # Fallback to standard TensorFlow operations
        if _config.fallback_to_standard:
            return tf.linalg.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
        else:
            raise tf.errors.UnimplementedError(
                None, None, "SM120 operations not available and fallback disabled")

def advanced_conv2d(
    input: tf.Tensor,
    filters: tf.Tensor,
    strides: Union[int, List[int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, List[int]]] = None,
    use_tensor_cores: Optional[bool] = None,
    optimization_level: Optional[int] = None,
    name: Optional[str] = None
) -> tf.Tensor:
    """
    Advanced 2D convolution optimized for RTX 50-series GPUs.
    
    Features sm_120 specific optimizations:
    - Tensor Core acceleration for supported precisions
    - Advanced memory coalescing and tiling
    - Optimized shared memory utilization
    - Multi-algorithm selection based on problem size
    
    Args:
        input: 4-D input tensor.
        filters: 4-D convolution filter.
        strides: Stride values for each spatial dimension.
        padding: Padding algorithm ("SAME" or "VALID").
        data_format: Data layout ("NHWC" or "NCHW").
        dilations: Dilation rates for each spatial dimension.
        use_tensor_cores: Enable Tensor Core acceleration.
        optimization_level: Optimization level (0-2).
        name: Optional name for the operation.
        
    Returns:
        Convolution result tensor.
    """
    with tf.name_scope(name or "sm120_advanced_conv2d"):
        # Normalize strides and dilations
        if isinstance(strides, int):
            strides = [1, strides, strides, 1]
        elif len(strides) == 2:
            strides = [1] + strides + [1]
        
        if dilations is None:
            dilations = [1, 1, 1, 1]
        elif isinstance(dilations, int):
            dilations = [1, dilations, dilations, 1]
        elif len(dilations) == 2:
            dilations = [1] + dilations + [1]
        
        # Use config defaults if not specified
        if use_tensor_cores is None:
            use_tensor_cores = _config.use_tensor_cores
        if optimization_level is None:
            optimization_level = _config.optimization_level
        
        # Validate inputs
        if len(input.shape) != 4:
            raise ValueError("Input must be 4-dimensional")
        if len(filters.shape) != 4:
            raise ValueError("Filters must be 4-dimensional")
        
        # Check if SM120 operations are available
        if _SM120_AVAILABLE and is_sm120_available():
            try:
                return _sm120_ops_so.sm120_advanced_conv2d(
                    input=input,
                    filter=filters,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilations=dilations,
                    use_tensor_cores=use_tensor_cores,
                    optimization_level=optimization_level
                )
            except Exception as e:
                if _config.fallback_to_standard:
                    warnings.warn(f"SM120 conv2d failed, falling back to standard: {e}")
                else:
                    raise
        
        # Fallback to standard TensorFlow operations
        if _config.fallback_to_standard:
            return tf.nn.conv2d(
                input=input,
                filters=filters,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=dilations
            )
        else:
            raise tf.errors.UnimplementedError(
                None, None, "SM120 operations not available and fallback disabled")

def flash_attention(
    queries: tf.Tensor,
    keys: tf.Tensor,
    values: tf.Tensor,
    attention_mask: Optional[tf.Tensor] = None,
    scale: Optional[float] = None,
    causal_mask: bool = False,
    dropout_rate: float = 0.0,
    name: Optional[str] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Flash Attention implementation optimized for RTX 50-series GPUs.
    
    This operation implements memory-efficient attention computation using:
    - Tiled computation to fit in shared memory
    - Online softmax computation
    - Reduced memory bandwidth requirements
    - sm_120 specific memory access patterns
    
    Args:
        queries: Query tensor [batch, heads, seq_len, head_dim].
        keys: Key tensor [batch, heads, seq_len, head_dim].
        values: Value tensor [batch, heads, seq_len, head_dim].
        attention_mask: Optional attention mask tensor.
        scale: Scaling factor for attention scores. Defaults to 1/sqrt(head_dim).
        causal_mask: Whether to apply causal masking.
        dropout_rate: Dropout rate for attention weights.
        name: Optional name for the operation.
        
    Returns:
        Tuple of (attention_output, attention_weights).
    """
    with tf.name_scope(name or "sm120_flash_attention"):
        # Validate input shapes
        if len(queries.shape) != 4 or len(keys.shape) != 4 or len(values.shape) != 4:
            raise ValueError("Queries, keys, and values must be 4-dimensional")
        
        batch_size, num_heads, seq_len, head_dim = queries.shape
        
        # Default scale
        if scale is None:
            scale = 1.0 / np.sqrt(float(head_dim))
        
        # Create dummy attention mask if not provided
        if attention_mask is None:
            attention_mask = tf.zeros([0], dtype=queries.dtype)
        
        # Check if SM120 operations are available
        if _SM120_AVAILABLE and is_sm120_available():
            try:
                return _sm120_ops_so.sm120_flash_attention(
                    queries=queries,
                    keys=keys,
                    values=values,
                    attention_mask=attention_mask,
                    scale=scale,
                    causal_mask=causal_mask,
                    dropout_rate=dropout_rate
                )
            except Exception as e:
                if _config.fallback_to_standard:
                    warnings.warn(f"SM120 flash attention failed, falling back to standard: {e}")
                else:
                    raise
        
        # Fallback to standard attention implementation
        if _config.fallback_to_standard:
            # Standard scaled dot-product attention
            scores = tf.matmul(queries, keys, transpose_b=True) * scale
            
            if causal_mask:
                # Apply causal mask
                mask_value = tf.constant(-1e9, dtype=scores.dtype)
                causal_mask_matrix = tf.linalg.band_part(
                    tf.ones([seq_len, seq_len], dtype=scores.dtype), -1, 0)
                causal_mask_matrix = tf.where(
                    tf.equal(causal_mask_matrix, 0), mask_value, 0.0)
                scores += causal_mask_matrix
            
            if attention_mask is not None and tf.size(attention_mask) > 0:
                scores += attention_mask
            
            attention_weights = tf.nn.softmax(scores, axis=-1)
            
            if dropout_rate > 0.0:
                attention_weights = tf.nn.dropout(attention_weights, dropout_rate)
            
            attention_output = tf.matmul(attention_weights, values)
            
            return attention_output, attention_weights
        else:
            raise tf.errors.UnimplementedError(
                None, None, "SM120 operations not available and fallback disabled")

class SM120Profiler:
    """Profiler for SM120 operations."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = {}
        
    def profile_operation(self, operation_name: str, operation_fn, *args, **kwargs):
        """Profile a SM120 operation and collect metrics."""
        if not self.enabled:
            return operation_fn(*args, **kwargs)
        
        start_time = tf.timestamp()
        result = operation_fn(*args, **kwargs)
        end_time = tf.timestamp()
        
        # Store timing information
        self.metrics[operation_name] = {
            'execution_time': end_time - start_time,
            'timestamp': start_time
        }
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected profiling metrics."""
        return self.metrics.copy()
    
    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.metrics.clear()

def benchmark_operation(
    operation_fn,
    *args,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark a SM120 operation.
    
    Args:
        operation_fn: Function to benchmark.
        *args: Arguments to pass to the function.
        num_iterations: Number of benchmark iterations.
        warmup_iterations: Number of warmup iterations.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Dictionary containing benchmark results.
    """
    # Warmup
    for _ in range(warmup_iterations):
        _ = operation_fn(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start_time = tf.timestamp()
        result = operation_fn(*args, **kwargs)
        end_time = tf.timestamp()
        times.append(end_time - start_time)
    
    times = tf.stack(times)
    
    return {
        'mean_time': tf.reduce_mean(times).numpy(),
        'std_time': tf.math.reduce_std(times).numpy(),
        'min_time': tf.reduce_min(times).numpy(),
        'max_time': tf.reduce_max(times).numpy(),
        'median_time': tf.nn.compute_average_loss(tf.nn.top_k(times, k=num_iterations//2).values).numpy()
    }

# Convenience functions for common operations
def optimized_dense(
    inputs: tf.Tensor,
    units: int,
    use_bias: bool = True,
    activation: Optional[str] = None,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    name: Optional[str] = None
) -> tf.Tensor:
    """
    Dense layer optimized with SM120 matrix multiplication.
    
    Args:
        inputs: Input tensor.
        units: Number of output units.
        use_bias: Whether to use bias.
        activation: Activation function name.
        kernel_initializer: Kernel initializer.
        bias_initializer: Bias initializer.
        name: Optional layer name.
        
    Returns:
        Output tensor.
    """
    with tf.name_scope(name or "sm120_dense"):
        input_shape = inputs.shape
        if len(input_shape) < 2:
            raise ValueError("Input must have at least 2 dimensions")
        
        # Flatten input if needed
        if len(input_shape) > 2:
            inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        
        # Create kernel
        kernel_shape = [input_shape[-1], units]
        kernel = tf.Variable(
            tf.keras.initializers.get(kernel_initializer)(kernel_shape),
            name="kernel"
        )
        
        # Optimized matrix multiplication
        outputs = advanced_matmul(inputs, kernel)
        
        # Add bias if requested
        if use_bias:
            bias = tf.Variable(
                tf.keras.initializers.get(bias_initializer)([units]),
                name="bias"
            )
            outputs = tf.nn.bias_add(outputs, bias)
        
        # Apply activation
        if activation:
            outputs = tf.keras.activations.get(activation)(outputs)
        
        # Reshape output to match input batch dimensions
        if len(input_shape) > 2:
            output_shape = input_shape[:-1].as_list() + [units]
            outputs = tf.reshape(outputs, output_shape)
        
        return outputs

# Export public API
__all__ = [
    'is_sm120_available',
    'get_sm120_device_info',
    'SM120Config',
    'get_config',
    'advanced_matmul',
    'advanced_conv2d', 
    'flash_attention',
    'SM120Profiler',
    'benchmark_operation',
    'optimized_dense'
]
