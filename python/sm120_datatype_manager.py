"""
SM120 Data Type Manager - Comprehensive Type Support and Automatic Casting
Handles FP32, FP16, BF16, FP8, FP4, and double precision with automatic optimization
Copyright 2024 - TensorFlow SM120 Optimization Project
"""

import tensorflow as tf
import numpy as np
from typing import Union, Optional, Dict, List, Tuple, Any
from enum import Enum
import warnings

try:
    import sm120_ops
    SM120_AVAILABLE = True
except ImportError:
    SM120_AVAILABLE = False

class SM120DataType(Enum):
    """Supported data types in SM120 operations."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    DOUBLE = "float64"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"
    INT32 = "int32"
    INT64 = "int64"

class SM120TypeProperties:
    """Properties and capabilities of SM120 data types."""
    
    TYPE_INFO = {
        SM120DataType.FLOAT32: {
            'size_bytes': 4,
            'tensor_core_support': True,
            'precision': 'high',
            'memory_efficiency': 1.0,
            'compute_efficiency': 1.0,
            'tf_dtype': tf.float32,
            'numpy_dtype': np.float32,
            'available_sm': (7, 0),  # Available from compute capability 7.0+
            'mixed_precision_primary': False
        },
        SM120DataType.FLOAT16: {
            'size_bytes': 2,
            'tensor_core_support': True,
            'precision': 'medium',
            'memory_efficiency': 2.0,
            'compute_efficiency': 1.8,
            'tf_dtype': tf.float16,
            'numpy_dtype': np.float16,
            'available_sm': (7, 0),
            'mixed_precision_primary': True
        },
        SM120DataType.BFLOAT16: {
            'size_bytes': 2,
            'tensor_core_support': True,
            'precision': 'medium',
            'memory_efficiency': 2.0,
            'compute_efficiency': 1.7,
            'tf_dtype': tf.bfloat16,
            'numpy_dtype': None,  # No direct numpy support
            'available_sm': (8, 0),  # Available from Ampere
            'mixed_precision_primary': True
        },
        SM120DataType.DOUBLE: {
            'size_bytes': 8,
            'tensor_core_support': False,
            'precision': 'very_high',
            'memory_efficiency': 0.5,
            'compute_efficiency': 0.3,
            'tf_dtype': tf.float64,
            'numpy_dtype': np.float64,
            'available_sm': (1, 0),  # Available on all GPUs
            'mixed_precision_primary': False
        },
        SM120DataType.FP8_E4M3: {
            'size_bytes': 1,
            'tensor_core_support': True,
            'precision': 'low',
            'memory_efficiency': 4.0,
            'compute_efficiency': 3.5,
            'tf_dtype': None,  # Custom type
            'numpy_dtype': None,
            'available_sm': (12, 0),  # Available from Blackwell/future
            'mixed_precision_primary': False
        },
        SM120DataType.FP8_E5M2: {
            'size_bytes': 1,
            'tensor_core_support': True,
            'precision': 'low',
            'memory_efficiency': 4.0,
            'compute_efficiency': 3.0,
            'tf_dtype': None,  # Custom type
            'numpy_dtype': None,
            'available_sm': (12, 0),
            'mixed_precision_primary': False
        },
        SM120DataType.FP4: {
            'size_bytes': 0.5,
            'tensor_core_support': True,
            'precision': 'very_low',
            'memory_efficiency': 8.0,
            'compute_efficiency': 4.0,
            'tf_dtype': None,  # Experimental
            'numpy_dtype': None,
            'available_sm': (12, 0),  # Future support
            'mixed_precision_primary': False
        },
        SM120DataType.INT32: {
            'size_bytes': 4,
            'tensor_core_support': False,
            'precision': 'exact',
            'memory_efficiency': 1.0,
            'compute_efficiency': 0.8,
            'tf_dtype': tf.int32,
            'numpy_dtype': np.int32,
            'available_sm': (1, 0),
            'mixed_precision_primary': False
        },
        SM120DataType.INT64: {
            'size_bytes': 8,
            'tensor_core_support': False,
            'precision': 'exact',
            'memory_efficiency': 0.5,
            'compute_efficiency': 0.4,
            'tf_dtype': tf.int64,
            'numpy_dtype': np.int64,
            'available_sm': (1, 0),
            'mixed_precision_primary': False
        }
    }

class SM120TypeManager:
    """Manager for SM120 data type operations and automatic optimization."""
    
    def __init__(self):
        self._gpu_compute_capability: Optional[Tuple[int, int]] = None
        self._supported_types: List[SM120DataType] = []
        self._optimal_types: Dict[str, SM120DataType] = {}
        self._type_conversion_cache: Dict[Tuple, Any] = {}
        self._mixed_precision_enabled = False
        self._initialize_gpu_capabilities()
    
    def _initialize_gpu_capabilities(self):
        """Initialize GPU capabilities and supported data types."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                device_details = tf.config.experimental.get_device_details(gpus[0])
                self._gpu_compute_capability = device_details.get('compute_capability', (7, 0))
            else:
                self._gpu_compute_capability = (7, 0)  # Default assumption
        except Exception:
            self._gpu_compute_capability = (7, 0)
        
        # Determine supported types based on compute capability
        for dtype, info in SM120TypeProperties.TYPE_INFO.items():
            required_sm = info['available_sm']
            if (self._gpu_compute_capability[0] > required_sm[0] or 
                (self._gpu_compute_capability[0] == required_sm[0] and 
                 self._gpu_compute_capability[1] >= required_sm[1])):
                self._supported_types.append(dtype)
        
        # Set optimal types for different operations
        self._determine_optimal_types()
    
    def _determine_optimal_types(self):
        """Determine optimal data types for different operation categories."""
        # Matrix multiplication - prioritize Tensor Core types
        if SM120DataType.BFLOAT16 in self._supported_types:
            self._optimal_types['matmul'] = SM120DataType.BFLOAT16
        elif SM120DataType.FLOAT16 in self._supported_types:
            self._optimal_types['matmul'] = SM120DataType.FLOAT16
        else:
            self._optimal_types['matmul'] = SM120DataType.FLOAT32
        
        # Convolution - similar to matmul
        self._optimal_types['conv'] = self._optimal_types['matmul']
        
        # Attention - memory bandwidth critical
        if SM120DataType.FP8_E4M3 in self._supported_types:
            self._optimal_types['attention'] = SM120DataType.FP8_E4M3
        elif SM120DataType.FLOAT16 in self._supported_types:
            self._optimal_types['attention'] = SM120DataType.FLOAT16
        else:
            self._optimal_types['attention'] = SM120DataType.FLOAT32
        
        # Normalization - precision important
        self._optimal_types['normalization'] = SM120DataType.FLOAT32
        
        # Activations - can use lower precision
        if SM120DataType.FLOAT16 in self._supported_types:
            self._optimal_types['activation'] = SM120DataType.FLOAT16
        else:
            self._optimal_types['activation'] = SM120DataType.FLOAT32
    
    def get_supported_types(self) -> List[SM120DataType]:
        """Get list of supported data types on current GPU."""
        return self._supported_types.copy()
    
    def is_type_supported(self, dtype: Union[SM120DataType, str, tf.DType]) -> bool:
        """Check if a data type is supported."""
        sm120_dtype = self._parse_dtype(dtype)
        return sm120_dtype in self._supported_types
    
    def get_optimal_type(self, operation_category: str) -> SM120DataType:
        """Get optimal data type for an operation category."""
        return self._optimal_types.get(operation_category, SM120DataType.FLOAT32)
    
    def get_type_properties(self, dtype: Union[SM120DataType, str, tf.DType]) -> Dict[str, Any]:
        """Get properties of a data type."""
        sm120_dtype = self._parse_dtype(dtype)
        return SM120TypeProperties.TYPE_INFO.get(sm120_dtype, {}).copy()
    
    def _parse_dtype(self, dtype: Union[SM120DataType, str, tf.DType]) -> SM120DataType:
        """Parse various dtype representations to SM120DataType."""
        if isinstance(dtype, SM120DataType):
            return dtype
        
        if isinstance(dtype, str):
            dtype_str = dtype.lower()
            for sm120_dtype in SM120DataType:
                if sm120_dtype.value == dtype_str:
                    return sm120_dtype
            
            # Handle aliases
            alias_map = {
                'half': SM120DataType.FLOAT16,
                'float': SM120DataType.FLOAT32,
                'double': SM120DataType.DOUBLE,
                'bf16': SM120DataType.BFLOAT16,
                'fp16': SM120DataType.FLOAT16,
                'fp32': SM120DataType.FLOAT32,
                'fp64': SM120DataType.DOUBLE
            }
            if dtype_str in alias_map:
                return alias_map[dtype_str]
        
        if hasattr(tf, 'DType') and isinstance(dtype, tf.DType):
            tf_to_sm120 = {
                tf.float32: SM120DataType.FLOAT32,
                tf.float16: SM120DataType.FLOAT16,
                tf.bfloat16: SM120DataType.BFLOAT16,
                tf.float64: SM120DataType.DOUBLE,
                tf.int32: SM120DataType.INT32,
                tf.int64: SM120DataType.INT64
            }
            if dtype in tf_to_sm120:
                return tf_to_sm120[dtype]
        
        # Default fallback
        return SM120DataType.FLOAT32
    
    def auto_cast_tensor(self, tensor: tf.Tensor, target_operation: str = 'general',
                        preserve_precision: bool = False) -> tf.Tensor:
        """Automatically cast tensor to optimal data type."""
        if not SM120_AVAILABLE:
            return tensor
        
        current_dtype = self._parse_dtype(tensor.dtype)
        
        if preserve_precision:
            # Maintain or increase precision
            if current_dtype == SM120DataType.DOUBLE:
                return tensor  # Already highest precision
            elif current_dtype in [SM120DataType.FP8_E4M3, SM120DataType.FP8_E5M2, SM120DataType.FP4]:
                # Promote low precision to higher precision
                optimal_dtype = self.get_optimal_type(target_operation)
                return self._cast_tensor(tensor, optimal_dtype)
        else:
            # Optimize for performance
            optimal_dtype = self.get_optimal_type(target_operation)
            if optimal_dtype != current_dtype and self.is_type_supported(optimal_dtype):
                return self._cast_tensor(tensor, optimal_dtype)
        
        return tensor
    
    def _cast_tensor(self, tensor: tf.Tensor, target_dtype: SM120DataType) -> tf.Tensor:
        """Cast tensor to target data type with proper handling."""
        target_properties = self.get_type_properties(target_dtype)
        tf_dtype = target_properties.get('tf_dtype')
        
        if tf_dtype is None:
            # Custom type - use SM120 conversion
            if SM120_AVAILABLE and hasattr(sm120_ops, 'cast_to_custom_type'):
                return sm120_ops.cast_to_custom_type(tensor, target_dtype.value)
            else:
                warnings.warn(f"Cannot cast to {target_dtype.value}, using original type")
                return tensor
        
        # Use TensorFlow casting
        cache_key = (id(tensor), target_dtype)
        if cache_key in self._type_conversion_cache:
            return self._type_conversion_cache[cache_key]
        
        result = tf.cast(tensor, tf_dtype)
        
        # Cache small tensors
        if tensor.shape.num_elements() < 10000:
            self._type_conversion_cache[cache_key] = result
        
        return result
    
    def enable_mixed_precision(self, policy: str = 'mixed_float16'):
        """Enable automatic mixed precision with SM120 optimizations."""
        self._mixed_precision_enabled = True
        
        # Map policy to SM120 optimal types
        if policy == 'mixed_float16' and self.is_type_supported(SM120DataType.FLOAT16):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            self._optimal_types['default'] = SM120DataType.FLOAT16
        elif policy == 'mixed_bfloat16' and self.is_type_supported(SM120DataType.BFLOAT16):
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
            self._optimal_types['default'] = SM120DataType.BFLOAT16
        elif policy == 'float32':
            tf.keras.mixed_precision.set_global_policy('float32')
            self._optimal_types['default'] = SM120DataType.FLOAT32
        else:
            warnings.warn(f"Mixed precision policy '{policy}' not supported, using float32")
            tf.keras.mixed_precision.set_global_policy('float32')
    
    def disable_mixed_precision(self):
        """Disable mixed precision."""
        self._mixed_precision_enabled = False
        tf.keras.mixed_precision.set_global_policy('float32')
    
    def get_memory_usage_estimate(self, tensor_shape: Tuple[int, ...], 
                                dtype: Union[SM120DataType, str, tf.DType]) -> int:
        """Estimate memory usage in bytes for a tensor with given shape and dtype."""
        sm120_dtype = self._parse_dtype(dtype)
        properties = self.get_type_properties(sm120_dtype)
        size_bytes = properties.get('size_bytes', 4)
        
        total_elements = np.prod(tensor_shape)
        return int(total_elements * size_bytes)
    
    def recommend_dtype_for_model(self, model: tf.keras.Model, 
                                 target_memory_gb: Optional[float] = None,
                                 prioritize_speed: bool = True) -> Dict[str, str]:
        """Recommend data types for model layers based on constraints."""
        recommendations = {}
        
        total_memory_usage = 0
        
        for layer in model.layers:
            layer_name = layer.name
            
            # Determine layer category
            if 'dense' in layer_name.lower() or 'matmul' in layer_name.lower():
                category = 'matmul'
            elif 'conv' in layer_name.lower():
                category = 'conv'
            elif 'attention' in layer_name.lower():
                category = 'attention'
            elif 'norm' in layer_name.lower():
                category = 'normalization'
            else:
                category = 'general'
            
            if prioritize_speed:
                recommended_dtype = self.get_optimal_type(category)
            else:
                # Prioritize memory efficiency
                if self.is_type_supported(SM120DataType.FP8_E4M3):
                    recommended_dtype = SM120DataType.FP8_E4M3
                elif self.is_type_supported(SM120DataType.FLOAT16):
                    recommended_dtype = SM120DataType.FLOAT16
                else:
                    recommended_dtype = SM120DataType.FLOAT32
            
            recommendations[layer_name] = recommended_dtype.value
            
            # Estimate memory usage
            if hasattr(layer, 'count_params'):
                params = layer.count_params()
                layer_memory = params * self.get_type_properties(recommended_dtype)['size_bytes']
                total_memory_usage += layer_memory
        
        # Check memory constraint
        if target_memory_gb is not None:
            target_memory_bytes = target_memory_gb * 1024**3
            if total_memory_usage > target_memory_bytes:
                # Reduce precision to meet memory constraint
                warnings.warn(f"Recommended types exceed memory target. Total: {total_memory_usage/1024**3:.2f}GB")
                
                # Iteratively reduce precision for largest layers
                for layer_name in recommendations:
                    current_dtype = self._parse_dtype(recommendations[layer_name])
                    if current_dtype == SM120DataType.FLOAT32:
                        recommendations[layer_name] = SM120DataType.FLOAT16.value
                    elif current_dtype == SM120DataType.FLOAT16:
                        if self.is_type_supported(SM120DataType.FP8_E4M3):
                            recommendations[layer_name] = SM120DataType.FP8_E4M3.value
        
        return recommendations
    
    def create_optimized_tensor(self, shape: Tuple[int, ...], 
                              operation_type: str = 'general',
                              initializer: str = 'zeros') -> tf.Tensor:
        """Create tensor with optimal data type for operation."""
        optimal_dtype = self.get_optimal_type(operation_type)
        tf_dtype = self.get_type_properties(optimal_dtype).get('tf_dtype', tf.float32)
        
        if initializer == 'zeros':
            return tf.zeros(shape, dtype=tf_dtype)
        elif initializer == 'ones':
            return tf.ones(shape, dtype=tf_dtype)
        elif initializer == 'random_normal':
            return tf.random.normal(shape, dtype=tf_dtype)
        elif initializer == 'random_uniform':
            return tf.random.uniform(shape, dtype=tf_dtype)
        else:
            return tf.zeros(shape, dtype=tf_dtype)
    
    def print_type_support_summary(self):
        """Print comprehensive summary of data type support."""
        print("\n" + "="*70)
        print("🔢 SM120 DATA TYPE SUPPORT SUMMARY")
        print("="*70)
        
        print(f"\n🖥️  GPU Information:")
        print(f"   Compute Capability: {self._gpu_compute_capability}")
        print(f"   Mixed Precision: {'Enabled' if self._mixed_precision_enabled else 'Disabled'}")
        
        print(f"\n📊 Supported Data Types ({len(self._supported_types)}/{len(SM120DataType)}):")
        print(f"{'Type':<12} {'Size':<6} {'Tensor Core':<12} {'Memory Eff':<12} {'Compute Eff':<12}")
        print("-" * 70)
        
        for dtype in SM120DataType:
            properties = self.get_type_properties(dtype)
            supported = "✅" if dtype in self._supported_types else "❌"
            tensor_core = "✅" if properties.get('tensor_core_support') else "❌"
            
            print(f"{dtype.value:<12} {properties.get('size_bytes', 0):<6} "
                  f"{tensor_core:<12} {properties.get('memory_efficiency', 0):<11.1f}x "
                  f"{properties.get('compute_efficiency', 0):<11.1f}x {supported}")
        
        print(f"\n🎯 Optimal Types by Operation:")
        for operation, dtype in self._optimal_types.items():
            print(f"   {operation.capitalize():<15}: {dtype.value}")
        
        print("="*70)

# Global instance
_type_manager = SM120TypeManager()

# Public API functions
def get_supported_types() -> List[str]:
    """Get list of supported data type names."""
    return [dtype.value for dtype in _type_manager.get_supported_types()]

def is_type_supported(dtype: Union[str, tf.DType]) -> bool:
    """Check if a data type is supported."""
    return _type_manager.is_type_supported(dtype)

def get_optimal_type(operation: str) -> str:
    """Get optimal data type for an operation."""
    return _type_manager.get_optimal_type(operation).value

def auto_cast(tensor: tf.Tensor, operation: str = 'general', 
              preserve_precision: bool = False) -> tf.Tensor:
    """Automatically cast tensor to optimal data type."""
    return _type_manager.auto_cast_tensor(tensor, operation, preserve_precision)

def enable_mixed_precision(policy: str = 'mixed_float16'):
    """Enable mixed precision with SM120 optimizations."""
    _type_manager.enable_mixed_precision(policy)

def disable_mixed_precision():
    """Disable mixed precision."""
    _type_manager.disable_mixed_precision()

def recommend_model_dtypes(model: tf.keras.Model, target_memory_gb: Optional[float] = None,
                          prioritize_speed: bool = True) -> Dict[str, str]:
    """Recommend data types for model layers."""
    return _type_manager.recommend_dtype_for_model(model, target_memory_gb, prioritize_speed)

def create_optimized_tensor(shape: Tuple[int, ...], operation_type: str = 'general',
                           initializer: str = 'zeros') -> tf.Tensor:
    """Create tensor with optimal data type."""
    return _type_manager.create_optimized_tensor(shape, operation_type, initializer)

def get_memory_usage(tensor_shape: Tuple[int, ...], dtype: str) -> int:
    """Get memory usage estimate in bytes."""
    return _type_manager.get_memory_usage_estimate(tensor_shape, dtype)

def print_type_summary():
    """Print data type support summary."""
    _type_manager.print_type_support_summary()

# Convenience functions for common operations
def optimize_for_training(tensors: List[tf.Tensor]) -> List[tf.Tensor]:
    """Optimize tensor dtypes for training operations."""
    return [auto_cast(tensor, 'matmul', preserve_precision=True) for tensor in tensors]

def optimize_for_inference(tensors: List[tf.Tensor]) -> List[tf.Tensor]:
    """Optimize tensor dtypes for inference (prioritize speed/memory)."""
    return [auto_cast(tensor, 'attention', preserve_precision=False) for tensor in tensors]

def optimize_for_memory(tensors: List[tf.Tensor]) -> List[tf.Tensor]:
    """Optimize tensor dtypes for memory efficiency."""
    result = []
    for tensor in tensors:
        if _type_manager.is_type_supported(SM120DataType.FP8_E4M3):
            result.append(_type_manager._cast_tensor(tensor, SM120DataType.FP8_E4M3))
        elif _type_manager.is_type_supported(SM120DataType.FLOAT16):
            result.append(_type_manager._cast_tensor(tensor, SM120DataType.FLOAT16))
        else:
            result.append(tensor)
    return result
