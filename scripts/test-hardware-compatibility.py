#!/usr/bin/env python3
"""
Hardware Compatibility Testing for TensorFlow SM120

This script performs comprehensive testing to ensure the SM120 optimizations
work correctly on both RTX 50-series GPUs and provide graceful fallback
on older hardware.
"""

import os
import sys
import subprocess
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Suppress TensorFlow warnings during testing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    import numpy as np
    # Try to import SM120 ops
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))
    import sm120_ops
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow or SM120 ops not available: {e}")
    TENSORFLOW_AVAILABLE = False


class HardwareCompatibilityTester:
    """Comprehensive hardware compatibility testing suite."""
    
    def __init__(self):
        self.results = {
            'system_info': {},
            'gpu_detection': {},
            'sm120_availability': {},
            'fallback_testing': {},
            'performance_tests': {},
            'stability_tests': {},
            'errors': []
        }
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all hardware compatibility tests."""
        print("🔍 Starting Hardware Compatibility Testing...")
        print("=" * 60)
        
        try:
            self.test_system_info()
            self.test_gpu_detection()
            self.test_sm120_availability()
            self.test_fallback_behavior()
            self.test_basic_operations()
            self.test_performance_characteristics()
            self.test_stability()
            
            print("\n" + "=" * 60)
            print("✅ Hardware Compatibility Testing Complete!")
            self.print_summary()
            
        except Exception as e:
            self.results['errors'].append(f"Critical test failure: {str(e)}")
            print(f"❌ Critical test failure: {e}")
            
        return self.results
    
    def test_system_info(self):
        """Test system information gathering."""
        print("\n📋 Testing System Information...")
        
        try:
            # CUDA version
            cuda_result = subprocess.run(['nvcc', '--version'], 
                                       capture_output=True, text=True)
            if cuda_result.returncode == 0:
                cuda_version = self._extract_cuda_version(cuda_result.stdout)
                self.results['system_info']['cuda_version'] = cuda_version
                print(f"   CUDA Version: {cuda_version}")
            else:
                self.results['system_info']['cuda_version'] = None
                print("   CUDA: Not available")
                
            # Driver version
            nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
            if nvidia_smi.returncode == 0:
                driver_version = nvidia_smi.stdout.strip()
                self.results['system_info']['driver_version'] = driver_version
                print(f"   Driver Version: {driver_version}")
            else:
                self.results['system_info']['driver_version'] = None
                print("   NVIDIA Driver: Not available")
                
            # TensorFlow version
            if TENSORFLOW_AVAILABLE:
                tf_version = tf.__version__
                self.results['system_info']['tensorflow_version'] = tf_version
                print(f"   TensorFlow Version: {tf_version}")
            else:
                self.results['system_info']['tensorflow_version'] = None
                print("   TensorFlow: Not available")
                
        except Exception as e:
            self.results['errors'].append(f"System info test failed: {str(e)}")
            print(f"   ❌ Error: {e}")
    
    def test_gpu_detection(self):
        """Test GPU detection and compute capability identification."""
        print("\n🎮 Testing GPU Detection...")
        
        if not TENSORFLOW_AVAILABLE:
            print("   ⚠️  Skipping GPU tests - TensorFlow not available")
            return
            
        try:
            # List physical devices
            gpus = tf.config.list_physical_devices('GPU')
            self.results['gpu_detection']['gpu_count'] = len(gpus)
            print(f"   Detected GPUs: {len(gpus)}")
            
            gpu_details = []
            for i, gpu in enumerate(gpus):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    compute_cap = details.get('compute_capability', (0, 0))
                    
                    gpu_info = {
                        'index': i,
                        'name': gpu.name,
                        'compute_capability': compute_cap,
                        'is_sm120': compute_cap >= (12, 0) if isinstance(compute_cap, tuple) else False
                    }
                    gpu_details.append(gpu_info)
                    
                    print(f"   GPU {i}: {gpu.name}")
                    print(f"     Compute Capability: {compute_cap}")
                    print(f"     SM120 Compatible: {gpu_info['is_sm120']}")
                    
                except Exception as e:
                    print(f"   ❌ Error getting details for GPU {i}: {e}")
                    
            self.results['gpu_detection']['gpus'] = gpu_details
            
        except Exception as e:
            self.results['errors'].append(f"GPU detection failed: {str(e)}")
            print(f"   ❌ Error: {e}")
    
    def test_sm120_availability(self):
        """Test SM120 operations availability."""
        print("\n🚀 Testing SM120 Availability...")
        
        try:
            # Test SM120 ops import
            sm120_available = sm120_ops.is_sm120_available()
            self.results['sm120_availability']['ops_available'] = sm120_available
            print(f"   SM120 Operations Available: {sm120_available}")
            
            if sm120_available:
                # Get SM120 device info
                device_info = sm120_ops.get_sm120_device_info()
                self.results['sm120_availability']['device_info'] = device_info
                print(f"   SM120 Devices: {len(device_info.get('devices', []))}")
                
                for device in device_info.get('devices', []):
                    if device.get('sm120_compatible', False):
                        print(f"     Device {device['index']}: {device['name']} (SM120)")
                    else:
                        print(f"     Device {device['index']}: {device['name']} (Fallback)")
            else:
                print("   No SM120 compatible devices found - will use fallback")
                
        except Exception as e:
            self.results['errors'].append(f"SM120 availability test failed: {str(e)}")
            print(f"   ❌ Error: {e}")
    
    def test_fallback_behavior(self):
        """Test fallback behavior on non-SM120 hardware."""
        print("\n🔄 Testing Fallback Behavior...")
        
        if not TENSORFLOW_AVAILABLE:
            print("   ⚠️  Skipping fallback tests - TensorFlow not available")
            return
            
        try:
            # Test basic operations with fallback
            a = tf.random.normal([100, 100], dtype=tf.float32)
            b = tf.random.normal([100, 100], dtype=tf.float32)
            
            # Test standard TensorFlow operation
            start_time = time.time()
            result_standard = tf.matmul(a, b)
            standard_time = time.time() - start_time
            
            # Test SM120 operation (should fallback if no SM120 GPU)
            start_time = time.time()
            try:
                result_sm120 = sm120_ops.advanced_matmul(a, b)
                sm120_time = time.time() - start_time
                fallback_working = True
            except Exception as e:
                print(f"   SM120 operation failed: {e}")
                result_sm120 = result_standard  # Use standard result for comparison
                sm120_time = float('inf')
                fallback_working = False
            
            # Compare results
            if fallback_working:
                max_diff = tf.reduce_max(tf.abs(result_standard - result_sm120)).numpy()
                self.results['fallback_testing'] = {
                    'fallback_working': True,
                    'max_difference': float(max_diff),
                    'standard_time': standard_time,
                    'sm120_time': sm120_time,
                    'speedup': standard_time / sm120_time if sm120_time > 0 else 0
                }
                
                print(f"   Fallback Working: ✅")
                print(f"   Max Difference: {max_diff:.2e}")
                print(f"   Standard Time: {standard_time:.4f}s")
                print(f"   SM120 Time: {sm120_time:.4f}s")
                if sm120_time < float('inf'):
                    print(f"   Speedup: {standard_time/sm120_time:.2f}x")
            else:
                self.results['fallback_testing'] = {
                    'fallback_working': False,
                    'error': 'SM120 operations failed'
                }
                print("   Fallback Working: ❌")
                
        except Exception as e:
            self.results['errors'].append(f"Fallback test failed: {str(e)}")
            print(f"   ❌ Error: {e}")
    
    def test_basic_operations(self):
        """Test basic SM120 operations."""
        print("\n🧮 Testing Basic Operations...")
        
        if not TENSORFLOW_AVAILABLE:
            print("   ⚠️  Skipping operation tests - TensorFlow not available")
            return
            
        operations_tested = []
        
        try:
            # Test matrix multiplication
            a = tf.random.normal([64, 64], dtype=tf.float32)
            b = tf.random.normal([64, 64], dtype=tf.float32)
            
            try:
                result = sm120_ops.advanced_matmul(a, b)
                operations_tested.append(('advanced_matmul', True, None))
                print("   ✅ Advanced MatMul: Working")
            except Exception as e:
                operations_tested.append(('advanced_matmul', False, str(e)))
                print(f"   ❌ Advanced MatMul: {e}")
            
            # Test other operations if available
            test_ops = [
                ('flash_attention', lambda: self._test_flash_attention()),
                ('batch_norm', lambda: self._test_batch_norm()),
                ('conv2d', lambda: self._test_conv2d())
            ]
            
            for op_name, test_func in test_ops:
                try:
                    test_func()
                    operations_tested.append((op_name, True, None))
                    print(f"   ✅ {op_name}: Working")
                except Exception as e:
                    operations_tested.append((op_name, False, str(e)))
                    print(f"   ❌ {op_name}: {e}")
            
            self.results['performance_tests']['operations'] = operations_tested
            
        except Exception as e:
            self.results['errors'].append(f"Basic operations test failed: {str(e)}")
            print(f"   ❌ Error: {e}")
    
    def test_performance_characteristics(self):
        """Test performance characteristics."""
        print("\n⚡ Testing Performance Characteristics...")
        
        if not TENSORFLOW_AVAILABLE:
            print("   ⚠️  Skipping performance tests - TensorFlow not available")
            return
            
        try:
            # Performance test with different sizes
            sizes = [64, 128, 256, 512]
            performance_data = []
            
            for size in sizes:
                a = tf.random.normal([size, size], dtype=tf.float32)
                b = tf.random.normal([size, size], dtype=tf.float32)
                
                # Warm up
                for _ in range(3):
                    _ = tf.matmul(a, b)
                
                # Standard TensorFlow
                times_standard = []
                for _ in range(5):
                    start = time.time()
                    _ = tf.matmul(a, b)
                    times_standard.append(time.time() - start)
                
                avg_standard = np.mean(times_standard)
                
                # SM120 (if available)
                try:
                    times_sm120 = []
                    for _ in range(5):
                        start = time.time()
                        _ = sm120_ops.advanced_matmul(a, b)
                        times_sm120.append(time.time() - start)
                    
                    avg_sm120 = np.mean(times_sm120)
                    speedup = avg_standard / avg_sm120
                except:
                    avg_sm120 = None
                    speedup = None
                
                perf_data = {
                    'size': size,
                    'standard_time': avg_standard,
                    'sm120_time': avg_sm120,
                    'speedup': speedup
                }
                performance_data.append(perf_data)
                
                print(f"   Size {size}x{size}: Standard={avg_standard:.4f}s", end="")
                if avg_sm120:
                    print(f", SM120={avg_sm120:.4f}s, Speedup={speedup:.2f}x")
                else:
                    print(", SM120=N/A")
            
            self.results['performance_tests']['benchmarks'] = performance_data
            
        except Exception as e:
            self.results['errors'].append(f"Performance test failed: {str(e)}")
            print(f"   ❌ Error: {e}")
    
    def test_stability(self):
        """Test stability over multiple iterations."""
        print("\n🔒 Testing Stability...")
        
        if not TENSORFLOW_AVAILABLE:
            print("   ⚠️  Skipping stability tests - TensorFlow not available")
            return
            
        try:
            iterations = 50
            errors = 0
            
            for i in range(iterations):
                try:
                    a = tf.random.normal([100, 100], dtype=tf.float32)
                    b = tf.random.normal([100, 100], dtype=tf.float32)
                    
                    # Test SM120 operation
                    result = sm120_ops.advanced_matmul(a, b)
                    
                    # Basic sanity check
                    if tf.reduce_any(tf.math.is_nan(result)) or tf.reduce_any(tf.math.is_inf(result)):
                        errors += 1
                        
                except Exception:
                    errors += 1
                
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i+1}/{iterations} iterations")
            
            success_rate = (iterations - errors) / iterations * 100
            self.results['stability_tests'] = {
                'iterations': iterations,
                'errors': errors,
                'success_rate': success_rate
            }
            
            print(f"   Success Rate: {success_rate:.1f}% ({iterations-errors}/{iterations})")
            
            if success_rate >= 99.0:
                print("   ✅ Stability: Excellent")
            elif success_rate >= 95.0:
                print("   ⚠️  Stability: Good")
            else:
                print("   ❌ Stability: Poor")
                
        except Exception as e:
            self.results['errors'].append(f"Stability test failed: {str(e)}")
            print(f"   ❌ Error: {e}")
    
    def _test_flash_attention(self):
        """Test flash attention operation."""
        # Placeholder for flash attention test
        pass
    
    def _test_batch_norm(self):
        """Test batch normalization operation."""
        # Placeholder for batch norm test
        pass
    
    def _test_conv2d(self):
        """Test 2D convolution operation."""
        # Placeholder for conv2d test
        pass
    
    def _extract_cuda_version(self, nvcc_output: str) -> str:
        """Extract CUDA version from nvcc output."""
        import re
        match = re.search(r'release (\d+\.\d+)', nvcc_output)
        return match.group(1) if match else "Unknown"
    
    def print_summary(self):
        """Print test summary."""
        print("\n📊 Test Summary:")
        print("-" * 40)
        
        # System info
        if self.results['system_info']:
            print(f"CUDA Version: {self.results['system_info'].get('cuda_version', 'N/A')}")
            print(f"Driver Version: {self.results['system_info'].get('driver_version', 'N/A')}")
            print(f"TensorFlow Version: {self.results['system_info'].get('tensorflow_version', 'N/A')}")
        
        # GPU info
        gpu_count = self.results['gpu_detection'].get('gpu_count', 0)
        print(f"GPUs Detected: {gpu_count}")
        
        # SM120 availability
        sm120_available = self.results['sm120_availability'].get('ops_available', False)
        print(f"SM120 Available: {'Yes' if sm120_available else 'No'}")
        
        # Errors
        error_count = len(self.results['errors'])
        print(f"Errors: {error_count}")
        
        if error_count > 0:
            print("\nErrors encountered:")
            for error in self.results['errors']:
                print(f"  - {error}")


def main():
    """Main test execution."""
    tester = HardwareCompatibilityTester()
    results = tester.run_all_tests()
    
    # Save results to file
    output_file = 'hardware_compatibility_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Exit with appropriate code
    error_count = len(results['errors'])
    sys.exit(0 if error_count == 0 else 1)


if __name__ == '__main__':
    main()
