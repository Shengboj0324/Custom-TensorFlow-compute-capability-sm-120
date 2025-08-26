"""
SM120 Performance Monitoring and Auto-Tuning API
Comprehensive performance analysis and automatic optimization for SM120 operations
Copyright 2024 - TensorFlow SM120 Optimization Project
"""

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    # Dependencies not available during linting - will be installed later
    tf = None
    np = None
import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import warnings

try:
    SM120_AVAILABLE = True
except ImportError:
    SM120_AVAILABLE = False


@dataclass
class KernelMetrics:
    """Detailed performance metrics for SM120 kernels."""

    kernel_name: str
    execution_time_ms: float
    memory_bandwidth_gb_s: float
    arithmetic_intensity: float
    occupancy_percent: float
    blocks_launched: int
    threads_per_block: int
    shared_memory_bytes: int
    register_count: int
    tile_size: int
    warp_efficiency: float
    memory_efficiency: float
    compute_utilization: float
    achieved_bandwidth: float
    theoretical_bandwidth: float
    flops_per_second: float
    input_shape: Tuple[int, ...]
    data_type: str
    timestamp: float


@dataclass
class OptimizationHints:
    """Auto-tuning hints for optimal kernel configuration."""

    optimal_tile_size: int
    optimal_block_size: int
    use_tensor_cores: bool
    enable_async_copy: bool
    preferred_data_type: str
    memory_coalescing_factor: float
    cooperative_groups_enabled: bool
    shared_memory_config: str
    register_pressure: str
    confidence_score: float


class SM120PerformanceProfiler:
    """Real-time performance profiler for SM120 operations."""

    def __init__(self):
        self._enabled = False
        self._metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._optimization_hints: Dict[str, OptimizationHints] = {}
        self._kernel_configs: Dict[str, Dict] = {}
        self._gpu_info: Optional[Dict] = None
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
        self._auto_tuning_enabled = False
        self._tuning_iterations = {}

    def enable_profiling(self, enable: bool = True):
        """Enable or disable performance profiling."""
        with self._lock:
            self._enabled = enable
            if enable and not self._gpu_info:
                self._initialize_gpu_info()

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    def _initialize_gpu_info(self):
        """Initialize GPU information for performance analysis."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return

            gpu = gpus[0]
            device_details = tf.config.experimental.get_device_details(gpu)

            self._gpu_info = {
                "name": gpu.name,
                "compute_capability": device_details.get("compute_capability", (0, 0)),
                "memory_limit": tf.config.experimental.get_memory_info(gpu.name).get(
                    "total", 0
                ),
                "peak_memory_bandwidth": self._estimate_peak_bandwidth(),
                "tensor_core_support": self._check_tensor_core_support(),
                "warp_size": 32,  # Standard for all modern GPUs
                "max_threads_per_block": 1024,
                "shared_memory_per_block": 49152,  # 48KB for most GPUs
            }
        except Exception as e:
            warnings.warn(f"Could not initialize GPU info: {e}")
            self._gpu_info = {}

    def _estimate_peak_bandwidth(self) -> float:
        """Estimate peak memory bandwidth based on GPU type."""
        # Simplified estimation - in practice would use CUDA device queries
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                device_details = tf.config.experimental.get_device_details(gpus[0])
                compute_capability = device_details.get("compute_capability", (0, 0))

                # Rough estimates based on compute capability
                if compute_capability >= (12, 0):  # RTX 50-series / Blackwell
                    return 2000.0  # ~2TB/s estimated
                elif compute_capability >= (9, 0):  # H100
                    return 3000.0
                elif compute_capability >= (8, 0):  # RTX 30/40 series
                    return 900.0
                elif compute_capability >= (7, 5):  # RTX 20 series
                    return 616.0
                else:
                    return 500.0
        except Exception:
            return 1000.0  # Conservative default

    def _check_tensor_core_support(self) -> bool:
        """Check if current GPU supports Tensor Cores."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                device_details = tf.config.experimental.get_device_details(gpus[0])
                compute_capability = device_details.get("compute_capability", (0, 0))
                return compute_capability[0] >= 7  # Volta and newer
        except Exception:
            return False
        return False

    def record_kernel_metrics(
        self,
        kernel_name: str,
        execution_time: float,
        input_shape: Tuple[int, ...],
        data_type: str,
        additional_metrics: Optional[Dict] = None,
    ):
        """Record performance metrics for a kernel execution."""
        if not self._enabled:
            return

        with self._lock:
            # Calculate derived metrics
            total_elements = np.prod(input_shape) if input_shape else 0
            data_size_bytes = total_elements * self._get_dtype_size(data_type)

            # Estimate memory bandwidth (simplified)
            memory_bandwidth = (
                (data_size_bytes * 3) / (execution_time / 1000) / 1e9
            )  # Read input, weights, write output

            # Create metrics object
            metrics = KernelMetrics(
                kernel_name=kernel_name,
                execution_time_ms=execution_time,
                memory_bandwidth_gb_s=min(
                    memory_bandwidth, self._gpu_info.get("peak_memory_bandwidth", 1000)
                ),
                arithmetic_intensity=self._estimate_arithmetic_intensity(
                    kernel_name, input_shape
                ),
                occupancy_percent=(
                    additional_metrics.get("occupancy", 0.0)
                    if additional_metrics
                    else 0.0
                ),
                blocks_launched=(
                    additional_metrics.get("blocks", 0) if additional_metrics else 0
                ),
                threads_per_block=(
                    additional_metrics.get("threads_per_block", 256)
                    if additional_metrics
                    else 256
                ),
                shared_memory_bytes=(
                    additional_metrics.get("shared_memory", 0)
                    if additional_metrics
                    else 0
                ),
                register_count=(
                    additional_metrics.get("registers", 0) if additional_metrics else 0
                ),
                tile_size=(
                    additional_metrics.get("tile_size", 16)
                    if additional_metrics
                    else 16
                ),
                warp_efficiency=(
                    additional_metrics.get("warp_efficiency", 0.0)
                    if additional_metrics
                    else 0.0
                ),
                memory_efficiency=memory_bandwidth
                / self._gpu_info.get("peak_memory_bandwidth", 1000)
                * 100,
                compute_utilization=(
                    additional_metrics.get("compute_utilization", 0.0)
                    if additional_metrics
                    else 0.0
                ),
                achieved_bandwidth=memory_bandwidth,
                theoretical_bandwidth=self._gpu_info.get("peak_memory_bandwidth", 1000),
                flops_per_second=self._estimate_flops(
                    kernel_name, input_shape, execution_time
                ),
                input_shape=input_shape,
                data_type=data_type,
                timestamp=time.time(),
            )

            # Store metrics
            self._metrics_history[kernel_name].append(metrics)

            # Trigger auto-tuning if enabled
            if self._auto_tuning_enabled:
                self._update_optimization_hints(kernel_name, metrics)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    warnings.warn(f"Callback error: {e}")

    def _get_dtype_size(self, data_type: str) -> int:
        """Get size in bytes for data type."""
        type_sizes = {
            "float32": 4,
            "float": 4,
            "float16": 2,
            "half": 2,
            "bfloat16": 2,
            "int32": 4,
            "int": 4,
            "int64": 8,
            "double": 8,
        }
        return type_sizes.get(data_type.lower(), 4)

    def _estimate_arithmetic_intensity(
        self, kernel_name: str, input_shape: Tuple[int, ...]
    ) -> float:
        """Estimate arithmetic intensity (FLOPs per byte)."""
        if not input_shape:
            return 0.0

        if "matmul" in kernel_name.lower() or "dense" in kernel_name.lower():
            # Matrix multiplication: 2*M*N*K FLOPs, ~(M*K + K*N + M*N) * 4 bytes
            if len(input_shape) >= 2:
                M, K = input_shape[-2], input_shape[-1]
                N = input_shape[-1]  # Assume square-ish
                flops = 2 * M * N * K
                bytes_transferred = (M * K + K * N + M * N) * 4
                return flops / bytes_transferred

        elif "conv" in kernel_name.lower():
            # Convolution: more complex, simplified estimate
            return 2.0  # Typical for convolution

        elif "attention" in kernel_name.lower():
            # Attention: depends on sequence length
            if len(input_shape) >= 2:
                seq_len = input_shape[-2] if len(input_shape) > 2 else input_shape[-1]
                return max(1.0, seq_len / 512.0)  # Scales with sequence length

        return 1.0  # Default

    def _estimate_flops(
        self, kernel_name: str, input_shape: Tuple[int, ...], execution_time_ms: float
    ) -> float:
        """Estimate FLOPS per second."""
        if not input_shape or execution_time_ms <= 0:
            return 0.0

        total_elements = np.prod(input_shape)

        if "matmul" in kernel_name.lower():
            # Rough estimate: 2 FLOPs per element for matrix multiplication
            flops = 2 * total_elements
        elif "conv" in kernel_name.lower():
            # Convolution typically has more operations per element
            flops = 10 * total_elements  # Rough estimate
        else:
            # Element-wise operations
            flops = total_elements

        return flops / (execution_time_ms / 1000.0)

    def _update_optimization_hints(self, kernel_name: str, metrics: KernelMetrics):
        """Update optimization hints based on performance metrics."""
        if kernel_name not in self._optimization_hints:
            self._optimization_hints[kernel_name] = OptimizationHints(
                optimal_tile_size=16,
                optimal_block_size=256,
                use_tensor_cores=self._gpu_info.get("tensor_core_support", False),
                enable_async_copy=True,
                preferred_data_type="float16",
                memory_coalescing_factor=1.0,
                cooperative_groups_enabled=True,
                shared_memory_config="prefer_shared",
                register_pressure="low",
                confidence_score=0.5,
            )

        hints = self._optimization_hints[kernel_name]
        history = self._metrics_history[kernel_name]

        if len(history) < 10:
            return  # Not enough data for tuning

        # Calculate average performance metrics
        recent_metrics = list(history)[-10:]
        avg_time = np.mean([m.execution_time_ms for m in recent_metrics])
        avg_bandwidth = np.mean([m.memory_bandwidth_gb_s for m in recent_metrics])
        avg_occupancy = np.mean([m.occupancy_percent for m in recent_metrics])

        # Auto-tune tile size based on performance
        if avg_occupancy < 50:
            # Low occupancy - try smaller tile size
            hints.optimal_tile_size = max(8, hints.optimal_tile_size // 2)
            hints.optimal_block_size = max(64, hints.optimal_block_size // 2)
        elif avg_occupancy > 95 and avg_time > 1.0:
            # High occupancy but slow - try larger tile size
            hints.optimal_tile_size = min(64, hints.optimal_tile_size * 2)
            hints.optimal_block_size = min(1024, hints.optimal_block_size * 2)

        # Adjust memory configuration
        if avg_bandwidth < self._gpu_info.get("peak_memory_bandwidth", 1000) * 0.3:
            hints.memory_coalescing_factor = 0.8
            hints.shared_memory_config = "prefer_cache"

        # Data type recommendation
        if metrics.data_type == "float32" and hints.use_tensor_cores:
            hints.preferred_data_type = "float16"

        # Update confidence score
        variance = np.var([m.execution_time_ms for m in recent_metrics])
        hints.confidence_score = max(0.1, min(1.0, 1.0 - variance / avg_time))

    def get_metrics(
        self, kernel_name: str, num_samples: int = 100
    ) -> List[KernelMetrics]:
        """Get recent performance metrics for a kernel."""
        with self._lock:
            if kernel_name not in self._metrics_history:
                return []

            history = list(self._metrics_history[kernel_name])
            return history[-num_samples:] if num_samples > 0 else history

    def get_average_metrics(self, kernel_name: str) -> Optional[Dict]:
        """Get average performance metrics for a kernel."""
        metrics = self.get_metrics(kernel_name)
        if not metrics:
            return None

        return {
            "kernel_name": kernel_name,
            "sample_count": len(metrics),
            "avg_execution_time_ms": np.mean([m.execution_time_ms for m in metrics]),
            "avg_memory_bandwidth_gb_s": np.mean(
                [m.memory_bandwidth_gb_s for m in metrics]
            ),
            "avg_occupancy_percent": np.mean([m.occupancy_percent for m in metrics]),
            "avg_memory_efficiency": np.mean([m.memory_efficiency for m in metrics]),
            "avg_flops_per_second": np.mean([m.flops_per_second for m in metrics]),
            "std_execution_time_ms": np.std([m.execution_time_ms for m in metrics]),
            "min_execution_time_ms": np.min([m.execution_time_ms for m in metrics]),
            "max_execution_time_ms": np.max([m.execution_time_ms for m in metrics]),
        }

    def get_optimization_hints(self, kernel_name: str) -> Optional[OptimizationHints]:
        """Get optimization hints for a kernel."""
        with self._lock:
            return self._optimization_hints.get(kernel_name)

    def enable_auto_tuning(self, enable: bool = True):
        """Enable automatic performance tuning."""
        with self._lock:
            self._auto_tuning_enabled = enable

    def add_performance_callback(self, callback: Callable[[KernelMetrics], None]):
        """Add callback function for performance events."""
        self._callbacks.append(callback)

    def print_performance_summary(self, top_k: int = 10):
        """Print comprehensive performance summary."""
        print("\n" + "=" * 80)
        print("🚀 SM120 PERFORMANCE SUMMARY")
        print("=" * 80)

        if not self._enabled:
            print(
                "❌ Performance profiling is disabled. Enable with profiler.enable_profiling(True)"
            )
            return

        if not self._metrics_history:
            print("📊 No performance data collected yet.")
            return

        # GPU Information
        if self._gpu_info:
            print("\n🖥️  GPU Information:")
            print(f"   Device: {self._gpu_info.get('name', 'Unknown')}")
            print(
                f"   Compute Capability: {self._gpu_info.get('compute_capability', 'Unknown')}"
            )
            print(
                f"   Peak Memory Bandwidth: "
                f"{self._gpu_info.get('peak_memory_bandwidth', 0):.1f} GB/s"
            )
            print(
                f"   Tensor Core Support: "
                f"{'Yes' if self._gpu_info.get('tensor_core_support') else 'No'}"
            )

        # Kernel Performance Rankings
        kernel_stats = []
        for kernel_name, history in self._metrics_history.items():
            if history:
                recent_metrics = list(history)[-50:]  # Last 50 executions
                avg_time = np.mean([m.execution_time_ms for m in recent_metrics])
                avg_bandwidth = np.mean(
                    [m.memory_bandwidth_gb_s for m in recent_metrics]
                )
                avg_efficiency = np.mean([m.memory_efficiency for m in recent_metrics])
                total_calls = len(history)

                kernel_stats.append(
                    {
                        "name": kernel_name,
                        "avg_time": avg_time,
                        "avg_bandwidth": avg_bandwidth,
                        "avg_efficiency": avg_efficiency,
                        "total_calls": total_calls,
                        "recent_samples": len(recent_metrics),
                    }
                )

        # Sort by average execution time (ascending)
        kernel_stats.sort(key=lambda x: x["avg_time"])

        print(f"\n📊 Top {min(top_k, len(kernel_stats))} Kernel Performance:")
        print(
            f"{'Rank':<4} {'Kernel Name':<25} {'Avg Time':<12} "
            f"{'Bandwidth':<12} {'Efficiency':<12} {'Calls':<8}"
        )
        print("-" * 80)

        for i, stats in enumerate(kernel_stats[:top_k]):
            print(
                f"{i+1:<4} {stats['name']:<25} {stats['avg_time']:<11.3f}ms "
                f"{stats['avg_bandwidth']:<11.1f}GB/s {stats['avg_efficiency']:<11.1f}% "
                f"{stats['total_calls']:<8}"
            )

        # Performance Insights
        print("\n💡 Performance Insights:")

        # Find bottlenecks
        slow_kernels = [s for s in kernel_stats if s["avg_time"] > 1.0]
        if slow_kernels:
            print(
                f"   ⚠️  Slow kernels (>1ms): {', '.join([k['name'] for k in slow_kernels[:3]])}"
            )

        # Find memory-bound operations
        memory_bound = [s for s in kernel_stats if s["avg_efficiency"] < 50]
        if memory_bound:
            print(
                f"   💾 Memory-bound ops (<50% efficiency): "
                f"{', '.join([k['name'] for k in memory_bound[:3]])}"
            )

        # Show optimization hints
        optimized_kernels = len(
            [
                k
                for k in self._optimization_hints.keys()
                if self._optimization_hints[k].confidence_score > 0.7
            ]
        )
        print(f"   🎯 Auto-tuned kernels: {optimized_kernels}/{len(kernel_stats)}")

        # Show data type usage
        fp16_usage = len(
            [
                h
                for h in self._metrics_history.values()
                if h and any(m.data_type in ["float16", "half"] for m in list(h)[-10:])
            ]
        )
        total_kernels = len(self._metrics_history)
        if total_kernels > 0:
            print(
                f"   🔢 FP16 utilization: {fp16_usage}/{total_kernels} kernels "
                f"({fp16_usage/total_kernels*100:.1f}%)"
            )

        print("=" * 80)

    def export_metrics(self, filename: str, format: str = "json"):
        """Export performance metrics to file."""
        with self._lock:
            data = {
                "gpu_info": self._gpu_info,
                "optimization_hints": {
                    k: asdict(v) for k, v in self._optimization_hints.items()
                },
                "metrics": {},
            }

            for kernel_name, history in self._metrics_history.items():
                data["metrics"][kernel_name] = [asdict(m) for m in history]

            if format.lower() == "json":
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def reset_metrics(self):
        """Reset all performance metrics."""
        with self._lock:
            self._metrics_history.clear()
            self._optimization_hints.clear()
            self._tuning_iterations.clear()


# Global profiler instance
_global_profiler = SM120PerformanceProfiler()


# Public API functions
def enable_profiling(enable: bool = True):
    """Enable SM120 performance profiling."""
    _global_profiler.enable_profiling(enable)


def is_profiling_enabled() -> bool:
    """Check if performance profiling is enabled."""
    return _global_profiler.is_enabled()


def enable_auto_tuning(enable: bool = True):
    """Enable automatic performance tuning."""
    _global_profiler.enable_auto_tuning(enable)


def get_kernel_metrics(kernel_name: str, num_samples: int = 100) -> List[Dict]:
    """Get performance metrics for a specific kernel."""
    metrics = _global_profiler.get_metrics(kernel_name, num_samples)
    return [asdict(m) for m in metrics]


def get_average_metrics(kernel_name: str) -> Optional[Dict]:
    """Get average performance metrics for a kernel."""
    return _global_profiler.get_average_metrics(kernel_name)


def get_optimization_hints(kernel_name: str) -> Optional[Dict]:
    """Get optimization hints for a kernel."""
    hints = _global_profiler.get_optimization_hints(kernel_name)
    return asdict(hints) if hints else None


def print_performance_summary(top_k: int = 10):
    """Print comprehensive performance summary."""
    _global_profiler.print_performance_summary(top_k)


def export_performance_data(filename: str, format: str = "json"):
    """Export performance data to file."""
    _global_profiler.export_metrics(filename, format)


def reset_performance_data():
    """Reset all performance data."""
    _global_profiler.reset_metrics()


def add_performance_callback(callback: Callable[[Dict], None]):
    """Add callback for performance events."""

    def wrapper(metrics: KernelMetrics):
        callback(asdict(metrics))

    _global_profiler.add_performance_callback(wrapper)


# Context manager for performance measurement
class SM120PerformanceContext:
    """Context manager for measuring SM120 operation performance."""

    def __init__(
        self,
        operation_name: str,
        input_shape: Tuple[int, ...] = (),
        data_type: str = "float32",
    ):
        self.operation_name = operation_name
        self.input_shape = input_shape
        self.data_type = data_type
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            execution_time = (
                time.perf_counter() - self.start_time
            ) * 1000  # Convert to ms
            _global_profiler.record_kernel_metrics(
                self.operation_name, execution_time, self.input_shape, self.data_type
            )


def measure_performance(
    operation_name: str, input_shape: Tuple[int, ...] = (), data_type: str = "float32"
):
    """Create performance measurement context."""
    return SM120PerformanceContext(operation_name, input_shape, data_type)


# Example usage functions
def benchmark_operation(
    operation_func: Callable,
    *args,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    operation_name: str = "custom_op",
) -> Dict:
    """Benchmark a specific operation."""
    if not SM120_AVAILABLE:
        raise RuntimeError("SM120 operations not available")

    # Warmup
    for _ in range(warmup_iterations):
        operation_func(*args)

    # Benchmark
    times = []
    for i in range(num_iterations):
        with measure_performance(f"{operation_name}_benchmark"):
            start = time.perf_counter()
            _ = operation_func(*args)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    return {
        "operation_name": operation_name,
        "num_iterations": num_iterations,
        "avg_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "median_time_ms": np.median(times),
    }


# Integration with SM120 layers
def monitor_model_performance(
    model: tf.keras.Model, sample_input: tf.Tensor, num_iterations: int = 10
) -> Dict:
    """Monitor performance of SM120 layers in a model."""
    if not SM120_AVAILABLE:
        raise RuntimeError("SM120 operations not available")

    enable_profiling(True)
    reset_performance_data()

    # Warmup
    for _ in range(3):
        _ = model(sample_input)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model(sample_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Collect SM120 layer metrics
    sm120_layers = []
    for layer in model.layers:
        if hasattr(layer, "use_sm120") and layer.use_sm120:
            sm120_layers.append(layer.name)

    return {
        "model_summary": {
            "total_layers": len(model.layers),
            "sm120_layers": len(sm120_layers),
            "sm120_layer_names": sm120_layers,
        },
        "timing": {
            "avg_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
        },
        "layer_metrics": {name: get_average_metrics(name) for name in sm120_layers},
    }
