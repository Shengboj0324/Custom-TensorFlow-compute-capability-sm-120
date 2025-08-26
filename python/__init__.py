"""
TensorFlow SM120 Optimization Suite

This package provides optimized TensorFlow operations for NVIDIA RTX 50-series GPUs
with compute capability sm_120.
"""

__version__ = "1.0.0"
__author__ = "TensorFlow SM120 Team"

# Import main modules for convenience
try:
    from . import benchmark
    from . import validate
except ImportError:
    # Handle case where modules are not yet built
    pass

__all__ = ["benchmark", "validate"]
