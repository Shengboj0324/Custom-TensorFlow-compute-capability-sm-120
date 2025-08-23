#!/usr/bin/env python3
"""
Setup script for TensorFlow SM120 optimizations.

This script builds and installs Python bindings for TensorFlow operations
optimized for RTX 50-series GPUs with compute capability 12.0.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Project metadata
__version__ = "1.0.0"
__description__ = "TensorFlow optimizations for RTX 50-series GPUs (sm_120)"
__author__ = "TensorFlow SM120 Project"
__email__ = "tensorflow-sm120@example.com"
__url__ = "https://github.com/yourusername/Custom-TensorFlow-compute-capability-sm-120"

# Check Python version
if sys.version_info < (3, 9):
    raise RuntimeError("Python 3.9 or higher is required")

# Check for CUDA
def check_cuda():
    """Check if CUDA is available and get version."""
    # Security: Use absolute paths and validate CUDA_HOME
    nvcc_paths = [
        '/usr/local/cuda/bin/nvcc',
        '/usr/local/cuda-12.8/bin/nvcc',
        '/usr/local/cuda-12/bin/nvcc',
    ]
    
    # Add CUDA_HOME path if it exists and is valid
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home and os.path.isdir(cuda_home):
        # Validate CUDA_HOME to prevent path injection
        if os.path.abspath(cuda_home).startswith(('/usr/local/', '/opt/', '/usr/')):
            nvcc_paths.insert(0, os.path.join(cuda_home, 'bin', 'nvcc'))
    
    # Try nvcc from PATH as fallback (less secure but sometimes necessary)
    import shutil
    nvcc_in_path = shutil.which('nvcc')
    if nvcc_in_path:
        nvcc_paths.append(nvcc_in_path)
    
    for nvcc_path in nvcc_paths:
        try:
            if not os.path.isfile(nvcc_path):
                continue
                
            result = subprocess.run([nvcc_path, '--version'], 
                                  capture_output=True, text=True, check=True, timeout=10)
            output = result.stdout
            
            # Extract CUDA version
            for line in output.split('\n'):
                if 'release' in line:
                    version_str = line.split('release ')[1].split(',')[0]
                    major, minor = map(int, version_str.split('.'))
                    if major >= 12 and minor >= 8:
                        return True, f"{major}.{minor}"
                    else:
                        print(f"Warning: CUDA {version_str} found, but 12.8+ required")
                        return False, version_str
            
            return False, "unknown"
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            continue
    
    return False, "not found"

# Check for RTX 50-series GPU
def check_sm120_gpu():
    """Check if RTX 50-series GPU is available."""
    # Security: Use absolute paths for nvidia-smi
    nvidia_smi_paths = [
        '/usr/bin/nvidia-smi',
        '/usr/local/cuda/bin/nvidia-smi',
        '/opt/nvidia/cuda/bin/nvidia-smi',
    ]
    
    # Add PATH lookup as fallback
    import shutil
    nvidia_smi_in_path = shutil.which('nvidia-smi')
    if nvidia_smi_in_path:
        nvidia_smi_paths.append(nvidia_smi_in_path)
    
    for nvidia_smi_path in nvidia_smi_paths:
        try:
            if not os.path.isfile(nvidia_smi_path):
                continue
                
            result = subprocess.run([nvidia_smi_path, '--query-gpu=compute_cap', 
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, check=True, timeout=15)
            
            compute_caps = result.stdout.strip().split('\n')
            has_sm120 = any('12.0' in cap for cap in compute_caps if cap.strip())
            
            return has_sm120, compute_caps
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return False, []

# Find TensorFlow
def find_tensorflow():
    """Find TensorFlow installation and get compile flags."""
    try:
        import tensorflow as tf
        
        # Get TensorFlow version
        tf_version = tf.__version__
        tf_major, tf_minor = map(int, tf_version.split('.')[:2])
        
        if tf_major < 2 or (tf_major == 2 and tf_minor < 10):
            raise RuntimeError(f"TensorFlow 2.10+ required, found {tf_version}")
        
        # Get include directories
        include_dirs = [
            tf.sysconfig.get_include(),
            os.path.join(tf.sysconfig.get_include(), 'external', 'nsync', 'public'),
        ]
        
        # Get library directories and flags
        lib_dirs = [tf.sysconfig.get_lib()]
        
        # Get compile flags
        compile_flags = tf.sysconfig.get_compile_flags()
        link_flags = tf.sysconfig.get_link_flags()
        
        return {
            'version': tf_version,
            'include_dirs': include_dirs,
            'lib_dirs': lib_dirs,
            'compile_flags': compile_flags,
            'link_flags': link_flags
        }
    except ImportError:
        raise RuntimeError("TensorFlow not found. Please install TensorFlow 2.10+")

# Find CUDA toolkit
def find_cuda_toolkit():
    """Find CUDA toolkit installation."""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    
    if not cuda_home:
        # Try common installation paths
        common_paths = [
            '/usr/local/cuda',
            '/usr/local/cuda-12.8',
            '/opt/cuda',
            'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8',
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                cuda_home = path
                break
    
    if not cuda_home or not os.path.exists(cuda_home):
        raise RuntimeError("CUDA toolkit not found. Please set CUDA_HOME environment variable")
    
    return {
        'home': cuda_home,
        'include_dir': os.path.join(cuda_home, 'include'),
        'lib_dir': os.path.join(cuda_home, 'lib64' if platform.system() != 'Windows' else 'lib\\x64'),
    }

# Custom build extension class
class SM120BuildExt(build_ext):
    """Custom build extension for SM120 operations."""
    
    def build_extensions(self):
        # Check system requirements
        print("Checking system requirements...")
        
        cuda_available, cuda_version = check_cuda()
        if not cuda_available:
            raise RuntimeError(f"CUDA 12.8+ required, found: {cuda_version}")
        
        sm120_available, compute_caps = check_sm120_gpu()
        if sm120_available:
            print(f"✓ RTX 50-series GPU detected with compute capabilities: {compute_caps}")
        else:
            print(f"⚠ No RTX 50-series GPU detected. Available: {compute_caps}")
            print("  Building with compatibility mode.")
        
        # Find dependencies
        tf_info = find_tensorflow()
        cuda_info = find_cuda_toolkit()
        
        print(f"✓ TensorFlow {tf_info['version']} found")
        print(f"✓ CUDA {cuda_version} found at {cuda_info['home']}")
        
        # Update extension configurations
        for ext in self.extensions:
            self.configure_extension(ext, tf_info, cuda_info, sm120_available)
        
        super().build_extensions()
    
    def configure_extension(self, ext, tf_info, cuda_info, has_sm120):
        """Configure extension with proper flags and libraries."""
        # Include directories
        ext.include_dirs.extend(tf_info['include_dirs'])
        ext.include_dirs.append(cuda_info['include_dir'])
        ext.include_dirs.append('src')
        
        # Library directories
        ext.library_dirs.extend(tf_info['lib_dirs'])
        ext.library_dirs.append(cuda_info['lib_dir'])
        
        # Libraries
        ext.libraries.extend([
            'tensorflow_framework',
            'cudart', 'cublas', 'cudnn', 'cufft', 'curand', 'cusolver', 'cusparse'
        ])
        
        # Compile flags
        compile_flags = [
            '-DGOOGLE_CUDA=1',
            '-DEIGEN_USE_GPU',
            '-DTENSORFLOW_USE_ROCM=0',
            '-std=c++17',
            '-O3',
            '-fPIC',
        ]
        
        if has_sm120:
            compile_flags.append('-DHAVE_SM120_GPU=1')
        
        # Platform-specific flags
        if platform.system() == 'Linux':
            compile_flags.extend(['-march=native', '-mtune=native'])
        elif platform.system() == 'Windows':
            compile_flags.extend(['/O2', '/DWIN32', '/D_WINDOWS'])
        
        ext.extra_compile_args.extend(compile_flags)
        
        # Link flags
        if platform.system() == 'Linux':
            ext.extra_link_args.extend(['-Wl,--as-needed', '-Wl,--no-undefined'])

# Define extensions
def get_extensions():
    """Get list of extensions to build."""
    
    # Source files (include CUDA sources - nvcc handles .cu files)
    sources = [
        'src/python_bindings/sm120_python_ops.cc',
        'src/tensorflow_ops/sm120_ops_fixed.cc',
        'src/cuda_kernels/sm120_optimized_kernels_fixed.cu',
        'src/tensorflow_ops/sm120_kernel_implementations.cu',
    ]
    
    # Main extension
    ext = Pybind11Extension(
        '_sm120_ops',
        sources=sources,
        include_dirs=[
            pybind11.get_include(),
            'src',
        ],
        libraries=[],
        library_dirs=[],
        extra_compile_args=[],
        extra_link_args=[],
        language='c++',
    )
    
    return [ext]

# Read README for long description
def get_long_description():
    """Get long description from README."""
    readme_path = Path(__file__).parent / 'README.md'
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return __description__

# Setup configuration
setup(
    name='tensorflow-sm120',
    version=__version__,
    author=__author__,
    author_email=__email__,
    url=__url__,
    description=__description__,
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    
    # Package configuration
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    include_package_data=True,
    # Security: Explicitly specify package data to prevent unintended file inclusion
    package_data={
        'tensorflow_sm120': ['_sm120_ops.so', '_sm120_ops.dll', '_sm120_ops.dylib']
    },
    
    # Extensions
    ext_modules=get_extensions(),
    cmdclass={'build_ext': SM120BuildExt},
    
    # Requirements
    python_requires='>=3.9',
    install_requires=[
        'tensorflow>=2.10.0',
        'numpy>=1.21.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.910',
        ],
        'benchmark': [
            'matplotlib>=3.3.0',
            'pandas>=1.3.0',
            'seaborn>=0.11.0',
        ],
    },
    
    # Metadata
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    keywords='tensorflow gpu cuda rtx-50 sm-120 machine-learning deep-learning',
    
    # Entry points
    entry_points={
        'console_scripts': [
            'sm120-benchmark=tensorflow_sm120.benchmark:main',
            'sm120-validate=tensorflow_sm120.validate:main',
        ],
    },
    
    # Package data
    package_data={
        'tensorflow_sm120': ['*.so', '*.dll', '*.dylib'],
    },
    
    # Build requirements
    setup_requires=[
        'pybind11>=2.10.0',
        'wheel',
    ],
    
    zip_safe=False,
)

# Post-installation message
def print_installation_info():
    """Print installation information."""
    print("\n" + "="*60)
    print("TensorFlow SM120 Installation Complete!")
    print("="*60)
    print("Next steps:")
    print("1. Validate installation: python -c 'import tensorflow_sm120; print(tensorflow_sm120.is_sm120_available())'")
    print("2. Run tests: python -m pytest tests/")
    print("3. Check examples: python examples/basic_usage.py")
    print("4. Benchmark performance: sm120-benchmark")
    print("\nFor documentation, visit:")
    print("https://github.com/yourusername/Custom-TensorFlow-compute-capability-sm-120")
    print("="*60)

if __name__ == '__main__':
    # Run setup
    try:
        # The setup() call is at the module level, so it runs automatically
        pass
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)
    
    # Print post-installation info if successful
    if len(sys.argv) > 1 and sys.argv[1] in ['install', 'develop']:
        print_installation_info()
