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
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

import pybind11


# Utility function for safe version parsing (prevents packaging.version.Version issues)
def parse_version_safely(version_str: str) -> tuple:
    """Safely parse version string without using packaging.version.Version."""
    try:
        # Handle version strings like "12.8" or "12.8.0"
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except (ValueError, IndexError):
        return (0, 0, 0)


def compare_versions(version1: str, version2: str) -> int:
    """Compare two version strings. Returns -1, 0, or 1."""
    v1 = parse_version_safely(version1)
    v2 = parse_version_safely(version2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def sanitize_path(path: str) -> str:
    """Sanitize path to prevent injection attacks."""
    if not path:
        return ""

    # Resolve to absolute path and normalize
    abs_path = os.path.abspath(os.path.expanduser(path))

    # Security: Only allow paths in safe locations
    safe_prefixes = [
        "/usr/local/",
        "/usr/bin/",
        "/usr/sbin/",
        "/opt/",
        "/snap/",
        os.path.expanduser("~/"),  # User home directory
    ]

    # On Windows, also allow standard locations
    if platform.system() == "Windows":
        safe_prefixes.extend(
            [
                "C:\\Program Files\\",
                "C:\\Program Files (x86)\\",
                "C:\\Windows\\System32\\",
                os.path.expanduser("~\\AppData\\"),
            ]
        )

    # Check if path starts with any safe prefix
    for prefix in safe_prefixes:
        if abs_path.startswith(os.path.abspath(prefix)):
            return abs_path

    # Path is not in a safe location
    raise ValueError(f"Path '{path}' is not in a safe location")


def find_executable_safely(name: str, env_paths: list = None) -> str:
    """Find executable with path sanitization."""
    import shutil

    # First try standard locations
    standard_locations = {
        "nvcc": [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
            "/usr/local/cuda-12/bin/nvcc",
            "/opt/cuda/bin/nvcc",
        ],
        "nvidia-smi": [
            "/usr/bin/nvidia-smi",
            "/usr/local/cuda/bin/nvidia-smi",
            "/opt/nvidia/cuda/bin/nvidia-smi",
        ],
    }

    # Try standard locations first
    if name in standard_locations:
        for location in standard_locations[name]:
            if os.path.isfile(location) and os.access(location, os.X_OK):
                return location

    # Try environment paths if provided
    if env_paths:
        for path in env_paths:
            try:
                safe_path = sanitize_path(path)
                full_path = os.path.join(safe_path, name)
                if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                    return full_path
            except ValueError:
                continue  # Skip unsafe paths

    # Finally try shutil.which as last resort (less safe)
    which_result = shutil.which(name)
    if which_result:
        try:
            return sanitize_path(which_result)
        except ValueError:
            pass  # PATH contains unsafe location

    return ""


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
    # Security: Use sanitized path finding
    cuda_home = os.environ.get("CUDA_HOME")
    env_paths = []
    if cuda_home:
        try:
            safe_cuda_home = sanitize_path(cuda_home)
            env_paths.append(os.path.join(safe_cuda_home, "bin"))
        except ValueError:
            print(f"Warning: CUDA_HOME path '{cuda_home}' is not safe, ignoring")

    nvcc_path = find_executable_safely("nvcc", env_paths)
    if not nvcc_path:
        return False, "not found"

    try:
        result = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        output = result.stdout

        # Extract CUDA version
        for line in output.split("\n"):
            if "release" in line:
                version_str = line.split("release ")[1].split(",")[0]
                # Use safe version parsing to avoid packaging.version.Version issues
                if compare_versions(version_str, "12.4") >= 0:
                    return True, version_str
                else:
                    print(f"Warning: CUDA {version_str} found, but 12.4+ required")
                    return False, version_str

        return False, "unknown"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
        ValueError,
    ):
        return False, "not found"


# Check for RTX 50-series GPU
def check_sm120_gpu():
    """Check if RTX 50-series GPU is available."""
    # Security: Use sanitized path finding
    nvidia_smi_path = find_executable_safely("nvidia-smi")
    if not nvidia_smi_path:
        return False, []

    try:
        result = subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )

        compute_caps = result.stdout.strip().split("\n")
        has_sm120 = any("12.0" in cap for cap in compute_caps if cap.strip())

        return has_sm120, compute_caps
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False, []


# Find TensorFlow
def find_tensorflow():
    """Find TensorFlow installation and get compile flags."""
    try:
        import tensorflow as tf

        # Get TensorFlow version
        tf_version = tf.__version__
        tf_major, tf_minor = map(int, tf_version.split(".")[:2])

        if tf_major < 2 or (tf_major == 2 and tf_minor < 10):
            raise RuntimeError(f"TensorFlow 2.10+ required, found {tf_version}")

        # Get include directories
        include_dirs = [
            tf.sysconfig.get_include(),
            os.path.join(tf.sysconfig.get_include(), "external", "nsync", "public"),
        ]

        # Get library directories and flags
        lib_dirs = [tf.sysconfig.get_lib()]

        # Get compile flags
        compile_flags = tf.sysconfig.get_compile_flags()
        link_flags = tf.sysconfig.get_link_flags()

        return {
            "version": tf_version,
            "include_dirs": include_dirs,
            "lib_dirs": lib_dirs,
            "compile_flags": compile_flags,
            "link_flags": link_flags,
        }
    except ImportError:
        raise RuntimeError("TensorFlow not found. Please install TensorFlow 2.10+")


# Find CUDA toolkit
def find_cuda_toolkit():
    """Find CUDA toolkit installation."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if not cuda_home:
        # Try common installation paths
        common_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12.4",
            "/opt/cuda",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
        ]

        for path in common_paths:
            if os.path.exists(path):
                cuda_home = path
                break

    if not cuda_home or not os.path.exists(cuda_home):
        raise RuntimeError(
            "CUDA toolkit not found. Please set CUDA_HOME environment variable"
        )

    return {
        "home": cuda_home,
        "include_dir": os.path.join(cuda_home, "include"),
        "lib_dir": os.path.join(
            cuda_home, "lib64" if platform.system() != "Windows" else "lib\\x64"
        ),
    }


# Custom build extension class
class SM120BuildExt(build_ext):
    """Custom build extension for SM120 operations."""

    def build_extensions(self):
        # Check system requirements
        print("Checking system requirements...")

        cuda_available, cuda_version = check_cuda()
        if not cuda_available:
            raise RuntimeError(f"CUDA 12.4+ required, found: {cuda_version}")

        sm120_available, compute_caps = check_sm120_gpu()
        if sm120_available:
            print(
                f"✓ RTX 50-series GPU detected with compute capabilities: {compute_caps}"
            )
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
        ext.include_dirs.extend(tf_info["include_dirs"])
        ext.include_dirs.append(cuda_info["include_dir"])
        ext.include_dirs.append("src")

        # Library directories
        ext.library_dirs.extend(tf_info["lib_dirs"])
        ext.library_dirs.append(cuda_info["lib_dir"])

        # Libraries
        ext.libraries.extend(
            [
                "tensorflow_framework",
                "cudart",
                "cublas",
                "cudnn",
                "cufft",
                "curand",
                "cusolver",
                "cusparse",
            ]
        )

        # Compile flags
        compile_flags = [
            "-DGOOGLE_CUDA=1",
            "-DEIGEN_USE_GPU",
            "-DTENSORFLOW_USE_ROCM=0",
            "-std=c++17",
            "-O3",
            "-fPIC",
        ]

        if has_sm120:
            compile_flags.append("-DHAVE_SM120_GPU=1")

        # Platform-specific flags
        if platform.system() == "Linux":
            compile_flags.extend(["-march=native", "-mtune=native"])
        elif platform.system() == "Windows":
            compile_flags.extend(["/O2", "/DWIN32", "/D_WINDOWS"])

        ext.extra_compile_args.extend(compile_flags)

        # Link flags
        if platform.system() == "Linux":
            ext.extra_link_args.extend(["-Wl,--as-needed", "-Wl,--no-undefined"])


# Define extensions
def get_extensions():
    """Get list of extensions to build."""

    # Source files (include CUDA sources - nvcc handles .cu files)
    sources = [
        "src/python_bindings/sm120_python_ops.cc",
        "src/tensorflow_ops/sm120_ops_fixed.cc",
        "src/cuda_kernels/sm120_optimized_kernels_fixed.cu",
        "src/tensorflow_ops/sm120_kernel_implementations.cu",
    ]

    # Main extension
    ext = Pybind11Extension(
        "_sm120_ops",
        sources=sources,
        include_dirs=[
            pybind11.get_include(),
            "src",
        ],
        libraries=[],
        library_dirs=[],
        extra_compile_args=[],
        extra_link_args=[],
        language="c++",
    )

    return [ext]


# Read README for long description
def get_long_description():
    """Get long description from README."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return __description__


# Setup configuration
setup(
    name="tensorflow-sm120",
    version=__version__,
    author=__author__,
    author_email=__email__,
    url=__url__,
    description=__description__,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    # Package configuration
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=True,
    # Security: Explicitly specify package data to prevent unintended file inclusion
    package_data={
        "tensorflow_sm120": ["_sm120_ops.so", "_sm120_ops.dll", "_sm120_ops.dylib"]
    },
    # Extensions
    ext_modules=get_extensions(),
    cmdclass={"build_ext": SM120BuildExt},
    # Requirements
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "benchmark": [
            "matplotlib>=3.3.0",
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
        ],
    },
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="tensorflow gpu cuda rtx-50 sm-120 machine-learning deep-learning",
    # Entry points
    entry_points={
        "console_scripts": [
            "sm120-benchmark=tensorflow_sm120.benchmark:main",
            "sm120-validate=tensorflow_sm120.validate:main",
        ],
    },
    # Package data
    package_data={
        "tensorflow_sm120": ["*.so", "*.dll", "*.dylib"],
    },
    # Build requirements
    setup_requires=[
        "pybind11>=2.10.0",
        "wheel",
    ],
    zip_safe=False,
)


# Post-installation message
def print_installation_info():
    """Print installation information."""
    print("\n" + "=" * 60)
    print("TensorFlow SM120 Installation Complete!")
    print("=" * 60)
    print("Next steps:")
    print(
        "1. Validate installation: python -c "
        "'import tensorflow_sm120; print(tensorflow_sm120.is_sm120_available())'"
    )
    print("2. Run tests: python -m pytest tests/")
    print("3. Check examples: python examples/basic_usage.py")
    print("4. Benchmark performance: sm120-benchmark")
    print("\nFor documentation, visit:")
    print("https://github.com/yourusername/Custom-TensorFlow-compute-capability-sm-120")
    print("=" * 60)


if __name__ == "__main__":
    # Run setup
    try:
        # The setup() call is at the module level, so it runs automatically
        pass
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)

    # Print post-installation info if successful
    if len(sys.argv) > 1 and sys.argv[1] in ["install", "develop"]:
        print_installation_info()
