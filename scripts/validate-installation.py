#!/usr/bin/env python3
"""
TensorFlow sm_120 Installation Validator

This script validates that TensorFlow is properly installed with RTX 50-series GPU support.
It performs comprehensive checks including GPU detection, compute capability verification,
and basic functionality tests.
"""

import sys

import subprocess

from typing import List, Dict, Optional


# Colors for terminal output
class Colors:
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


class ValidationResult:
    def __init__(
        self, name: str, passed: bool, message: str, details: Optional[Dict] = None
    ):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}


class TensorFlowValidator:
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.tf_available = False
        self.tf_version = None
        self.tf_module = None

    def add_result(self, result: ValidationResult) -> None:
        self.results.append(result)
        if result.passed:
            log_success(f"{result.name}: {result.message}")
        else:
            log_error(f"{result.name}: {result.message}")

    def check_python_version(self) -> ValidationResult:
        """Check if Python version is supported by TensorFlow."""
        version_info = sys.version_info
        version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

        if version_info.major == 3 and 9 <= version_info.minor <= 13:
            return ValidationResult(
                "Python Version",
                True,
                f"Python {version_str} is supported",
                {
                    "version": version_str,
                    "major": version_info.major,
                    "minor": version_info.minor,
                },
            )
        else:
            return ValidationResult(
                "Python Version",
                False,
                f"Python {version_str} is not supported (requires 3.9-3.13)",
                {
                    "version": version_str,
                    "major": version_info.major,
                    "minor": version_info.minor,
                },
            )

    def check_tensorflow_import(self) -> ValidationResult:
        """Check if TensorFlow can be imported."""
        try:
            import tensorflow as tf

            self.tf_module = tf
            self.tf_available = True
            self.tf_version = tf.__version__

            return ValidationResult(
                "TensorFlow Import",
                True,
                f"TensorFlow {tf.__version__} imported successfully",
                {"version": tf.__version__, "location": tf.__file__},
            )
        except ImportError as e:
            return ValidationResult(
                "TensorFlow Import",
                False,
                f"Failed to import TensorFlow: {str(e)}",
                {"error": str(e)},
            )

    def check_cuda_availability(self) -> ValidationResult:
        """Check if CUDA is available to TensorFlow."""
        if not self.tf_available:
            return ValidationResult(
                "CUDA Availability",
                False,
                "Cannot check CUDA (TensorFlow not available)",
                {},
            )

        try:
            cuda_available = self.tf_module.test.is_built_with_cuda()
            if cuda_available:
                return ValidationResult(
                    "CUDA Support",
                    True,
                    "TensorFlow was built with CUDA support",
                    {"cuda_built": True},
                )
            else:
                return ValidationResult(
                    "CUDA Support",
                    False,
                    "TensorFlow was not built with CUDA support",
                    {"cuda_built": False},
                )
        except Exception as e:
            return ValidationResult(
                "CUDA Support",
                False,
                f"Error checking CUDA support: {str(e)}",
                {"error": str(e)},
            )

    def check_gpu_devices(self) -> ValidationResult:
        """Check for available GPU devices."""
        if not self.tf_available:
            return ValidationResult(
                "GPU Devices",
                False,
                "Cannot check GPU devices (TensorFlow not available)",
                {},
            )

        try:
            gpus = self.tf_module.config.list_physical_devices("GPU")
            if gpus:
                gpu_info = []
                for i, gpu in enumerate(gpus):
                    try:
                        details = self.tf_module.config.experimental.get_device_details(
                            gpu
                        )
                        gpu_info.append(
                            {
                                "index": i,
                                "name": gpu.name,
                                "device_type": gpu.device_type,
                                "details": details,
                            }
                        )
                    except Exception as e:
                        gpu_info.append(
                            {
                                "index": i,
                                "name": gpu.name,
                                "device_type": gpu.device_type,
                                "error": str(e),
                            }
                        )

                return ValidationResult(
                    "GPU Devices",
                    True,
                    f"Found {len(gpus)} GPU device(s)",
                    {"count": len(gpus), "devices": gpu_info},
                )
            else:
                return ValidationResult(
                    "GPU Devices", False, "No GPU devices found", {"count": 0}
                )
        except Exception as e:
            return ValidationResult(
                "GPU Devices",
                False,
                f"Error checking GPU devices: {str(e)}",
                {"error": str(e)},
            )

    def check_compute_capability(self) -> ValidationResult:
        """Check if any GPU has compute capability 12.0 (sm_120)."""
        if not self.tf_available:
            return ValidationResult(
                "Compute Capability sm_120",
                False,
                "Cannot check compute capability (TensorFlow not available)",
                {},
            )

        try:
            gpus = self.tf_module.config.list_physical_devices("GPU")
            if not gpus:
                return ValidationResult(
                    "Compute Capability sm_120",
                    False,
                    "No GPU devices available to check",
                    {},
                )

            sm120_found = False
            capabilities = []

            for gpu in gpus:
                try:
                    details = self.tf_module.config.experimental.get_device_details(gpu)
                    compute_cap = details.get("compute_capability")
                    capabilities.append(
                        {"device": gpu.name, "compute_capability": compute_cap}
                    )

                    if compute_cap and (
                        compute_cap == (12, 0) or compute_cap == "12.0"
                    ):
                        sm120_found = True
                except Exception as e:
                    capabilities.append({"device": gpu.name, "error": str(e)})

            if sm120_found:
                return ValidationResult(
                    "Compute Capability sm_120",
                    True,
                    "Found GPU with compute capability 12.0 (sm_120)",
                    {"capabilities": capabilities, "sm120_found": True},
                )
            else:
                return ValidationResult(
                    "Compute Capability sm_120",
                    False,
                    "No GPU with compute capability 12.0 found",
                    {"capabilities": capabilities, "sm120_found": False},
                )
        except Exception as e:
            return ValidationResult(
                "Compute Capability sm_120",
                False,
                f"Error checking compute capability: {str(e)}",
                {"error": str(e)},
            )

    def check_nvidia_smi(self) -> ValidationResult:
        """Check nvidia-smi output for RTX 50-series GPUs."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []
                rtx50_found = False

                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            name, compute_cap, driver_version = (
                                parts[0],
                                parts[1],
                                parts[2],
                            )
                            gpus.append(
                                {
                                    "name": name,
                                    "compute_cap": compute_cap,
                                    "driver_version": driver_version,
                                }
                            )

                            # Check for RTX 50-series
                            if "RTX 50" in name or compute_cap == "12.0":
                                rtx50_found = True

                if rtx50_found:
                    return ValidationResult(
                        "NVIDIA RTX 50-series Detection",
                        True,
                        "RTX 50-series GPU detected via nvidia-smi",
                        {"gpus": gpus, "rtx50_found": True},
                    )
                else:
                    return ValidationResult(
                        "NVIDIA RTX 50-series Detection",
                        False,
                        "No RTX 50-series GPU detected",
                        {"gpus": gpus, "rtx50_found": False},
                    )
            else:
                return ValidationResult(
                    "NVIDIA RTX 50-series Detection",
                    False,
                    f"nvidia-smi failed: {result.stderr}",
                    {"error": result.stderr},
                )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                "NVIDIA RTX 50-series Detection",
                False,
                "nvidia-smi command timed out",
                {"error": "timeout"},
            )
        except FileNotFoundError:
            return ValidationResult(
                "NVIDIA RTX 50-series Detection",
                False,
                "nvidia-smi command not found",
                {"error": "command_not_found"},
            )
        except Exception as e:
            return ValidationResult(
                "NVIDIA RTX 50-series Detection",
                False,
                f"Error running nvidia-smi: {str(e)}",
                {"error": str(e)},
            )

    def check_basic_operations(self) -> ValidationResult:
        """Test basic TensorFlow operations on GPU."""
        if not self.tf_available:
            return ValidationResult(
                "Basic GPU Operations",
                False,
                "Cannot test operations (TensorFlow not available)",
                {},
            )

        try:
            # Check if GPUs are available
            gpus = self.tf_module.config.list_physical_devices("GPU")
            if not gpus:
                return ValidationResult(
                    "Basic GPU Operations", False, "No GPU available for testing", {}
                )

            # Test basic matrix multiplication on GPU
            with self.tf_module.device("/GPU:0"):
                # Create test matrices
                a = self.tf_module.constant([[1.0, 2.0], [3.0, 4.0]])
                b = self.tf_module.constant([[2.0, 0.0], [0.0, 2.0]])

                # Perform matrix multiplication
                c = self.tf_module.matmul(a, b)
                result = c.numpy()

                expected = [[2.0, 4.0], [6.0, 8.0]]

                if result.tolist() == expected:
                    return ValidationResult(
                        "Basic GPU Operations",
                        True,
                        "Matrix multiplication on GPU successful",
                        {"result": result.tolist(), "expected": expected},
                    )
                else:
                    return ValidationResult(
                        "Basic GPU Operations",
                        False,
                        "Matrix multiplication result incorrect",
                        {"result": result.tolist(), "expected": expected},
                    )
        except Exception as e:
            return ValidationResult(
                "Basic GPU Operations",
                False,
                f"Error during GPU operation test: {str(e)}",
                {"error": str(e)},
            )

    def run_all_checks(self) -> None:
        """Run all validation checks."""
        log_header("TensorFlow sm_120 Installation Validation")

        # System checks
        log_info("Checking system requirements...")
        self.add_result(self.check_python_version())

        # TensorFlow checks
        log_info("Checking TensorFlow installation...")
        self.add_result(self.check_tensorflow_import())
        self.add_result(self.check_cuda_availability())

        # GPU checks
        log_info("Checking GPU configuration...")
        self.add_result(self.check_gpu_devices())
        self.add_result(self.check_compute_capability())
        self.add_result(self.check_nvidia_smi())

        # Functionality checks
        log_info("Testing basic functionality...")
        self.add_result(self.check_basic_operations())

    def print_detailed_report(self) -> None:
        """Print a detailed validation report."""
        log_header("Detailed Validation Report")

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\n{Colors.WHITE}Overall Status: {Colors.NC}", end="")
        if passed == total:
            print(f"{Colors.GREEN}PASSED ({passed}/{total}){Colors.NC}")
        else:
            print(f"{Colors.RED}FAILED ({passed}/{total}){Colors.NC}")

        print(f"\n{Colors.WHITE}Test Results:{Colors.NC}")
        for result in self.results:
            status_color = Colors.GREEN if result.passed else Colors.RED
            status_text = "PASS" if result.passed else "FAIL"
            print(f"  {status_color}[{status_text}]{Colors.NC} {result.name}")
            print(f"        {result.message}")

            # Print additional details for failed tests
            if not result.passed and result.details:
                for key, value in result.details.items():
                    if key != "error":
                        print(f"        {key}: {value}")

        # GPU Information Summary
        gpu_results = [
            r for r in self.results if "GPU" in r.name or "Compute Capability" in r.name
        ]
        if any(r.passed for r in gpu_results):
            print(f"\n{Colors.WHITE}GPU Information Summary:{Colors.NC}")

            for result in self.results:
                if result.name == "GPU Devices" and result.passed:
                    devices = result.details.get("devices", [])
                    for device in devices:
                        print(f"  Device: {device.get('name', 'Unknown')}")
                        if "details" in device:
                            details = device["details"]
                            if "compute_capability" in details:
                                cc = details["compute_capability"]
                                if isinstance(cc, tuple):
                                    cc_str = f"{cc[0]}.{cc[1]}"
                                else:
                                    cc_str = str(cc)
                                print(f"    Compute Capability: {cc_str}")

    def print_recommendations(self) -> None:
        """Print recommendations based on validation results."""
        log_header("Recommendations")

        failed_results = [r for r in self.results if not r.passed]

        if not failed_results:
            log_success(
                "All checks passed! Your TensorFlow sm_120 installation is working correctly."
            )
            print(f"\n{Colors.WHITE}Next Steps:{Colors.NC}")
            print("  1. Try running some TensorFlow models to test performance")
            print("  2. Use the examples in the examples/ directory")
            print("  3. Monitor GPU utilization with nvidia-smi during training")
            return

        print(f"\n{Colors.WHITE}Issues Found:{Colors.NC}")

        for result in failed_results:
            print(f"\n{Colors.RED}• {result.name}{Colors.NC}")
            print(f"  Problem: {result.message}")

            # Provide specific recommendations
            if "TensorFlow Import" in result.name:
                print("  Solution: Install TensorFlow using the provided wheel:")
                print("    pip install path/to/tensorflow-*sm120*.whl")

            elif "CUDA Support" in result.name:
                print(
                    "  Solution: Ensure you're using the custom-built TensorFlow with CUDA support"
                )
                print("    The pre-built TensorFlow may not have CUDA enabled")

            elif "GPU Devices" in result.name:
                print("  Solution: Check NVIDIA drivers and CUDA installation:")
                print("    - Ensure NVIDIA drivers 570.x+ are installed")
                print("    - Verify CUDA 12.8+ runtime is available")
                print("    - Run 'nvidia-smi' to check GPU status")

            elif "Compute Capability" in result.name:
                print(
                    "  Solution: This may be expected if you don't have RTX 50-series GPU"
                )
                print(
                    "    The custom build will still work on other GPUs with reduced optimization"
                )

            elif "Basic GPU Operations" in result.name:
                print("  Solution: Check GPU memory and driver compatibility:")
                print("    - Ensure sufficient GPU memory is available")
                print("    - Restart the system if drivers were recently updated")


def main():
    """Main validation function."""
    validator = TensorFlowValidator()

    try:
        validator.run_all_checks()
        validator.print_detailed_report()
        validator.print_recommendations()

        # Exit with appropriate code
        failed_count = sum(1 for r in validator.results if not r.passed)
        if failed_count == 0:
            log_success("Validation completed successfully!")
            sys.exit(0)
        else:
            log_error(f"Validation failed with {failed_count} issues")
            sys.exit(1)

    except KeyboardInterrupt:
        log_warning("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error during validation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
