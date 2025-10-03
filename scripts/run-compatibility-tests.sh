#!/bin/bash

# Hardware Compatibility Test Runner for TensorFlow SM120
# This script runs comprehensive hardware compatibility tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "  TensorFlow SM120 Hardware Compatibility Testing Suite"
    echo "=================================================================="
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Python version: $python_version"
    
    # Check if we're in the right directory
    if [[ ! -f "setup.py" ]] || [[ ! -d "src" ]]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check if TensorFlow is available
    if python3 -c "import tensorflow" 2>/dev/null; then
        tf_version=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
        log_info "TensorFlow version: $tf_version"
    else
        log_warning "TensorFlow not available - some tests will be skipped"
    fi
    
    # Check NVIDIA tools
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA tools available"
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || log_warning "GPU query failed"
    else
        log_warning "nvidia-smi not found - GPU tests may be limited"
    fi
    
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        log_info "CUDA version: $cuda_version"
    else
        log_warning "CUDA compiler not found"
    fi
}

# Run hardware compatibility tests
run_tests() {
    log_info "Starting hardware compatibility tests..."
    
    # Set up Python path
    export PYTHONPATH="${PWD}/python:${PYTHONPATH}"
    
    # Run the test script
    if python3 scripts/test-hardware-compatibility.py; then
        log_success "Hardware compatibility tests completed successfully"
        return 0
    else
        log_error "Hardware compatibility tests failed"
        return 1
    fi
}

# Generate test report
generate_report() {
    log_info "Generating test report..."
    
    if [[ -f "hardware_compatibility_results.json" ]]; then
        # Create a simple HTML report
        cat > hardware_compatibility_report.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>TensorFlow SM120 Hardware Compatibility Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { background-color: #d4edda; border-color: #c3e6cb; }
        .warning { background-color: #fff3cd; border-color: #ffeaa7; }
        .error { background-color: #f8d7da; border-color: #f5c6cb; }
        .code { background-color: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TensorFlow SM120 Hardware Compatibility Report</h1>
        <p>Generated on: $(date)</p>
    </div>
    
    <div class="section">
        <h2>Test Results Summary</h2>
        <p>Detailed results are available in <code>hardware_compatibility_results.json</code></p>
    </div>
    
    <div class="section">
        <h2>System Information</h2>
        <div class="code">
            <strong>OS:</strong> $(uname -a)<br>
            <strong>Python:</strong> $(python3 --version)<br>
            <strong>CUDA:</strong> $(nvcc --version 2>/dev/null | grep "release" || echo "Not available")<br>
            <strong>Driver:</strong> $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null || echo "Not available")
        </div>
    </div>
    
    <div class="section">
        <h2>GPU Information</h2>
        <div class="code">
EOF

        # Add GPU information
        if command -v nvidia-smi &> /dev/null; then
            echo "            <strong>GPUs Detected:</strong><br>" >> hardware_compatibility_report.html
            nvidia-smi --query-gpu=index,name,compute_cap --format=csv 2>/dev/null | sed 's/^/            /' | sed 's/$/\<br\>/' >> hardware_compatibility_report.html || echo "            GPU query failed<br>" >> hardware_compatibility_report.html
        else
            echo "            No NVIDIA GPUs detected or nvidia-smi not available<br>" >> hardware_compatibility_report.html
        fi

        cat >> hardware_compatibility_report.html << 'EOF'
        </div>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            <li>For RTX 50-series GPUs: Ensure CUDA 12.8+ and cuDNN 9.8+ are installed</li>
            <li>For older GPUs: SM120 operations will fallback to standard TensorFlow implementations</li>
            <li>Check the JSON results file for detailed performance metrics</li>
        </ul>
    </div>
</body>
</html>
EOF

        log_success "HTML report generated: hardware_compatibility_report.html"
    else
        log_warning "No test results file found - report generation skipped"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup tasks here
}

# Main execution
main() {
    print_banner
    
    # Set up error handling
    trap cleanup EXIT
    
    check_prerequisites
    
    if run_tests; then
        generate_report
        log_success "All tests completed successfully!"
        echo ""
        echo "📊 Results available in:"
        echo "  - hardware_compatibility_results.json (detailed JSON)"
        echo "  - hardware_compatibility_report.html (summary report)"
        echo ""
        echo "🚀 Next steps:"
        echo "  - Review the test results"
        echo "  - If SM120 is available, run performance benchmarks"
        echo "  - If fallback mode, verify standard TensorFlow operations work correctly"
        exit 0
    else
        log_error "Tests failed - check the output above for details"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --quiet, -q   Run in quiet mode"
        echo ""
        echo "This script runs comprehensive hardware compatibility tests for TensorFlow SM120."
        exit 0
        ;;
    --quiet|-q)
        exec > /dev/null 2>&1
        ;;
esac

# Run main function
main "$@"
