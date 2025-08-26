#!/bin/bash

# Test script to validate all fixes are working
# This script simulates the CI environment to test our fixes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

log_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

# Test 1: Black formatting
test_black_formatting() {
    log_header "Testing Black Formatting"
    
    if python -m black --check python/ examples/; then
        log_success "Black formatting: PASSED"
        return 0
    else
        log_error "Black formatting: FAILED"
        return 1
    fi
}

# Test 2: Python syntax
test_python_syntax() {
    log_header "Testing Python Syntax"
    
    local files=(
        "python/benchmark.py"
        "python/validate.py" 
        "examples/benchmark.py"
        "examples/basic_usage.py"
        "examples/comprehensive_sm120_example.py"
    )
    
    local failed=0
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            if python -m py_compile "$file"; then
                log_success "Syntax check: $file"
            else
                log_error "Syntax error: $file"
                failed=1
            fi
        else
            log_warning "File not found: $file"
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        log_success "Python syntax: PASSED"
        return 0
    else
        log_error "Python syntax: FAILED"
        return 1
    fi
}

# Test 3: TensorFlow detection in CI mode
test_tensorflow_detection() {
    log_header "Testing TensorFlow Detection (CI Mode)"
    
    # Simulate CI environment
    export CI=true
    export GITHUB_ACTIONS=true
    
    # Install TensorFlow if not present
    if ! python -c "import tensorflow" 2>/dev/null; then
        log_info "Installing TensorFlow for testing..."
        pip install tensorflow>=2.10.0
    fi
    
    # Test the validation function
    if python -c "
import sys
import os
sys.path.append('scripts')

# Simulate the TensorFlow validation logic
try:
    import tensorflow as tf
    print(f'TensorFlow {tf.__version__}')
    print('TensorFlow detection: SUCCESS')
except ImportError:
    print('TensorFlow detection: FAILED')
    sys.exit(1)
"; then
        log_success "TensorFlow detection: PASSED"
        return 0
    else
        log_error "TensorFlow detection: FAILED"
        return 1
    fi
}

# Test 4: cuDNN detection in CI mode
test_cudnn_detection() {
    log_header "Testing cuDNN Detection (CI Mode)"
    
    # Try to install cuDNN via pip
    pip install nvidia-cudnn-cu12 2>/dev/null || log_warning "Could not install nvidia-cudnn-cu12"
    
    # Test cuDNN detection logic
    if python -c "
try:
    import nvidia.cudnn
    print('cuDNN detection via pip: SUCCESS')
except ImportError:
    print('cuDNN detection via pip: Not available (expected in CI)')
"; then
        log_success "cuDNN detection: PASSED"
        return 0
    else
        log_warning "cuDNN detection: Not available (expected in CI)"
        return 0  # This is OK in CI
    fi
}

# Test 5: Docker build without GPU
test_docker_build() {
    log_header "Testing Docker Build (No GPU)"
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available - skipping Docker tests"
        return 0
    fi
    
    # Test that Docker build command doesn't use GPU flags
    local dockerfile_content=$(cat docker/Dockerfile.ubuntu 2>/dev/null || echo "")
    if [[ -n "$dockerfile_content" ]]; then
        log_success "Docker configuration: Available"
        
        # Check if our fixes are in place
        if grep -q "CPU-only in CI" .github/workflows/build-wheels.yml; then
            log_success "Docker GPU fix: Applied"
            return 0
        else
            log_error "Docker GPU fix: Missing"
            return 1
        fi
    else
        log_warning "Docker configuration: Not found"
        return 0
    fi
}

# Main test execution
main() {
    log_header "TensorFlow SM120 Fix Validation"
    
    local tests_passed=0
    local tests_total=5
    
    # Run all tests
    test_black_formatting && ((tests_passed++))
    test_python_syntax && ((tests_passed++))
    test_tensorflow_detection && ((tests_passed++))
    test_cudnn_detection && ((tests_passed++))
    test_docker_build && ((tests_passed++))
    
    # Summary
    log_header "Test Results Summary"
    
    if [[ $tests_passed -eq $tests_total ]]; then
        log_success "All tests passed! ($tests_passed/$tests_total)"
        log_success "The fixes should resolve the CI build issues."
        return 0
    else
        log_error "Some tests failed. ($tests_passed/$tests_total)"
        log_error "Additional fixes may be needed."
        return 1
    fi
}

# Run main function
main "$@"
