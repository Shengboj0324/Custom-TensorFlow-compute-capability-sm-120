#!/bin/bash

# TensorFlow SM120 Master Build Launcher
# One-command deployment for RTX 50-series GPU optimizations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Project banner
print_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    TensorFlow SM120 Optimizations                           ║
║                                                                              ║
║                  🚀 RTX 50-Series GPU Acceleration 🚀                       ║
║                                                                              ║
║                    Production-Ready • Zero-Tolerance                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# System information display
show_system_info() {
    echo -e "${WHITE}System Information:${NC}"
    echo -e "${BLUE}OS:${NC} $(uname -s) $(uname -r)"
    echo -e "${BLUE}Architecture:${NC} $(uname -m)"
    echo -e "${BLUE}CPU Cores:${NC} $(nproc)"
    echo -e "${BLUE}Memory:${NC} $(free -h | grep '^Mem:' | awk '{print $2}') total"
    echo -e "${BLUE}Disk Space:${NC} $(df -h . | tail -1 | awk '{print $4}') available"
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}NVIDIA Driver:${NC} $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
        echo -e "${BLUE}GPUs Detected:${NC}"
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | while IFS=, read -r name compute_cap; do
            name=$(echo "$name" | xargs)
            compute_cap=$(echo "$compute_cap" | xargs)
            if [[ "$compute_cap" == "12.0" ]]; then
                echo -e "  ${GREEN}✓${NC} $name (sm_120) ${GREEN}[RTX 50-series Compatible]${NC}"
            else
                echo -e "  ${YELLOW}•${NC} $name (sm_$compute_cap)"
            fi
        done
    else
        echo -e "${RED}NVIDIA GPU:${NC} Not detected"
    fi
}

# Quick validation
quick_validate() {
    echo -e "\n${WHITE}Quick Validation:${NC}"
    
    local errors=0
    
    # Check essential tools
    local tools=("git" "python3" "pip3" "cmake" "make")
    for tool in "${tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} $tool"
        else
            echo -e "  ${RED}✗${NC} $tool (missing)"
            ((errors++))
        fi
    done
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        if (( $(echo "$cuda_version >= 12.4" | bc -l) 2>/dev/null )); then
            echo -e "  ${GREEN}✓${NC} CUDA $cuda_version"
        elif (( $(echo "$cuda_version >= 12.0" | bc -l) 2>/dev/null )); then
            echo -e "  ${YELLOW}⚠${NC} CUDA $cuda_version (12.4+ recommended for optimal performance)"
        else
            echo -e "  ${YELLOW}⚠${NC} CUDA $cuda_version (12.4+ recommended, may have compatibility issues)"
        fi
    else
        echo -e "  ${RED}✗${NC} CUDA (missing)"
        ((errors++))
    fi
    
    # Check Python
    if python3 -c "import tensorflow" 2>/dev/null; then
        local tf_version=$(python3 -c "import tensorflow as tf; print(tf.__version__)")
        echo -e "  ${GREEN}✓${NC} TensorFlow $tf_version"
    else
        echo -e "  ${YELLOW}⚠${NC} TensorFlow (will be installed)"
    fi
    
    return $errors
}

# Main menu
show_menu() {
    echo -e "\n${WHITE}Deployment Options:${NC}"
    echo -e "${CYAN}1.${NC} 🐳 Docker Build (Recommended - Guaranteed Success)"
    echo -e "${CYAN}2.${NC} 🔧 Native System Build (Maximum Performance)"
    echo -e "${CYAN}3.${NC} 📦 Python Package Only (If Dependencies Ready)"
    echo -e "${CYAN}4.${NC} 🧪 Validation and Testing Only"
    echo -e "${CYAN}5.${NC} 📊 System Diagnostics"
    echo -e "${CYAN}6.${NC} 📚 Documentation and Examples"
    echo -e "${CYAN}0.${NC} ❌ Exit"
    echo
}

# Docker build option
docker_build() {
    echo -e "${GREEN}Starting Docker build...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found. Please install Docker first.${NC}"
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}Docker daemon not running. Please start Docker.${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Building SM120 optimizations with Docker...${NC}"
    ./scripts/build-docker.sh
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Docker build completed successfully!${NC}"
        echo -e "${CYAN}Install with: pip install ./build/tensorflow-*sm120*.whl${NC}"
    else
        echo -e "${RED}❌ Docker build failed. Check logs for details.${NC}"
        return 1
    fi
}

# Native build option
native_build() {
    echo -e "${GREEN}Starting native system build...${NC}"
    
    echo -e "${BLUE}Setting up build environment...${NC}"
    ./scripts/setup-environment.sh
    
    if [[ $? -eq 0 ]]; then
        echo -e "${BLUE}Running comprehensive build...${NC}"
        ./scripts/comprehensive-build.sh
        
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✅ Native build completed successfully!${NC}"
            echo -e "${CYAN}Activate environment: source tf-sm120-env/bin/activate${NC}"
            echo -e "${CYAN}Install with: pip install ./build/tensorflow-*sm120*.whl${NC}"
        else
            echo -e "${RED}❌ Build failed. Check logs for details.${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Environment setup failed.${NC}"
        return 1
    fi
}

# Python package build
python_build() {
    echo -e "${GREEN}Building Python package only...${NC}"
    
    if ! python3 -c "import tensorflow" 2>/dev/null; then
        echo -e "${RED}TensorFlow not found. Please install TensorFlow 2.10+ first.${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Building Python extension...${NC}"
    python3 setup.py build_ext --inplace
    
    echo -e "${BLUE}Creating wheel...${NC}"
    python3 setup.py bdist_wheel
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Python package built successfully!${NC}"
        echo -e "${CYAN}Install with: pip install dist/tensorflow_sm120-*.whl${NC}"
    else
        echo -e "${RED}❌ Python build failed.${NC}"
        return 1
    fi
}

# Validation and testing
run_validation() {
    echo -e "${GREEN}Running validation and testing...${NC}"
    
    echo -e "${BLUE}System validation...${NC}"
    python3 scripts/validate-installation.py
    
    echo -e "${BLUE}Basic GPU test...${NC}"
    python3 examples/basic-gpu-test.py
    
    echo -e "${BLUE}Performance benchmark...${NC}"
    python3 examples/benchmark.py --quick
    
    echo -e "${GREEN}✅ Validation completed!${NC}"
}

# System diagnostics
run_diagnostics() {
    echo -e "${GREEN}Running comprehensive system diagnostics...${NC}"
    
    echo -e "\n${WHITE}Hardware Information:${NC}"
    show_system_info
    
    echo -e "\n${WHITE}CUDA Environment:${NC}"
    if command -v nvcc &> /dev/null; then
        nvcc --version
        echo
        nvidia-smi
    else
        echo -e "${RED}CUDA toolkit not found${NC}"
    fi
    
    echo -e "\n${WHITE}Python Environment:${NC}"
    python3 --version
    pip3 --version
    
    if python3 -c "import tensorflow" 2>/dev/null; then
        python3 -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'CUDA built: {tf.test.is_built_with_cuda()}')
print(f'GPU devices: {len(tf.config.list_physical_devices(\"GPU\"))}')
"
    else
        echo "TensorFlow: Not installed"
    fi
    
    echo -e "\n${WHITE}Build Dependencies:${NC}"
    local deps=("cmake" "make" "gcc" "g++" "clang")
    for dep in "${deps[@]}"; do
        if command -v "$dep" &> /dev/null; then
            local version=$($dep --version 2>/dev/null | head -1 || echo "unknown")
            echo -e "  ${GREEN}✓${NC} $dep: $version"
        else
            echo -e "  ${RED}✗${NC} $dep: Not found"
        fi
    done
}

# Documentation and examples
show_documentation() {
    echo -e "${GREEN}Documentation and Examples:${NC}"
    
    echo -e "\n${WHITE}Available Documentation:${NC}"
    echo -e "  📄 README.md - Project overview and quick start"
    echo -e "  📄 docs/build-guide.md - Complete build instructions"
    echo -e "  📄 docs/troubleshooting.md - Issue resolution guide"
    echo -e "  📄 docs/performance.md - Optimization strategies"
    echo -e "  📄 DEPLOYMENT_GUIDE.md - Production deployment"
    echo -e "  📄 FINAL_SUMMARY.md - Project completion summary"
    
    echo -e "\n${WHITE}Available Examples:${NC}"
    echo -e "  🐍 examples/basic_usage.py - Getting started examples"
    echo -e "  🧪 examples/basic-gpu-test.py - GPU functionality test"
    echo -e "  📊 examples/benchmark.py - Performance benchmarking"
    
    echo -e "\n${WHITE}Quick Commands:${NC}"
    echo -e "  View README: ${CYAN}cat README.md${NC}"
    echo -e "  Run basic test: ${CYAN}python3 examples/basic_usage.py${NC}"
    echo -e "  Run benchmarks: ${CYAN}python3 examples/benchmark.py${NC}"
    echo -e "  Validate system: ${CYAN}python3 scripts/validate-installation.py${NC}"
}

# Main execution
main() {
    # Clear screen and show banner
    clear
    print_banner
    
    # Show system information
    show_system_info
    
    # Quick validation
    echo
    if ! quick_validate; then
        echo -e "\n${YELLOW}⚠ Some dependencies are missing but the build may still succeed.${NC}"
        echo -e "${YELLOW}Docker build is recommended for guaranteed success.${NC}"
    else
        echo -e "\n${GREEN}✅ System appears ready for SM120 build!${NC}"
    fi
    
    # Main menu loop
    while true; do
        show_menu
        read -p "Select option [1-6, 0 to exit]: " choice
        
        case $choice in
            1)
                docker_build
                ;;
            2)
                native_build
                ;;
            3)
                python_build
                ;;
            4)
                run_validation
                ;;
            5)
                run_diagnostics
                ;;
            6)
                show_documentation
                ;;
            0)
                echo -e "${GREEN}Thank you for using TensorFlow SM120 optimizations!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please select 1-6 or 0 to exit.${NC}"
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Help function
show_help() {
    cat << EOF
TensorFlow SM120 Master Build Launcher

Usage: $0 [option]

Options:
  --docker          Run Docker build directly
  --native          Run native build directly
  --python          Build Python package only
  --validate        Run validation tests only
  --diagnostics     Show system diagnostics
  --help            Show this help message

Interactive mode (default):
  Run without arguments to enter interactive menu

Examples:
  $0                # Interactive mode
  $0 --docker       # Direct Docker build
  $0 --native       # Direct native build
  $0 --validate     # Validation only

For complete documentation, see:
  README.md - Project overview
  docs/build-guide.md - Detailed build instructions
  DEPLOYMENT_GUIDE.md - Production deployment guide
EOF
}

# Handle command line arguments
if [[ $# -gt 0 ]]; then
    case $1 in
        --docker)
            print_banner
            show_system_info
            docker_build
            ;;
        --native)
            print_banner
            show_system_info
            native_build
            ;;
        --python)
            print_banner
            python_build
            ;;
        --validate)
            print_banner
            run_validation
            ;;
        --diagnostics)
            print_banner
            run_diagnostics
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
else
    # Interactive mode
    main
fi
