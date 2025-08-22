#!/bin/bash

# TensorFlow sm_120 Patch Application Script
# Applies all necessary patches for RTX 50-series GPU support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORFLOW_DIR="${1:-tensorflow}"
PATCHES_DIR="$SCRIPT_DIR"

# Patch definitions with descriptions
declare -A PATCHES=(
    ["fix-matrix-naming.patch"]="Fixes matrix function naming conflicts with LLVM 22"
    ["fix-template-errors.patch"]="Resolves template instantiation errors in GPU kernels"
    ["fix-c23-extensions.patch"]="Adds compiler flags to handle C23 extensions warnings"
    ["fix-sm120-support.patch"]="Adds explicit support for compute capability 12.0 (sm_120)"
)

# Check if TensorFlow directory exists
check_tensorflow_dir() {
    if [[ ! -d "$TENSORFLOW_DIR" ]]; then
        log_error "TensorFlow directory not found: $TENSORFLOW_DIR"
        log_info "Please clone TensorFlow first or specify the correct path"
        log_info "Usage: $0 [tensorflow_directory]"
        exit 1
    fi
    
    if [[ ! -d "$TENSORFLOW_DIR/.git" ]]; then
        log_error "Directory $TENSORFLOW_DIR is not a Git repository"
        exit 1
    fi
    
    log_success "TensorFlow directory found: $TENSORFLOW_DIR"
}

# Check if patch can be applied
can_apply_patch() {
    local patch_file="$1"
    local patch_path="$PATCHES_DIR/$patch_file"
    
    if [[ ! -f "$patch_path" ]]; then
        log_error "Patch file not found: $patch_path"
        return 1
    fi
    
    # Test if patch can be applied
    cd "$TENSORFLOW_DIR"
    if git apply --check "$patch_path" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Apply a single patch
apply_patch() {
    local patch_file="$1"
    local description="$2"
    local patch_path="$PATCHES_DIR/$patch_file"
    
    log_info "Applying patch: $patch_file"
    log_info "Description: $description"
    
    cd "$TENSORFLOW_DIR"
    
    # Check if patch can be applied
    if can_apply_patch "$patch_file"; then
        if git apply "$patch_path"; then
            log_success "Successfully applied: $patch_file"
            return 0
        else
            log_error "Failed to apply: $patch_file"
            return 1
        fi
    else
        # Check if patch is already applied
        if git apply --reverse --check "$patch_path" 2>/dev/null; then
            log_warning "Patch already applied: $patch_file"
            return 0
        else
            log_error "Cannot apply patch: $patch_file"
            log_info "This may be due to:"
            log_info "  - Different TensorFlow version"
            log_info "  - Files already modified"
            log_info "  - Conflicting changes"
            return 1
        fi
    fi
}

# Create backup of original files
create_backup() {
    log_info "Creating backup of original files..."
    
    cd "$TENSORFLOW_DIR"
    local backup_dir="../tensorflow-backup-$(date +%Y%m%d-%H%M%S)"
    
    # Create backup of key files that will be modified
    mkdir -p "$backup_dir"
    
    # Files that will be modified by patches
    local files_to_backup=(
        "third_party/xla/xla/service/gpu/fusions/triton/triton_support.cc"
        "tensorflow/core/kernels/gpu_utils.h"
        "tensorflow/core/kernels/cuda_solvers.h"
        "third_party/xla/xla/service/gpu/gpu_compiler.cc"
        "tensorflow/core/platform/default/build_config.bzl"
        "third_party/xla/xla/service/gpu/BUILD"
        "tensorflow/core/kernels/BUILD"
        "tensorflow/stream_executor/cuda/cuda_gpu_executor.cc"
        "third_party/xla/xla/service/gpu/nvptx_compiler.cc"
        "tensorflow/python/framework/config.py"
    )
    
    for file in "${files_to_backup[@]}"; do
        if [[ -f "$file" ]]; then
            local dir_path=$(dirname "$backup_dir/$file")
            mkdir -p "$dir_path"
            cp "$file" "$backup_dir/$file"
        fi
    done
    
    log_success "Backup created: $backup_dir"
    echo "$backup_dir" > .patch-backup-location
}

# Verify TensorFlow version compatibility
verify_tensorflow_version() {
    cd "$TENSORFLOW_DIR"
    
    local tf_version=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD)
    log_info "TensorFlow version: $tf_version"
    
    # Check for known compatible versions
    local compatible_versions=("r2.19" "r2.20" "r2.21" "main" "master")
    local is_compatible=false
    
    for version in "${compatible_versions[@]}"; do
        if [[ "$tf_version" == *"$version"* ]]; then
            is_compatible=true
            break
        fi
    done
    
    if [[ "$is_compatible" == false ]]; then
        log_warning "TensorFlow version $tf_version may not be compatible"
        log_warning "Tested versions: ${compatible_versions[*]}"
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Aborting patch application"
            exit 1
        fi
    else
        log_success "TensorFlow version is compatible"
    fi
}

# Apply all patches
apply_all_patches() {
    log_info "Applying all patches for sm_120 support..."
    
    local success_count=0
    local total_count=${#PATCHES[@]}
    local failed_patches=()
    
    # Apply patches in specific order
    local patch_order=(
        "fix-c23-extensions.patch"
        "fix-matrix-naming.patch"
        "fix-template-errors.patch"
        "fix-sm120-support.patch"
    )
    
    for patch_file in "${patch_order[@]}"; do
        local description="${PATCHES[$patch_file]}"
        
        if apply_patch "$patch_file" "$description"; then
            ((success_count++))
        else
            failed_patches+=("$patch_file")
        fi
        echo
    done
    
    log_info "Patch application summary:"
    log_info "  Total patches: $total_count"
    log_success "  Successfully applied: $success_count"
    
    if [[ ${#failed_patches[@]} -gt 0 ]]; then
        log_error "  Failed patches: ${#failed_patches[@]}"
        for patch in "${failed_patches[@]}"; do
            log_error "    - $patch"
        done
        
        log_warning "Some patches failed to apply. The build may not work correctly."
        log_info "You may need to apply these patches manually or use a different TensorFlow version."
        return 1
    else
        log_success "All patches applied successfully!"
        return 0
    fi
}

# Generate patch report
generate_report() {
    local report_file="$TENSORFLOW_DIR/sm120-patches-applied.txt"
    
    cat > "$report_file" << EOF
TensorFlow sm_120 Patches Applied
================================

Applied on: $(date)
TensorFlow version: $(cd "$TENSORFLOW_DIR" && git describe --tags 2>/dev/null || git rev-parse --short HEAD)
Patches directory: $PATCHES_DIR

Applied Patches:
EOF

    for patch_file in "${!PATCHES[@]}"; do
        echo "- $patch_file: ${PATCHES[$patch_file]}" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

Build Instructions:
1. Configure TensorFlow: ./configure
2. Build with Bazel: bazel build --config=opt --config=cuda //tensorflow:libtensorflow.so
3. Build Python wheel: ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tf_pkg

For more information, see the project README.md
EOF

    log_success "Patch report generated: $report_file"
}

# Restore from backup
restore_backup() {
    if [[ -f "$TENSORFLOW_DIR/.patch-backup-location" ]]; then
        local backup_dir=$(cat "$TENSORFLOW_DIR/.patch-backup-location")
        
        if [[ -d "$backup_dir" ]]; then
            log_info "Restoring from backup: $backup_dir"
            
            cd "$TENSORFLOW_DIR"
            cp -r "$backup_dir"/* .
            
            log_success "Backup restored successfully"
            rm -f .patch-backup-location
        else
            log_error "Backup directory not found: $backup_dir"
            exit 1
        fi
    else
        log_error "No backup location found"
        exit 1
    fi
}

# Show help
show_help() {
    cat << EOF
TensorFlow sm_120 Patch Application Script

Usage: $0 [options] [tensorflow_directory]

Options:
  --help, -h        Show this help message
  --list            List available patches
  --check           Check if patches can be applied (dry run)
  --restore         Restore from backup
  --force           Apply patches without version check

Examples:
  $0                          # Apply patches to ./tensorflow
  $0 /path/to/tensorflow      # Apply patches to specific directory
  $0 --check tensorflow       # Check if patches can be applied
  $0 --restore tensorflow     # Restore from backup

Available Patches:
EOF

    for patch_file in "${!PATCHES[@]}"; do
        echo "  $patch_file: ${PATCHES[$patch_file]}"
    done
}

# List patches
list_patches() {
    log_info "Available patches for sm_120 support:"
    echo
    
    for patch_file in "${!PATCHES[@]}"; do
        local patch_path="$PATCHES_DIR/$patch_file"
        
        echo "Patch: $patch_file"
        echo "Description: ${PATCHES[$patch_file]}"
        
        if [[ -f "$patch_path" ]]; then
            local line_count=$(wc -l < "$patch_path")
            echo "Size: $line_count lines"
        else
            echo "Status: File not found"
        fi
        echo
    done
}

# Check patches (dry run)
check_patches() {
    log_info "Checking if patches can be applied (dry run)..."
    
    local applicable_count=0
    local total_count=${#PATCHES[@]}
    
    for patch_file in "${!PATCHES[@]}"; do
        echo -n "Checking $patch_file... "
        
        if can_apply_patch "$patch_file"; then
            echo -e "${GREEN}OK${NC}"
            ((applicable_count++))
        else
            # Check if already applied
            cd "$TENSORFLOW_DIR"
            if git apply --reverse --check "$PATCHES_DIR/$patch_file" 2>/dev/null; then
                echo -e "${YELLOW}Already Applied${NC}"
                ((applicable_count++))
            else
                echo -e "${RED}Failed${NC}"
            fi
        fi
    done
    
    echo
    log_info "Check summary:"
    log_info "  Total patches: $total_count"
    log_info "  Applicable: $applicable_count"
    
    if [[ $applicable_count -eq $total_count ]]; then
        log_success "All patches can be applied successfully!"
        return 0
    else
        log_warning "Some patches cannot be applied"
        return 1
    fi
}

# Main function
main() {
    local check_only=false
    local restore_mode=false
    local force_apply=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --list)
                list_patches
                exit 0
                ;;
            --check)
                check_only=true
                ;;
            --restore)
                restore_mode=true
                ;;
            --force)
                force_apply=true
                ;;
            -*)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                TENSORFLOW_DIR="$1"
                ;;
        esac
        shift
    done
    
    # Check TensorFlow directory
    check_tensorflow_dir
    
    if [[ "$restore_mode" == true ]]; then
        restore_backup
        exit 0
    fi
    
    if [[ "$check_only" == true ]]; then
        check_patches
        exit $?
    fi
    
    # Verify TensorFlow version unless forced
    if [[ "$force_apply" == false ]]; then
        verify_tensorflow_version
    fi
    
    # Create backup
    create_backup
    
    # Apply patches
    if apply_all_patches; then
        generate_report
        log_success "sm_120 patches applied successfully!"
        log_info "Next steps:"
        log_info "  1. Configure TensorFlow: cd $TENSORFLOW_DIR && ./configure"
        log_info "  2. Build TensorFlow: bazel build --config=opt --config=cuda //tensorflow:libtensorflow.so"
        log_info "  3. Build Python wheel: ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tf_pkg"
    else
        log_error "Some patches failed to apply"
        log_info "Use --restore to revert changes if needed"
        exit 1
    fi
}

# Run main function
main "$@"
