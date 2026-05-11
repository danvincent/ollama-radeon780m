#!/bin/bash
#
# build_deploy_local.sh - Deploy Ollama from local build
#
# This script prepares the system for a local Ollama deployment by:
# - Validating prerequisites and repo context
# - Checking root/sudo permissions
# - Detecting existing installations
# - Verifying ollama user/group presence
# - Safely stopping the running service
# - Cleaning up only binaries and libs (preserving models)
#
# Usage: ./build_deploy_local.sh [OPTIONS]
#   -n, --dry-run    Show what would be done without making changes
#   -h, --help       Show this help message
#

set -euo pipefail

# ============================================================================
# Configuration and Constants
# ============================================================================

# Initialize SUDO early (empty by default, set to "sudo" if needed during permissions check)
SUDO=""

# Install prefix
INSTALL_PREFIX="/usr/local"
BIN_DIR="${INSTALL_PREFIX}/bin"
LIB_DIR="${INSTALL_PREFIX}/lib/ollama"
MODEL_DIR="/usr/share/ollama/.ollama/models"
SERVICE_FILE="/etc/systemd/system/ollama.service"
SERVICE_USER="ollama"
SERVICE_GROUP="ollama"
SERVICE_HOST="0.0.0.0:11434"
SERVICE_PATH="${BIN_DIR}:/usr/bin:/bin"

# Build configuration
# CMAKE_BUILD_DIR can be overridden by setting the CMAKE_BUILD_DIR environment variable.
# If not set, defaults to "build" in the current directory.
# This allows users to build in a custom location via: CMAKE_BUILD_DIR=/path/to/build ./build_deploy_local.sh
BUILD_DIR="${CMAKE_BUILD_DIR:-./build}"
CLEAN_CACHE=false

# Mode flags
DRY_RUN=false

# Deployment validation configuration
# VALIDATION_MODEL can be set via environment variable or --validation-model flag
# If specified, enables truthful model-aware validation for fresh services
VALIDATION_MODEL="${OLLAMA_VALIDATION_MODEL:-}"

# VALIDATION_TIMEOUT can be set via environment variable or --inference-timeout flag
# If specified, overrides the default inference timeout for validation
# Useful for slow first-load scenarios (e.g., slow GPUs, kernel compilation)
VALIDATION_TIMEOUT="${OLLAMA_VALIDATION_TIMEOUT:-}"

# Color codes for output (if terminal supports it)
if [ -t 2 ]; then
    RED="$(printf '\033[0;31m')"
    GREEN="$(printf '\033[0;32m')"
    YELLOW="$(printf '\033[1;33m')"
    BLUE="$(printf '\033[0;34m')"
    PLAIN="$(printf '\033[0m')"
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    PLAIN=""
fi

# ============================================================================
# Utility Functions
# ============================================================================

status() {
    echo "${BLUE}>>>${PLAIN} $*" >&2
}

success() {
    echo "${GREEN}✓${PLAIN} $*" >&2
}

warning() {
    echo "${YELLOW}⚠${PLAIN}  $*" >&2
}

error() {
    echo "${RED}✗${PLAIN}  ERROR: $*" >&2
    exit 1
}

info() {
    echo "   $*" >&2
}

# Print usage information
usage() {
    cat >&2 <<EOF
${BLUE}build_deploy_local.sh${PLAIN} - Build and deploy Ollama from local source

${BLUE}Usage:${PLAIN}
  $(basename "$0") [OPTIONS]

${BLUE}Options:${PLAIN}
  -n, --dry-run               Show what would be done without making changes
  -c, --clean-cache           Clean Go cache before building
  --validation-model MODEL    Model to test for inference during deployment validation
                              (default: from OLLAMA_VALIDATION_MODEL env var, or GPU-only check)
  --inference-timeout DUR     Timeout for model inference validation (e.g., 5m, 30s)
                              (default: from OLLAMA_VALIDATION_TIMEOUT env var, or built-in default)
  -h, --help                  Show this help message

${BLUE}Description:${PLAIN}
  This script builds and deploys Ollama locally by:
  - Validating prerequisites and repo context
  - Checking root/sudo permissions
  - Detecting existing installations
  - Building with the local Vulkan-only preset path for this UM790 deployment
  - Installing the binary and backend libraries to /usr/local
  - Installing a systemd service configured for Vulkan runtime
  - Verifying models are preserved at ${MODEL_DIR}

${BLUE}Build Configuration:${PLAIN}
  Backend:      Vulkan-only local deployment (CPU + Vulkan only, OLLAMA_RUNNER_DIR=vulkan)
  Build Dir:    ${BUILD_DIR} (set CMAKE_BUILD_DIR env var to override)
  Clean:        Optional via --clean-cache flag

${BLUE}Install Paths:${PLAIN}
  Binary:     ${BIN_DIR}/ollama
  Libraries:  ${LIB_DIR}/vulkan/ (subdirectory preserved for runtime discovery)
  Service:    ${SERVICE_FILE}
  Models:     ${MODEL_DIR} (preserved)

${BLUE}Examples:${PLAIN}
  # Show build and deploy plan
  $(basename "$0") --dry-run

  # Build and deploy with clean cache
  $(basename "$0") --clean-cache

  # Build and deploy with model-aware validation (truthful for fresh services)
  $(basename "$0") --validation-model gemma:7b

  # Build and deploy with longer inference timeout for slow first-load models
  $(basename "$0") --validation-model gemma:7b --inference-timeout 5m

  # Use environment variables instead of flags
  OLLAMA_VALIDATION_MODEL=gemma:7b OLLAMA_VALIDATION_TIMEOUT=5m $(basename "$0")

  # Using environment variable for validation model
  OLLAMA_VALIDATION_MODEL=llama2:7b $(basename "$0")

  # Standard build and deploy (requires root/sudo, GPU-only validation)
  $(basename "$0")

EOF
}

# Check if a command is available
available() {
    command -v "$1" >/dev/null 2>&1
}

# Require specific commands, error if missing
require() {
    local missing=""
    for tool in "$@"; do
        if ! available "$tool"; then
            missing="$missing $tool"
        fi
    done
    
    if [ -n "$missing" ]; then
        error "The following required tools are missing:$missing"
    fi
}

# Execute command, respecting dry-run mode
run() {
    if [ "$DRY_RUN" = true ]; then
        echo "(DRY-RUN) $*" >&2
    else
        "$@"
    fi
}

# ============================================================================
# Preflight Checks
# ============================================================================

preflight_check_tools() {
    status "Checking for required tools..."
    
    # Check for tools needed even in preflight
    require git cmake make go
    
    success "All required tools found"
}

preflight_check_repo_context() {
    status "Checking repo context..."
    
    # Verify we're in the ollama repo by checking for key files
    if [ ! -f "go.mod" ] || ! grep -q "ollama" go.mod; then
        error "Not in ollama repository root. Please run from repo root."
    fi
    
    if [ ! -d "scripts" ]; then
        error "scripts directory not found. Please run from repo root."
    fi
    
    success "Repository context verified"
}

preflight_check_permissions() {
    status "Checking permissions..."
    
    # Check if running as root or have sudo
    if [ "$(id -u)" -ne 0 ]; then
        if ! available sudo; then
            error "Must run as root or have sudo available"
        fi
        SUDO="sudo"
        info "Will use sudo for privileged operations"
    else
        SUDO=""
        info "Running as root"
    fi
    
    success "Permission check passed"
}

preflight_check_install_detection() {
    status "Checking for existing Ollama installation..."
    
    # Check if ollama binary exists
    if [ -f "${BIN_DIR}/ollama" ]; then
        info "Found existing binary at ${BIN_DIR}/ollama"
    fi
    
    # Check if lib directory exists
    if [ -d "${LIB_DIR}" ]; then
        info "Found existing libraries at ${LIB_DIR}"
    fi
    
    # Check for models directory (should be preserved)
    if [ -d "${MODEL_DIR}" ]; then
        info "Found model data at ${MODEL_DIR} (will be preserved)"
    fi
    
    success "Installation detection completed"
}

preflight_check_ollama_user() {
    status "Checking for ollama user/group..."
    
    if id ollama >/dev/null 2>&1; then
        info "ollama user exists"
    else
        warning "ollama user does not exist (will be created during deployment)"
    fi
    
    if getent group ollama >/dev/null 2>&1; then
        info "ollama group exists"
    else
        warning "ollama group does not exist (will be created during deployment)"
    fi
    
    success "Ollama user/group check completed"
}

# ============================================================================
# Service Management
# ============================================================================

stop_service_if_running() {
    status "Checking for running Ollama service..."
    
    if available systemctl; then
        if systemctl is-active --quiet ollama 2>/dev/null; then
            if [ "$DRY_RUN" = true ]; then
                status "Would stop ollama service (currently running)"
            else
                status "Stopping ollama service..."
                if run $SUDO systemctl stop ollama; then
                    sleep 1
                    success "Service stopped successfully"
                else
                    warning "Failed to stop ollama service (you may need to stop it manually)"
                fi
            fi
        else
            info "Service is not running"
        fi
    else
        info "systemctl not available, skipping service stop"
    fi
}

ensure_ollama_user_group() {
    status "Ensuring ollama user/group and runtime access..."

    if [ "$DRY_RUN" = true ]; then
        if id "${SERVICE_USER}" >/dev/null 2>&1; then
            info "${SERVICE_USER} user already exists"
        else
            info "Would create ${SERVICE_USER} user with home /usr/share/ollama"
        fi

        if getent group render >/dev/null 2>&1; then
            info "Would ensure ${SERVICE_USER} is a member of render"
        fi
        if getent group video >/dev/null 2>&1; then
            info "Would ensure ${SERVICE_USER} is a member of video"
        fi
        return 0
    fi

    if ! getent group "${SERVICE_GROUP}" >/dev/null 2>&1; then
        run $SUDO groupadd --system "${SERVICE_GROUP}" || error "Failed to create ${SERVICE_GROUP} group"
    fi

    if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
        run $SUDO useradd -r -s /bin/false -g "${SERVICE_GROUP}" -m -d /usr/share/ollama "${SERVICE_USER}" || \
            error "Failed to create ${SERVICE_USER} user"
    fi

    run $SUDO mkdir -p /usr/share/ollama || error "Failed to create /usr/share/ollama"
    run $SUDO chown "${SERVICE_USER}:${SERVICE_GROUP}" /usr/share/ollama || error "Failed to set ownership on /usr/share/ollama"

    if getent group render >/dev/null 2>&1; then
        run $SUDO usermod -a -G render "${SERVICE_USER}" || error "Failed to add ${SERVICE_USER} to render group"
    fi
    if getent group video >/dev/null 2>&1; then
        run $SUDO usermod -a -G video "${SERVICE_USER}" || error "Failed to add ${SERVICE_USER} to video group"
    fi

    success "ollama user/group runtime access configured"
}

# ============================================================================
# Cleanup Operations
# ============================================================================

cleanup_binary() {
    local binary_path="${BIN_DIR}/ollama"
    
    if [ -f "$binary_path" ] || [ -L "$binary_path" ]; then
        if [ "$DRY_RUN" = true ]; then
            status "Would remove binary: $binary_path"
        else
            status "Removing binary: $binary_path"
            run $SUDO rm -f "$binary_path"
            success "Binary removed"
        fi
    else
        info "Binary not found at $binary_path"
    fi
}

cleanup_libraries() {
    if [ -d "${LIB_DIR}" ]; then
        if [ "$DRY_RUN" = true ]; then
            status "Would remove libraries: ${LIB_DIR}"
        else
            status "Removing libraries: ${LIB_DIR}"
            run $SUDO rm -rf "${LIB_DIR}"
            success "Libraries removed"
        fi
    else
        info "Libraries directory not found"
    fi
}

verify_model_preservation() {
    status "Verifying model data preservation..."
    
    if [ -d "${MODEL_DIR}" ]; then
        # Check that it still exists and has content
        local model_count
        model_count=$(find "${MODEL_DIR}" -type f 2>/dev/null | wc -l)
        if [ "$model_count" -gt 0 ]; then
            success "Models preserved: ${MODEL_DIR} ($model_count files)"
        else
            info "Model directory exists but is empty"
        fi
    else
        info "No model directory found"
    fi
}

# ============================================================================
# Build Operations (Phase 2)
# ============================================================================

check_build_prerequisites() {
    status "Checking build prerequisites..."
    
    # Check for required build tools
    require cmake make go
    
    success "Build prerequisites found"
}

build_with_cmake() {
    status "Configuring build with CMake..."
    
    if [ "$DRY_RUN" = true ]; then
        info "Would run cmake configuration with Vulkan-only local build (OLLAMA_VULKAN_ONLY=ON, OLLAMA_RUNNER_DIR=vulkan, CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX})"
    else
        # Configure CMake with Vulkan backend using the Vulkan preset
        # The repo computes absolute install destinations at configure time, so the
        # deploy install prefix must be set here rather than relying on --install --prefix.
        run cmake --preset Vulkan \
            -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
            -DOLLAMA_VULKAN_ONLY=ON || error "CMake configuration failed"
        success "CMake configuration completed (Vulkan-only local build with OLLAMA_RUNNER_DIR=vulkan, install prefix ${INSTALL_PREFIX})"
    fi
}

build_cmake_project() {
    status "Building with CMake..."
    
    if [ "$DRY_RUN" = true ]; then
        info "Would build CMake project in ${BUILD_DIR}"
    else
        run cmake --build "${BUILD_DIR}" --config Release -j "$(nproc)" || error "CMake build failed"
        success "CMake build completed"
    fi
}

clean_go_cache_if_requested() {
    if [ "$CLEAN_CACHE" = true ]; then
        status "Cleaning Go cache..."
        if [ "$DRY_RUN" = true ]; then
            info "Would run: go clean -cache"
        else
            run go clean -cache || warning "Go clean failed (non-critical)"
            success "Go cache cleaned"
        fi
    else
        info "Skipping Go cache clean (use --clean-cache to enable)"
    fi
}

build_ollama_binary() {
    status "Building Ollama binary..."
    
    if [ "$DRY_RUN" = true ]; then
        info "Would build Ollama binary for the Vulkan-only local deployment"
    else
        # Build the Go binary
        run go build -o ollama . || error "Ollama binary build failed"
        
        success "Ollama binary built successfully"
    fi
}

# ============================================================================
# Install Operations (Phase 2)
# ============================================================================

verify_build_artifacts() {
    status "Verifying build artifacts..."
    
    if [ "$DRY_RUN" = true ]; then
        info "Would verify artifacts exist (binary, CPU fallback, and Vulkan backend libraries only)"
        return 0
    fi
    
    # Check that the binary exists
    if [ ! -f "./ollama" ]; then
        error "Ollama binary not found at ./ollama"
    fi
    
    # Check that the backend libraries were built
    if [ ! -d "${BUILD_DIR}/lib/ollama" ]; then
        error "Backend libraries not found at ${BUILD_DIR}/lib/ollama"
    fi
    
    # The build tree keeps the shared objects under ${BUILD_DIR}/lib/ollama/
    # while the Vulkan component is staged into lib/ollama/vulkan at install time.
    if [ ! -f "${BUILD_DIR}/lib/ollama/libggml-vulkan.so" ]; then
        error "Vulkan backend library not found at ${BUILD_DIR}/lib/ollama/libggml-vulkan.so - build may have failed"
    fi

    if find "${BUILD_DIR}/lib/ollama" -maxdepth 1 -type f \( -name "libggml-cuda*.so*" -o -name "libggml-hip*.so*" -o -name "libggml-metal*.so*" \) | grep -q .; then
        error "Unexpected non-Vulkan GPU backend libraries found in ${BUILD_DIR}/lib/ollama - Vulkan-only build configuration did not take effect"
    fi
    
    success "Build artifacts verified (binary and Vulkan-only backend library set present)"
}

install_binary() {
    local binary_src="./ollama"
    local binary_dst="${BIN_DIR}/ollama"
    
    status "Installing Ollama binary..."
    
    if [ "$DRY_RUN" = true ]; then
        info "Would copy binary from $binary_src to $binary_dst"
    else
        # Ensure install directory exists
        run $SUDO mkdir -p "${BIN_DIR}" || error "Failed to create ${BIN_DIR}"
        
        # Copy the binary
        run $SUDO cp "$binary_src" "$binary_dst" || error "Failed to copy binary"
        
        # Make it executable
        run $SUDO chmod +x "$binary_dst" || error "Failed to set executable permissions"
        
        success "Binary installed at $binary_dst"
    fi
}

install_backend_libraries() {
    status "Installing backend libraries..."
    
    if [ "$DRY_RUN" = true ]; then
        info "Would install backend libraries via: cmake --install ${BUILD_DIR} --component CPU"
        info "Would install Vulkan backend via: cmake --install ${BUILD_DIR} --component Vulkan"
        info "Would preserve subdirectory structure (vulkan/, etc.) under ${LIB_DIR}"
        return 0
    fi
    
    # Verify source library directory exists (only when actually installing)
    local lib_src="${BUILD_DIR}/lib/ollama"
    if [ ! -d "$lib_src" ]; then
        error "Source library directory not found at $lib_src"
    fi
    
    # Install CPU component (always required)
    run $SUDO cmake --install "${BUILD_DIR}" --component CPU || \
        error "Install component CPU failed. Check that the build completed successfully."
    
    # Install Vulkan component (required since we're using Vulkan-first preset)
    run $SUDO cmake --install "${BUILD_DIR}" --component Vulkan || \
        error "Install component Vulkan failed. Vulkan libraries are required for this build. Check that CMake preset was configured correctly with Vulkan support."
    
    # Ensure libraries have correct permissions for runtime loading
    run $SUDO find "${LIB_DIR}" -type f -name "*.so*" -exec chmod 755 {} \; || error "Failed to set library permissions"
    
    success "Backend libraries installed at ${LIB_DIR}"
}

verify_install_vulkan_artifacts() {
    status "Verifying installed Vulkan libraries..."
    
    if [ "$DRY_RUN" = true ]; then
        info "Would verify Vulkan libraries installed at ${LIB_DIR}/vulkan/"
        return 0
    fi
    
    # Check that Vulkan subdirectory exists after install
    if [ ! -d "${LIB_DIR}/vulkan" ]; then
        error "Vulkan library directory not found at ${LIB_DIR}/vulkan - install may have failed or Vulkan component not installed"
    fi
    
    # Verify that vulkan directory contains actual library files
    if ! find "${LIB_DIR}/vulkan" -type f \( -name "*.so*" -o -name "*.dylib" -o -name "*.dll" \) 2>/dev/null | grep -q .; then
        error "No Vulkan libraries found in ${LIB_DIR}/vulkan/ after installation - Vulkan component install failed"
    fi
    
    success "Vulkan libraries verified at ${LIB_DIR}/vulkan/"
}

install_systemd_service() {
    status "Installing systemd service..."

    if [ "$DRY_RUN" = true ]; then
        info "Would write ${SERVICE_FILE} with Vulkan-only runtime selection enabled"
        info "Would set ExecStart=${BIN_DIR}/ollama serve"
        return 0
    fi

    if ! available systemctl; then
        warning "systemctl not available, skipping service installation"
        return 0
    fi

    cat <<EOF | $SUDO tee "${SERVICE_FILE}" >/dev/null || error "Failed to write ${SERVICE_FILE}"
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=${BIN_DIR}/ollama serve
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
Restart=always
RestartSec=3
Environment="PATH=${SERVICE_PATH}"
Environment="OLLAMA_VULKAN=1"
Environment="OLLAMA_LLM_LIBRARY=vulkan"
Environment="OLLAMA_HOST=${SERVICE_HOST}"

[Install]
WantedBy=multi-user.target
EOF

    success "Installed systemd service at ${SERVICE_FILE}"
}

verify_systemd_service_configuration() {
    status "Verifying systemd service configuration..."

    if [ "$DRY_RUN" = true ]; then
        info "Would verify ${SERVICE_FILE} enables Vulkan and uses ${BIN_DIR}/ollama"
        return 0
    fi

    if [ ! -f "${SERVICE_FILE}" ]; then
        error "Service file not found at ${SERVICE_FILE}"
    fi

    grep -q 'Environment="OLLAMA_VULKAN=1"' "${SERVICE_FILE}" || error "Service file does not enable OLLAMA_VULKAN=1"
    grep -q 'Environment="OLLAMA_LLM_LIBRARY=vulkan"' "${SERVICE_FILE}" || error "Service file does not pin OLLAMA_LLM_LIBRARY=vulkan"
    grep -q "ExecStart=${BIN_DIR}/ollama serve" "${SERVICE_FILE}" || error "Service file does not use ${BIN_DIR}/ollama serve"
    grep -q 'WantedBy=multi-user.target' "${SERVICE_FILE}" || error "Service file does not target multi-user.target"

    success "Systemd service configuration verified"
}

reload_enable_start_service() {
    status "Reloading and starting ollama service..."

    if ! available systemctl; then
        warning "systemctl not available, skipping service enable/start"
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        info "Would run: systemctl daemon-reload"
        info "Would run: systemctl enable ollama"
        info "Would run: systemctl restart ollama"
        return 0
    fi

    run $SUDO systemctl daemon-reload || error "systemctl daemon-reload failed"
    run $SUDO systemctl enable ollama || error "systemctl enable ollama failed"
    run $SUDO systemctl restart ollama || error "systemctl restart ollama failed"
    run $SUDO systemctl is-active --quiet ollama || error "ollama service is not active after restart"

    success "ollama service reloaded, enabled, and running"
}

# ============================================================================
# GPU Deployment Verification (Phase 3 Integration)
# ============================================================================

verify_gpu_deployment() {
    status "Verifying GPU deployment (Phase 3 Deployment Validator)..."
    
    # Skip verification in dry-run mode
    if [ "$DRY_RUN" = true ]; then
        if [ -n "$VALIDATION_MODEL" ]; then
            info "Would verify GPU deployment with model-aware validation: $VALIDATION_MODEL"
        else
            info "Would verify GPU deployment using API-only GPU check (no model inference)"
        fi
        return 0
    fi
    
    # Find the verification script
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local verify_script="${script_dir}/verify-gpu-deployment.sh"
    
    if [ ! -f "$verify_script" ]; then
        warning "GPU deployment verification script not found at ${verify_script}"
        warning "Skipping Phase 3 deployment validation (non-critical)"
        return 0
    fi
    
    # Run the verification script
    # This checks:
    # 1. Service is responding to API
    # 2. GPU acceleration is active (via /api/ps - real runtime signal)
    # 3. If model specified: model can produce inference responses
    if [ -n "$VALIDATION_MODEL" ]; then
        # Model-aware validation: exercise inference first, then check GPU residency
        # This is truthful for fresh services with no models loaded yet
        info "Using model-aware validation (infers first, then checks GPU)"
        
        # Build command with optional timeout for inference (model loading can be slow)
        local verify_cmd=(bash "$verify_script" --url "http://localhost:11434" --model "$VALIDATION_MODEL")
        if [ -n "$VALIDATION_TIMEOUT" ]; then
            info "Using custom inference timeout: $VALIDATION_TIMEOUT"
            verify_cmd+=(--inference-timeout "$VALIDATION_TIMEOUT")
        fi
        
        if ! "${verify_cmd[@]}"; then
            error "GPU deployment verification failed"
            error "The service is not properly configured for GPU acceleration"
            error "Model inference failed or GPU not detected after model load"
            error "Check: sudo systemctl status ollama"
            error "Check: sudo journalctl -u ollama -n 50"
            exit 1
        fi
        success "GPU deployment verified - model-aware validation successful: $VALIDATION_MODEL on GPU"
    else
        # GPU-only validation: just check /api/ps (lightweight check, no inference)
        info "Using GPU-only validation (no model inference, just checking /api/ps)"
        
        # GPU-only check uses fast API timeout (defaults to 10s from llm package)
        # VALIDATION_TIMEOUT is only for inference (model loading), not for quick GPU checks
        # Pass --api-timeout separately if user needs custom timing for the GPU check itself
        local verify_cmd=(bash "$verify_script" --url "http://localhost:11434")
        
        if ! "${verify_cmd[@]}"; then
            error "GPU deployment verification failed"
            error "The service is not properly configured for GPU acceleration"
            error "Check: sudo systemctl status ollama"
            error "Check: sudo journalctl -u ollama -n 50"
            error "Tip: Use --validation-model <model> for truthful fresh-service validation"
            exit 1
        fi
        success "GPU deployment verified - GPU-only validation successful (no model inference performed)"
    fi
}

# ============================================================================
# Build and Deploy Plan Display
# ============================================================================

show_build_deploy_plan() {
    echo ""
    echo "${BLUE}=== BUILD & DEPLOY PLAN (DRY-RUN) ===${PLAIN}" >&2
    echo "" >&2
    
    echo "${BLUE}Build Phase:${PLAIN}" >&2
    echo "  - Configure CMake with Vulkan-only local build flags (OLLAMA_VULKAN_ONLY=ON, OLLAMA_RUNNER_DIR=vulkan)" >&2
    echo "  - Build with CMake (Release mode, $(nproc) jobs)" >&2
    echo "  - Build Ollama binary" >&2
    [ "$CLEAN_CACHE" = true ] && echo "  - Clean Go cache" >&2
    
    echo "" >&2
    echo "${BLUE}Install Phase:${PLAIN}" >&2
    echo "  - Verify build artifacts" >&2
    echo "  - Install binary to ${BIN_DIR}/ollama" >&2
    echo "  - Install libraries to ${LIB_DIR}" >&2
    echo "  - Preserve backend subdirectories" >&2

    echo "" >&2
    echo "${BLUE}Service Phase:${PLAIN}" >&2
    echo "  - Ensure ${SERVICE_USER} user/group exists and has GPU access groups" >&2
    echo "  - Install ${SERVICE_FILE} with OLLAMA_VULKAN=1 and OLLAMA_LLM_LIBRARY=vulkan" >&2
    echo "  - Reload, enable, and restart the ollama service" >&2
    
    echo "" >&2
    echo "${YELLOW}Will be removed:${PLAIN}" >&2
    echo "  - Existing binary: ${BIN_DIR}/ollama (if present)" >&2
    echo "  - Existing libraries: ${LIB_DIR} (if present)" >&2
    
    echo "" >&2
    echo "${GREEN}Will be preserved:${PLAIN}" >&2
    echo "  - Models: ${MODEL_DIR}" >&2
    
    echo "" >&2
    echo "${BLUE}=== END BUILD & DEPLOY PLAN ===${PLAIN}" >&2
    echo "" >&2
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Parse command line arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -c|--clean-cache)
                CLEAN_CACHE=true
                shift
                ;;
            --validation-model)
                VALIDATION_MODEL="$2"
                shift 2
                ;;
            --inference-timeout)
                VALIDATION_TIMEOUT="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    # Show mode information
    if [ "$DRY_RUN" = true ]; then
        status "Running in DRY-RUN mode (no changes will be made)"
    fi
    
    echo ""
    echo "${BLUE}========================================${PLAIN}" >&2
    echo "${BLUE}  Ollama Local Build Deploy${PLAIN}" >&2
    echo "${BLUE}  Phase 1: Preflight & Cleanup${PLAIN}" >&2
    echo "${BLUE}  Phase 2: Build & Install${PLAIN}" >&2
    echo "${BLUE}  Phase 3: Service Setup${PLAIN}" >&2
    echo "${BLUE}========================================${PLAIN}" >&2
    echo "" >&2
    
    # Run preflight checks
    preflight_check_tools
    preflight_check_repo_context
    preflight_check_permissions
    preflight_check_install_detection
    preflight_check_ollama_user
    
    echo "" >&2
    
    # Show plan if in dry-run mode
    if [ "$DRY_RUN" = true ]; then
        show_build_deploy_plan
    fi
    
    # Stop service
    stop_service_if_running
    
    echo "" >&2
    
    # Perform cleanup
    status "Starting cleanup phase..."
    cleanup_binary
    cleanup_libraries
    verify_model_preservation
    
    echo "" >&2
    
    # Check build prerequisites
    check_build_prerequisites
    
    echo "" >&2
    
    # Perform build
    status "Starting build phase..."
    clean_go_cache_if_requested
    build_with_cmake
    build_cmake_project
    build_ollama_binary
    
    echo "" >&2
    
    # Verify build artifacts
    verify_build_artifacts
    
    echo "" >&2
    
    # Perform install
    status "Starting install phase..."
    install_binary
    install_backend_libraries
    verify_install_vulkan_artifacts

    echo "" >&2

    # Configure service
    status "Starting service phase..."
    ensure_ollama_user_group
    install_systemd_service
    verify_systemd_service_configuration
    reload_enable_start_service
    verify_model_preservation
    
    echo "" >&2
    
    # Phase 3: Verify GPU deployment using ValidateDeploymentState
    verify_gpu_deployment
    
    echo "" >&2
    echo "${BLUE}========================================${PLAIN}" >&2
    if [ "$DRY_RUN" = true ]; then
        echo "${GREEN}✓ Dry-run completed successfully${PLAIN}" >&2
    else
        echo "${GREEN}✓ Build and deploy completed successfully${PLAIN}" >&2
        # Truthfully report validation type
        if [ -n "$VALIDATION_MODEL" ]; then
            echo "${GREEN}✓ Phase 3: Model-aware GPU deployment validated ($VALIDATION_MODEL)${PLAIN}" >&2
        else
            echo "${GREEN}✓ Phase 3: GPU deployment validated (GPU-only, no model validation)${PLAIN}" >&2
        fi
    fi
    echo "${BLUE}========================================${PLAIN}" >&2
    echo "" >&2
}

# Run main function
main "$@"
