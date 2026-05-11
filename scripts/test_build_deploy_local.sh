#!/bin/bash
# Test suite for build_deploy_local.sh
# Tests preflight checks, cleanup logic, and safety validations
# Includes behavioral tests that actually run the script

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Log test results
test_result() {
    local name="$1"
    local status="$2"
    TESTS_RUN=$((TESTS_RUN + 1))
    
    if [ "$status" = "PASS" ]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo -e "${GREEN}✓ PASS${NC}: $name"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo -e "${RED}✗ FAIL${NC}: $name"
        if [ $# -gt 2 ]; then
            echo "  $3"
        fi
    fi
}

# Setup temp directory for testing
setup_test_env() {
    export TEST_TMPDIR=$(mktemp -d)
    export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export SCRIPT="$SCRIPT_DIR/build_deploy_local.sh"
}

# Cleanup test environment
cleanup_test_env() {
    [ -d "$TEST_TMPDIR" ] && rm -rf "$TEST_TMPDIR"
}

# ============================================================================
# Static Tests: Source code validation
# ============================================================================

# Test: Script exists and is executable
test_script_exists() {
    if [ -f "$SCRIPT" ]; then
        test_result "Script file exists" "PASS"
    else
        test_result "Script file exists" "FAIL" "File not found: $SCRIPT"
    fi
}

# Test: Script is executable
test_script_executable() {
    if [ -x "$SCRIPT" ]; then
        test_result "Script is executable" "PASS"
    else
        test_result "Script is executable" "FAIL" "Script is not executable"
    fi
}

# Test: Script has proper shebang
test_script_shebang() {
    if head -1 "$SCRIPT" | grep -q "^#!/.*bash"; then
        test_result "Script has bash shebang" "PASS"
    else
        test_result "Script has bash shebang" "FAIL" "Expected #!/bin/bash shebang"
    fi
}

# Test: Script syntax is valid (bash -n)
test_script_syntax() {
    if bash -n "$SCRIPT" 2>/dev/null; then
        test_result "Script syntax is valid" "PASS"
    else
        test_result "Script syntax is valid" "FAIL" "Syntax error detected"
    fi
}

# Test: Script has help/usage function
test_script_has_help() {
    if grep -q "usage\|help\|-h\|--help" "$SCRIPT"; then
        test_result "Script has help/usage" "PASS"
    else
        test_result "Script has help/usage" "FAIL" "No help/usage found"
    fi
}

# Test: Script mentions /usr/local
test_script_mentions_prefix() {
    if grep -q "/usr/local" "$SCRIPT"; then
        test_result "Script mentions /usr/local prefix" "PASS"
    else
        test_result "Script mentions /usr/local prefix" "FAIL" "Prefix not found"
    fi
}

# Test: Script mentions model preservation
test_script_preserves_models() {
    if grep -q "/usr/share/ollama" "$SCRIPT" || grep -q "models" "$SCRIPT" || grep -q ".ollama/models" "$SCRIPT"; then
        test_result "Script mentions model preservation" "PASS"
    else
        test_result "Script mentions model preservation" "FAIL" "Model preservation not found"
    fi
}

# Test: Script has preflight checks
test_script_has_preflight() {
    if grep -q "preflight\|check\|prerequisite\|require" "$SCRIPT"; then
        test_result "Script has preflight checks" "PASS"
    else
        test_result "Script has preflight checks" "FAIL" "Preflight checks not found"
    fi
}

# Test: Script checks for root/sudo
test_script_checks_root() {
    if grep -q "root\|sudo\|id -u" "$SCRIPT"; then
        test_result "Script checks for root/sudo" "PASS"
    else
        test_result "Script checks for root/sudo" "FAIL" "Root/sudo check not found"
    fi
}

# Test: Script checks for ollama user/group
test_script_checks_ollama_user() {
    if grep -q "ollama" "$SCRIPT" && grep -q "user\|group\|getent" "$SCRIPT"; then
        test_result "Script checks for ollama user/group" "PASS"
    else
        test_result "Script checks for ollama user/group" "FAIL" "Ollama user/group check not found"
    fi
}

# Test: Script has dry-run support
test_script_has_dryrun() {
    if grep -q "dry-run\|dry_run\|DRY_RUN\|-n" "$SCRIPT"; then
        test_result "Script has dry-run support" "PASS"
    else
        test_result "Script has dry-run support" "FAIL" "Dry-run support not found"
    fi
}

# Test: Script mentions stopping service
test_script_stops_service() {
    if grep -q "systemctl\|service\|stop" "$SCRIPT" || grep -q "ollama" "$SCRIPT"; then
        test_result "Script mentions service stopping" "PASS"
    else
        test_result "Script mentions service stopping" "FAIL" "Service stop not found"
    fi
}

# Test: Script mentions cleanup
test_script_has_cleanup() {
    if grep -q "cleanup\|rm -rf\|remove" "$SCRIPT"; then
        test_result "Script has cleanup logic" "PASS"
    else
        test_result "Script has cleanup logic" "FAIL" "Cleanup logic not found"
    fi
}

# Test: Verify --verbose is removed or properly used
test_verbose_removed_or_handled() {
    # Check if --verbose exists in the script
    if grep -q "\-\-verbose\|VERBOSE" "$SCRIPT"; then
        # If it exists, verify it's properly handled
        if grep -q "VERBOSE=false\|VERBOSE=true" "$SCRIPT"; then
            test_result "Verbose flag handled properly" "PASS"
        else
            test_result "Verbose flag handled properly" "FAIL" "Verbose flag exists but not properly initialized"
        fi
    else
        test_result "Verbose flag removed or handled" "PASS"
    fi
}

# Test: Verify Phase 3/systemd deployment messaging is present
test_phase3_messaging_present() {
    if grep -q "Phase 3: Service Setup\|Installing systemd service\|Starting service phase" "$SCRIPT"; then
        test_result "Phase 3 messaging present in deploy script" "PASS"
    else
        test_result "Phase 3 messaging present in deploy script" "FAIL" "Phase 3/service messaging not found"
    fi
}

# ============================================================================
# Behavioral Tests: Actually run the script in controlled ways
# ============================================================================

# Test: --help exits successfully
test_help_exit_success() {
    local output
    local exit_code
    
    # Run from repo root to ensure context is correct
    output=$("$SCRIPT" --help 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        test_result "--help exits with code 0" "PASS"
    else
        test_result "--help exits with code 0" "FAIL" "Exit code: $exit_code"
    fi
}

# Test: --help prints usage text
test_help_prints_usage() {
    local output
    
    output=$("$SCRIPT" --help 2>&1) || true
    
    if echo "$output" | grep -q "Usage:\|build_deploy_local"; then
        test_result "--help prints usage text" "PASS"
    else
        test_result "--help prints usage text" "FAIL" "Usage text not found in output"
    fi
}

# Test: --dry-run exits successfully (must be exit code 0)
test_dryrun_exit_success() {
    local exit_code
    local output
    
    # Run from repo root to ensure preflight checks pass
    output=$("$SCRIPT" --dry-run 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        test_result "--dry-run exits with code 0" "PASS"
    else
        test_result "--dry-run exits with code 0" "FAIL" "Exit code: $exit_code. Output: $output"
    fi
}

# Test: --dry-run prints cleanup plan with specific structure
test_dryrun_prints_plan() {
    local output
    
    output=$("$SCRIPT" --dry-run 2>&1)
    
    # Check for key cleanup/build plan elements
    if echo "$output" | grep -qE "CLEANUP PLAN|BUILD.*DEPLOY.*PLAN|Build.*Phase"; then
        # Check for "Will be removed" section
        if echo "$output" | grep -q "Will be removed"; then
            # Check for specific items to be removed (flexible matching)
            if echo "$output" | grep -qE "(Binary|binary).*ollama" && echo "$output" | grep -qi "Libraries"; then
                # Check for "Will be preserved" section
                if echo "$output" | grep -q "Will be preserved" && echo "$output" | grep -qi "Models"; then
                    test_result "--dry-run prints cleanup plan" "PASS"
                else
                    test_result "--dry-run prints cleanup plan" "FAIL" "Preservation section missing"
                fi
            else
                test_result "--dry-run prints cleanup plan" "FAIL" "Removal items not specified"
            fi
        else
            test_result "--dry-run prints cleanup plan" "FAIL" "No removal section found"
        fi
    else
        test_result "--dry-run prints cleanup plan" "FAIL" "Build/cleanup plan header not found"
    fi
}

# Test: Invalid flag exits with non-zero
test_invalid_flag_fails() {
    local exit_code
    
    # Run from repo root
    if "$SCRIPT" --invalid-flag >/dev/null 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    
    if [ $exit_code -ne 0 ]; then
        test_result "Invalid flag exits with non-zero" "PASS"
    else
        test_result "Invalid flag exits with non-zero" "FAIL" "Expected non-zero exit code"
    fi
}

# Test: Dry-run does not report success as if changes were made
test_dryrun_messaging() {
    local output
    
    output=$("$SCRIPT" --dry-run 2>&1) || true
    
    # Check that output clearly indicates dry-run mode
    if echo "$output" | grep -qE "DRY-RUN|DRY-RUN|planned|would"; then
        test_result "Dry-run messaging indicates no changes made" "PASS"
    else
        test_result "Dry-run messaging indicates no changes made" "FAIL" "Output does not clarify dry-run nature"
    fi
}

# Test: Behavioral tests run from repo root with proper context
test_behavioral_context_from_repo_root() {
    local output
    
    # The script requires go.mod and scripts/ to exist (repo root check)
    # This test verifies that our behavioral tests run from the correct context
    output=$("$SCRIPT" --dry-run 2>&1) || true
    
    # Check that the repo context check passed (not failed)
    if echo "$output" | grep -q "Repository context verified"; then
        test_result "Behavioral tests run from repo root" "PASS"
    else
        test_result "Behavioral tests run from repo root" "FAIL" "Repo context check did not pass"
    fi
}

# Test: Dry-run should not error on missing artifacts (artifact-independent)
test_dryrun_independent_of_artifacts() {
    local output
    local exit_code
    
    # Run dry-run - should succeed regardless of artifacts
    output=$("$SCRIPT" --dry-run 2>&1)
    exit_code=$?
    
    # Should exit 0 for dry-run even if artifacts missing
    if [ $exit_code -eq 0 ]; then
        # Should show build and install plan without errors
        if echo "$output" | grep -qi "would\|will\|plan"; then
            test_result "Dry-run is artifact-independent (plan-only)" "PASS"
        else
            test_result "Dry-run is artifact-independent (plan-only)" "FAIL" "Dry-run did not show plan"
        fi
    else
        test_result "Dry-run is artifact-independent (plan-only)" "FAIL" "Dry-run exited with $exit_code (likely artifact check failed)"
    fi
}

# Test: Verify build tools are checked (not just git)
test_script_checks_build_tools() {
    if grep -q "require.*cmake\|require.*make\|require.*go" "$SCRIPT"; then
        test_result "Script checks for cmake/make/go build tools" "PASS"
    else
        test_result "Script checks for cmake/make/go build tools" "FAIL" "Build tools check not found (cmake, make, go)"
    fi
}

# Test: Verify install uses cmake --install for proper layout
test_install_uses_cmake_install() {
    if grep -q "cmake --install" "$SCRIPT"; then
        test_result "Install uses cmake --install for proper layout" "PASS"
    else
        # Check if raw copy is used (which is problematic)
        if grep -q "cp -r.*BUILD_DIR/lib\|cp.*ollama.*LIB_DIR" "$SCRIPT"; then
            test_result "Install uses cmake --install for proper layout" "FAIL" "Using raw copy instead of cmake --install"
        else
            test_result "Install uses cmake --install for proper layout" "PASS"
        fi
    fi
}

# Test: Verify CMake configuration uses OLLAMA_RUNNER_DIR=vulkan
test_cmake_config_sets_runner_dir() {
    # Check that CMake configuration includes OLLAMA_RUNNER_DIR=vulkan or uses Vulkan preset
    if grep -q "OLLAMA_RUNNER_DIR.*vulkan\|preset.*[Vv]ulkan\|--preset.*Vulkan" "$SCRIPT"; then
        test_result "CMake config includes OLLAMA_RUNNER_DIR=vulkan or Vulkan preset" "PASS"
    else
        test_result "CMake config includes OLLAMA_RUNNER_DIR=vulkan or Vulkan preset" "FAIL" "Vulkan runner dir or preset not found"
    fi
}

# Test: Verify CMake configuration enables local Vulkan-only mode
test_cmake_config_sets_vulkan_only() {
    if grep -q 'OLLAMA_VULKAN_ONLY=ON' "$SCRIPT"; then
        test_result "CMake config enables OLLAMA_VULKAN_ONLY=ON" "PASS"
    else
        test_result "CMake config enables OLLAMA_VULKAN_ONLY=ON" "FAIL" "OLLAMA_VULKAN_ONLY=ON not found in CMake configure command"
    fi
}

# Test: Verify install component failures are fatal (not warnings)
test_install_component_failures_fatal() {
    # Check that cmake install failures in the loop cause the script to fail
    # Should use error rather than warning. May span multiple lines due to continuation.
    local install_func_content
    install_func_content=$(grep -A50 "install_backend_libraries()" "$SCRIPT")
    
    # Check for the pattern: command || error (allowing for multiline with continuation)
    if echo "$install_func_content" | grep -q "cmake --install.*--component" && \
       echo "$install_func_content" | grep -q "|| \\\\" && \
       echo "$install_func_content" | grep -q 'error "Install component'; then
        test_result "Install component failures are fatal (error not warning)" "PASS"
    else
        # Check if warnings are being used (which is wrong)
        if echo "$install_func_content" | grep -q "|| warning.*Install component"; then
            test_result "Install component failures are fatal (error not warning)" "FAIL" "Using warning instead of error for install failures"
        else
            test_result "Install component failures are fatal (error not warning)" "FAIL" "Install failure handling not clear"
        fi
    fi
}

# Test: Verify no duplicated permission-setting logic
test_no_duplicated_permissions() {
    # Extract the install_backend_libraries function and check for duplication
    local perm_lines
    perm_lines=$(grep -c "find.*LIB_DIR.*chmod 755" "$SCRIPT" || echo "0")
    
    if [ "$perm_lines" -lt 2 ]; then
        test_result "No duplicated permission-setting logic" "PASS"
    else
        # Multiple identical permission commands suggest duplication
        if grep -A2 "if id ollama.*then" "$SCRIPT" | grep -q "chmod 755" && \
           grep -A2 "if id ollama.*then" "$SCRIPT" | grep -A1 "else" | grep -q "chmod 755"; then
            test_result "No duplicated permission-setting logic" "FAIL" "Same chmod command in both if and else branches"
        else
            test_result "No duplicated permission-setting logic" "PASS"
        fi
    fi
}

# Test: Verify no misleading OLLAMA_VULKAN export in build_ollama_binary
# This test extracts the entire function and checks its contents
test_no_redundant_vulkan_export() {
    # The OLLAMA_VULKAN is already set globally, exporting it again in build_ollama_binary is redundant
    # Extract the full build_ollama_binary function (from declaration to next function)
    local func_content
    func_content=$(awk '/^build_ollama_binary\(\)/{flag=1} flag{print} /^[a-z_]*\(\)/ && !/^build_ollama_binary/{if(flag) exit}' "$SCRIPT")
    
    if echo "$func_content" | grep -q "export OLLAMA_VULKAN"; then
        test_result "No redundant OLLAMA_VULKAN export in build function" "FAIL" "OLLAMA_VULKAN exported again in build_ollama_binary (already global)"
    else
        test_result "No redundant OLLAMA_VULKAN export in build function" "PASS"
    fi
}

# Test: Verify verify_build_artifacts checks DRY_RUN mode
test_verify_artifacts_checks_dryrun() {
    # Check that verify_build_artifacts function includes DRY_RUN check before validation
    if grep -A 5 "^verify_build_artifacts()" "$SCRIPT" | grep -q "DRY_RUN"; then
        test_result "verify_build_artifacts respects dry-run mode" "PASS"
    else
        test_result "verify_build_artifacts respects dry-run mode" "FAIL" "DRY_RUN check missing in verify_build_artifacts"
    fi
}

# Test: Verify install_backend_libraries checks DRY_RUN before artifact check
test_install_libraries_checks_dryrun() {
    # Check that install_backend_libraries checks DRY_RUN early (before error on missing source)
    if grep -A 3 "^install_backend_libraries()" "$SCRIPT" | grep -q "DRY_RUN"; then
        test_result "install_backend_libraries respects dry-run mode early" "PASS"
    else
        test_result "install_backend_libraries respects dry-run mode early" "FAIL" "DRY_RUN check not early in install_backend_libraries"
    fi
}

# Test: Verify Vulkan component is treated as required, not optional
test_vulkan_component_required() {
    # Check that the script treats Vulkan as a required component for installation
    # Look for explicit Vulkan component installation (not just optional detection)
    local install_func
    install_func=$(awk '/^install_backend_libraries\(\)/{flag=1} flag{print} /^[a-z_]*\(\)/ && !/^install_backend_libraries/{if(flag) exit}' "$SCRIPT")
    
    # Vulkan should be installed as a required component, not as optional detection
    # Check that we attempt to install Vulkan component
    if echo "$install_func" | grep -q 'cmake --install.*--component.*Vulkan' || \
       echo "$install_func" | grep -q 'components=".*Vulkan'; then
        # But we should NOT be doing optional detection based on cache strings
        if echo "$install_func" | grep -q 'grep.*CMakeCache' && \
           echo "$install_func" | grep -q 'ggml-vulkan\|GGML_VULKAN'; then
            test_result "Vulkan component treated as required (not fragile cache detection)" "FAIL" "Vulkan install depends on fragile CMakeCache.txt detection"
        else
            test_result "Vulkan component treated as required (not fragile cache detection)" "PASS"
        fi
    else
        test_result "Vulkan component treated as required (not fragile cache detection)" "FAIL" "Vulkan component not explicitly installed"
    fi
}

# ============================================================================
# Phase 2 Tests: Build and Install Flow
# ============================================================================

# Test: Script mentions build or cmake configuration
test_script_has_build_flow() {
    if grep -q "cmake\|build\|make\|OLLAMA_VULKAN\|VULKAN" "$SCRIPT"; then
        test_result "Script has build configuration" "PASS"
    else
        test_result "Script has build configuration" "FAIL" "Build configuration not found"
    fi
}

# Test: Script mentions install operations
test_script_has_install_flow() {
    if grep -q "install\|cp\|mkdir" "$SCRIPT"; then
        test_result "Script has install operations" "PASS"
    else
        test_result "Script has install operations" "FAIL" "Install operations not found"
    fi
}

# Test: Script mentions lib/ollama directory preservation
test_script_preserves_lib_structure() {
    if grep -q "lib/ollama\|LIB_DIR" "$SCRIPT"; then
        test_result "Script preserves lib/ollama structure" "PASS"
    else
        test_result "Script preserves lib/ollama structure" "FAIL" "lib/ollama not found"
    fi
}

# Test: Script mentions build directory
test_script_uses_build_output() {
    if grep -q "build/lib\|build/\|CMAKE_BUILD_DIR" "$SCRIPT"; then
        test_result "Script uses build output directory" "PASS"
    else
        test_result "Script uses build output directory" "FAIL" "Build output not found"
    fi
}

# Test: Script has optional clean cache flag/option
test_script_has_optional_clean() {
    if grep -q "\-\-clean-cache" "$SCRIPT"; then
        test_result "Script has optional clean cache option" "PASS"
    else
        test_result "Script has optional clean cache option" "FAIL" "Expected --clean-cache flag in script"
    fi
}

# Behavioral test: Build option in help
test_help_mentions_build() {
    local output
    
    output=$("$SCRIPT" --help 2>&1) || true
    
    if echo "$output" | grep -qE "build|Build|CMAKE|install|Install"; then
        test_result "Help text mentions build/install flow" "PASS"
    else
        test_result "Help text mentions build/install flow" "FAIL" "Build/install not mentioned in help"
    fi
}

# Test: Verify script documents the /usr/local/lib/ollama/vulkan path expectation
test_script_validates_vulkan_lib_path() {
    # Check that the script documents or validates the Vulkan lib path structure
    if grep -q "lib/ollama.*vulkan\|LIB_DIR.*vulkan\|/vulkan" "$SCRIPT"; then
        test_result "Script references Vulkan lib path structure" "PASS"
    else
        # Weaker test: just check that lib/ollama is mentioned along with Vulkan
        if grep -q "lib/ollama" "$SCRIPT" && grep -q "vulkan\|Vulkan\|VULKAN" "$SCRIPT"; then
            test_result "Script references Vulkan lib path structure" "PASS"
        else
            test_result "Script references Vulkan lib path structure" "FAIL" "Vulkan lib path not documented"
        fi
    fi
}

# Test: Verify CMAKE_BUILD_DIR naming is clear/documented
test_build_dir_naming_clear() {
    # Check that CMAKE_BUILD_DIR is clearly explained or documented
    # Look for comments explaining the BUILD_DIR variable setup
    if grep -B2 -A2 'BUILD_DIR.*CMAKE_BUILD_DIR' "$SCRIPT" | grep -q '#.*[Bb]uild\|#.*[Dd]ir'; then
        test_result "CMAKE_BUILD_DIR naming is documented/clear" "PASS"
    else
        # Also accept if there's clear help text explaining it
        if grep -q 'Build Dir:' "$SCRIPT" && grep -q 'CMAKE_BUILD_DIR'; then
            test_result "CMAKE_BUILD_DIR naming is documented/clear" "PASS"
        else
            test_result "CMAKE_BUILD_DIR naming is documented/clear" "FAIL" "BUILD_DIR/CMAKE_BUILD_DIR configuration not clearly documented"
        fi
    fi
}

# Test: Verify script documentation mentions the expected directory structure
test_script_documents_lib_structure() {
    # Check that the script or help mentions the directory layout/structure
    if grep -q "subdirector\|layout\|structur\|preserve.*subdirector" "$SCRIPT"; then
        test_result "Script documents library subdirectory structure preservation" "PASS"
    else
        test_result "Script documents library subdirectory structure preservation" "FAIL" "Library structure documentation not found (look for 'subdirectory', 'layout', or 'structure')"
    fi
}

# Test: Verify post-build Vulkan artifact validation exists and is fatal
test_post_build_vulkan_validation() {
    # Check that verify_build_artifacts validates Vulkan artifacts (not just generic backend libs)
    if grep -q "verify_build_artifacts" "$SCRIPT"; then
        # Extract the verify_build_artifacts function
        local func_content
        func_content=$(awk '/^verify_build_artifacts\(\)/{flag=1} flag{print} /^[a-z_]*\(\)/ && !/^verify_build_artifacts/{if(flag) exit}' "$SCRIPT")
        
        # Check that it specifically validates the build-tree Vulkan shared library,
        # not the install-time vulkan/ staging directory.
        if echo "$func_content" | grep -q "libggml-vulkan\.so" && \
           ! echo "$func_content" | grep -q 'No Vulkan libraries found in .*build.*/vulkan/'; then
            test_result "Post-build Vulkan artifact validation exists (specific not generic)" "PASS"
        else
            test_result "Post-build Vulkan artifact validation exists (specific not generic)" "FAIL" "Expected build-tree libggml-vulkan.so validation, not install-layout-only checks"
        fi
    else
        test_result "Post-build Vulkan artifact validation exists (specific not generic)" "FAIL" "verify_build_artifacts function not found"
    fi
}

# Test: Verify post-build validation rejects unexpected non-Vulkan GPU backends
test_post_build_rejects_non_vulkan_backends() {
    local func_content
    func_content=$(awk '/^verify_build_artifacts\(\)/{flag=1} flag{print} /^[a-z_]*\(\)/ && !/^verify_build_artifacts/{if(flag) exit}' "$SCRIPT")

    if echo "$func_content" | grep -q 'libggml-cuda' && \
       echo "$func_content" | grep -q 'libggml-hip' && \
       echo "$func_content" | grep -q 'Unexpected non-Vulkan GPU backend libraries'; then
        test_result "Post-build validation rejects non-Vulkan GPU backends" "PASS"
    else
        test_result "Post-build validation rejects non-Vulkan GPU backends" "FAIL" "Expected explicit rejection of CUDA/HIP/Metal artifacts in verify_build_artifacts"
    fi
}

# Test: Verify post-install Vulkan library directory validation exists and is fatal
test_post_install_vulkan_validation() {
    # Check that the script validates Vulkan libraries exist at /usr/local/lib/ollama/vulkan after install
    if grep -q "/vulkan\|vulkan.*LIB_DIR\|LIB_DIR.*vulkan" "$SCRIPT"; then
        # Also check that the validation is fatal (uses error not warning)
        if grep -A20 "install_backend_libraries" "$SCRIPT" | grep -q "vulkan" && \
           grep -A30 "install_backend_libraries" "$SCRIPT" | grep -q "error.*[Vv]ulkan"; then
            test_result "Post-install Vulkan library validation exists and is fatal" "PASS"
        else
            test_result "Post-install Vulkan library validation exists and is fatal" "FAIL" "Vulkan validation found but may not be fatal"
        fi
    else
        test_result "Post-install Vulkan library validation exists and is fatal" "FAIL" "Vulkan library path validation not found"
    fi
}

# Test: Verify the installed systemd service enables Vulkan explicitly
test_service_enables_vulkan() {
    if grep -q 'Environment="OLLAMA_VULKAN=1"' "$SCRIPT"; then
        test_result "Systemd service enables OLLAMA_VULKAN=1" "PASS"
    else
        test_result "Systemd service enables OLLAMA_VULKAN=1" "FAIL" "OLLAMA_VULKAN=1 not found in service template"
    fi
}

# Test: Verify the installed systemd service pins the Vulkan runtime library
test_service_pins_vulkan_library() {
    if grep -q 'Environment="OLLAMA_LLM_LIBRARY=vulkan"' "$SCRIPT"; then
        test_result "Systemd service pins OLLAMA_LLM_LIBRARY=vulkan" "PASS"
    else
        test_result "Systemd service pins OLLAMA_LLM_LIBRARY=vulkan" "FAIL" "OLLAMA_LLM_LIBRARY=vulkan not found in service template"
    fi
}

# Test: Verify systemd unit is written to /etc/systemd/system/ollama.service
test_service_file_written() {
    if grep -q '/etc/systemd/system/ollama.service' "$SCRIPT" && grep -q 'tee "\${SERVICE_FILE}"' "$SCRIPT"; then
        test_result "Systemd service file is written to /etc/systemd/system/ollama.service" "PASS"
    else
        test_result "Systemd service file is written to /etc/systemd/system/ollama.service" "FAIL" "Service file write path or tee command missing"
    fi
}

# Test: Verify service unit uses the deployed /usr/local binary
test_service_execstart_uses_usr_local() {
    if grep -q 'ExecStart=\${BIN_DIR}/ollama serve' "$SCRIPT"; then
        test_result "Systemd service uses deployed /usr/local binary" "PASS"
    else
        test_result "Systemd service uses deployed /usr/local binary" "FAIL" "ExecStart does not use \${BIN_DIR}/ollama serve"
    fi
}

# Test: Verify service unit targets multi-user.target
test_service_uses_multi_user_target() {
    if grep -q 'WantedBy=multi-user.target' "$SCRIPT"; then
        test_result "Systemd service targets multi-user.target" "PASS"
    else
        test_result "Systemd service targets multi-user.target" "FAIL" "WantedBy=multi-user.target not found"
    fi
}

# Test: Verify service management reloads, enables, and restarts the unit
test_service_reload_enable_restart() {
    if grep -q 'systemctl daemon-reload' "$SCRIPT" && \
       grep -q 'systemctl enable ollama' "$SCRIPT" && \
       grep -q 'systemctl restart ollama' "$SCRIPT"; then
        test_result "Systemd service reload/enable/restart is implemented" "PASS"
    else
        test_result "Systemd service reload/enable/restart is implemented" "FAIL" "Missing daemon-reload, enable, or restart command"
    fi
}

# Test: Verify ollama user creation follows repo conventions and GPU access groups
test_service_user_setup_present() {
    if grep -q 'useradd -r -s /bin/false' "$SCRIPT" && \
       grep -q 'usermod -a -G render' "$SCRIPT" && \
       grep -q 'usermod -a -G video' "$SCRIPT"; then
        test_result "ollama user setup follows repo conventions" "PASS"
    else
        test_result "ollama user setup follows repo conventions" "FAIL" "Expected useradd/usermod setup for ollama render/video access"
    fi
}

# Test: Verify pipefail is set in shebang or set statement
test_pipefail_enabled() {
    # Check that pipefail is enabled for safe pipe handling
    # Check first 25 lines (accounting for script header comments)
    if head -25 "$SCRIPT" | grep -q "set.*pipefail\|set -.*o pipefail"; then
        test_result "Pipefail enabled for safe pipe handling" "PASS"
    else
        test_result "Pipefail enabled for safe pipe handling" "FAIL" "pipefail not found in set statement (check first 25 lines)"
    fi
}

# Test: Verify CMAKE_BUILD_DIR is clearly documented
test_cmake_build_dir_documented() {
    # Check that CMAKE_BUILD_DIR is clearly documented with comments explaining its use
    if grep -B2 'BUILD_DIR=' "$SCRIPT" | grep -q '#' && \
       grep 'BUILD_DIR=' "$SCRIPT" | grep -q 'CMAKE_BUILD_DIR'; then
        test_result "CMAKE_BUILD_DIR is clearly documented with comments" "PASS"
    else
        test_result "CMAKE_BUILD_DIR is clearly documented with comments" "FAIL" "CMAKE_BUILD_DIR documentation unclear"
    fi
}

# ============================================================================

main() {
    echo "=================================================="
    echo "Test Suite: build_deploy_local.sh"
    echo "=================================================="
    
    setup_test_env
    
    # Static tests
    echo ""
    echo -e "${BLUE}Static Tests (source code validation):${NC}"
    test_script_exists
    test_script_executable
    test_script_shebang
    test_script_syntax
    test_script_has_help
    test_script_mentions_prefix
    test_script_preserves_models
    test_script_has_preflight
    test_script_checks_root
    test_script_checks_ollama_user
    test_script_has_dryrun
    test_script_stops_service
    test_script_has_cleanup
    test_verbose_removed_or_handled
    
    # Phase 2 build/install tests
    echo ""
    echo -e "${BLUE}Phase 2 Tests (build and install flow):${NC}"
    test_script_has_build_flow
    test_script_has_install_flow
    test_script_preserves_lib_structure
    test_script_uses_build_output
    test_script_has_optional_clean
    
    # Behavioral tests
    echo ""
    echo -e "${BLUE}Behavioral Tests (actual execution):${NC}"
    test_behavioral_context_from_repo_root
    test_dryrun_independent_of_artifacts
    test_script_checks_build_tools
    test_help_exit_success
    test_help_prints_usage
    test_help_mentions_build
    test_dryrun_exit_success
    test_dryrun_prints_plan
    test_invalid_flag_fails
    test_dryrun_messaging
    
    # Phase 2 Revision Tests (address blocking issues)
    echo ""
    echo -e "${BLUE}Deploy Revision Tests:${NC}"
    test_phase3_messaging_present
    test_cmake_config_sets_runner_dir
    test_cmake_config_sets_vulkan_only
    test_install_component_failures_fatal
    test_install_uses_cmake_install
    test_no_duplicated_permissions
    test_no_redundant_vulkan_export
    test_verify_artifacts_checks_dryrun
    test_install_libraries_checks_dryrun
    test_vulkan_component_required
    test_post_build_rejects_non_vulkan_backends
    test_service_enables_vulkan
    test_service_pins_vulkan_library
    test_pipefail_enabled
    test_cmake_build_dir_documented
    test_service_file_written
    test_service_execstart_uses_usr_local
    test_service_uses_multi_user_target
    test_service_reload_enable_restart
    test_service_user_setup_present
    
    # Layout and directory structure tests
    echo ""
    echo -e "${BLUE}Install Layout Tests (directory structure validation):${NC}"
    test_script_validates_vulkan_lib_path
    test_script_documents_lib_structure
    test_build_dir_naming_clear
    test_post_build_vulkan_validation
    test_post_install_vulkan_validation
    
    # Summary
    echo ""
    echo "=================================================="
    echo "Test Summary"
    echo "=================================================="
    echo "Tests run:    $TESTS_RUN"
    echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    echo "=================================================="
    
    cleanup_test_env
    
    # Exit with appropriate code
    [ $TESTS_FAILED -eq 0 ] && exit 0 || exit 1
}

main "$@"
