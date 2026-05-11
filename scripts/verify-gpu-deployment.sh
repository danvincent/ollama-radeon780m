#!/bin/bash
#
# verify-gpu-deployment.sh - Verify that Ollama GPU deployment is working correctly
#
# This script is a thin wrapper that calls the unified Go deployment validator.
# All validation logic is in the llm package (llm/deployment_integration.go):
# - ValidateDeploymentViaAPI is the unified entry point
# - This script just handles shell integration and exit-code propagation
#
# Validation flow:
# 1. Script changes to repo directory
# 2. Runs: go run ./cmd/deploy-check/main.go
# 3. CLI (cmd/deploy-check/main.go) calls: llm.ValidateDeploymentViaAPI
# 4. ValidateDeploymentViaAPI performs unified validation (model-aware):
#    - If model specified: exercise inference first, then check GPU via /api/ps
#    - If no model: just check GPU via /api/ps (lightweight check)
#    - Returns (valid, description, error)
# 5. Script propagates the exit code (0/1/2)
#
# Usage: ./verify-gpu-deployment.sh [--url SERVICE_URL] [--model MODEL_NAME] [--inference-timeout DURATION] [--api-timeout DURATION]
#   --url SERVICE_URL: URL of Ollama service (default: http://localhost:11434)
#   --model MODEL_NAME: Model to test for inference (optional, enables truthful validation for fresh services)
#   --inference-timeout DURATION: Timeout for inference requests (e.g., 5m, 30s) - default: 2m
#   --api-timeout DURATION: Timeout for fast API calls like /api/ps (e.g., 30s, 1m) - default: 10s
#
# Legacy usage (deprecated): ./verify-gpu-deployment.sh [SERVICE_URL]
#   SERVICE_URL: URL of Ollama service (positional, deprecated - use --url instead)
#
# Exit codes:
#   0 = Deployment is valid (GPU-accelerated with responses)
#   1 = Deployment is invalid (CPU-only, no response, or inference failed)
#   2 = Error gathering deployment state (API unreachable, timeout, etc.)

set -euo pipefail

# Default values
SERVICE_URL="http://localhost:11434"
MODEL=""
INFERENCE_TIMEOUT=""
API_TIMEOUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)
            SERVICE_URL="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --inference-timeout)
            INFERENCE_TIMEOUT="$2"
            shift 2
            ;;
        --api-timeout)
            API_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            cat >&2 <<'EOF'
verify-gpu-deployment.sh - Verify Ollama GPU deployment

Usage:
  ./verify-gpu-deployment.sh [--url SERVICE_URL] [--model MODEL_NAME] [--inference-timeout DURATION] [--api-timeout DURATION]

Options:
  --url SERVICE_URL              URL of Ollama service (default: http://localhost:11434)
  --model MODEL_NAME             Model to test for inference (optional, enables truthful fresh-service validation)
  --inference-timeout DURATION   Timeout for inference requests (e.g., 5m, 30s) - default: 2m
  --api-timeout DURATION         Timeout for fast API calls like /api/ps (e.g., 30s, 1m) - default: 10s
  -h, --help                     Show this help message

Exit codes:
  0 = Deployment is valid (GPU-accelerated)
  1 = Deployment is invalid (CPU-only, no response, or inference failed)
  2 = Error gathering deployment state (API unreachable, timeout, etc.)
EOF
            exit 0
            ;;
        *)
            # Support legacy positional argument for SERVICE_URL
            if [[ "$SERVICE_URL" == "http://localhost:11434" ]]; then
                SERVICE_URL="$1"
                shift
            else
                echo "ERROR: Unknown argument: $1" >&2
                exit 2
            fi
            ;;
    esac
done

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
status() {
    echo -e "${BLUE}>>>${NC} $*" >&2
}

success() {
    echo -e "${GREEN}✓${NC} $*" >&2
}

warning() {
    echo -e "${YELLOW}⚠${NC}  $*" >&2
}

error() {
    echo -e "${RED}✗${NC}  ERROR: $*" >&2
}

# Main verification
main() {
    status "Verifying GPU deployment using API-based validation"
    status "URL: $SERVICE_URL"
    if [ -n "$MODEL" ]; then
        status "Model: $MODEL (performing inference-first validation)"
    else
        status "Model: (none - GPU-only validation)"
    fi
    if [ -n "$INFERENCE_TIMEOUT" ]; then
        status "Inference timeout: $INFERENCE_TIMEOUT"
    fi
    if [ -n "$API_TIMEOUT" ]; then
        status "API timeout: $API_TIMEOUT"
    fi
    
    echo "" >&2
    
    # Call the Go deployment validator CLI
    # This uses /api/ps for real-time GPU state detection (not logs)
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    
    # Change to script directory first
    cd "$script_dir" || {
        error "Failed to change to script directory"
        exit 2
    }
    
    # Run the Go validator and capture its exit code
    # Use set +e to temporarily disable strict mode so we can capture the exit code
    # without the script terminating on non-zero exit.
    set +e
    
    # Build the command with optional arguments
    local cmd=(go run ./cmd/deploy-check/main.go -url "$SERVICE_URL")
    
    if [ -n "$MODEL" ]; then
        cmd+=(-model "$MODEL")
    fi
    
    if [ -n "$INFERENCE_TIMEOUT" ]; then
        cmd+=(-inference-timeout "$INFERENCE_TIMEOUT")
    fi
    
    if [ -n "$API_TIMEOUT" ]; then
        cmd+=(-api-timeout "$API_TIMEOUT")
    fi
    
    "${cmd[@]}"
    local exit_code=$?
    set -e
    
    # The Go CLI is now responsible for all output (success/failure banners)
    # The shell script only propagates the exit code
    exit $exit_code
}

main "$@"

