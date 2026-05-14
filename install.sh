#!/usr/bin/env bash

# Strict mode
set -euo pipefail

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_error() {
  echo -e "${RED}ERROR: $1${NC}" >&2
}

print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_info() {
  echo -e "${GREEN}→ $1${NC}"
}

# Step 1: Check if running as root
if [[ $EUID -ne 0 ]]; then
  print_error "This script must be run as root or with sudo"
  exit 1
fi

print_info "Running as root - proceeding with installation"

# Step 2: Detect script directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
print_info "Script directory: $SCRIPT_DIR"

# Step 3: Verify prerequisites
if [[ ! -f "$SCRIPT_DIR/ollama" ]]; then
  print_error "Custom ollama binary not found at $SCRIPT_DIR/ollama"
  exit 1
fi

if [[ ! -d "$SCRIPT_DIR/build/lib/ollama" ]]; then
  print_error "Custom ollama libs not found at $SCRIPT_DIR/build/lib/ollama"
  exit 1
fi

print_success "Prerequisites verified: binary and libs found"

# Step 4: Stop service
print_info "Stopping ollama service..."
systemctl stop ollama || true

# Step 5: Remove apt package
print_info "Removing apt-installed ollama package..."
apt-get remove --purge -y ollama || true

# Step 6: Remove old binaries
print_info "Removing old ollama binaries..."
rm -f /usr/bin/ollama || true
rm -f /usr/local/bin/ollama || true

# Step 7: Remove old libs
print_info "Removing old ollama libraries..."
rm -rf /usr/lib/ollama || true
rm -rf /usr/local/lib/ollama || true

# Step 8: Install binary
print_info "Installing custom ollama binary..."
cp "$SCRIPT_DIR/ollama" /usr/local/bin/ollama
chown root:root /usr/local/bin/ollama
chmod 755 /usr/local/bin/ollama
print_success "Binary installed at /usr/local/bin/ollama"

# Step 9: Install libs
print_info "Installing custom ollama libraries..."
mkdir -p /usr/local/lib/ollama
cp -r "$SCRIPT_DIR/build/lib/ollama/." /usr/local/lib/ollama/
# Set permissions: dirs 755, files 644 for .so files, keep executable bit where needed
chmod -R a+rX /usr/local/lib/ollama/
print_success "Libraries installed at /usr/local/lib/ollama"

# Step 10: Verify service config
print_info "Checking service configuration..."
if [[ -f /etc/systemd/system/ollama.service ]]; then
  if grep -q "OLLAMA_VULKAN=1" /etc/systemd/system/ollama.service; then
    print_success "Service config already has OLLAMA_VULKAN=1"
  else
    print_warning "Service config missing OLLAMA_VULKAN=1 - you may need to add it manually"
  fi
else
  print_warning "Service file not found at /etc/systemd/system/ollama.service"
fi

# Step 11: Reload and restart service
print_info "Reloading systemd configuration and restarting service..."
systemctl daemon-reload
systemctl enable ollama
systemctl restart ollama
print_success "Service restarted"

# Step 12: Health check - poll API for readiness
print_info "Waiting for ollama service to be ready..."
HEALTH_CHECK_MAX_ATTEMPTS=30
HEALTH_CHECK_INTERVAL=1
attempt=0

while [[ $attempt -lt $HEALTH_CHECK_MAX_ATTEMPTS ]]; do
  echo -n "."
  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo ""
    print_success "Service is healthy"
    break
  fi
  sleep "$HEALTH_CHECK_INTERVAL"
  ((attempt++))
done

if [[ $attempt -eq $HEALTH_CHECK_MAX_ATTEMPTS ]]; then
  echo ""
  print_error "Service health check timed out after ${HEALTH_CHECK_MAX_ATTEMPTS}s"
  exit 1
fi

# Step 13: Smoke test Gemma4
print_info "Running smoke test for Gemma4..."
GEMMA4_OUTPUT=$(/usr/local/bin/ollama run gemma4 "What is 2+2? Answer in one word." 2>&1 || true)
echo "Gemma4 output: $GEMMA4_OUTPUT"

if echo "$GEMMA4_OUTPUT" | grep -qi "4"; then
  print_success "Gemma4 smoke test PASSED"
  GEMMA4_PASS=true
else
  print_error "Gemma4 smoke test FAILED - output did not contain '4'"
  GEMMA4_PASS=false
fi

# Step 14: Smoke test Qwen3
print_info "Running smoke test for Qwen3..."
QWEN3_OUTPUT=$(/usr/local/bin/ollama run qwen3:latest "What is 2+2? Answer in one word." 2>&1 || true)
echo "Qwen3 output: $QWEN3_OUTPUT"

if echo "$QWEN3_OUTPUT" | grep -qi "4"; then
  print_success "Qwen3 smoke test PASSED"
  QWEN3_PASS=true
else
  print_error "Qwen3 smoke test FAILED - output did not contain '4'"
  QWEN3_PASS=false
fi

# Step 15: Final summary
echo ""
print_info "=== INSTALLATION SUMMARY ==="
echo "Installed binary: $(/usr/local/bin/ollama --version)"

if [[ "$GEMMA4_PASS" == true && "$QWEN3_PASS" == true ]]; then
  print_success "All tests PASSED - Installation successful!"
  exit 0
else
  print_error "Some tests FAILED - Installation may have issues"
  exit 1
fi
