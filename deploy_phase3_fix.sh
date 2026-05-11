#!/bin/bash
# Deploy Phase 3 fix (full offload retry logic) for Gemma4+Vulkan
# This script deploys the newly built binary with the Phase 3 fix to production

set -euo pipefail

REPO_DIR="/home/daniel/source/ollama"
SOURCE_BINARY="${REPO_DIR}/ollama"
SOURCE_VULKAN_LIB="${REPO_DIR}/build/lib/ollama/libggml-vulkan.so"

DEST_BINARY="/usr/local/bin/ollama"
DEST_VULKAN_LIB="/usr/local/lib/ollama/vulkan/libggml-vulkan.so"

OLLAMA_SERVICE="ollama.service"

echo "=== Phase 3 Fix Deployment Script ==="
echo ""
echo "Verifying source binaries exist..."
if [ ! -f "$SOURCE_BINARY" ]; then
    echo "ERROR: Source binary not found at $SOURCE_BINARY"
    exit 1
fi
if [ ! -f "$SOURCE_VULKAN_LIB" ]; then
    echo "ERROR: Source Vulkan library not found at $SOURCE_VULKAN_LIB"
    exit 1
fi

echo "✓ Source binaries found"
echo "  Binary: $(stat -c '%y' "$SOURCE_BINARY" | cut -d. -f1)"
echo "  Vulkan: $(stat -c '%y' "$SOURCE_VULKAN_LIB" | cut -d. -f1)"
echo ""

echo "Stopping Ollama service..."
sudo systemctl stop "$OLLAMA_SERVICE"
sleep 2

echo "Deploying Phase 3 fix binary..."
sudo cp "$SOURCE_BINARY" "$DEST_BINARY"
sudo cp "$SOURCE_VULKAN_LIB" "$DEST_VULKAN_LIB"
echo "✓ Binaries deployed"

echo ""
echo "Starting Ollama service..."
sudo systemctl start "$OLLAMA_SERVICE"
sleep 5

if systemctl is-active --quiet "$OLLAMA_SERVICE"; then
    echo "✓ Service started successfully"
else
    echo "✗ Service failed to start - checking logs:"
    journalctl -u "$OLLAMA_SERVICE" -n 10 --no-pager
    exit 1
fi

echo ""
echo "=== Deployment Complete ==="
echo "Phase 3 fix is now active on the UM790 Ollama installation."
echo ""
echo "Next: Run validation tests to verify Gemma4+Vulkan behavior:"
echo "  ollama run gemma4:e2b 'What is the capital of Iceland?'"
echo "  ollama run gemma4:e2b 'list the states of America'"
