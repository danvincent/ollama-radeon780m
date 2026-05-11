# Ollama Development: GPU Investigation Shortcuts

This file documents the exact compile, run, and debug shortcuts used during GPU support investigation on this machine (RADV/Vulkan-first resolution for gfx1103 Phoenix). Use these commands for rapid development iteration and troubleshooting.

---

## Quick Reference: Build Shortcuts

### Vulkan-First Path (RECOMMENDED)

**IMPORTANT:** Vulkan discovery is gated by `OLLAMA_VULKAN=1`. This environment variable must be set to enable Vulkan device discovery (prerequisite for the flow).

**Vulkan build and source-run serve:**
```bash
# 1. Clean rebuild (clears CGO cache)
go clean -cache

# 2. Configure and build Vulkan backend (source build layout)
cmake -B build
cmake --build build

# 3. Run Ollama server with Vulkan enabled (development)
OLLAMA_VULKAN=1 go run . serve

# 4. In another terminal: Check available models
curl -s http://localhost:11434/api/tags | jq .
```

**Quick Vulkan symbols check (libggml-vulkan.so):**
```bash
# After cmake build, inspect symbol table
nm -D build/lib/ollama/libggml-vulkan.so | grep -i "vulkan\|matmul" | head -20

# Validate no unresolved BF16 coopmat references (Phase 1 fix verification)
objdump -T build/lib/ollama/libggml-vulkan.so | grep -i "coopmat\|bf16" || echo "✓ No unresolved BF16 coopmat symbols"
```

---

## Alternative Builds

### ROCm Build (Fallback Path)

**ROCm requires extra flags and will stall at runtime on gfx1103 (known external limitation):**
```bash
# Clean rebuild
go clean -cache

# Configure with Ninja and Clang (required for ROCm)
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release

# Serve (will detect gfx1103 but runtime stalls during model execution)
go run . serve
# Check logs for ROCm device detection: "devices: [gfx1103 Phoenix]"
```

### Go-Only Build (No GPU Acceleration)

**CPU-only fallback if both Vulkan and ROCm builds fail:**
```bash
# Skip CMake entirely
go run . serve
```

---

## Debug: Library and Symbol Verification

### Verify Backend Library Discovery (Source Build)

After a source build with CMake, the development layout should place libraries at `build/lib/ollama/`:

```bash
# Check library resolution in source builds
ls -lh build/lib/ollama/
  # Expected: libggml-base.so, libggml-vulkan.so, libggml-hip.so (if ROCm built), etc.

# Inspect a built library's symbols
nm -D build/lib/ollama/libggml-vulkan.so | wc -l  # Total symbols
nm -D build/lib/ollama/libggml-vulkan.so | grep -i "vk_" | head -5  # Vulkan symbols
```

### Check Running Models

**Check which models are currently loaded:**
```bash
# Terminal 1: Start development runner (with Vulkan enabled)
OLLAMA_VULKAN=1 go run . serve

# Terminal 2: Query loaded models and resource info
curl -s http://localhost:11434/api/ps | jq .

# Example response shows currently loaded models and VRAM usage
# Device detection occurs at server startup (check logs for "detected device: RADV PHOENIX")
```

---

## Environment Variables

### Vulkan Discovery Enablement vs. Library Variant Selection

Two separate mechanisms control Vulkan support:

1. **`OLLAMA_VULKAN=1`** (Discovery Enablement—Required)
   - Gates whether Vulkan backends are discovered at all
   - Without this, discover/runner.go skips Vulkan library directories entirely
   - **Must be set to enable Vulkan discovery**

2. **`OLLAMA_LLM_LIBRARY=vulkan`** (Variant Selection—Optional)
   - After discovery is enabled, selects which specific library variant to use
   - Only useful if multiple backends are discovered and you want to force one
   - Auto-discovery (Phase 2) usually handles this automatically

**Recommended Vulkan-first flow:**
```bash
# Vulkan discovery is enabled via OLLAMA_VULKAN=1; specific library selection is optional
OLLAMA_VULKAN=1 go run . serve
```

### Advanced: Override a Specific Backend (Rarely Needed)

After Phase 2 fix, source builds automatically discover backends. Use only to force a specific library variant:

```bash
# Force a specific backend variant (e.g., "vulkan", "rocm", "cuda_v12") after discovery is enabled
export OLLAMA_VULKAN=1  # Still required for Vulkan discovery
export OLLAMA_LLM_LIBRARY=vulkan
go run . serve

# Or select a different backend entirely
export OLLAMA_LLM_LIBRARY=rocm
go run . serve
```

### Debug Logging

```bash
# Enable debug logging with Vulkan support (includes device detection, backend loading info)
export OLLAMA_VULKAN=1
export OLLAMA_DEBUG=1
go run . serve

# Example log output:
# - "detected device: RADV PHOENIX (VK_DRIVER_ID_RADV)" (Vulkan)
# - "devices: [gfx1103 Phoenix]" (ROCm)
# - Backend library discovery logs
```

---

## Known Pitfalls & Investigation Outcomes

### Vulkan Path: BF16 Cooperative-Matrix Shader Issue (Phase 1, RESOLVED)

**Symptom:** libggml-vulkan.so fails to load with unresolved symbol `_Z...coopmat...@GLIBCXX`.

**Root cause:** Shader generator emitted BF16 cooperative-matrix header declarations but never generated matching shader definitions.

**Fix:** Disable `GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT` in vulkan-shaders-gen.cpp and matching runtime paths in ggml-vulkan.cpp (Phase 1).

**Verification:**
```bash
# Before fix: objdump shows unresolved BF16 coopmat symbol
# After Phase 1: No unresolved symbols
objdump -T build/lib/ollama/libggml-vulkan.so | grep -i "UND\|coopmat"
```

### Vulkan Path: Source-Build Backend Discovery (Phase 2, RESOLVED)

**Symptom:** Source-built Vulkan backend not discovered; runner reports CPU-only or empty devices.

**Root cause:** Source builds did not auto-locate `build/lib/ollama`; required manual `OLLAMA_LLM_LIBRARY` override.

**Fix:** Created `ml/path.go` to detect repo root and resolve backend library path automatically (Phase 2).

**Verification:**
```bash
# After Phase 2: No OLLAMA_LLM_LIBRARY needed, but OLLAMA_VULKAN=1 is still required for discovery
export OLLAMA_VULKAN=1
go run . serve
# Check logs for "detected device" message
```

### ROCm Path: gfx1103 Runtime Stall (Phase 3, EXTERNAL)

**Symptom:** ROCm detects gfx1103 successfully (`rocminfo` shows device, phase tests pass), but execution hangs during model inference.

**Root cause:** External to repo logic—likely GPU driver, ROCm runtime, or hardware-specific resource allocation issue.

**Conclusion:** Phase 3 added boundary and integration tests confirming Ollama-controlled discovery/environment handoff works correctly. Remaining stall is outside repo-controlled code paths.

**Validation:**
```bash
# Phase 3 tests verify discovery and env handoff complete; stall occurs after
go test ./ml -v -run TestGetDevicesFromRunnerWithROCmDevice
go test ./ml -v -run TestROCmDiscoveryFallbackPath
# Tests pass → confirms repo-level paths are correct
# Model execution still hangs → confirms issue is external
```

### Reference: LM Studio Works via Vulkan/RADV

On this machine, LM Studio successfully runs models via Vulkan on RADV. This validates:
- RADV Vulkan driver is functional
- GPU hardware is capable
- Vulkan approach is the practical working solution

Use LM Studio's Vulkan path as a reference if debugging Ollama's Vulkan integration further.

---

## Advanced: Shader and Backend Inspection

### List Available Vulkan Shaders (After Build)

```bash
find build -name "*.spv" -o -name "*vulkan*.glsl" | head -20
ls -lh build/lib/ollama/*.so* | grep vulkan
```

### Inspect Compiled Shader Metadata

```bash
# Inspect compiled SPIR-V shaders (requires spirv-tools to be installed)
# Shaders are stored as .spv files in the build output:
find build -name "*.spv" -type f | head -5

# Disassemble a specific shader (using find to select robustly)
spirv-dis "$(find build -name "*.spv" -type f | head -1)" 2>/dev/null | head -50
```

### Rebuild Just the Vulkan Backend (Faster Iteration)

```bash
# After initial cmake config, rebuild only Vulkan library target
cmake --build build --target ggml-vulkan

# Or rebuild the shader generator (for changes to shader compilation):
cmake --build build --target vulkan-shaders-gen
```

---

## Full Test Suite

### Run All Tests (Post-Build Validation)

```bash
# Run all Go tests (includes ml, discover, runner, backend tests)
go test ./...

# Run only GPU/backend-related tests
go test ./ml ./discover -v

# Run only Vulkan discovery tests
go test ./ml/backend/ggml -v -run "Vulkan"
go test ./discover -v -run "Vulkan"

# Run only ROCm tests (Phase 3) - all in ml package
go test ./ml -v -run "ROCm"
```

---

## Summary: Typical Debug Session

1. **Edit code** (e.g., fix shader generation, backend loader)
2. **Clean and rebuild:**
   ```bash
   go clean -cache
   cmake -B build && cmake --build build
   ```
3. **Run server with Vulkan enabled:**
   ```bash
   OLLAMA_VULKAN=1 go run . serve
   ```
4. **Check logs for device detection:**
   ```bash
   # In another terminal, check the server output for:
   # "detected device: RADV PHOENIX" or similar
   ```
5. **Inspect symbols if build fails:**
   ```bash
   nm -D build/lib/ollama/libggml-vulkan.so | grep UND
   ```
6. **Run tests to validate:**
   ```bash
   go test ./ml ./discover -v
   ```

---

## Additional Resources

- **docs/development.md**: General Ollama build documentation
- **Phase 1 (Vulkan Shader Fix)**: plans/ollama-gpu-resolution-phase-1-complete.md
- **Phase 2 (Backend Discovery)**: plans/ollama-gpu-resolution-phase-2-complete.md
- **Phase 3 (ROCm Investigation)**: plans/ollama-gpu-resolution-phase-3-complete.md
- **ml/path.go**: Library path resolution logic (Phase 2)
- **ml/backend/ggml/ggml/src/ggml-vulkan/**: Vulkan backend source
- **ml/backend/ggml/ggml/src/ggml-hip/**: ROCm/HIP backend source
