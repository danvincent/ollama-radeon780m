# Plan Complete: Resolve Ollama 780M GPU Support

This plan delivered a Vulkan-first recovery path for Ollama on the AMD Radeon 780M by fixing a broken Vulkan backend library, restoring source-build backend discovery, documenting a validated GPU debugging workflow, and clarifying that the remaining ROCm gfx1103 stall is outside repo-controlled logic. The result is a repo with stronger GPU backend coverage, a working development-layout Vulkan path, and a machine-specific Copilot guide for compiling and debugging on this hardware.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Fix Vulkan shader generation
2. ✅ Phase 2: Validate Vulkan backend registration
3. ✅ Phase 3: Investigate ROCm fallback path
4. ✅ Phase 4: Add Copilot instructions for build and debug shortcuts

**All Files Created/Modified:**
- ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
- ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp
- ml/path.go
- ml/path_test.go
- ml/backend/ggml/ggml/src/ggml.go
- ml/backend/ggml/vulkan_discovery_test.go
- discover/vulkan_discovery_test.go
- ml/device_rocm_boundary_test.go
- ml/device_rocm_env_test.go
- ml/device_rocm_filter_test.go
- ml/device_rocm_fallback_integration_test.go
- ml/device_rocm_phases_test.go (deleted)
- .github/copilot-instructions.md
- plans/ollama-gpu-resolution-plan.md
- plans/ollama-gpu-resolution-phase-1-complete.md
- plans/ollama-gpu-resolution-phase-2-complete.md
- plans/ollama-gpu-resolution-phase-3-complete.md
- plans/ollama-gpu-resolution-phase-4-complete.md

**Key Functions/Classes Added:**
- matmul_shaders in ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
- BF16 cooperative-matrix pipeline selection/setup paths in ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp
- ResolveLibraryPath in ml/path.go
- findRepoRoot in ml/path.go
- GGML backend library loading path resolution in ml/backend/ggml/ggml/src/ggml.go
- TestGetDevicesFromRunnerWithROCmDevice in ml/device_rocm_boundary_test.go
- TestGetDevicesFromRunnerBoundaryWithTimeout in ml/device_rocm_boundary_test.go
- TestGetDevicesFromRunnerBoundaryWithRunnerExit in ml/device_rocm_boundary_test.go

**Test Coverage:**
- Total tests written: 20
- All tests passing: ✅

**Recommendations for Next Steps:**
- Prefer the Vulkan path on this machine and keep OLLAMA_VULKAN=1 in source-build debug workflows.
- Treat the remaining ROCm gfx1103 model-load stall as an external runtime/driver issue unless new repo-local evidence appears.
- Optionally follow up on minor cleanup noted in reviews, such as trimming duplicated helper-test coverage or keeping the Copilot instructions maximally concise.
