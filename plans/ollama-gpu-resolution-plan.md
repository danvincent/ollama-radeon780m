## Plan: Resolve Ollama 780M GPU Support

Prioritize a Vulkan-first recovery path because ROCm now detects gfx1103 but hangs during runtime, while Vulkan appears blocked by a more actionable build and shader-generation defect. In parallel, preserve a ROCm fallback investigation path and include a repo-level Copilot instructions file with the exact compile and debug shortcuts used for this machine.

**Phases**
1. **Phase 1: Fix Vulkan shader generation**
    - **Objective:** Eliminate the unresolved-symbol failure in libggml-vulkan.so by disabling or correcting the false-positive BF16/cooperative-matrix shader path.
    - **Files/Functions to Modify/Create:** ml/backend/ggml/ggml/src/ggml-vulkan/CMakeLists.txt; ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt; ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
    - **Tests to Write:** Targeted coverage for Vulkan shader-generation gating and/or build-time generation expectations around coopmat2/BF16 variants.
    - **Steps:**
        1. Write tests or assertions that fail when header declarations are emitted without matching generated shader definitions.
        2. Implement the minimal change to disable or correctly gate GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT and related BF16 paths for this environment.
        3. Rebuild the Vulkan backend and verify the generated library no longer has unresolved shader symbols.
        4. Re-run targeted tests, then broader relevant tests.

2. **Phase 2: Validate Vulkan backend registration**
    - **Objective:** Confirm the repaired Vulkan backend is actually discovered and registers the RADV Phoenix device instead of silently falling back to CPU.
    - **Files/Functions to Modify/Create:** ml/backend/ggml/ggml/src/ggml-backend-reg.cpp; ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp; discover/runner.go; ml/path.go
    - **Tests to Write:** Focused tests around backend discovery/load behavior and any new guardrails or diagnostics added.
    - **Steps:**
        1. Write tests covering backend load and discovery behavior for a valid Vulkan backend layout.
        2. Add the minimum code or diagnostics needed to ensure a loadable Vulkan backend is surfaced correctly.
        3. Verify the runner /info path reports a Vulkan device instead of [] or CPU-only behavior.
        4. Re-run targeted tests, then broader relevant tests.

3. **Phase 3: Investigate ROCm fallback path**
    - **Objective:** Keep a secondary path open if Vulkan still fails by narrowing the ROCm runtime stall after successful gfx1103 detection.
    - **Files/Functions to Modify/Create:** CMakePresets.json; CMakeLists.txt; envconfig/config.go; discover/runner.go; service/debug documentation only if code changes are warranted.
    - **Tests to Write:** Targeted tests only for any code or config behavior changed; avoid speculative code churn if the issue remains external or runtime-only.
    - **Steps:**
        1. Write tests first for any proposed config or environment-handling changes.
        2. Implement only the minimal safe changes needed for ROCm-specific tuning or diagnostics.
        3. Verify whether reduced allocation pressure or compatibility flags improve runner startup behavior.
        4. Re-run targeted tests, then broader relevant tests.

4. **Phase 4: Add Copilot instructions for build and debug shortcuts [PARALLEL]**
    - **Objective:** Create a repo instructions file with the exact compile, run, and debug shortcuts for this GPU investigation.
    - **Files/Functions to Modify/Create:** .github/copilot-instructions.md
    - **Tests to Write:** None unless the repo has explicit documentation validation for this file.
    - **Steps:**
        1. Gather the final Vulkan-safe, ROCm, runner, serve, symbol-audit, and direct-probe commands used during this work.
        2. Write .github/copilot-instructions.md with machine-specific build and debug recipes, backend layout notes, and known pitfalls.
        3. Verify the commands and paths documented match the implemented approach.

**Open Questions**
1. Should Phase 1 disable only GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT, or also disable BF16 shader generation up front for the first conservative rebuild?
2. If Vulkan becomes loadable but still does not enumerate the device, should Phase 2 stop at diagnostics or include a small loader or discovery fix if one is clearly identified?
