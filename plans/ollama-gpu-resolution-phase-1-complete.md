## Phase 1 Complete: Fix Vulkan shader generation

**Plan:** ollama-gpu-resolution
**Phase:** 1 of 4
**Status:** APPROVED

---

### TL;DR

Phase 1 fixed the broken Vulkan shared library by disabling BF16 cooperative-matrix shader generation and the matching runtime pipeline paths that referenced symbols never emitted by the shader generator. The Vulkan target now builds cleanly and libggml-vulkan.so loads without the prior unresolved BF16 cooperative-matrix symbols.

---

### Files Changed

| File | Created / Modified |
|------|--------------------|
| ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp | Modified |
| ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp | Modified |

---

### Functions Changed

| Function / Path | File | Notes |
|-----------------|------|-------|
| `matmul_shaders` | ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp | Disabled BF16 cooperative-matrix shader generation to eliminate unresolved-symbol emission |
| BF16 cooperative-matrix pipeline selection/setup paths | ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp | Disabled matching runtime pipeline paths; scalar fallback retained |

---

### Tests Changed

No new source test files were added in this phase.

- Targeted Vulkan build and shared-library validation passed.
- Relevant Go package tests passed for `discover`, `llm`, `ml/backend/ggml`, and broader impacted `ml` packages.

---

### Review Status

**APPROVED** with minor recommendations.

- Both reviews approved the fix.
- Minor recommendations focused on future cleanup of `#if 0` dead-code guards and redundant/unused BF16 coopmat-related state.
- No blocking correctness issues were found.

---

### Git Commit Message

```
fix: disable broken bf16 vulkan coopmat path

- stop generating BF16 cooperative-matrix shaders that were leaving unresolved symbols
- disable matching Vulkan BF16 coopmat runtime pipeline paths and keep scalar fallback
- restore a loadable libggml-vulkan.so and keep relevant tests passing
```

---

### Next Phase

**Phase 2: Validate Vulkan backend registration** — Confirm the repaired Vulkan backend is actually discovered and registers the RADV Phoenix device instead of silently falling back to CPU.
