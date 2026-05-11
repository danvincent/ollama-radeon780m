## Phase 4 Complete: Add Copilot instructions for build and debug shortcuts

**Plan:** ollama-gpu-resolution
**Phase:** 4 of 4
**Status:** APPROVED

---

### TL;DR

Phase 4 created .github/copilot-instructions.md as a repo-scoped GPU investigation guide covering Vulkan-first and ROCm fallback workflows, build/debug shortcuts, backend library checks, and the key findings from prior phases. The file was iteratively corrected to match the actual repo endpoints, test locations, library names, Vulkan enablement gate, and ROCm/HIP paths, and the approved result is accurate to the current codebase.

---

### Files Changed

| File | Created / Modified |
|------|--------------------|
| .github/copilot-instructions.md | Created |

---

### Functions Changed

| Function / Path | File | Notes |
|-----------------|------|-------|
| N/A | — | No code functions changed in this phase |

---

### Tests Changed

| Test | File | Notes |
|------|------|-------|
| N/A | — | No test files changed in this phase |
| Documentation validation | — | Confirmed the file exists and that endpoints, environment variables, test commands, CMake targets, library names, and repo paths match the codebase |

---

### Review Status

**APPROVED** with minor recommendations.

- Both reviews approved the phase.
- Minor recommendations focused on optional doc-polish follow-ups such as trimming narrative background, clarifying broader ggml test coverage versus narrow Vulkan filters, and keeping Vulkan-first examples consistently gated by OLLAMA_VULKAN=1 in future edits.
- No blocking correctness issues remained.

---

### Git Commit Message

```
chore: add gpu debugging copilot guide

- add repo-scoped build and debug shortcuts for Vulkan-first and ROCm investigation flows
- document validated endpoints, test commands, backend library checks, and environment variable usage
- capture the key findings and pitfalls from the 780M GPU investigation in copilot instructions
```

---

### Next Phase

This was the final phase of the **ollama-gpu-resolution** plan. See `plans/ollama-gpu-resolution-complete.md` for the overall summary when produced.
