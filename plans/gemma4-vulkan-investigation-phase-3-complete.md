# Phase 3 Complete: Audit Ollama Layout / Memory / Partial-Offload Logic

**Plan:** gemma4-vulkan-investigation  
**Phase:** 3  
**Status:** APPROVED

---

### TL;DR

Phase 3 implemented the minimal production fix in Ollama's layout logic so Gemma4+Vulkan no longer drops directly to CPU when partial offload is blocked; it first retries a safe full-offload layout and only falls back to CPU if full offload cannot fit. The phase also added assertion-based regression coverage for the retry and integration paths.

> **Scope note:** This fix is a **layout strategy improvement** — it makes Ollama choose a safer and more capable offload mode before surrendering to CPU. It is **not** a proven root-cause kernel fix. The underlying reason partial Vulkan offload is incorrect for Gemma4 (memory layout corruption, shader limitation, or scheduler bug) remains unverified and is deferred to Phase 4 runtime experiments.

---

### Files Changed

| File | Created / Modified |
|------|--------------------|
| `llm/server.go` | Modified |
| `llm/server_test.go` | Modified |
| `llm/gemma4_phase3_investigation_test.go` | Created |

---

### Functions Changed

| Function | File | Notes |
|----------|------|-------|
| `buildLayout` | `llm/server.go` | Added full-offload retry step before CPU fallback when partial Vulkan offload is suppressed for Gemma4 |
| `shouldAvoidPartialVulkanOffload` | `llm/server.go` | Retained as-is; partial-offload safety guard kept in place for correctness |

---

### Tests Changed

| Test | File | Purpose |
|------|------|---------|
| `TestLLMServerAvoidsPartialGemma4VulkanOffload` | `llm/server_test.go` | Confirms server path still avoids partial Gemma4 Vulkan offload after the retry logic was introduced |
| `TestPhase3Gemma4VulkanFullOffloadFallback` | `llm/gemma4_phase3_investigation_test.go` | Validates that `buildLayout` retries full offload before falling back to CPU when partial Vulkan offload is blocked |
| `TestPhase3Gemma4PartialOffloadGuardCornerCases` | `llm/gemma4_phase3_investigation_test.go` | Validates guard edge cases: single-layer models, near-full offload, non-Gemma4 models, and non-Vulkan backends |
| `TestPhase3Gemma4LayoutRetryBehavior` | `llm/gemma4_phase3_investigation_test.go` | Asserts the retry path is exercised and that the final layout reflects a full-offload decision, not a CPU-fallback decision |
| `TestPhase3Gemma4CreateLayoutIntegration` | `llm/gemma4_phase3_investigation_test.go` | End-to-end integration path check: Gemma4+Vulkan reaches the correct layout outcome through the production `buildLayout` call chain |

---

### Review Status

**APPROVED**

- Fix is narrowly scoped: only the CPU-fallback path inside `buildLayout` is changed; the partial-offload guard itself is untouched.
- Regression coverage added for all directly affected code paths.
- No speculative changes; fix is conservative and reversible.

---

### Git Commit Message

```
fix: retry full gemma4 vulkan offload

- retry full offload before cpu fallback for gemma4 vulkan
- add regression coverage for layout retry and integration paths
- keep partial-offload safety guard in place for correctness
```
