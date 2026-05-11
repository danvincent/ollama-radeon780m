## Phase 2 Complete: Audit llama.cpp Vulkan Gemma4 Support

**Plan:** gemma4-vulkan-investigation
**Phase:** 2
**Status:** APPROVED

---

### TL;DR

Phase 2 completed a source-level audit of the pinned llama.cpp Vulkan path and found no explicit source-level capability block for Gemma4-relevant attention shapes. Production code remained unchanged; test/documentation changes now accurately describe this as a source audit plus Ollama gating validation, not runtime proof.

---

### Files Changed

| File | Created / Modified |
|------|--------------------|
| llm/gemma4_vulkan_gating_test.go | Created |
| plans/gemma4-vulkan-investigation-phase-2-complete.md | Modified |

> `llm/phase2_vulkan_audit_test.go` was removed and replaced by `llm/gemma4_vulkan_gating_test.go` — the phase-numbered file was maintenance debt; the new file names the subject under test rather than the phase.

**No production code changes were made.**

---

### Functions Changed

| Function / Path | File | Notes |
|-----------------|------|-------|
| `shouldAvoidPartialVulkanOffload` | llm/server.go | Validated by source audit and automated tests; not modified |

---

### Tests Changed

| Test | File | Notes |
|------|------|-------|
| `TestShouldAvoidPartialVulkanOffloadGemma4` | llm/gemma4_vulkan_gating_test.go | Validates Ollama's gating logic avoids partial Vulkan offload for Gemma4 while allowing full and no-offload; does not prove llama.cpp runtime correctness |
| `TestShouldAvoidPartialVulkanOffloadEdgeCases` | llm/gemma4_vulkan_gating_test.go | Validates edge cases: single-layer models, near-full offload, non-Gemma4 models, alternative backends |

---

### Review Status

**APPROVED** with minor recommendations.

- Source audit of `ggml-vulkan.cpp` found no explicit rejection of 512×512 attention shapes; workgroup adjustment at that size is a performance tuning, not a capability block.
- Automated tests confirm Ollama's gating logic correctly implements its conservative strategy.
- Minor recommendations noted: runtime correctness of full Vulkan offload and root cause of partial-offload failure remain unverified and are deferred to Phase 3+.

---

### Git Commit Message

```
test: clarify gemma4 vulkan audit scope

- replace phase-scoped audit tests with stable gating coverage
- separate source-audit findings from automated test evidence
- keep phase 2 audit-only for production code
```

---

### Next Phase

**Phase 3** — Audit Ollama Layout / Memory / Partial-Offload Logic

Focus on determining whether the root cause of Gemma4+Vulkan partial-offload failure is memory layout corruption, scheduler bug, or shader limitation. May also verify whether full Vulkan offload is actually safe or should also be disabled.
