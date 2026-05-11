## Phase 1 Complete: Establish Exact Divergence Path

**Plan:** gemma4-vulkan-investigation
**Phase:** 1
**Status:** APPROVED

---

### TL;DR

Phase 1 established the live Gemma4 decision points in Ollama, including the production engine-attempt predicate, pre-Turing CUDA flash-attention gating, and Gemma4 Vulkan partial-offload avoidance. The result is a test-backed baseline that distinguishes code-inspected facts from runtime uncertainties for later phases.

---

### Files Changed

| File | Created / Modified |
|------|--------------------|
| llm/server.go | Modified |
| llm/gemma4_investigation_test.go | Created |
| fs/ggml/ggml.go | Modified |
| plans/PHASE_1_SUMMARY.md | Created |

---

### Functions Changed

| Function / Path | File | Notes |
|-----------------|------|-------|
| `NewLlamaServer` | llm/server.go | Reviewed as part of engine-selection divergence mapping |
| `shouldTryOllamaEngine` | llm/server.go | Production predicate governing whether the Ollama engine is attempted |
| `isPreTuringCUDAGPU` | llm/server.go | Gates flash-attention on pre-Turing CUDA devices for Gemma4 |
| `shouldAvoidPartialVulkanOffload` | llm/server.go | Avoids partial Vulkan offload specifically for Gemma4 models |
| `buildLayout` | llm/server.go | Affected by partial-offload avoidance fallback path |
| `OllamaEngineRequired` | fs/ggml/ggml.go | Model-level predicate consulted during engine selection |

---

### Tests Changed

| Test | File | Notes |
|------|------|-------|
| `TestGemma4ServerSelection` | llm/gemma4_investigation_test.go | Covers engine-attempt predicate for Gemma4 model type |
| `TestGemma4FlashAttentionPreTuringLogic` | llm/gemma4_investigation_test.go | Validates pre-Turing CUDA flash-attention gating logic |
| `TestGemma4VulkanPartialOffloadAvoidance` | llm/gemma4_investigation_test.go | Verifies Gemma4 Vulkan partial-offload is avoided |
| `TestGemma4VulkanPartialOffloadLayoutFallback` | llm/gemma4_investigation_test.go | Validates layout fallback when partial Vulkan offload is suppressed |
| `TestLLMServerAvoidsPartialGemma4VulkanOffload` | llm/gemma4_investigation_test.go | End-to-end check that the server path avoids partial Gemma4 Vulkan offload |

---

### Review Status

**APPROVED** with minor recommendations.

- Phase reviewed and approved with no blocking correctness issues.
- Minor recommendations noted for future phases around runtime validation of guarded claims.

---

### Git Commit Message

```
test: document gemma4 vulkan decisions

- add test-backed seams for Gemma4 engine and FA gating
- cover Vulkan partial offload avoidance and layout fallback
- record phase 1 findings with guarded runtime claims
```

---

### Next Phase

**Phase 2** — Validate runtime behaviour against the code-inspected decision points established in Phase 1.
