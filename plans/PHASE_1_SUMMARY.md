# Phase 1: Gemma4 Vulkan Investigation - Implementation Summary

## Objective
Investigate which Ollama execution path Gemma4 is taking and identify any decision points that affect behavior.

## Status: ✅ COMPLETE

### Key Findings

#### 1. Server Selection Path (ollamaServer vs llamaServer)

**By code inspection**: `NewLlamaServer()` in `llm/server.go` calls `shouldTryOllamaEngine(envconfig.NewEngine(), f.KV())` to decide whether to attempt the Ollama engine. The helper returns `true` if either the `OLLAMA_NEW_ENGINE` environment variable is set or the model's KV metadata reports `OllamaEngineRequired() == true`. Gemma4 is listed in `OllamaEngineRequired()` in `fs/ggml/ggml.go`.

When `shouldTryOllamaEngine()` returns `true`, `NewLlamaServer()` attempts tokenizer initialization via `model.NewTextProcessor()`:
- Success → **ollamaServer** (new Ollama engine)
- Failure → **llamaServer** (legacy llama.cpp runner, fallback)

**Code locations**: `llm/server.go` — `NewLlamaServer()` and `shouldTryOllamaEngine()` helper; `fs/ggml/ggml.go` — `OllamaEngineRequired()`

**Test coverage**: `TestGemma4ServerSelection` exercises the `shouldTryOllamaEngine()` helper directly and confirms:
- Gemma4 is included in `OllamaEngineRequired()` — the KV-based branch returns `true`
- The env-override branch (`shouldTryOllamaEngine(true, any_kv)`) returns `true`
- A non-Ollama-engine model (`shouldTryOllamaEngine(false, non_ollama_kv)`) returns `false`

These tests verify the helper's logic in isolation. That `NewLlamaServer()` calls this helper is established by code inspection, not by tests that invoke `NewLlamaServer()` end-to-end.

---

#### 2. Flash Attention Pre-Turing GPU Logic

**By code inspection**: Gemma4 enables flash attention by default (per `FlashAttention()` in `fs/ggml/ggml.go`). A GPU architecture check in `NewLlamaServer()` then overrides this for CUDA devices:
- Pre-Turing CUDA (Maxwell compute 5.x, Pascal 6.x, Volta 7.0–7.2): FA disabled
- Turing+ CUDA (compute ≥ 7.5): FA remains enabled
- Non-CUDA backends: unaffected (the override is CUDA-specific)

**Code locations**: `llm/server.go` — flash-attention section of `NewLlamaServer()`; `isPreTuringCUDAGPU()` helper

**Test coverage**: `TestGemma4FlashAttentionPreTuringLogic` exercises the `isPreTuringCUDAGPU()` helper across pre-Turing inputs (Maxwell, Pascal, Volta), Turing+ inputs (7.5, 7.6, Ampere), and a non-CUDA input (Vulkan), confirming the predicate classifies each case correctly.

---

#### 3. Vulkan Partial Offload Behavior

**By code inspection**: `buildLayout()` in `llm/server.go` calls `shouldAvoidPartialVulkanOffload()`. When that helper returns `true` (Gemma4 + Vulkan backend), partial GPU offload is suppressed — all layers fall back to CPU. Full offload (all layers on GPU) and no-offload (all layers on CPU) are not affected. Other models and the CUDA backend are not affected.

**Code locations**: `llm/server.go` — `buildLayout()` and `shouldAvoidPartialVulkanOffload()` helper

**Test coverage**:
- `TestGemma4VulkanPartialOffloadAvoidance`: exercises `shouldAvoidPartialVulkanOffload()` for Gemma4 partial/full/no-offload cases, a non-Gemma4 model, and the CUDA backend
- `TestGemma4VulkanPartialOffloadLayoutFallback`: integration tests that construct real layout objects and confirm GPU layer count is zero for Gemma4 partial-Vulkan scenarios

**Note**: This is a conservative safety measure in the code. The root cause of the restriction is not yet characterized and remains a Phase 2/3 objective.

---

### Code Changes

#### Files Modified

1. **llm/server.go**
   - Added `modelArch` field to `llmServer` struct for architecture tracking
   - Refactored engine-selection logic in `NewLlamaServer()` to call the extracted `shouldTryOllamaEngine()` helper rather than inlining the predicate
   - Added `isPreTuringCUDAGPU()` production helper (shared with tests) driving the flash-attention override
   - Added `shouldTryOllamaEngine(envNewEngine bool, kv ggml.KV)` production helper (called by `NewLlamaServer()`)
   - Added/retained `shouldAvoidPartialVulkanOffload()` helper

2. **fs/ggml/ggml.go**
   - Updated `OllamaEngineRequired()` comments to describe observable behavior rather than asserting root causes
   - Updated `FlashAttention()` comments similarly

3. **llm/gemma4_investigation_test.go**
   - `TestGemma4ServerSelection`: exercises all three branches of `shouldTryOllamaEngine()` (env-override, KV-based, fallback)
   - `TestGemma4FlashAttentionPreTuringLogic`: exercises `isPreTuringCUDAGPU()` across GPU architecture variants
   - `TestGemma4VulkanPartialOffloadAvoidance`: exercises `shouldAvoidPartialVulkanOffload()` across offload and backend variants
   - `TestGemma4VulkanPartialOffloadLayoutFallback`: integration tests for layout fallback behavior

---

### Decision Points Identified

```
GEMMA4 EXECUTION DECISION TREE
(code inspection + helper-level tests)

1. Engine Selection  [NewLlamaServer]
   ├─ shouldTryOllamaEngine(envconfig.NewEngine(), f.KV())
   │  ├─ OLLAMA_NEW_ENGINE set → true  (all models)       [test-verified]
   │  ├─ model in OllamaEngineRequired() → true  (Gemma4) [test-verified]
   │  └─ neither → false  (legacy llama.cpp path)         [test-verified]
   └─ If true: attempt model.NewTextProcessor()
      ├─ Success → ollamaServer
      └─ Failure → llamaServer (fallback)                  [code inspection only]

2. Flash Attention   [NewLlamaServer, CUDA only]
   ├─ CUDA + pre-Turing (compute < 7.5) → FA disabled     [test-verified]
   ├─ CUDA + Turing+ (compute ≥ 7.5)   → FA enabled       [test-verified]
   └─ Non-CUDA backend                 → FA state unchanged [test-verified]

3. Vulkan Partial Offload  [buildLayout]
   ├─ Gemma4 + Vulkan + partial layers → CPU fallback      [test-verified]
   ├─ Gemma4 + Vulkan + full/no offload → unaffected       [test-verified]
   └─ Other models / CUDA              → unaffected        [test-verified]
```

---

### What Is Established vs What Remains Uncertain

**Established by code inspection**:
- `NewLlamaServer()` calls `shouldTryOllamaEngine()` to gate engine selection
- Gemma4 is listed in `OllamaEngineRequired()`
- The flash-attention pre-Turing override exists in `NewLlamaServer()` and uses `isPreTuringCUDAGPU()`
- `buildLayout()` calls `shouldAvoidPartialVulkanOffload()` and suppresses GPU layers when it returns `true`

**Verified by tests** (production helper functions exercised directly):
- All three branches of `shouldTryOllamaEngine()` return the expected value
- `isPreTuringCUDAGPU()` correctly classifies all relevant GPU architectures
- `shouldAvoidPartialVulkanOffload()` correctly identifies Gemma4 + Vulkan + partial-offload cases
- Layout integration produces zero GPU layers for Gemma4 partial-Vulkan scenarios

**Not tested / runtime-only**:
- The end-to-end path through `NewLlamaServer()` for a real Gemma4 model file
- Whether tokenizer initialization succeeds or falls back in practice
- Actual inference correctness on affected GPU configurations

---

### Test Results

All Phase 1 tests pass:
```
✓ TestGemma4ServerSelection
  - shouldTryOllamaEngine: env-override branch, KV-based branch, fallback branch

✓ TestGemma4FlashAttentionPreTuringLogic
  - isPreTuringCUDAGPU: Maxwell, Pascal, Volta (pre-Turing)
                        7.5, 7.6, Ampere (Turing+)
                        Vulkan (non-CUDA, unaffected)

✓ TestGemma4VulkanPartialOffloadAvoidance
  - shouldAvoidPartialVulkanOffload: Gemma4 partial, full, no-offload
                                     other models, CUDA backend

✓ TestGemma4VulkanPartialOffloadLayoutFallback
  - Layout: Gemma4 partial Vulkan → GPU=0 (fallback)
            Gemma4 full Vulkan   → GPU=max (allowed)
            Gemma3 partial Vulkan → GPU>0 (other models unaffected)

Full llm test suite: PASS — no regressions
```

---

### Open Questions for Phase 2/3

1. **Tokenizer initialization**: Under what conditions does `model.NewTextProcessor()` fail for Gemma4, and what is the performance difference between the ollamaServer and llamaServer paths?
2. **Vulkan partial-offload root cause**: Why is partial Vulkan offload problematic for Gemma4 specifically?
3. **Flash attention on pre-Turing GPUs**: Can alternative kernels be used? What is the performance impact of disabling FA?
4. **LM Studio comparison**: How does LM Studio handle the same model/hardware scenarios?

---

## Summary

Phase 1 documents three observable Gemma4 decision points identified through code inspection and validated by targeted helper-level tests:

1. **Engine selection**: Gemma4 is configured to attempt the Ollama engine path. By code inspection, `NewLlamaServer()` delegates to `shouldTryOllamaEngine()`; tests verify all three branches of that helper return correct values.
2. **Flash attention gating**: A CUDA GPU architecture check in `NewLlamaServer()` disables flash attention for pre-Turing devices. The `isPreTuringCUDAGPU()` helper driving this check is test-verified across all relevant GPU tiers.
3. **Vulkan partial-offload guard**: `buildLayout()` disables partial Vulkan offload for Gemma4 via `shouldAvoidPartialVulkanOffload()`, which is test-verified; integration tests confirm the resulting zero-GPU-layer fallback.

All findings are grounded either in direct code inspection or in tests that exercise production helper functions. End-to-end runtime behavior through `NewLlamaServer()` has not been exercised by tests and remains a Phase 2/3 objective.
