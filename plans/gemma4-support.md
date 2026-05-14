# Gemma4 Vulkan Support — Investigation Log

> **Living reference document.** Updated as new findings are confirmed.  
> All facts below have been verified against source code or observed behaviour.

---

## Environment

| Item | Value |
|---|---|
| GPU | AMD Radeon 780M / RADV PHOENIX (gfx1103) |
| Ollama repo | `/run/media/daniel/94f067f6-2bbf-4cd3-b4a3-94e3ec9bb7c5/daniel/source/ollama` |
| Working reference llama.cpp | `~/source-usb/llama.cpp/` (Gemma4 runs correctly on Vulkan here) |
| Model under test | `gemma4` (8B Q4_K_M) |
| Vulkan enabled via | `OLLAMA_VULKAN=1` |
| Dev run command | `OLLAMA_VULKAN=1 go run . serve` |

---

## Plan File

See [plans/gemma4-vulkan-plan.md](gemma4-vulkan-plan.md) for the current investigation plan.

## Architecture Notes

### How llama.cpp integrates with Ollama

- `llama/llama.cpp/` is a **vendored git submodule** of a modified llama.cpp
- Compiled as shared libraries: `libggml-base.so`, `libggml-vulkan.so`, etc.
- The **Go engine** reimplements model logic in Go (`model/models/gemma4/model_text.go`) and calls GGML ops via CGo
- The **C++ llama.cpp model code** (`llama/llama.cpp/src/models/gemma4.cpp`) is NOT on the runtime path when the Go engine is active
- `llm/server.go` selects Go engine when `OllamaEngineRequired()` returns true for the GGUF, or when `OLLAMA_NEW_ENGINE=1`
- The `~/source-usb/llama.cpp/` is a standalone reference build — NOT the same codebase, cannot be used as a drop-in library

### Can we use upstream llama.cpp as a library?

No. The ollama-embedded llama.cpp has diverged from upstream (custom patches at `llama/patches/`), uses a different build system, and has a different ABI. The reference build at `~/source-usb/llama.cpp/` is useful for:
- Running `llama-server` to get ground-truth outputs
- Reading the C++ code to understand correct model architecture
- NOT as a drop-in replacement

### Previously Committed Fixes (may or may not be on active runtime path)

| Commit | Fix | On Runtime Path? |
|--------|-----|-----------------|
| `9744dab` | Vulkan MoE stride fix (RC2) + FFN F32 precision (RC1) | RC2: Yes (Vulkan backend). RC1: No (Vulkan ignores F32 prec hint) |
| `abf74fae` | FA guard for Vulkan (RC3) + ggml_cont on C++ MoE views (RC4) + compat FA fix | RC3: Yes (confirmed in logs). RC4: No (C++ path not used). Compat FA fix: Only if Go tokenizer fails |

---

## Symptom

- **CPU run:** Also gibberish (different wrong languages) — same code path as Vulkan for the PLE bug
- **Vulkan run:** Thai/off-topic fantasy prose completely unrelated to the prompt.
- **Vulkan IS active (confirmed):** 42/43 layers offloaded to GPU; `library=Vulkan`; `AMD Radeon 780M Graphics (RADV PHOENIX)` detected in logs.
- **Large allocation failure observed:** `alloc_tensor_range: failed to allocate Vulkan0 buffer of size 5637144576` (~5.25 GiB).
- Since llama.cpp on GPU works, the bug is in the Ollama fork's divergence from upstream — **NOT** in the Vulkan shaders.

---

## Fixes Already Applied (did not resolve live output)

The following changes are committed in the working tree but did **not** fix the gibberish Vulkan output on their own:

1. **`build_ffn()` in `llama/llama.cpp/src/llama-graph.cpp`** — Gemma4 now forces F32 on `up`, `gate`, and `down` FFN projections.
2. **`build_moe_ffn()`** — MoE expert projections use `ggml_mul_mat_id_set_prec(GGML_PREC_F32)`.
3. **ISWA attention path** — Gemma4 forces F32 on the attention output projection.
4. **`ggml_mul_mat_id_set_prec()` API added to GGML** (`ggml.h`, `ggml.c`) and wired into the Vulkan backend (`ggml-vulkan.cpp`).
5. **Phase 1 original: `ml/path.go`** — source-build library path auto-discovery.
6. **Phase 2 original: ISWA attention F32 precision.**

---

## Final Resolution (CONFIRMED FIXED — commit `786f76ff`)

### Root Cause: Non-contiguous Per-Layer Input tensor in Vulkan element-wise ops

**File:** `model/models/gemma4/model_text.go`

#### What was wrong

`PerLayerProjector.Forward` returned the per-layer inputs tensor with shape `[pleDim, nLayer, nTokens]`.
In `TextModel.Forward`, each layer's PLE slice was extracted as a **strided non-contiguous 2D view**:
- Stride for the token dimension = `pleDim × nLayer × sizeof(float)` (not the natural `pleDim × sizeof`)
- This non-contiguous tensor was then used in `ggml_geglu_split` (element-wise GELU-gate × perLayerInput)

The Vulkan GGML backend's element-wise kernels do not correctly handle non-contiguous tensors, producing
garbage values for all 42 Gemma4 layers. (The CPU backend handles non-contiguous strides natively, but
still produced wrong output because both paths hit the same strided view.)

#### What llama.cpp does (the reference)

`project_per_layer_inputs()` in `gemma4.cpp` explicitly calls:
```cpp
inp_per_layer = ggml_cont(ggml_permute(inp_per_layer, 0, 2, 1, 3));
```
This permutes `[pleDim, nLayer, nTokens]` → `[pleDim, nTokens, nLayer]` and makes it contiguous,
so each per-layer slice `[pleDim, nTokens]` extracted by `view_2d_slice` is fully contiguous.

#### The fix

In `PerLayerProjector.Forward`, changed the return to:
```go
return perLayerProjection.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
```

In `TextModel.Forward`, updated the View extraction to match the new `[pleDim, nTokens, nLayer]` layout:
```go
// perLayerInputs: [pleDim, nTokens, nLayer] — contiguous, per-layer slices on 3rd axis
perLayerInput = perLayerInputs.View(ctx, i*perLayerInputs.Stride(2),
    perLayerInputs.Dim(0), perLayerInputs.Stride(1), perLayerInputs.Dim(1))
```
`Stride(1) = pleDim × sizeof` is the natural contiguous stride — resulting view IS contiguous. ✓

#### Verification

- **Gemma4 8B Q4_K_M** on RADV PHOENIX (Vulkan): correctly listed all 50 US states in English ✅
- **Qwen3 regression test**: `2+2=4` ✅
- Commit: `786f76ff` on branch `radeon-780m-support-attempt-1`

---

## Root Cause (Confirmed)

### Global RoPE frequency factors loader bug (fixed)

- `TextSelfAttention.RopeFactors` was previously tagged as `gguf:"rope_freqs.weight"` inside `Layers []TextLayer \`gguf:"blk"\``, which made GGUF lookup search for nonexistent per-layer tensors like `blk.N.rope_freqs.weight`.
- The actual tensor is model-level `rope_freqs.weight`, so global attention layers were always missing proportional RoPE factors and therefore applied incorrect attention.
- Fix applied in `model/models/gemma4/model_text.go`: load `RopeFactors` on `TextModel`, copy it into `TextOptions` at the start of `TextModel.Forward`, remove the nested attention field, and consume `opts.RopeFactors` in `TextSelfAttention.Forward`.


### Primary Bug — Wrong `batch_stride_a` for non-contiguous-dim2 MoE weight views

**File:** `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp`  
**Lines:** ~7685 (`ggml_vk_mul_mat_id_q_f16` prefill path) and ~7918 (`ggml_vk_mul_mat_vec_id_q_f16` decode path)

#### What happens

- `gemma4.cpp` lines 169–186 create `gate_exps` and `up_exps` as `ggml_view_3d` views from a single fused `ffn_gate_up_exps` tensor.
- These views are **dim01-contiguous** (`nb[1] == ne[0] * nb[0]`), so `ggml_vk_dim01_contiguous()` returns `true`.
- However, they are **NOT fully contiguous**: `nb[2] == 2 × ne[1] × nb[1]` — experts are spaced twice as far apart as in a standalone tensor.
- Both Vulkan `mul_mat_id` kernel paths compute the expert stride as `ne00 * ne01` (the fully-contiguous assumption).
- The GPU shader reads expert `e` at offset `e × ne00 × ne01` instead of the correct `e × (nb[2] / nb[0])`.
- This reads interleaved gate+up weights for every expert except expert 0 → completely wrong MoE computation → gibberish output on GPU.
- The CPU path uses `nb[2]` natively → correct output.

#### Why the reference llama.cpp works on Vulkan

Recent upstream llama.cpp either stores separate `ffn_gate_exps` / `ffn_up_exps` tensors (no fused view) or has already patched this stride calculation. The fused-tensor loading path was backported into this Ollama fork's `gemma4.cpp` without the corresponding Vulkan stride fix.

---

### Secondary Issue — `GGML_PREC_F32` overrides are no-ops on Vulkan

- `ggml_mul_mat_set_prec()` and `ggml_mul_mat_id_set_prec()` write their precision flag into `dst->op_params[0]`.
- The Vulkan matmul shader uses a **specialization constant** for accumulation precision (baked at pipeline-creation time) and never reads `op_params[0]`.
- This may introduce subtle numerical errors on top of the MoE stride bug, but it is **not** the primary cause of complete gibberish output.
- Long-term fix: plumb `GGML_PREC_F32` through to the Vulkan matmul shader specialisation constants.

---

### Tertiary — No Vulkan FA guard in `llm/server.go`

- Lines 213–223 guard Gemma4 from flash-attention only for CUDA on pre-Turing hardware.
- There is no equivalent guard for the Vulkan backend.
- For RDNA3, the FA scalar path with HSK=512 compiles and runs correctly via spec constants → **not** the cause of gibberish, but parity with the CUDA guard is warranted.

---

## Fix Options

### Option A — Targeted `batch_stride_a` fix *(Recommended long-term)*

In both `ggml_vk_mul_mat_id_q_f16` (line ~7685) and `ggml_vk_mul_mat_vec_id_q_f16` (line ~7918), replace the hardcoded `ne00 * ne01` expert stride with an `nb[2]`-derived stride when data has **not** been copied to a scratch buffer:

- If `qx_needs_dequant || x_non_contig` → data was already copied to a contiguous buffer → use `ne00 * ne01` (current behaviour, correct).
- Otherwise → compute actual stride from `src0->nb[2]`:
  ```
  stride_factor   = nb[2] / (nb[1] * ne01)
  stride_batch_x  = ne00 * ne01 * stride_factor
  ```

### Option B — Force copy for non-fully-contiguous weights *(Recommended first)*

Add `!ggml_is_contiguous(src0)` to the `x_non_contig` check in both `mul_mat_id` paths. This forces a D2D copy to a contiguous buffer for view tensors — correct by construction. Estimated ~2 % performance overhead. Simpler to implement; no stride formula needed.

### Option C — Upstream workaround in `gemma4.cpp`

In `gemma4.cpp` lines 169–186, replace the `ggml_view_3d` views with `ggml_cont()` copies. Avoids the non-contiguous stride entirely at the model graph level. Extra memory cost: duplicate expert weights resident in GPU VRAM.

---

**Recommended approach:** Implement **Option B** first (safest, easiest to verify correct), then follow up with **Option A** as a performance optimisation once Gemma4 is confirmed working end-to-end.

---

## Files to Modify for Fix

| File | Purpose |
|---|---|
| `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Primary fix target — lines ~7685 and ~7918 |
| `llama/llama.cpp/src/models/gemma4.cpp` | Option C alternative only |
| `llm/server.go` | Add Vulkan FA guard parity (tertiary, not blocking) |

---

## Open Questions

1. Does `nb[2] / (nb[1] * ne01)` hold for all quantized types (Q4_K, Q6_K)?
2. Do standard Gemma4 GGUF models from HuggingFace use fused `ffn_gate_up_exps` or separate tensors? (If separate, the bug only affects users with specific GGUF pipelines.)
3. Is there a similar non-contiguous dim2 issue in the `ggml_vk_mul_mat` (non-ID) path?
4. After the stride fix, does fp16 accumulation in MoE matmuls still cause wrong values? (This would determine whether the secondary PREC_F32 fix is also required.)

---

## Reference — Useful Commands

```bash
# Build Vulkan backend
cmake -B build && cmake --build build

# Run isolated server on port 11435 (repo-local, not system service)
pkill ollama || true
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_VULKAN=1 OLLAMA_DEBUG=1 go run . serve > /tmp/ollama-vulkan-test.log 2>&1 &

# Test prompt (exact model name required)
OLLAMA_HOST=127.0.0.1:11435 ./ollama run gemma4 "name the 50 states of america"

# CPU-only comparison
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_GPU=0 go run . serve > /tmp/ollama-cpu-test.log 2>&1 &
OLLAMA_HOST=127.0.0.1:11435 ./ollama run gemma4 "name the 50 states of america"

# Check Vulkan evidence in logs
grep -i "vulkan\|radv\|phoenix\|offload\|library=" /tmp/ollama-vulkan-test.log | head -20

# Compare against working reference
diff ~/source-usb/llama.cpp/src/models/gemma4.cpp \
     llama/llama.cpp/src/models/gemma4.cpp
diff ~/source-usb/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp \
     ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp
```
