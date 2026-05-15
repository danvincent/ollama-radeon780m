## Plan: Gemma4 Vulkan Runtime Fix

We now have a real failing runtime case on Gemma4 8B Q4_K_M with Vulkan active, so the next pass is optimized for the fastest likely correctness fix rather than broad cleanup. We will fix the most likely remaining Gemma4 prefill precision gap first, validate against the live failing prompt set starting with the 50 states prompt, then commit that baseline and continue improving if further issues remain.

**Phases**
1. **Phase 1: Fix Gemma4 FFN prefill precision first**
    - **Objective:** Eliminate the most likely remaining live-runtime corruption source by forcing F32 on the missing Gemma4 build_ffn() up and gate projections during prefill.
    - **Files/Functions to Modify/Create:** llama/llama.cpp/src/llama-graph.cpp, especially build_ffn(...), plus targeted Gemma4 precision tests in llama/ or llm/ as appropriate.
    - **Tests to Write:** failing tests for Gemma4 FFN precision on up, gate, and down outputs in the real graph-builder path.
    - **Steps:**
        1. Add failing tests that prove Gemma4 build_ffn() still leaves up and gate on default precision.
        2. Patch build_ffn() so Gemma4 sets F32 on all required FFN projections.
        3. Re-run targeted tests and the affected suites until passing.

2. **Phase 2: Prove Vulkan output correctness on the live failing case and commit baseline**
    - **Objective:** Use the exact failing runtime scenario to verify whether the FFN fix resolves the wrong-answer behavior on Vulkan, then establish a committed baseline if it does.
    - **Files/Functions to Modify/Create:** no production files unless runtime validation reveals another directly related defect; tests/log capture support as needed.
    - **Tests to Write:** runtime-focused validation for Gemma4 8B Q4_K_M on Vulkan versus CPU quality, starting with the 50 states prompt and then expanding to a small deeper prompt set if the initial prompt is fixed.
    - **Steps:**
        1. Run targeted tests first, then the relevant broader suites.
        2. Re-run OLLAMA_VULKAN=1 OLLAMA_DEBUG=1 ./ollama run gemma4 with the failing prompt and compare to CPU-only output.
        3. If fixed, provide a commit message for the baseline and continue to the next phase.

3. **Phase 3: Add Gemma4 attention-side F32 only if Phase 2 still fails or further improvement is needed**
    - **Objective:** If Vulkan output is still corrupted, or if post-baseline improvement is needed, extend Gemma4-specific F32 handling to the remaining likely attention projections used during prefill.
    - **Files/Functions to Modify/Create:** likely llama/llama.cpp/src/models/gemma4.cpp and llama/llama.cpp/src/llama-graph.cpp, especially build_attn(...).
    - **Tests to Write:** failing tests for Gemma4 attention projection precision in the actual path used by Gemma4 8B.
    - **Steps:**
        1. Add failing tests for missing attention-side precision enforcement.
        2. Implement the smallest Gemma4-specific fix in the actual attention path.
        3. Re-run targeted suites and the live Vulkan runtime checks.

4. **Phase 4: Final regression and closeout**
    - **Objective:** Lock in the runtime fix with regression coverage representing the real failure class: Vulkan active, coherent output required.
    - **Files/Functions to Modify/Create:** final touched test files across llama, ml/backend/ggml, and llm as needed.
    - **Tests to Write:** regression coverage for the live wrong-answer Vulkan case as closely as the repository can support.
    - **Steps:**
        1. Tighten tests around the final identified defect or defects.
        2. Run targeted suites and broader affected suites.
        3. Confirm Vulkan now produces correct output for the original failing case and deeper prompt set.

**Open Questions**
1. Resolved: start with the 50 states prompt, then expand to a deeper prompt set after the first live fix works.
2. Resolved: commit the initial successful live fix as a baseline, then continue improving.
