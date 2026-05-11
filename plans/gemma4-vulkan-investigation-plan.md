## Plan: Identify Gemma4 Vulkan Failure

Treat LM Studio as the known-good behavioral baseline and isolate where Ollama diverges: runner selection, Vulkan flash-attention/kernel support, memory/layout fitting, or Gemma4-specific execution assumptions. The goal is to prove the failing layer first, then make the smallest fix that restores correctness without unnecessary CPU fallback, and end with a fully tested build ready to deploy.

**Phases**
1. **Phase 1: Establish Exact Divergence Path**
    - **Objective:** Prove which Ollama execution path Gemma4 is taking and where it diverges from the LM Studio behavior we can observe.
    - **Files/Functions to Modify/Create:** llm/server.go, discover/runner.go, service/deploy config if needed for traceability.
    - **Tests to Write:** Targeted tests around runner selection, backend selection, and Gemma4 load-request shaping.
    - **Steps:**
        1. Add failing tests for Gemma4-specific backend/load selection expectations.
        2. Trace whether Gemma4 is using the legacy llama.cpp path or Ollama engine path in deployed/runtime conditions.
        3. Capture the exact decision points for Vulkan enablement, flash attention, and partial-offload avoidance.

2. **Phase 2: Audit llama.cpp Vulkan Gemma4 Support [PARALLEL]**
    - **Objective:** Determine whether upstream/pinned llama.cpp Vulkan actually supports the Gemma4 attention/kernel shapes Ollama is asking it to run.
    - **Files/Functions to Modify/Create:** llama/ submodule Vulkan backend sources and local patch set, especially Vulkan attention/kernel dispatch.
    - **Tests to Write:** Narrow regression tests or assertions for supported Gemma4 Vulkan attention/kernel configurations.
    - **Steps:**
        1. Add failing coverage for any unsupported Gemma4 Vulkan FA/kernel combinations discovered.
        2. Compare pinned llama.cpp Vulkan support against Gemma4 requirements.
        3. Identify whether Ollama is enabling a Vulkan path LM Studio likely avoids or patches.
        4. Run this phase in its own git worktree if executed in parallel.

3. **Phase 3: Audit Ollama Layout / Memory / Partial-Offload Logic [PARALLEL]**
    - **Objective:** Determine whether Ollama's memory estimation and partial-offload strategy are producing an invalid-but-fast Gemma4 Vulkan execution mode.
    - **Files/Functions to Modify/Create:** llm/server.go, ml/backend/ggml/ggml.go, related memory/layout code.
    - **Tests to Write:** Failing tests around Gemma4 Vulkan layout fitting, FA gating, and partial/full offload decisions.
    - **Steps:**
        1. Add failing tests that model the bad partial-Vulkan fit behavior.
        2. Verify whether Gemma4-specific graph sizing or layout backoff is wrong.
        3. Isolate whether the corruption is from partial Vulkan itself or from a bad decision to allow it.
        4. Run this phase in its own git worktree if executed in parallel.

4. **Phase 4: Reproduce and Narrow with Controlled Runtime Experiments**
    - **Objective:** Use the live repro to prove which switch changes behavior: FA, runner path, full vs partial offload, or Gemma4-specific config.
    - **Files/Functions to Modify/Create:** Test harnesses and minimal debug instrumentation only where necessary.
    - **Tests to Write:** Reproducible regression test for the chosen root cause and the user's known prompt/repro class.
    - **Steps:**
        1. Start with small-context/full-fit-oriented experiments and move upward.
        2. Test FA on/off, legacy vs Ollama engine path, and partial-offload boundaries.
        3. Convert the confirmed failing scenario into automated regression coverage.

5. **Phase 5: Implement Minimal Fix and Harden Deployment**
    - **Objective:** Restore the fastest correct Gemma4 Vulkan path available on this UM790, without broad risky changes.
    - **Files/Functions to Modify/Create:** Only the confirmed root-cause files from Phases 2–4, plus deploy/runtime safeguards if needed.
    - **Tests to Write:** Targeted tests first, then affected suite coverage, then deploy/runtime checks.
    - **Steps:**
        1. Add the failing regression test(s) first.
        2. Implement the narrowest fix that restores correct Vulkan behavior or safe optimized fallback behavior.
        3. Validate with the Gemma4 repro and keep the Vulkan-only UM790 deployment robust.

**Open Questions**
1. Is deployed Gemma4 actually running through llamaServer or ollamaServer in the failing case?
2. Is the corruption caused by Vulkan flash attention or kernel coverage for Gemma4 shapes, or by partial offload scheduling?
3. Does LM Studio avoid the bad path by using a different llama.cpp revision or patch set, or simply different runtime settings?
4. Is there any full Vulkan Gemma4 fit available on this machine, or do we need to make the partial path correct?
