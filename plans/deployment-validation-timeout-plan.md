## Plan: Deployment validation timeout fix

Fix the model-aware deployment validator so first-load inference for Gemma/Vulkan does not time out after 10 seconds and get misreported as a hard deployment failure. The approach is to separate fast API timeouts from slow inference timeouts, plumb timeout configuration through the CLI and shell deploy path, and make error/final-status messaging truthful.

**Phases**
1. **Phase 1: Split validation timeouts**
    - **Objective:** Introduce distinct fast API and inference timeouts in the llm deployment validator so model-aware validation can tolerate first-load latency without weakening quick health checks.
    - **Files/Functions to Modify/Create:** `llm/deployment_integration.go`; deployment validation option structs/defaults; `CheckInferenceViaAPI`; `CheckGPUViaAPI`; relevant `llm` tests.
    - **Tests to Write:** `TestCheckInferenceViaAPI_UsesInferenceTimeout`; `TestValidateDeploymentViaAPI_AllowsSlowFirstLoad`; updates to existing API/deployment validation tests.
    - **Steps:**
        1. Write failing tests that simulate a slow `/api/generate` response and verify fast `/api/ps` checks remain short.
        2. Add a dedicated inference timeout field/default and update inference requests to use it.
        3. Run the targeted llm tests to confirm the new timeout behavior passes.

2. **Phase 2: Expose timeout controls in the deploy path**
    - **Objective:** Make the timeout configurable through the real user-facing validation path so deploy scripts can validate slow first-load models without code edits.
    - **Files/Functions to Modify/Create:** `cmd/deploy-check/main.go`; `cmd/deploy-check/main_test.go`; `scripts/verify-gpu-deployment.sh`; `scripts/build_deploy_local.sh`.
    - **Tests to Write:** CLI option-mapping tests for inference timeout; lightweight script argument-flow coverage if practical.
    - **Steps:**
        1. Write failing tests for CLI option mapping and expected timeout propagation.
        2. Add CLI flags/default handling for inference timeout and wire them into API validation options.
        3. Add shell/script passthrough using named flags and/or environment variables, then run targeted CLI tests and shell syntax validation.

3. **Phase 3: Make timeout and validation messaging truthful**
    - **Objective:** Distinguish timeout/state-gathering errors from real validation failures so deployment output tells the user whether to retry with a longer timeout or fix GPU configuration.
    - **Files/Functions to Modify/Create:** `cmd/deploy-check/main.go`; `scripts/build_deploy_local.sh`; relevant deployment validation tests.
    - **Tests to Write:** Tests covering timeout-oriented error messaging and truthful final deploy messaging for model-aware vs GPU-only validation.
    - **Steps:**
        1. Write failing tests for timeout-specific output and final deploy-status wording.
        2. Update CLI and deploy-script messaging to separate timeout/error conditions from validation failures.
        3. Run targeted tests again to confirm the user-facing output matches the real validation state.

4. **Phase 4: Final verification and review sweep**
    - **Objective:** Validate the full timeout fix across llm, CLI, and shell deploy wiring and ensure no regressions remain.
    - **Files/Functions to Modify/Create:** No new production files expected; test files may be updated if needed.
    - **Tests to Write:** None expected beyond final targeted/full-suite execution.
    - **Steps:**
        1. Run targeted tests first across `./llm` and `./cmd/deploy-check`, then the broader relevant suites.
        2. Run shell syntax checks for both validation/deploy scripts.
        3. Hand the completed change through both code-review agents and capture any required revisions before completion.

**Open Questions**
1. Should the default inference timeout be 2 minutes or 5 minutes for first-load Gemma4 on UM790/Vulkan?
2. Should timeout-related errors remain exit code 2, or is a separate retryable code worth the compatibility cost?
3. Should `build_deploy_local.sh` expose timeout via `--validation-timeout`, `--inference-timeout`, env vars, or both?
4. Is model-name normalization needed beyond exact `.Model` / `.Name` matching for the user's Gemma4 tag?
