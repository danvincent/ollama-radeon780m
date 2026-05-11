## Plan: Gemma4 Vulkan Runtime Fix

Fix the false-positive GPU validation path first, then diagnose and repair the real Gemma4 Vulkan runtime failure so deployment claims match actual UM790 behavior. The work should tighten validation, fix the runtime path causing CPU fallback or empty output, and finish with end-to-end deployment verification.

**Phases**
1. **Phase 1: Make Validation Truthful**
    - **Objective:** Remove false GPU success signals and make the scripts fail on CPU fallback or empty output.
    - **Files/Functions to Modify/Create:** deployment/validation scripts related to the local build and Phase 5 validation, including likely `scripts/build_deploy_local.sh` and any repo validation helpers used to detect GPU state.
    - **Tests to Write:** checks for CPU fallback detection, empty-response failure detection, misleading generic `GPU`/`offload` grep matches, and required success log matching.
    - **Steps:**
        1. Write tests first for the known false-positive cases: `0/36 layers`, empty output, and generic `GPU` text that does not prove acceleration.
        2. Update validation to require real success signals instead of permissive grep matches.
        3. Run targeted tests, then broader relevant test coverage, and confirm the validation path fails correctly on CPU-only or empty-output runs.

2. **Phase 2: Diagnose and Fix Runtime Fallback**
    - **Objective:** Identify why the deployed Phase 3 build still lands on CPU fallback and sometimes returns no response, then implement the minimal runtime fix.
    - **Files/Functions to Modify/Create:** likely `llm/server.go`, related load/layout/runtime surfaces, and any deployment/runtime logging helpers implicated by the investigation.
    - **Tests to Write:** regression tests for the confirmed failing runtime branch and no-response behavior.
    - **Steps:**
        1. Reproduce the failing runtime path using the deployed service behavior and logs.
        2. Add failing tests for the confirmed branch or behavior before changing production code.
        3. Implement the minimal code change, rerun targeted tests, and then rerun broader relevant coverage.

3. **Phase 3: Harden Deployment Verification**
    - **Objective:** Ensure deployment claims match the actual running service and model behavior.
    - **Files/Functions to Modify/Create:** deployment scripts, validation scripts, and any status/reporting surfaces that claim GPU usage.
    - **Tests to Write:** checks that deployment validation rejects stale, misconfigured, or CPU-only service states.
    - **Steps:**
        1. Write tests for stale binary and misconfigured service scenarios.
        2. Add checks for live service configuration and post-load runtime state.
        3. Re-run deployment validation flow and confirm it fails fast on invalid states.

4. **Phase 4: Verify End-to-End on UM790**
    - **Objective:** Prove the final build is ready by validating real prompts and real runtime state.
    - **Files/Functions to Modify/Create:** minimal documentation or reporting updates only if needed.
    - **Tests to Write:** only gap-filling regression tests if Phase 4 reveals a missing automated check.
    - **Steps:**
        1. Re-deploy the fixed build.
        2. Validate `What is the capital of Iceland?` and `list the states of America` using the hardened validation path.
        3. Confirm actual GPU versus CPU state from reliable runtime signals only.
        4. Record completion details for the full plan.

**Open Questions**
1. Is the no-response failure caused by the same runtime path as the CPU fallback, or a second issue after load?
2. Should final validation require `ollama ps` or API VRAM evidence in addition to log evidence?
3. Is the current deployed service definitely using the intended Vulkan environment on every restart?
