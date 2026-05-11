# Phase 4 Final Revision: Assert Branch Execution Through Strict Log Assertions

**Plan:** gemma4-vulkan-investigation  
**Phase:** 4  
**Status:** COMPLETE (REVISED)

---

## Executive Summary

Phase 4 revised with **strict assertions**: Tests now FAIL if expected branch logs are absent. This phase documents **code path behavior** of the Phase 3 fix by executing `buildLayout()` directly and asserting on specific slog.Info messages the code must emit when the guard branch executes.

**Key change:** Tests use explicit `t.Errorf()` assertions on required logs — tests fail if evidence is missing, not just if they're observed.

**Tests**: 
1. `TestPhase4RuntimeCodePathTracingWithLogObservation` - Single scenario that asserts guard branch execution via logs
2. `TestPhase4ControlVariableEffect` - Differential control proving guard effect (Gemma4 vs Gemma3)

---

## Scope Clarification: What Phase 4 Tests vs. What It Doesn't

### ✓ What Phase 4 DOES Test

- Direct execution of `buildLayout()` with controlled memory scenarios
- **Log assertion** - Code must emit specific slog.Info messages to pass (tests use `t.Errorf()` for missing logs)
- Observation of which code path the Phase 3 fix executes (lines 1057-1072 in server.go)
- Verification that `shouldAvoidPartialVulkanOffload()` guard function triggers when partial is detected
- Control comparison: Effect of the guard variable isolated (Gemma4 vs Gemma3)

### ✗ What Phase 4 Does NOT Test

- **No production running** - We do NOT start the Ollama server
- **No actual model loading** - We do NOT run Gemma4 model files
- **No UM790 hardware** - Tests run on test infrastructure, not production hardware
- **No output correctness** - We do NOT verify model produces correct answers
- **No performance measurement** - We do NOT measure inference speed
- **No real prompts** - We do NOT test with actual LLM queries
- **No happy-path scenario** - We do NOT test successful full offload on first try (does not exercise guard branch)

---

## Asserted Branch-Proof Scenario

### Test: `TestPhase4RuntimeCodePathTracingWithLogObservation/gemma4_tight_memory_guard_branch_execution_proven`

**Setup:**
- Model: Gemma4, 10 layers × 120MB = 1200MB needed
- GPU: Vulkan with 600MB free. After 457MB minimum reserve, ~143MB available (fits ~1 layer of 120MB)
- Expected code path: partial detected → full retry → fails → CPU fallback

**Required Assertions (Tests FAIL if any missing):**

1. **Guard Trigger Log REQUIRED:**
   ```
   Log string must contain: "disabling partial Vulkan offload for model"
   Assertion: t.Errorf() if missing
   ```
   Proves line 1058 in server.go executed (guard detected partial)

2. **Retry Failure Log REQUIRED:**
   ```
   Log string must contain: "gemma4 vulkan full offload fallback not possible"
   Assertion: t.Errorf() if missing
   ```
   Proves line 1069 in server.go executed (retry attempted and failed)

3. **CPU Fallback Result REQUIRED:**
   ```
   Result must equal 0/10 (no GPU layers)
   Assertion: t.Errorf() if result != 0
   ```
   Proves line 1070 fallback occurred (libraryGpuLayers = nil)

**What this proves:**
- ✓ Phase 3 guard code (lines 1057-1072) actually executes when partial offload is detected
- ✓ Guard correctly identifies Gemma4 + Vulkan partial offload scenario
- ✓ Full offload retry is attempted (log evidence)
- ✓ When full offload is impossible, guard correctly falls back to CPU
- ✓ Log signals are emitted at correct points with correct information

**What remains unproven (Phase 5):**
- Whether CPU-only execution is acceptable performance for user's use case
- Whether there are other fallback options before CPU

---

## Deterministic Control: Guard Variable Effect

### Test Group: `TestPhase4ControlVariableEffect`

**Identical memory setup for both tests:**
- Both: 10 layers × 100MB = 1000MB total needed
- Both: Vulkan GPU with 600MB free. After 457MB minimum reserve, ~143MB available (fits ~1 layer)
- Difference: Gemma4 has guard enabled, Gemma3 has guard disabled

#### Test 1: `gemma4_with_guard_cpu_fallback_asserted`
**Assertion:** Gemma4 MUST get CPU fallback (0/10)
- Assertion: `t.Errorf()` if result != 0
- Proves: Guard variable causally prevents partial offload for Gemma4

#### Test 2: `gemma3_without_guard_no_cpu_fallback_asserted`
**Assertion:** Gemma3 MUST NOT fall back to CPU (result > 0)
- Assertion: `t.Errorf()` if result == 0
- Full offload (10/10) is acceptable; partial offload (>0 and <10) is acceptable
- Only CPU fallback (0) fails the test
- Proves: Without guard, Gemma3 achieves GPU allocation under identical memory constraints

**Combined proving:**
- ✓ Guard is correctly model-architecture-specific
- ✓ Gemma4 behavior differs from Gemma3 under identical conditions
- ✓ Guard causally affects outcomes (one variable changed)

---

## Files Changed

| File | Change | Reason |
|------|--------|--------|
| `llm/gemma4_phase4_runtime_test.go` | Revised | Replaced soft t.Logf() with strict t.Errorf() assertions; removed happy-path scenario |
| `plans/gemma4-vulkan-investigation-phase-4-complete.md` | Revised | Updated to describe only asserted scenarios; removed unproven claims |

**Key improvements:**
- Removed scenario_1 (sufficient memory/happy path) - does not exercise guard branch
- Converted scenario_2 to strict assertions - tests FAIL if required logs absent or result != 0
- Converted Gemma3 control to deterministic comparison - must allow GPU allocation (> 0 layers)
- Removed soft "Note:" observations - only hard assertions remain

**Result:** Tests now prove branch execution through strict log assertions, not soft observation

---

## Tests Changed

| Test | File | Purpose | Status |
|------|------|---------|--------|
| `TestPhase4RuntimeCodePathTracingWithLogObservation/gemma4_tight_memory_guard_branch_execution_proven` | `llm/gemma4_phase4_runtime_test.go` | Assert guard branch execution via logs (FAIL if logs absent) | ✓ PASS |
| `TestPhase4ControlVariableEffect/gemma4_with_guard_cpu_fallback_asserted` | `llm/gemma4_phase4_runtime_test.go` | Assert Gemma4 CPU fallback (FAIL if not 0) | ✓ PASS |
| `TestPhase4ControlVariableEffect/gemma3_without_guard_no_cpu_fallback_asserted` | `llm/gemma4_phase4_runtime_test.go` | Assert Gemma3 no CPU fallback (FAIL if result == 0) | ✓ PASS |

All Phase 4 tests pass; all Phase 3 tests still pass (no regressions).

---

## Phase 3 vs Phase 4 Evidence Distinction

### Phase 3: Unit Tests (Regression Coverage)
- File: `gemma4_phase3_investigation_test.go`
- 5 tests covering guard logic, full/CPU decisions, controls
- Assertion-based, deterministic outcomes
- Tests that the fix prevents partial and chooses full/CPU correctly

### Phase 4: Code Path Tracing (Strict Assertions)
- File: `llm/gemma4_phase4_runtime_test.go`
- 1 scenario with strict log assertions + 2 control tests
- **Strict log-based assertions** - tests FAIL if expected logs absent
- Tests that code reaches intended branches and emits required signals
- Scenario: Guard branch (partial detected, full retry fails, CPU fallback)
- Control: Differential effect (Gemma4 vs Gemma3 under identical memory)

### Phase 5: Deployment (NOT DONE YET)
- Hardware validation on UM790
- Real model inference
- Output correctness
- Performance measurement

---

## Key Findings

### 1. Phase 3 Guard Code is Present and Reachable
- Lines 1057-1072 in server.go execute as designed
- Guard detects partial Vulkan offload for Gemma4
- Retry logic is called with requireFull=true
- **Proven by:** Required logs are present when tests run

### 2. Guard Has Intended Differential Effect
- **Scenario (tight memory):** Guard triggers, retry attempted, CPU fallback occurs
- **Control:** Gemma4 (guard=true) prevents partial; Gemma3 (guard=false) allows GPU allocation
- **Variable causally affects outcomes**

### 3. Log Signals Generated Correctly
- "disabling partial Vulkan offload for model" - proves guard detection
- "gemma4 vulkan full offload fallback not possible" - proves full retry failed
- Signals appear at right points (lines 1058, 1069)
- **Proven by:** Strict assertions require logs to be present

### 4. No Regressions
- Phase 3 tests pass (5 tests)
- Phase 4 tests pass (3 tests: 1 scenario + 2 controls)
- Existing LLM tests pass
- No production behavior changed

---

## What Phase 4 DOES NOT Prove

1. ✗ Output correctness on any hardware
2. ✗ Performance characteristics
3. ✗ Real UM790 VRAM requirements
4. ✗ Hardware deployment success
5. ✗ Root cause of partial-offload problem
6. ✗ Happy-path full offload (not in Phase 4 scope)

---

## Phase 5 Target

**Objective:** Hardware validation and output correctness

**Scope:**
1. Deploy to UM790 + Radeon 780M
2. Run Gemma4 with Phase 3 fix active
3. Test both: full Vulkan offload scenario, CPU fallback scenario
4. Verify output correctness with known-good prompts
5. Check log signals match Phase 4 documentation

**Success:** Phase 3 fix resolves the issue OR needs extension

---

## Summary

**Phase 4 Status:** COMPLETE (REVISED) ✓

Phase 4 uses **strict log assertions** to prove the guard branch executes. Tests FAIL if required logs are absent. The Phase 3 fix is correctly implemented and reaches intended code paths.

**Key accomplishment:** Confirmed Phase 3 guard code executes when partial is detected, retry is attempted, and CPU fallback occurs — all proven by mandatory log assertions.

**Distinction from Phase 3:** Phase 3 tests guard logic; Phase 4 tests guard branch execution through log assertions.

**Next step:** Phase 5 hardware deployment and output verification.

---

## Git Commit Message

```
refactor(phase-4): strict log assertions prove guard branch execution

Phase 4 revised to use strict assertions (t.Errorf) instead of soft logging.
Tests now FAIL if required branch logs are absent.

Key changes:
- TestPhase4RuntimeCodePathTracingWithLogObservation: Single scenario with
  strict assertions:
  * MUST find "disabling partial Vulkan offload for model" (guard branch)
  * MUST find "gemma4 vulkan full offload fallback not possible" (retry failed)
  * MUST get result = 0/10 (CPU fallback)
  * Tests FAIL with t.Errorf() if any assertion fails
- TestPhase4ControlVariableEffect: Deterministic control comparison
  * Gemma4 MUST get CPU fallback (0/10) - strict assertion
  * Gemma3 allowed GPU allocation - deterministic observation
- Removed happy-path scenario (sufficient memory) - does not exercise guard
- Removed soft t.Logf() observations - only hard assertions remain

Evidence collected (strict proof):
✓ Guard branch code executes when partial is detected (log-proven)
✓ Retry with requireFull=true is attempted (log-proven)
✓ CPU fallback occurs when full retry fails (assertion-proven)
✓ Guard has differential effect on Gemma4 vs Gemma3 (control-proven)
✓ No regressions (Phase 3 tests pass, all LLM tests pass)

Files: llm/gemma4_phase4_runtime_test.go
Tests: 1 scenario + 2 controls with strict assertions, all passing
Regressions: None
```
