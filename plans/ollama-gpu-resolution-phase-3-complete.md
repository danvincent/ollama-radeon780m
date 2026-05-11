## Phase 3 Complete: Investigate ROCm fallback path

**Plan:** ollama-gpu-resolution
**Phase:** 3 of 4
**Status:** APPROVED

---

### TL;DR

Phase 3 determined that no repo-controlled production change is warranted for the remaining gfx1103 ROCm stall. Instead, it added boundary-level and helper-level ROCm tests that verify Ollama-controlled discovery, validation, and environment handoff complete successfully before runtime execution, supporting the conclusion that the remaining stall is external to repo-controlled logic.

---

### Files Changed

| File | Created / Modified |
|------|--------------------|
| ml/device_rocm_boundary_test.go | Created |
| ml/device_rocm_env_test.go | Created |
| ml/device_rocm_filter_test.go | Created |
| ml/device_rocm_fallback_integration_test.go | Created |
| ml/device_rocm_phases_test.go | Deleted |

---

### Functions Changed

| Function / Path | File | Notes |
|-----------------|------|-------|
| `TestGetDevicesFromRunnerWithROCmDevice` | ml/device_rocm_boundary_test.go | Boundary test asserting real GetDevicesFromRunner() completes against an httptest-backed runner serving a synthetic ROCm device response |
| `TestGetDevicesFromRunnerBoundaryWithTimeout` | ml/device_rocm_boundary_test.go | Boundary test asserting GetDevicesFromRunner() returns a clean error rather than hanging when the runner does not respond within the deadline |
| `TestGetDevicesFromRunnerBoundaryWithRunnerExit` | ml/device_rocm_boundary_test.go | Boundary test asserting GetDevicesFromRunner() returns a clean error when the runner exits unexpectedly during the request |
| ROCm helper/env validation tests | ml/device_rocm_env_test.go | Tests covering NeedsInitValidation, AddInitValidation, PreferredLibrary, Compute string formatting, and RunnerEnvOverrides propagation for ROCm devices |
| ROCm fallback path tests | ml/device_rocm_fallback_integration_test.go | Tests covering the end-to-end ROCm fallback path through discovery, validation, and environment handoff |

---

### Tests Changed

| Test | File | Notes |
|------|------|-------|
| `TestGetDevicesFromRunnerWithROCmDevice` | ml/device_rocm_boundary_test.go | Added; asserts real GetDevicesFromRunner() handles a synthetic ROCm device response correctly using an httptest-backed runner |
| `TestGetDevicesFromRunnerBoundaryWithTimeout` | ml/device_rocm_boundary_test.go | Added; asserts GetDevicesFromRunner() surfaces a clean error rather than stalling when the runner is unresponsive |
| `TestGetDevicesFromRunnerBoundaryWithRunnerExit` | ml/device_rocm_boundary_test.go | Added; asserts GetDevicesFromRunner() surfaces a clean error when the runner process exits mid-request |
| ROCm helper tests | ml/device_rocm_env_test.go | Added/retained; cover NeedsInitValidation, AddInitValidation, PreferredLibrary, Compute formatting, and RunnerEnvOverrides propagation |
| ROCm device filter tests | ml/device_rocm_filter_test.go | Added; cover ROCm device filtering behavior |
| ROCm fallback integration tests | ml/device_rocm_fallback_integration_test.go | Added; validate the full ROCm fallback path through discovery, validation, and environment handoff |
| Phase-level ROCm tests | ml/device_rocm_phases_test.go | Removed; replaced by stronger boundary and helper coverage across the new files |
| Broader test suite validation | ml, discover, runner packages | Targeted ROCm tests plus broader ml, discover, and runner package test suites validated |

---

### Review Status

**APPROVED** with minor recommendations.

- Both reviews approved the phase.
- Minor recommendations focused on optional follow-up cleanup such as reducing duplicate helper-test coverage, softening a low-probability timing flake in the runner-exit boundary test, and trimming unnecessary platform skips and non-idiomatic test logging.
- No blocking correctness issues remained.

---

### Git Commit Message

```
chore: add rocm boundary coverage

- add real GetDevicesFromRunner boundary tests for ROCm discovery handoff
- validate ROCm env and filtering behavior without changing production logic
- document by test evidence that the remaining gfx1103 stall is outside repo control
```

---

### Next Phase

**Phase 4** — To be defined by the Conductor.
