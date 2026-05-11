## Phase 1 Complete: Add deploy-script preflight and cleanup

**Plan:** ollama-build-deploy
**Phase:** 1 of 4
**Status:** APPROVED

---

### TL;DR

Phase 1 created the initial local deployment script with safe preflight checks, dry-run support, and narrowly scoped cleanup for /usr/local installs while explicitly preserving model data under /usr/share/ollama. The phase also added a behavioral test harness so help, dry-run, invalid-flag handling, repo-root execution, and cleanup-plan messaging are validated rather than inferred from source text.

---

### Files Changed

| File | Created / Modified |
|------|--------------------|
| scripts/build_deploy_local.sh | Created |
| scripts/test_build_deploy_local.sh | Created |

---

### Functions Changed

| Function / Path | File | Notes |
|-----------------|------|-------|
| `preflight_check_tools` | scripts/build_deploy_local.sh | Validates required tools are present before any destructive steps |
| `preflight_check_repo_context` | scripts/build_deploy_local.sh | Confirms the script is executed from the repository root |
| `preflight_check_permissions` | scripts/build_deploy_local.sh | Checks for root or sudo privileges required for /usr/local writes |
| `preflight_check_install_detection` | scripts/build_deploy_local.sh | Detects an existing /usr/local install layout before cleanup |
| `preflight_check_ollama_user` | scripts/build_deploy_local.sh | Verifies the ollama user/group exists prior to service operations |
| `stop_service_if_running` | scripts/build_deploy_local.sh | Gracefully stops the ollama systemd service if active |
| `cleanup_binary` | scripts/build_deploy_local.sh | Removes /usr/local/bin/ollama only; no model paths touched |
| `cleanup_libraries` | scripts/build_deploy_local.sh | Removes /usr/local/lib/ollama only; no model paths touched |
| `verify_model_preservation` | scripts/build_deploy_local.sh | Asserts /usr/share/ollama is untouched after cleanup steps |
| `show_cleanup_plan` | scripts/build_deploy_local.sh | Prints what would be removed in dry-run mode without executing |

---

### Tests Changed

Added behavioral and static validation in scripts/test_build_deploy_local.sh.

- Validated `--help` success path and exit code.
- Validated `--dry-run` success and cleanup-plan output messaging.
- Validated invalid-flag failure behavior and non-zero exit code.
- Validated executable bit, bash syntax check, repo-root execution requirement, and explicit model-preservation messaging in script output.

---

### Review Status

**APPROVED** with minor recommendations.

- Both reviews approved the phase.
- Minor recommendations focused on future cleanup such as tightening shell test hygiene, optionally adding `pipefail` in the main script, making test harness repo-root setup more explicit, and cleaning up a few non-blocking shell-style nits.
- No blocking correctness or safety issues remained.

---

### Git Commit Message

```
chore: add deploy preflight and cleanup script

- add a local deployment script with safe /usr/local cleanup and model preservation
- add dry-run, preflight checks, and graceful service-stop handling
- add behavioral tests for help, dry-run, invalid flags, and cleanup-plan output
```

---

### Next Phase

**Phase 2: Add Vulkan-first build and install flow** — Build the custom Ollama binary and native libraries for this machine, then install the binary and backend libraries into /usr/local with the correct preserved layout.
