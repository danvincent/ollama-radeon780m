## Phase 2 Complete: Add Vulkan-first build and install flow

**Plan:** ollama-build-deploy
**Phase:** 2 of 4
**Status:** APPROVED

---

### TL;DR

Phase 2 extended the local deployment script to build Ollama with the repo's Vulkan-first configuration and install the binary and backend libraries into /usr/local using the repo-supported CMake install flow. The phase added optional cache cleaning, artifact verification, fatal Vulkan post-build/post-install checks, and stronger shell tests that validate dry-run safety and the expected Vulkan layout assumptions.

---

### Files Changed

| File | Created / Modified |
|------|--------------------|
| scripts/build_deploy_local.sh | Modified |
| scripts/test_build_deploy_local.sh | Modified |

---

### Functions Changed

| Function / Path | File | Notes |
|-----------------|------|-------|
| `check_build_prerequisites` | scripts/build_deploy_local.sh | Validates CMake, Go, and Vulkan toolchain prerequisites are present before any build step |
| `clean_go_cache_if_requested` | scripts/build_deploy_local.sh | Optionally runs go clean -cache when the --clean-cache flag is provided; skipped by default |
| `build_with_cmake` | scripts/build_deploy_local.sh | Invokes CMake configure with the Vulkan-first preset for this machine |
| `build_cmake_project` | scripts/build_deploy_local.sh | Runs the CMake build step and captures output for artifact verification |
| `build_ollama_binary` | scripts/build_deploy_local.sh | Compiles the Ollama Go binary against the freshly built native libraries |
| `verify_build_artifacts` | scripts/build_deploy_local.sh | Asserts expected Vulkan backend artifacts are present after the build; fatal on failure |
| `install_binary` | scripts/build_deploy_local.sh | Copies the compiled binary to /usr/local/bin/ollama |
| `install_backend_libraries` | scripts/build_deploy_local.sh | Installs the backend library tree into /usr/local/lib/ollama using the CMake install flow |
| `verify_install_vulkan_artifacts` | scripts/build_deploy_local.sh | Asserts the Vulkan backend libraries are present under the install prefix after install; fatal on failure |

---

### Tests Changed

Added Phase 2 behavioral and static validation in scripts/test_build_deploy_local.sh.

- Validated help/usage output reflects build/install flow flags including --clean-cache.
- Validated build configuration and CMake preset selection behavior.
- Validated install flow and expected Vulkan backend library layout under /usr/local/lib/ollama.
- Validated fatal post-build and post-install Vulkan artifact check behavior.
- Validated optional --clean-cache handling is skipped by default and activated only when the flag is present.
- Validated dry-run safety: no build, install, or destructive steps execute without explicit confirmation flags.
- Validated no premature Phase 3/systemd deployment behavior is triggered by the Phase 2 code path.

---

### Review Status

**APPROVED** with minor recommendations.

- Both reviews approved the phase.
- Minor recommendations focused on follow-up polish such as simplifying a few redundant build flags/checks, tightening one post-install shell test, and optionally documenting shell-test coverage methodology.
- No blocking correctness issues remained.

---

### Git Commit Message

```
feat: add local build and install flow

- extend the deploy script with Vulkan-first cmake and go build steps
- install the binary and backend libraries into /usr/local using cmake install
- add dry-run-safe Vulkan artifact validation and stronger shell tests
```

---

### Next Phase

**Phase 3: Add systemd service deployment** — Write and install the service definition, include the required environment for Vulkan discovery, reload systemd, and enable/start the service.
