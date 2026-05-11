## Plan: Build and Deploy Custom Ollama

Create a repo-local Linux deployment script that safely replaces the existing Ollama binary and backend libraries under /usr/local, rebuilds the custom source tree with the Vulkan-first path for this machine, writes a matching systemd unit, and installs everything without deleting model data. The script should follow repo conventions, check for the existing ollama user/group, and overwrite the existing systemd unit while preserving local models.

**Phases**
1. **Phase 1: Add deploy-script preflight and cleanup**
    - **Objective:** Create a repo-local script that validates prerequisites, detects the /usr/local install layout, stops the running service, and removes only the currently installed binary/libs while preserving models.
    - **Files/Functions to Modify/Create:** scripts/build_deploy_local.sh
    - **Tests to Write:** Shell-level safety/behavior checks where practical via script-logic validation patterns already used in repo shell scripts.
    - **Steps:**
        1. Write tests or validation scaffolding for install-prefix handling, service-stop behavior, and model-preservation assumptions.
        2. Implement preflight checks for required tools, repo root, root/sudo handling, existing install detection, and ollama user/group presence.
        3. Implement safe cleanup of /usr/local/bin/ollama and /usr/local/lib/ollama without removing /usr/share/ollama/.ollama/models.
        4. Re-run targeted validation of cleanup logic.

2. **Phase 2: Add Vulkan-first build and install flow**
    - **Objective:** Build the custom Ollama binary and native libraries for this machine, then install the binary and backend libraries into /usr/local with the correct preserved layout.
    - **Files/Functions to Modify/Create:** scripts/build_deploy_local.sh
    - **Tests to Write:** Validation that the script builds with the selected preset, installs the binary, and preserves backend subdirectory structure under the install lib path.
    - **Steps:**
        1. Write tests or validation scaffolding for build preset selection and install-path layout.
        2. Implement the build flow using the repo's CMake + Go toolchain, with a Vulkan-first default for this machine.
        3. Implement install steps for the main binary and lib/ollama backend tree so runtime backend discovery continues to work after deployment.
        4. Re-run targeted validation of build/install behavior.

3. **Phase 3: Add systemd service deployment**
    - **Objective:** Write and install the service definition, include the required environment for Vulkan discovery, reload systemd, and enable/start the service.
    - **Files/Functions to Modify/Create:** scripts/build_deploy_local.sh
    - **Tests to Write:** Validation that the generated service file contains the correct ExecStart, user/group, install target, and Vulkan discovery environment.
    - **Steps:**
        1. Write tests or assertions for the generated unit content.
        2. Implement service-user checks/creation and service-file installation.
        3. Include the required OLLAMA_VULKAN=1 service environment and align the binary/lib install paths with the unit.
        4. Re-run targeted validation of service generation and install behavior.

4. **Phase 4: Add deploy-script usage docs and guardrails**
    - **Objective:** Document how to use the script and add clear guardrails for destructive or package-manager-related operations.
    - **Files/Functions to Modify/Create:** scripts/build_deploy_local.sh, .github/copilot-instructions.md
    - **Tests to Write:** None unless existing docs validation applies.
    - **Steps:**
        1. Add usage/help output and clear warnings for optional destructive steps like removing an apt-installed package.
        2. Update the Copilot instructions with the new build/deploy shortcut once the script behavior is finalized.
        3. Re-run applicable validation and ensure the final usage text matches actual script behavior.

**Open Questions**
1. The script will install under /usr/local and overwrite /etc/systemd/system/ollama.service — this is now treated as approved.
2. The script will preserve models/data under /usr/share/ollama and only replace the binary and backend libraries.
3. The script should check for the ollama user/group and create them only if missing.
4. The script should follow repo service conventions for the unit target.
5. go clean -cache should be optional rather than always-on.
6. The preferred backend for this machine is Vulkan, and the deployed service should include OLLAMA_VULKAN=1.
