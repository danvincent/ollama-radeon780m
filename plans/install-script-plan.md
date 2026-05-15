## Plan: Ollama Custom Build Install Script

Create a single `install.sh` script in the repo root that cleanly replaces the system Ollama install with the locally-built custom binary and Vulkan libs, then restarts and self-tests.

**Phases (1)**

1. **Phase 1: Create install.sh**
    - **Objective:** Write a robust install script that stops the service, removes the existing install (apt + manual binary/lib cleanup), installs the new binary and libs, verifies the systemd service config includes OLLAMA_VULKAN=1, restarts, and runs a smoke test against Gemma4 and Qwen3.
    - **Files/Functions to Modify/Create:** `install.sh` (repo root)
    - **Tests to Write:** N/A — this is a shell script; validation is done by running the script against the live system and checking exit codes and model output.
    - **Steps:**
        1. Create `install.sh` with bash strict mode (`set -euo pipefail`)
        2. Detect script directory using `$(cd "$(dirname "$0")" && pwd)` for reliable relative paths
        3. Stop ollama systemd service gracefully
        4. Run `sudo apt-get remove --purge ollama` (continue on failure if not apt-installed)
        5. Remove `/usr/bin/ollama` and `/usr/local/bin/ollama` if they exist
        6. Remove `/usr/lib/ollama/` and `/usr/local/lib/ollama/` directories
        7. Copy built binary `./ollama` to `/usr/local/bin/ollama` with chmod 755
        8. Create `/usr/local/lib/ollama/` and copy all `build/lib/ollama/` contents recursively
        9. Verify `/etc/systemd/system/ollama.service` contains `OLLAMA_VULKAN=1` — print warning if missing but continue
        10. Run `systemctl daemon-reload && systemctl restart ollama`
        11. Poll `http://localhost:11434/api/tags` in a loop (30s timeout) until server responds
        12. Run smoke test: `ollama run gemma4 "What is 2+2? Answer in one word."` — print result
        13. Run regression test: `ollama run qwen3:latest "What is 2+2? Answer in one word."` — print result
        14. Print success summary with installed binary version
