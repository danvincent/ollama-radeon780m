# Plan: Gemma4 Vulkan Port

Port the working upstream llama.cpp Gemma4 ggml runtime support into Ollama's vendored llama.cpp layer so Gemma4 can run through the ggml/Vulkan path at all. The approach is to add Gemma4 architecture/model support first, then validate the Vulkan runtime behavior on top of that, since the current highest-signal gap is missing Gemma4 model support rather than the Vulkan backend alone.

---

## Phases (4)

### Phase 1: Port Gemma4 architecture and model metadata

**Objective:** Add Gemma4 architecture registration and the required model/hparam fields to Ollama's vendored llama.cpp so Gemma4 models can be recognized and loaded.

**Files/Functions to Modify/Create:**
- `llama/llama.cpp/src/llama-arch.h`
- `llama/llama.cpp/src/llama-arch.cpp`
- `llama/llama.cpp/src/llama-hparams.h`
- `llama/llama.cpp/src/llama-model.h`
- Related load/dispatch points in `llama/llama.cpp/src/llama-model.cpp`

**Tests to Write:** Targeted model-arch and metadata-loading tests if the vendored tree has existing coverage hooks; otherwise focused build/parse validation in the relevant existing test surfaces.

**Steps:**
1. Write failing tests or validation points for Gemma4 architecture recognition and required metadata fields.
2. Port the minimal upstream Gemma4 architecture identifiers and supporting hparam/model fields.
3. Run targeted tests/build checks for the vendored llama.cpp layer to confirm Gemma4 is recognized.

---

### Phase 2: Port Gemma4 tensor loading and graph construction

**Objective:** Port the upstream Gemma4 model-loading and graph-builder logic into Ollama's vendored llama.cpp integration so Gemma4 inference can execute through ggml.

**Files/Functions to Modify/Create:**
- `llama/llama.cpp/src/llama-model.cpp`
- Any new Gemma4 model implementation file needed under `llama/llama.cpp/src/models/`
- Gemma4-related tensor/graph helpers

**Tests to Write:** Focused inference/model-construction tests for Gemma4 load paths and any existing vendored llama.cpp test hooks that can cover graph creation.

**Steps:**
1. Write failing coverage for the Gemma4 load/graph path.
2. Port the upstream Gemma4 tensor-loading and graph-construction logic, adapting it to Ollama's vendored structure.
3. Run targeted tests/build checks to confirm Gemma4 graph construction works.

---

### Phase 3: Validate Vulkan runtime compatibility for Gemma4

**Objective:** Confirm the existing Vulkan backend is sufficient for the new Gemma4 ggml path and make only the minimal runtime/backend adjustments required for correct execution.

**Files/Functions to Modify/Create:** Vendored ggml/Vulkan files only if required after validation; likely `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp` and directly related tests.

**Tests to Write:** Targeted Vulkan/runtime validation tests or focused backend checks only if a concrete incompatibility is found.

**Steps:**
1. Run targeted build/runtime validation for Gemma4 on the Vulkan path.
2. If a concrete backend incompatibility appears, add a failing test or reproducer and implement the minimal fix.
3. Re-run targeted validation to confirm Gemma4 can execute on Vulkan as expected.

---

### Phase 4: Final deploy-path validation and review sweep

**Objective:** Ensure the deploy/check path truthfully validates the now-working Gemma4 Vulkan flow and complete final tests/reviews.

**Files/Functions to Modify/Create:** Deployment validation files only if needed after runtime validation; otherwise no new production files expected.

**Tests to Write:** Final targeted deploy-check and llm validation runs; no new tests expected unless wiring changes are required.

**Steps:**
1. Run targeted tests/builds across vendored llama.cpp, llm validation, and deploy-check surfaces.
2. Run shell syntax checks for the deploy scripts if touched.
3. Pass the completed change through both review agents and capture any required revisions before completion.

---

## Open Questions (4)

1. Is upstream `src/models/gemma4.cpp` best ported as a new file, or should it be adapted into Ollama's existing monolithic `llama-model.cpp` pattern?
2. Which existing vendored llama.cpp tests can most directly cover Gemma4 architecture recognition and graph construction?
3. Will AMD 780M Vulkan need only partial CPU fallback on DK=512 flash-attention layers, or does a concrete backend fix become necessary?
4. Are any additional Ollama-side model-name or Modelfile changes required once Gemma4 works through the vendored ggml path?
