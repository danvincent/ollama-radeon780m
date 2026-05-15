## Plan: Copilot streaming and prompt limits

Tighten the OpenAI-compatible streaming path for VS Code, then clamp prompt context to the runner's actual loaded limit so prompt construction stops overshooting and causing journal truncation warnings. While touching the SSE writers, include the matching completions path and cheap hot-loop cleanups that shave a bit of latency and allocation overhead.

**Phases 4**
1. **Phase 1: Fix chat and completions SSE buffering**
    - **Objective:** Add anti-buffering headers and per-event flush behavior for streamed `/v1/chat/completions` and `/v1/completions`.
    - **Files/Functions to Modify/Create:** `middleware/openai.go` (`ChatMiddleware`, `CompletionsMiddleware`, `ChatWriter.writeResponse`, `CompleteWriter.writeResponse`), `middleware/openai_test.go`
    - **Tests to Write:** `TestChatMiddleware_StreamingResponseHeaders`, `TestChatMiddleware_NonStreamingNoAntiBufferHeaders`, `TestChatWriter_StreamFlushesAfterEachChunk`, completions-path equivalents
    - **Steps:**
        1. Write failing middleware/writer tests for streaming headers and flush behavior.
        2. Add `Cache-Control`, `Connection`, `X-Accel-Buffering` for streaming responses only.
        3. Flush after each SSE event and terminal frame for both chat and completions.

2. **Phase 2: Trim SSE hot-loop overhead**
    - **Objective:** Remove small per-token waste in the streaming writers.
    - **Files/Functions to Modify/Create:** `middleware/openai.go`, `middleware/openai_test.go`
    - **Tests to Write:** Regression coverage for unchanged SSE payload formatting on both writers
    - **Steps:**
        1. Write tests preserving exact SSE framing.
        2. Move `Content-Type` header setup out of the per-chunk loop.
        3. Replace `fmt.Sprintf("data: %s\n\n", ...)` with direct writes.

3. **Phase 3: Clamp chat prompt context to loaded model context**
    - **Objective:** Ensure chat prompt truncation uses the runner's actual loaded `ContextLength()` rather than an oversized requested/default `num_ctx`.
    - **Files/Functions to Modify/Create:** `server/routes.go` (`ChatHandler`), `server/routes_generate_test.go`
    - **Tests to Write:** `TestChatHandlerClampsNumCtxToRunnerContextLength`
    - **Steps:**
        1. Add a failing handler test with a mock runner exposing a smaller actual context length.
        2. Clamp `opts.NumCtx` after runner scheduling and before prompt construction.
        3. Re-run targeted chat handler tests.

4. **Phase 4: Clamp generate path and capture follow-up speed work**
    - **Objective:** Apply the same prompt-limit fix to generate and leave the codebase ready for a second pass of small latency wins.
    - **Files/Functions to Modify/Create:** `server/routes.go` (`GenerateHandler`), `server/routes_generate_test.go`
    - **Tests to Write:** `TestGenerateHandlerClampsNumCtxToRunnerContextLength`
    - **Steps:**
        1. Add failing generate-path regression coverage.
        2. Apply the same post-load clamp.
        3. Leave isolated follow-up candidates documented in the completion summary.

**Open Questions**
1. None at approval time; proceed with the SSE hot-path work now and defer model metadata caching to a follow-up.
