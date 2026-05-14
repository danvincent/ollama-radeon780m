# Plan Complete: Copilot Streaming and Prompt Limits

Improved OpenAI-compatible streaming responsiveness for VS Code / GitHub Copilot by fixing buffering and flushing behavior on both chat and completions endpoints. Reduced SSE hot-path overhead by replacing fmt-based frame assembly with direct writes while preserving exact framing. Added handler-side `num_ctx` clamping for both `ChatHandler` and `GenerateHandler` so prompt construction uses the runner's actual loaded context length. During validation, a follow-up finding was identified: the runner warning `truncating input prompt` can still occur when the final prompt itself exceeds the true context budget (e.g. an oversized last message, or generate-context / token-accounting differences); this is separate from the handler clamp fixes and warrants dedicated follow-up work.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Fix chat and completions SSE buffering
2. ✅ Phase 2: Trim SSE hot-loop overhead
3. ✅ Phase 3: Clamp chat prompt context to loaded model context
4. ✅ Phase 4: Clamp generate path context to loaded model context

**All Files Created/Modified:**
- `middleware/openai.go`
- `middleware/openai_test.go`
- `server/routes.go`
- `server/routes_generate_test.go`
- `plans/copilot-streaming-prompt-limits-plan.md`
- `plans/copilot-streaming-prompt-limits-complete.md`

**Key Functions/Classes Added:**
- `ChatMiddleware` — flush-on-every-token SSE fix for chat endpoint
- `CompletionsMiddleware` — flush-on-every-token SSE fix for completions endpoint
- `ChatWriter.writeResponse` — direct-write SSE frame assembly (no fmt overhead)
- `CompleteWriter.writeResponse` — direct-write SSE frame assembly (no fmt overhead)
- `ChatHandler` — added `num_ctx` clamp to runner's loaded context length
- `GenerateHandler` — added `num_ctx` clamp to runner's loaded context length
- `mockRunner.ContextLength` — test helper exposing context length for clamp validation

**Test Coverage:**
- Total tests written: 14
- Targeted middleware and server test suites passing: ✅
- Note: the full repository test suite has pre-existing, unrelated failures in `cmd/`, `ml/`, and `gemma4` areas; those failures pre-date this work and were not introduced or fixed here.

**Recommendations for Next Steps:**
- Investigate runner-side prompt-budget mismatch that causes `truncating input prompt` even with limit-aligned handlers; likely causes include an oversized final message, deprecated generate `context` accounting, BOS/tokenization differences, or multimodal token-estimate heuristics.
- Investigate `ollama pull gemma4:e2b` manifest `412` version-gate failure separately; it is unrelated to this Copilot streaming work.
- Consider a follow-up micro-optimization pass for model metadata and template caching if additional latency reduction is needed.
