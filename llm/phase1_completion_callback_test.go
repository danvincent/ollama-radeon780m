package llm

import (
	"fmt"
	"strings"
	"testing"
)

// PHASE 1 CALLBACK LOGIC TESTS
// =============================
// These tests validate the core callback and validation logic using
// the mockCompletionServer harness which duplicates the exact validation
// logic from server.go:Completion(). This focuses on the critical aspects:
// - Empty/whitespace-only responses are rejected
// - Missing Done chunks are errors
// - Valid responses stream correctly
//
// Production tests (phase1_completion_production_test.go) cover the full
// HTTP integration for these same scenarios using real mock servers.

// Helper mock for llmServer when we can't easily inject a full real one
// This provides a minimal harness to test the callback invocation logic
type mockCompletionServer struct {
	responses []CompletionResponse
}

func (m *mockCompletionServer) simulateCompletion(fn func(CompletionResponse)) error {
	// Simulate the exact Completion() logic for validating responses
	var totalResponseContent strings.Builder
	hasValidActivity := false

	for _, c := range m.responses {
		if !c.Done {
			// Streaming protocol: always stream non-Done chunks to callback
			fn(c)

			// Validation bookkeeping: track valid activity separately
			if hasValidCompletionActivity(c) {
				hasValidActivity = true
				totalResponseContent.WriteString(c.Content)
			}
		}

		if c.Done {
			// Final chunk handling: validate BEFORE calling callback
			if hasValidCompletionActivity(c) {
				hasValidActivity = true
				totalResponseContent.WriteString(c.Content)
			}

			// Validate with accumulated content
			collectedResponse := totalResponseContent.String()
			if !hasValidActivity && !ValidateResponsePresence(collectedResponse) {
				// Validation failed: return error WITHOUT calling callback
				return fmt.Errorf("model produced empty response: no content generated")
			}

			// Validation passed: stream the final Done=true chunk exactly once
			fn(c)
			return nil
		}
	}

	return fmt.Errorf("no Done chunk received")
}

// TestCompletionCallbackLogic tests the callback invocation logic directly
func TestCompletionCallbackLogic(t *testing.T) {
	t.Run("callback_invocation_matches_real_completion_logic", func(t *testing.T) {
		// This test verifies our understanding of the real Completion() callback logic
		// by simulating it in isolation

		mock := &mockCompletionServer{
			responses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " ", Done: false},
				{Content: "world", Done: true},
			},
		}

		var receivedResponses []CompletionResponse
		callback := func(resp CompletionResponse) {
			receivedResponses = append(receivedResponses, resp)
		}

		err := mock.simulateCompletion(callback)
		if err != nil {
			t.Errorf("Expected success, got error: %v", err)
		}

		// Should have received all 3 responses
		if len(receivedResponses) != 3 {
			t.Errorf("Expected 3 callbacks, got %d", len(receivedResponses))
		}

		// Verify streaming contract: whitespace token reaches callback
		if len(receivedResponses) > 1 && receivedResponses[1].Content != " " {
			t.Error("Whitespace-only token should reach callback")
		}

		// Verify Done=true chunk reaches callback
		if len(receivedResponses) > 2 && !receivedResponses[2].Done {
			t.Error("Done=true chunk should reach callback")
		}
	})

	t.Run("callback_logic_rejects_whitespace_only", func(t *testing.T) {
		mock := &mockCompletionServer{
			responses: []CompletionResponse{
				{Content: "   ", Done: false},
				{Content: "\n", Done: true},
			},
		}

		var receivedResponses []CompletionResponse
		callback := func(resp CompletionResponse) {
			receivedResponses = append(receivedResponses, resp)
		}

		err := mock.simulateCompletion(callback)
		if err == nil {
			t.Error("Expected error for whitespace-only response")
		}

		// Callbacks should have been made for non-Done chunks (streaming protocol)
		// but NOT for the Done chunk (validation failed)
		if len(receivedResponses) == 0 {
			t.Error("Non-Done whitespace chunk should have reached callback")
		}
		if len(receivedResponses) > 1 {
			t.Error("Done chunk should not reach callback when validation fails")
		}
		// Verify exactly 1 callback (just the non-Done whitespace chunk)
		if len(receivedResponses) != 1 {
			t.Errorf("Expected exactly 1 callback, got %d", len(receivedResponses))
		}
	})

	t.Run("callback_logic_rejects_empty_response", func(t *testing.T) {
		mock := &mockCompletionServer{
			responses: []CompletionResponse{
				{Content: "", Done: true},
			},
		}

		var receivedResponses []CompletionResponse
		callback := func(resp CompletionResponse) {
			receivedResponses = append(receivedResponses, resp)
		}

		err := mock.simulateCompletion(callback)
		if err == nil {
			t.Error("Expected error for empty response")
		}
		if !strings.Contains(err.Error(), "empty response") {
			t.Errorf("Expected 'empty response' in error, got: %v", err)
		}

		// No callbacks should be made for invalid empty response
		if len(receivedResponses) > 0 {
			t.Errorf("Expected no callbacks for invalid empty response, got %d", len(receivedResponses))
		}
	})

	t.Run("callback_logic_rejects_no_done_chunk", func(t *testing.T) {
		mock := &mockCompletionServer{
			responses: []CompletionResponse{
				{Content: "hello", Done: false},
				{Content: " world", Done: false},
				// Missing Done chunk
			},
		}

		var receivedResponses []CompletionResponse
		callback := func(resp CompletionResponse) {
			receivedResponses = append(receivedResponses, resp)
		}

		err := mock.simulateCompletion(callback)
		if err == nil {
			t.Error("Expected error when no Done chunk received")
		}
		if !strings.Contains(err.Error(), "Done") {
			t.Errorf("Expected 'Done' in error, got: %v", err)
		}

		// Non-Done callbacks should have been made (streaming protocol)
		if len(receivedResponses) != 2 {
			t.Errorf("Expected 2 callbacks for non-Done chunks, got %d", len(receivedResponses))
		}
	})

	t.Run("callback_logic_done_chunk_single_invocation", func(t *testing.T) {
		mock := &mockCompletionServer{
			responses: []CompletionResponse{
				{Content: "test", Done: false},
				{Content: ".", Done: true},
			},
		}

		var doneChunkCount int
		callback := func(resp CompletionResponse) {
			if resp.Done {
				doneChunkCount++
			}
		}

		err := mock.simulateCompletion(callback)
		if err != nil {
			t.Errorf("Expected success, got error: %v", err)
		}

		// CRITICAL: Done chunk invoked exactly once
		if doneChunkCount != 1 {
			t.Errorf("Done chunk should invoke callback exactly 1 time, got %d", doneChunkCount)
		}
	})
}
