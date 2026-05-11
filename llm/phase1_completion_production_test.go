package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"testing"

	"golang.org/x/sync/semaphore"
)

// newMockLlamaServer creates a minimal HTTP server that provides the status and completion endpoints
func newMockLlamaServer(completionHandler func(http.ResponseWriter, *http.Request)) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/status", "/health":
			// Return a ready status (ServerStatusReady = 0)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(ServerStatusResponse{
				Status:   ServerStatusReady,
				Progress: 0,
			})
		case "/completion":
			completionHandler(w, r)
		default:
			http.NotFound(w, r)
		}
	}))
}

// newTestLLMServer creates a minimal mock llmServer for testing
func newTestLLMServer(url string) *llmServer {
	// Create a mock cmd - we use a command that will block (like cat waiting for stdin)
	// but don't actually run it - we just need the fields to be non-nil
	cmd := &exec.Cmd{
		Path: "/bin/sleep",
		Args: []string{"sleep", "60"},
	}
	
	return &llmServer{
		port: parsePort(url),
		sem:  semaphore.NewWeighted(1),
		cmd:  cmd, // ProcessState will be nil since we didn't Start() it
	}
}

// parsePort extracts port number from a test server URL (e.g., http://127.0.0.1:12345)
func parsePort(url string) int {
	parts := strings.Split(url, ":")
	if len(parts) < 3 {
		return 8000
	}
	port := 0
	fmt.Sscanf(parts[2], "%d", &port)
	return port
}

// TestCompletionRealPathSingleChunkWithContent tests that a single Done chunk with content
// calls the callback exactly once and returns the full response.
func TestCompletionRealPathSingleChunkWithContent(t *testing.T) {
	// Create a mock HTTP server that responds with a single completion chunk
	completionHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := CompletionResponse{
			Content:    "hello",
			Done:       true,
			EvalCount:  5,
			DoneReason: DoneReasonStop,
		}
		data, _ := json.Marshal(response)
		fmt.Fprintf(w, "data: %s\n", string(data))
	}
	server := newMockLlamaServer(completionHandler)
	defer server.Close()

	// Extract port and create mock llmServer
	llm := newTestLLMServer(server.URL)

	callCount := 0
	var lastResponse CompletionResponse

	err := llm.Completion(context.Background(), CompletionRequest{
		Prompt: "test",
	}, func(resp CompletionResponse) {
		callCount++
		lastResponse = resp
	})

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if callCount != 1 {
		t.Fatalf("expected 1 callback, got %d", callCount)
	}
	if lastResponse.Content != "hello" {
		t.Fatalf("expected content 'hello', got %q", lastResponse.Content)
	}
	if lastResponse.EvalCount != 5 {
		t.Fatalf("expected EvalCount 5, got %d", lastResponse.EvalCount)
	}
	if !lastResponse.Done {
		t.Fatalf("expected Done=true")
	}
}

// TestCompletionRealPathStreamWithWhitespace tests that whitespace tokens are delivered
// to the callback while streaming, and only count as valid activity if there's actual content.
func TestCompletionRealPathStreamWithWhitespace(t *testing.T) {
	// Create a mock HTTP server that responds with multiple chunks: whitespace, then content
	completionHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("ResponseWriter does not support Flusher interface")
		}

		chunks := []CompletionResponse{
			{Content: "  ", Done: false}, // whitespace token
			{Content: "he", Done: false},
			{Content: "llo", Done: false},
			{Content: "world", Done: true, EvalCount: 4},
		}

		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n", string(data))
			flusher.Flush()
		}
	}
	server := newMockLlamaServer(completionHandler)
	defer server.Close()

	llm := newTestLLMServer(server.URL)

	callCount := 0
	var responses []CompletionResponse
	var lastResponse CompletionResponse

	err := llm.Completion(context.Background(), CompletionRequest{
		Prompt: "test",
	}, func(resp CompletionResponse) {
		callCount++
		responses = append(responses, resp)
		lastResponse = resp
	})

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// Should be called 4 times: once for each chunk
	if callCount != 4 {
		t.Fatalf("expected 4 callbacks, got %d", callCount)
	}

	// First callback should be the whitespace token
	if responses[0].Content != "  " {
		t.Fatalf("expected first chunk '  ', got %q", responses[0].Content)
	}

	// Last callback should be the final Done chunk
	if lastResponse.Content != "world" {
		t.Fatalf("expected last content 'world', got %q", lastResponse.Content)
	}
	if !lastResponse.Done {
		t.Fatalf("expected last chunk Done=true")
	}

	// Verify full response preservation - last chunk should have EvalCount
	if lastResponse.EvalCount != 4 {
		t.Fatalf("expected last chunk EvalCount 4, got %d", lastResponse.EvalCount)
	}
}

// TestCompletionRealPathEmptyResponseError tests that a whitespace-only response
// returns an error about no content being generated AND does not call the callback
// for the invalid Done chunk.
func TestCompletionRealPathEmptyResponseError(t *testing.T) {
	// Create a mock HTTP server that responds with only whitespace chunks
	completionHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("ResponseWriter does not support Flusher interface")
		}

		chunks := []CompletionResponse{
			{Content: "  ", Done: false},
			{Content: "\n", Done: false},
			{Content: "", Done: true, EvalCount: 2},
		}

		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n", string(data))
			flusher.Flush()
		}
	}
	server := newMockLlamaServer(completionHandler)
	defer server.Close()

	llm := newTestLLMServer(server.URL)

	callCount := 0
	var finalCallDone bool

	err := llm.Completion(context.Background(), CompletionRequest{
		Prompt: "test",
	}, func(resp CompletionResponse) {
		callCount++
		if resp.Done {
			finalCallDone = true
		}
	})

	// Should have called the callback for the non-Done chunks (whitespace and newline are still streamed)
	// but NOT for the invalid Done chunk
	if callCount != 2 {
		t.Fatalf("expected 2 callbacks (non-Done chunks only), got %d", callCount)
	}

	// Verify no Done callback was made for the invalid chunk
	if finalCallDone {
		t.Errorf("expected no Done callback for invalid empty response")
	}

	// Should return an error about empty response
	if err == nil {
		t.Fatalf("expected error for empty response, got nil")
	}
	if !strings.Contains(err.Error(), "no content generated") {
		t.Fatalf("expected 'no content generated' error, got: %v", err)
	}
}

// TestCompletionRealPathNoDoubleDone tests that a Done chunk is not called twice.
// When we have a Done chunk with content, we should call the callback once with
// the full response, not once as a non-Done callback and again as a Done callback.
func TestCompletionRealPathNoDoubleDone(t *testing.T) {
	completionHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := CompletionResponse{
			Content:    "output",
			Done:       true,
			EvalCount:  10,
			DoneReason: DoneReasonStop,
		}
		data, _ := json.Marshal(response)
		fmt.Fprintf(w, "data: %s\n", string(data))
	}
	server := newMockLlamaServer(completionHandler)
	defer server.Close()

	llm := newTestLLMServer(server.URL)

	callCount := 0
	contentsSeen := []string{}

	err := llm.Completion(context.Background(), CompletionRequest{
		Prompt: "test",
	}, func(resp CompletionResponse) {
		callCount++
		contentsSeen = append(contentsSeen, resp.Content)
	})

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// Should be called exactly once
	if callCount != 1 {
		t.Fatalf("expected 1 callback, got %d", callCount)
	}

	// Should not see "output" twice in our callbacks
	count := 0
	for _, content := range contentsSeen {
		if content == "output" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected 'output' to appear exactly once, got %d times", count)
	}
}

// TestCompletionRealPathFullFieldPreservation tests that all CompletionResponse fields
// are preserved and delivered to the callback, not just Content and Logprobs.
func TestCompletionRealPathFullFieldPreservation(t *testing.T) {
	completionHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("ResponseWriter does not support Flusher interface")
		}

		// Non-Done chunk with Image and Step info (e.g., image generation)
		chunk1 := CompletionResponse{
			Content:    "",
			Image:      "base64data",
			Step:       5,
			TotalSteps: 20,
			Done:       false,
		}
		data1, _ := json.Marshal(chunk1)
		fmt.Fprintf(w, "data: %s\n", string(data1))
		flusher.Flush()

		// Final chunk
		chunk2 := CompletionResponse{
			Content:    "final",
			Done:       true,
			EvalCount:  1,
			DoneReason: DoneReasonStop,
		}
		data2, _ := json.Marshal(chunk2)
		fmt.Fprintf(w, "data: %s\n", string(data2))
		flusher.Flush()
	}
	server := newMockLlamaServer(completionHandler)
	defer server.Close()

	llm := newTestLLMServer(server.URL)

	var responses []CompletionResponse

	err := llm.Completion(context.Background(), CompletionRequest{
		Prompt: "test",
	}, func(resp CompletionResponse) {
		responses = append(responses, resp)
	})

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(responses) != 2 {
		t.Fatalf("expected 2 callbacks, got %d", len(responses))
	}

	// First callback should preserve Image, Step, TotalSteps
	if responses[0].Image != "base64data" {
		t.Fatalf("expected Image 'base64data', got %q", responses[0].Image)
	}
	if responses[0].Step != 5 {
		t.Fatalf("expected Step 5, got %d", responses[0].Step)
	}
	if responses[0].TotalSteps != 20 {
		t.Fatalf("expected TotalSteps 20, got %d", responses[0].TotalSteps)
	}

	// Second callback should have final content and metadata
	if responses[1].Content != "final" {
		t.Fatalf("expected content 'final', got %q", responses[1].Content)
	}
	if responses[1].EvalCount != 1 {
		t.Fatalf("expected EvalCount 1, got %d", responses[1].EvalCount)
	}
}

// TestCompletionRealPathStreamWithoutDone tests that if the stream ends
// without a Done chunk (e.g., connection drop, truncation), an error is returned.
func TestCompletionRealPathStreamWithoutDone(t *testing.T) {
	// Create a mock HTTP server that responds with content chunks but no Done
	completionHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("ResponseWriter does not support Flusher interface")
		}

		chunks := []CompletionResponse{
			{Content: "hello", Done: false},
			{Content: " ", Done: false},
			{Content: "world", Done: false}, // No Done chunk - stream is incomplete
		}

		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n", string(data))
			flusher.Flush()
		}
		// Stream ends without sending a Done chunk - simulating truncation/connection drop
	}
	server := newMockLlamaServer(completionHandler)
	defer server.Close()

	llm := newTestLLMServer(server.URL)

	callCount := 0
	err := llm.Completion(context.Background(), CompletionRequest{
		Prompt: "test",
	}, func(resp CompletionResponse) {
		callCount++
	})

	// Should return an error about missing Done chunk
	if err == nil {
		t.Fatalf("expected error for stream without Done chunk, got nil")
	}
	if !strings.Contains(err.Error(), "Done") && !strings.Contains(err.Error(), "chunk") {
		t.Fatalf("expected error mentioning 'Done' or 'chunk', got: %v", err)
	}

	// Callbacks should have been made for the non-Done chunks (streaming protocol)
	if callCount != 3 {
		t.Fatalf("expected 3 callbacks for non-Done chunks, got %d", callCount)
	}
}
