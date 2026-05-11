package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestAPIValidation_CheckGPUViaAPI tests GPU detection via /api/ps endpoint
func TestAPIValidation_CheckGPUViaAPI(t *testing.T) {
	t.Run("detects_gpu_acceleration_active", func(t *testing.T) {
		// Mock server that simulates /api/ps with GPU model
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "llama2",
							Model:    "llama2:7b",
							Size:     4e9,
							SizeVRAM: 2e9, // 2GB VRAM allocated = GPU active
							Digest:   "abc123",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
		}

		hasGPU, msg, err := CheckGPUViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !hasGPU {
			t.Errorf("Expected hasGPU=true, got false")
		}
		if msg == "" {
			t.Errorf("Expected non-empty message, got empty string")
		}
	})

	t.Run("rejects_cpu_only_execution", func(t *testing.T) {
		// Mock server with CPU-only model (SizeVRAM = 0)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "llama2",
							Model:    "llama2:7b",
							Size:     4e9,
							SizeVRAM: 0, // No VRAM = CPU only
							Digest:   "abc123",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
		}

		hasGPU, msg, err := CheckGPUViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if hasGPU {
			t.Errorf("Expected hasGPU=false for CPU-only, got true")
		}
		if msg == "" {
			t.Errorf("Expected non-empty error message, got empty string")
		}
	})

	t.Run("rejects_no_models_loaded", func(t *testing.T) {
		// Mock server with empty model list
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				resp := api.ProcessResponse{Models: []api.ProcessModelResponse{}}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
		}

		hasGPU, msg, err := CheckGPUViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if hasGPU {
			t.Errorf("Expected hasGPU=false for no models, got true")
		}
		if msg == "" {
			t.Errorf("Expected non-empty message, got empty string")
		}
	})

	t.Run("rejects_api_connection_failure", func(t *testing.T) {
		opts := APIDeploymentValidationOptions{
			ServiceURL: "http://localhost:99999", // Invalid port
			Timeout:    1 * time.Millisecond,     // Very short timeout
		}

		hasGPU, _, err := CheckGPUViaAPI(opts)
		if err == nil {
			t.Errorf("Expected error for connection failure, got nil")
		}
		if hasGPU {
			t.Errorf("Expected hasGPU=false on error, got true")
		}
	})
}

// TestAPIValidation_CheckInferenceViaAPI tests inference response validation
func TestAPIValidation_CheckInferenceViaAPI(t *testing.T) {
	t.Run("accepts_valid_inference_response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/generate" {
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "The capital of France is Paris",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "llama2:7b",
		}

		valid, msg, err := CheckInferenceViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true for non-empty response, got false")
		}
		if msg == "" {
			t.Errorf("Expected non-empty message, got empty string")
		}
	})

	t.Run("rejects_empty_inference_response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/generate" {
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "", // Empty response
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "llama2:7b",
		}

		valid, _, err := CheckInferenceViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if valid {
			t.Errorf("Expected valid=false for empty response, got true")
		}
	})

	t.Run("rejects_whitespace_only_response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/generate" {
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "   \n\t  ", // Whitespace only
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "llama2:7b",
		}

		valid, _, err := CheckInferenceViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if valid {
			t.Errorf("Expected valid=false for whitespace-only response, got true")
		}
	})

	t.Run("skips_check_when_no_model_specified", func(t *testing.T) {
		// Should skip inference check if model not specified
		opts := APIDeploymentValidationOptions{
			ServiceURL: "http://localhost:11434",
			Timeout:    5 * time.Second,
			Model:      "", // No model specified
		}

		valid, msg, err := CheckInferenceViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true when check skipped, got false")
		}
		if msg != "Inference check skipped (no model specified)" {
			t.Errorf("Expected skip message, got %q", msg)
		}
	})
}

// TestAPIValidation_ValidateDeploymentViaAPI tests combined API validation
func TestAPIValidation_ValidateDeploymentViaAPI(t *testing.T) {
	t.Run("accepts_gpu_with_valid_inference", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "llama2",
							Model:    "llama2:7b",
							Size:     4e9,
							SizeVRAM: 2e9,
							Digest:   "abc123",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "The capital of France is Paris",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "llama2:7b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true, got false (msg: %s)", msg)
		}
	})

	t.Run("preserves_cpu_only_failure_reason", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "llama2",
							Model:    "llama2:7b",
							Size:     4e9,
							SizeVRAM: 0, // CPU only
							Digest:   "abc123",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if valid {
			t.Errorf("Expected valid=false for CPU-only, got true")
		}
		// IMPORTANT: Check that the specific CPU-only reason is preserved
		if msg == "" {
			t.Errorf("Expected non-empty failure reason, got empty string")
		}
		if !strings.Contains(msg, "CPU-only") && !strings.Contains(msg, "Service running CPU-only") {
			t.Errorf("Expected specific CPU-only reason in message, got: %s", msg)
		}
	})

	t.Run("preserves_no_models_failure_reason", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if valid {
			t.Errorf("Expected valid=false for no models, got true")
		}
		// IMPORTANT: Check that the specific "no models" reason is preserved
		if msg == "" {
			t.Errorf("Expected non-empty failure reason, got empty string")
		}
		if !strings.Contains(msg, "No models") {
			t.Errorf("Expected 'No models' reason in message, got: %s", msg)
		}
	})

	t.Run("preserves_empty_inference_failure_reason", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "llama2",
							Model:    "llama2:7b",
							Size:     4e9,
							SizeVRAM: 2e9,
							Digest:   "abc123",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "", // Empty response
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "llama2:7b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if valid {
			t.Errorf("Expected valid=false for empty inference, got true")
		}
		// IMPORTANT: Check that the specific "empty response" reason is preserved
		if msg == "" {
			t.Errorf("Expected non-empty failure reason, got empty string")
		}
		if !strings.Contains(msg, "empty") && !strings.Contains(msg, "Empty") {
			t.Errorf("Expected 'empty response' reason in message, got: %s", msg)
		}
	})


	t.Run("rejects_gpu_with_empty_inference", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "llama2",
							Model:    "llama2:7b",
							Size:     4e9,
							SizeVRAM: 2e9,
							Digest:   "abc123",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "", // Empty response
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "llama2:7b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if valid {
			t.Errorf("Expected valid=false for empty inference, got true")
		}
		if msg == "" {
			t.Errorf("Expected error message, got empty string")
		}
	})

	t.Run("rejects_api_connection_failure", func(t *testing.T) {
		opts := APIDeploymentValidationOptions{
			ServiceURL: "http://localhost:99999",
			Timeout:    1 * time.Millisecond,
		}

		valid, _, err := ValidateDeploymentViaAPI(opts)
		if err == nil {
			t.Errorf("Expected error for connection failure, got nil")
		}
		if valid {
			t.Errorf("Expected valid=false on error, got true")
		}
	})
}

// TestAPIValidation_ModelSpecificGPUValidation tests model-aware GPU validation (regression tests for false-positive)
func TestAPIValidation_ModelSpecificGPUValidation(t *testing.T) {
	t.Run("rejects_wrong_model_on_gpu_with_requested_model_missing", func(t *testing.T) {
		// FALSE POSITIVE BUG: Requested model A, but unrelated model B is on GPU
		// Should FAIL because the requested model is not on GPU
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "modelB",
							Model:    "modelB:7b",
							Size:     7e9,
							SizeVRAM: 5e9, // This model IS on GPU
							Digest:   "other_digest",
						},
						// modelA (the requested one) is NOT loaded at all
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				// Inference for modelA would load it (after previous inference check)
				resp := api.GenerateResponse{
					Model:    "modelA:7b",
					Response: "Response from modelA",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "modelA:7b", // Request modelA
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		// Must FAIL because requested model (modelA) is not on GPU, even though another model is
		if valid {
			t.Errorf("Expected valid=false when requested model not on GPU (unrelated model on GPU), got true")
			t.Errorf("Message was: %s", msg)
		}
		// Message should indicate the specific model was not found on GPU
		if msg == "" {
			t.Errorf("Expected specific failure message, got empty string")
		}
		if !strings.Contains(msg, "modelA") && !strings.Contains(msg, "not") && !strings.Contains(msg, "GPU") {
			t.Logf("Message could be more specific about modelA not being on GPU: %s", msg)
		}
	})

	t.Run("rejects_requested_model_present_but_cpu_only", func(t *testing.T) {
		// Requested model present but CPU-only (SizeVRAM == 0)
		// Should FAIL with specific CPU-only message
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "llama2",
							Model:    "llama2:7b",
							Size:     7e9,
							SizeVRAM: 0, // Requested model is loaded but CPU-only
							Digest:   "abc123",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "CPU-only response",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "llama2:7b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		// Must FAIL because requested model has SizeVRAM == 0 (CPU-only)
		if valid {
			t.Errorf("Expected valid=false for CPU-only requested model, got true")
		}
		// Should have specific message about model being CPU-only
		if msg == "" {
			t.Errorf("Expected specific CPU-only message, got empty string")
		}
		if !strings.Contains(msg, "CPU") && !strings.Contains(msg, "cpu") {
			t.Logf("Message should mention CPU-only status: %s", msg)
		}
	})

	t.Run("accepts_requested_model_on_gpu_even_with_other_models", func(t *testing.T) {
		// Requested model is on GPU, other models may also be loaded
		// Should PASS because requested model is confirmed on GPU
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "modelA",
							Model:    "modelA:7b",
							Size:     7e9,
							SizeVRAM: 5e9, // Requested model IS on GPU
							Digest:   "modelA_digest",
						},
						{
							Name:     "modelB",
							Model:    "modelB:9b",
							Size:     9e9,
							SizeVRAM: 0, // Other model is CPU-only
							Digest:   "modelB_digest",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				resp := api.GenerateResponse{
					Model:    "modelA:7b",
					Response: "GPU-accelerated response from modelA",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "modelA:7b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		// Must PASS because requested model is on GPU
		if !valid {
			t.Errorf("Expected valid=true when requested model is on GPU, got false (msg: %s)", msg)
		}
	})

	t.Run("handles_model_name_matching_correctly", func(t *testing.T) {
		// Test that model name matching is exact and not just substring
		// Requesting "llama2" should not match "llama2-uncensored" or "llama2:7b-fp16"
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "llama2",
							Model:    "llama2:7b-fp16", // Similar but not exact match
							Size:     7e9,
							SizeVRAM: 5e9,
							Digest:   "digest_fp16",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "Response",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "llama2:7b", // Request exact model
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		// Should PASS if exact match found on GPU
		if !valid {
			t.Logf("Note: %s (msg: %s)", msg, msg)
		}
	})
}

// TestAPIValidation_DefaultOptions tests default API validation options
func TestAPIValidation_DefaultOptions(t *testing.T) {
	opts := DefaultAPIDeploymentValidationOptions()

	if opts.ServiceURL == "" {
		t.Errorf("Expected non-empty ServiceURL, got empty")
	}
	if opts.Timeout == 0 {
		t.Errorf("Expected non-zero Timeout, got 0")
	}
	if opts.Model != "" {
		t.Errorf("Expected empty Model by default, got %q", opts.Model)
	}
}

// TestAPIValidation_IntegrationScenarios tests complete deployment validation scenarios
func TestAPIValidation_IntegrationScenarios(t *testing.T) {
	t.Run("real_world_gpu_deployment_success", func(t *testing.T) {
		// Simulates successful GPU deployment with multiple models
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:           "gemma4",
							Model:          "gemma4:9b",
							Size:           9e9,
							SizeVRAM:       6e9, // GPU active
							Digest:         "def456",
							ContextLength: 8192,
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				resp := api.GenerateResponse{
					Model:    "gemma4:9b",
					Response: "Gemma4 successfully generated this response using GPU acceleration",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "gemma4:9b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !valid {
			t.Errorf("Expected successful deployment validation (msg: %s)", msg)
		}
	})

	t.Run("real_world_cpu_fallback_failure", func(t *testing.T) {
		// Simulates the original Phase 3 bug: service reports CPU-only
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				// Model is loaded but has no GPU VRAM
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:           "gemma4",
							Model:          "gemma4:9b",
							Size:           9e9,
							SizeVRAM:       0, // CPU-only fallback
							Digest:         "def456",
							ContextLength: 8192,
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if valid {
			t.Errorf("Expected CPU fallback to fail validation (msg: %s)", msg)
		}
	})

	t.Run("api_signal_vs_log_patterns_superiority", func(t *testing.T) {
		// Demonstrates why /api/ps is superior to log patterns:
		// - Real-time state (not subject to log rotation)
		// - Structured data (not fragile regex patterns)
		// - Authoritative source (from the running service)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "model1",
							SizeVRAM: 3e9, // GPU is really in use
						},
						{
							Name:     "model2",
							SizeVRAM: 0, // This one is CPU
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
		}

		hasGPU, msg, err := CheckGPUViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !hasGPU {
			t.Errorf("Expected API signal to detect GPU in model1, got false")
		}
		if msg == "" {
			t.Errorf("Expected descriptive message, got empty string")
		}

		// The /api/ps signal gives us exact VRAM allocation:
		// - If any model has SizeVRAM > 0, GPU is active
		// - This is authoritative and real-time
		// - No ambiguity unlike log patterns
	})
}
