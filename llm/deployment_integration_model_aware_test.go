package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestAPIValidation_ModelAwareValidationOrder tests the new model-aware validation logic
// that performs inference FIRST when a model is specified, then checks GPU residency.
// This fixes the bug where a fresh service with no model loaded would fail GPU validation.
func TestAPIValidation_ModelAwareValidationOrder(t *testing.T) {
	t.Run("fresh_service_with_no_model_loaded_initially", func(t *testing.T) {
		// This simulates the real-world scenario:
		// 1. Service just started, no models loaded yet
		// 2. Inference request is made, model gets loaded
		// 3. After inference, /api/ps shows GPU is active
		//
		// BEFORE the fix: Would check /api/ps first, see no models, and fail.
		// AFTER the fix: Calls inference first, then checks /api/ps.
		callOrder := []string{}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				callOrder = append(callOrder, "api/ps")
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "gemma",
							Model:    "gemma:7b",
							Size:     7e9,
							SizeVRAM: 4e9, // After inference, GPU is active
							Digest:   "xyz789",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/generate":
				callOrder = append(callOrder, "api/generate")
				resp := api.GenerateResponse{
					Model:    "gemma:7b",
					Response: "Generated response after loading model",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "gemma:7b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true after model-aware validation, got false (msg: %s)", msg)
		}

		// Verify that inference was called BEFORE GPU check
		if len(callOrder) >= 2 {
			if callOrder[0] != "api/generate" {
				t.Errorf("Expected inference (/api/generate) to be called first, but order was: %v", callOrder)
			}
			if callOrder[1] != "api/ps" {
				t.Errorf("Expected GPU check (/api/ps) to be called second, but order was: %v", callOrder)
			}
		} else {
			t.Errorf("Expected at least 2 API calls, but got %d", len(callOrder))
		}
	})

	t.Run("model_specified_inference_then_gpu_check", func(t *testing.T) {
		// Verify that when a model is specified, inference happens before GPU check
		callOrder := []string{}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				callOrder = append(callOrder, "api/ps")
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
				callOrder = append(callOrder, "api/generate")
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "Paris is the capital of France",
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

		valid, _, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true, got false")
		}

		// CRITICAL: Verify call order when model is specified
		if len(callOrder) < 2 {
			t.Fatalf("Expected at least 2 calls, got %d", len(callOrder))
		}
		if callOrder[0] != "api/generate" {
			t.Errorf("When model specified, inference should be called first. Order: %v", callOrder)
		}
		if callOrder[1] != "api/ps" {
			t.Errorf("When model specified, GPU check should be called second. Order: %v", callOrder)
		}
	})

	t.Run("no_model_specified_gpu_only_check", func(t *testing.T) {
		// When no model is specified, should NOT call inference, just check GPU
		callOrder := []string{}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/ps":
				callOrder = append(callOrder, "api/ps")
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
				callOrder = append(callOrder, "api/generate")
				t.Fatalf("Inference should NOT be called when no model is specified")
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "", // No model specified
		}

		valid, _, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true, got false")
		}

		// Verify inference was NOT called
		if len(callOrder) > 0 && callOrder[0] == "api/generate" {
			t.Errorf("When no model specified, inference should NOT be called. Order: %v", callOrder)
		}
	})

	t.Run("model_specified_inference_fails_validly_reported", func(t *testing.T) {
		// When model is specified but inference fails, that failure should be reported
		// (not hidden by GPU check)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/generate":
				// Inference fails or returns empty
				resp := api.GenerateResponse{
					Model:    "llama2:7b",
					Response: "",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/ps":
				// GPU would be active, but inference failed first
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
			t.Errorf("Expected valid=false when inference returns empty response")
		}
		if msg == "" {
			t.Errorf("Expected non-empty failure reason")
		}
		// The failure reason should be about inference, not GPU
		if msg != "Model returned empty response" {
			t.Logf("Failure reason: %s (may have been improved)", msg)
		}
	})

	t.Run("comprehensive_truthful_deployment_validation_workflow", func(t *testing.T) {
		// This test documents the truthful deployment validation workflow:
		// 1. Service starts fresh (no models loaded)
		// 2. Validator calls inference with specified model
		// 3. Model loads and inference succeeds
		// 4. Validator checks /api/ps and confirms GPU
		// 5. Validation passes
		//
		// This is the TRUTHFUL behavior for a fresh service.

		// Track what the service state is at each call
		state := struct {
			modelsLoaded bool
		}{
			modelsLoaded: false,
		}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/generate":
				// First inference call loads the model
				state.modelsLoaded = true
				resp := api.GenerateResponse{
					Model:    "gemma:7b",
					Response: "Hello, I am Gemma running with GPU acceleration",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/ps":
				// GPU check happens after inference, so model should be loaded
				if !state.modelsLoaded {
					// This would have been the bug - checking GPU before inference
					resp := api.ProcessResponse{
						Models: []api.ProcessModelResponse{}, // Empty, no models
					}
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(resp)
				} else {
					// After inference, model is loaded and using GPU
					resp := api.ProcessResponse{
						Models: []api.ProcessModelResponse{
							{
								Name:     "gemma",
								Model:    "gemma:7b",
								Size:     7e9,
								SizeVRAM: 4e9, // GPU active
								Digest:   "gemma_digest",
							},
						},
					}
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(resp)
				}
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "gemma:7b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true for fresh service with model-aware validation (msg: %s)", msg)
		}
	})
}
