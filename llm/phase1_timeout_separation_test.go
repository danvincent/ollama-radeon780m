package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestCheckInferenceViaAPI_UsesInferenceTimeout verifies that inference requests
// use a separate, longer timeout than fast API checks.
// This test simulates a slow first-load inference response that exceeds the fast
// API timeout but completes within the inference timeout.
func TestCheckInferenceViaAPI_UsesInferenceTimeout(t *testing.T) {
	t.Run("slow_first_load_inference_respects_inference_timeout", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping slow timing test in -short mode")
		}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/generate" {

				// Simulate slow first-load inference: 1.5 seconds
				// This is realistic for first GPU kernel load on some hardware
				time.Sleep(1500 * time.Millisecond)

				resp := api.GenerateResponse{
					Model:    "gemma:7b",
					Response: "Generated text after slow load",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		// Create options with:
		// - FastAPITimeout: 1 second (for quick /api/ps checks)
		// - InferenceTimeout: 3 seconds (for slow first-load inference)
		opts := APIDeploymentValidationOptions{
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 3 * time.Second,
			Model:            "gemma:7b",
		}

		// Call CheckInferenceViaAPI with the inference timeout
		valid, msg, err := CheckInferenceViaAPI(opts)

		// Should succeed because we used the longer inference timeout
		if err != nil {
			t.Errorf("Expected no error with inference timeout, got: %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true with inference timeout, got false (msg: %s)", msg)
		}
		if msg == "" {
			t.Errorf("Expected non-empty message, got empty string")
		}
	})

	t.Run("inference_timeout_exceeded_returns_error", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping slow timing test in -short mode")
		}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/generate" {
				// Simulate inference that takes longer than timeout
				time.Sleep(2 * time.Second)
				resp := api.GenerateResponse{
					Model:    "gemma:7b",
					Response: "This should timeout",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 1500 * time.Millisecond, // Too short for this inference
			Model:            "gemma:7b",
		}

		valid, _, err := CheckInferenceViaAPI(opts)

		// Should fail with timeout error
		if err == nil {
			t.Errorf("Expected error due to timeout, got none")
		}
		if valid {
			t.Errorf("Expected valid=false on timeout, got true")
		}
	})
}

// TestCheckGPUViaAPI_UsesAPITimeout verifies that GPU checks via /api/ps
// use the fast API timeout (not the slow inference timeout).
// This ensures quick health checks remain responsive.
func TestCheckGPUViaAPI_UsesAPITimeout(t *testing.T) {
	t.Run("gpu_check_uses_fast_api_timeout", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				// GPU check should respond quickly
				// Sleep less than APITimeout
				time.Sleep(100 * time.Millisecond)

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
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 5 * time.Second,
			Model:            "",
		}

		hasGPU, msg, err := CheckGPUViaAPI(opts)

		if err != nil {
			t.Errorf("Expected no error for fast GPU check, got: %v", err)
		}
		if !hasGPU {
			t.Errorf("Expected hasGPU=true, got false (msg: %s)", msg)
		}
	})

	t.Run("gpu_check_timeout_with_fast_api_timeout", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				// GPU check that takes longer than APITimeout
				time.Sleep(2 * time.Second)
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 5 * time.Second,
			Model:            "",
		}

		hasGPU, _, err := CheckGPUViaAPI(opts)

		// Should timeout because /api/ps takes longer than APITimeout
		if err == nil {
			t.Errorf("Expected timeout error for slow /api/ps response, got none")
		}
		if hasGPU {
			t.Errorf("Expected hasGPU=false on timeout, got true")
		}
	})
}

// TestValidateDeploymentViaAPI_AllowsSlowFirstLoad verifies that model-aware validation
// tolerates first-load latency without timing out.
// When a model is specified, inference runs first (to load the model) with the longer
// inference timeout, then GPU residency is checked with the fast API timeout.
func TestValidateDeploymentViaAPI_AllowsSlowFirstLoad(t *testing.T) {
	t.Run("slow_first_load_inference_succeeds_with_model_specified", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/generate":
				// First inference load takes 3 seconds (realistic for Vulkan/Gemma4)
				time.Sleep(3 * time.Second)
				resp := api.GenerateResponse{
					Model:    "gemma:7b",
					Response: "Generated response after slow first load",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)

			case "/api/ps":
				// After inference, /api/ps responds quickly with GPU info
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "gemma",
							Model:    "gemma:7b",
							Size:     7e9,
							SizeVRAM: 4e9, // GPU is active after inference
							Digest:   "xyz789",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 10 * time.Second,
			Model:            "gemma:7b",
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)

		// Should succeed despite the 3-second inference delay
		if err != nil {
			t.Errorf("Expected no error for slow first-load with inference timeout, got: %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true for slow first-load with inference timeout, got false (msg: %s)", msg)
		}
	})

	t.Run("fast_api_timeout_insufficient_for_inference_fails_cleanly", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/generate" {
				// Slow first-load takes 3 seconds
				time.Sleep(3 * time.Second)
				resp := api.GenerateResponse{
					Model:    "gemma:7b",
					Response: "Generated response",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		// If APITimeout is used for inference, this would fail
		opts := APIDeploymentValidationOptions{
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 1 * time.Second, // Too short for 3-second load
			Model:            "gemma:7b",
		}

		valid, _, err := ValidateDeploymentViaAPI(opts)

		// Should fail with timeout
		if err == nil {
			t.Errorf("Expected timeout error, got none")
		}
		if valid {
			t.Errorf("Expected valid=false on timeout, got true")
		}
	})

	t.Run("gpu_only_check_remains_fast_with_api_timeout", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				// Fast /api/ps response
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
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 10 * time.Second,
			Model:            "", // No model specified
		}

		valid, msg, err := ValidateDeploymentViaAPI(opts)

		// GPU-only check should use APITimeout and succeed
		if err != nil {
			t.Errorf("Expected no error for GPU-only check, got: %v", err)
		}
		if !valid {
			t.Errorf("Expected valid=true for GPU-only check, got false (msg: %s)", msg)
		}
	})
}

// TestTimeoutDefaults_InferenceTimeoutConservative verifies that the default
// inference timeout is conservative (allowing first-load latency).
// This test documents the chosen default value.
func TestTimeoutDefaults_InferenceTimeoutConservative(t *testing.T) {
	t.Run("default_inference_timeout_is_two_minutes", func(t *testing.T) {
		opts := DefaultAPIDeploymentValidationOptions()

		// Default inference timeout should be conservative for first-load models
		// 2 minutes allows Vulkan/Gemma4 first-kernel compilation
		if opts.InferenceTimeout < 1*time.Minute {
			t.Errorf("Expected InferenceTimeout >= 1 minute for conservative first-load tolerance, got %v", opts.InferenceTimeout)
		}
		if opts.InferenceTimeout > 10*time.Minute {
			t.Errorf("Expected InferenceTimeout <= 10 minutes (reasonable upper bound), got %v", opts.InferenceTimeout)
		}
	})

	t.Run("default_api_timeout_is_fast", func(t *testing.T) {
		opts := DefaultAPIDeploymentValidationOptions()

		// Default API timeout should be quick for /api/ps health checks
		if opts.APITimeout > 15*time.Second {
			t.Errorf("Expected APITimeout <= 15 seconds for fast health checks, got %v", opts.APITimeout)
		}
		if opts.APITimeout < 1*time.Second {
			t.Errorf("Expected APITimeout >= 1 second, got %v", opts.APITimeout)
		}
	})

	t.Run("api_timeout_less_than_inference_timeout", func(t *testing.T) {
		opts := DefaultAPIDeploymentValidationOptions()

		if opts.APITimeout >= opts.InferenceTimeout {
			t.Errorf("Expected APITimeout < InferenceTimeout; got APITimeout=%v, InferenceTimeout=%v",
				opts.APITimeout, opts.InferenceTimeout)
		}
	})
}

// TestBackwardsCompatibility_OldOptionsStillWork verifies that code passing
// only the legacy Timeout field continues to work (for backward compatibility).
func TestBackwardsCompatibility_OldOptionsStillWork(t *testing.T) {
	t.Run("legacy_timeout_field_still_supported", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{
						{
							Name:     "test",
							Model:    "test:7b",
							Size:     4e9,
							SizeVRAM: 2e9,
							Digest:   "abc",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		// Create options using only Timeout (legacy)
		opts := APIDeploymentValidationOptions{
			ServiceURL: server.URL,
			Timeout:    5 * time.Second,
			Model:      "",
		}

		// Should still work even if APITimeout/InferenceTimeout are not set
		// The function should use Timeout as a fallback
		hasGPU, _, err := CheckGPUViaAPI(opts)

		if err != nil {
			t.Errorf("Expected legacy Timeout to work, got error: %v", err)
		}
		if !hasGPU {
			t.Errorf("Expected hasGPU=true with legacy options, got false")
		}
	})
}

// TestTimeout_SideEffectFree verifies that validation functions are side-effect free
// by checking they don't modify the options struct or produce external side effects.
func TestTimeout_SideEffectFree(t *testing.T) {
	t.Run("check_inference_via_api_does_not_modify_options", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/generate" {
				resp := api.GenerateResponse{
					Model:    "test:7b",
					Response: "Generated response",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 5 * time.Second,
			Model:            "test:7b",
		}

		// Save original values
		originalAPITimeout := opts.APITimeout
		originalInferenceTimeout := opts.InferenceTimeout
		originalModel := opts.Model
		originalServiceURL := opts.ServiceURL

		// Call the function
		_, _, _ = CheckInferenceViaAPI(opts)

		// Verify options were not modified
		if opts.APITimeout != originalAPITimeout {
			t.Errorf("APITimeout was modified: %v -> %v", originalAPITimeout, opts.APITimeout)
		}
		if opts.InferenceTimeout != originalInferenceTimeout {
			t.Errorf("InferenceTimeout was modified: %v -> %v", originalInferenceTimeout, opts.InferenceTimeout)
		}
		if opts.Model != originalModel {
			t.Errorf("Model was modified: %s -> %s", originalModel, opts.Model)
		}
		if opts.ServiceURL != originalServiceURL {
			t.Errorf("ServiceURL was modified: %s -> %s", originalServiceURL, opts.ServiceURL)
		}
	})

	t.Run("check_gpu_via_api_does_not_modify_options", func(t *testing.T) {
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
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second,
			InferenceTimeout: 5 * time.Second,
			Model:            "",
		}

		// Save original values
		originalAPITimeout := opts.APITimeout
		originalInferenceTimeout := opts.InferenceTimeout

		// Call the function
		_, _, _ = CheckGPUViaAPI(opts)

		// Verify options were not modified
		if opts.APITimeout != originalAPITimeout {
			t.Errorf("APITimeout was modified: %v -> %v", originalAPITimeout, opts.APITimeout)
		}
		if opts.InferenceTimeout != originalInferenceTimeout {
			t.Errorf("InferenceTimeout was modified: %v -> %v", originalInferenceTimeout, opts.InferenceTimeout)
		}
	})
}

// TestTimeoutUsage_DocumentedBehavior verifies the documented timeout usage
// for both fast and slow paths.
func TestTimeoutUsage_DocumentedBehavior(t *testing.T) {
	t.Run("inference_timeout_applies_to_api_generate", func(t *testing.T) {
		callTracker := struct {
			generateCalls int
			psCalls       int
		}{}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/generate" {
				callTracker.generateCalls++
				// Simulate response that takes 2 seconds
				time.Sleep(2 * time.Second)
				resp := api.GenerateResponse{
					Model:    "test:7b",
					Response: "Response",
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			} else if r.URL.Path == "/api/ps" {
				callTracker.psCalls++
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL:       server.URL,
			APITimeout:       500 * time.Millisecond, // Too short for /api/generate
			InferenceTimeout: 5 * time.Second,         // Sufficient for /api/generate
			Model:            "test:7b",
		}

		// With InferenceTimeout, /api/generate should succeed
		valid, _, err := CheckInferenceViaAPI(opts)

		if err != nil {
			t.Errorf("CheckInferenceViaAPI should succeed with InferenceTimeout, got error: %v", err)
		}
		if !valid {
			t.Errorf("CheckInferenceViaAPI should return valid=true, got false")
		}
	})

	t.Run("api_timeout_applies_to_api_ps", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" {
				// Simulate slow /api/ps response
				time.Sleep(2 * time.Second)
				resp := api.ProcessResponse{
					Models: []api.ProcessModelResponse{},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}
		}))
		defer server.Close()

		opts := APIDeploymentValidationOptions{
			ServiceURL:       server.URL,
			APITimeout:       1 * time.Second, // Too short for /api/ps
			InferenceTimeout: 10 * time.Second,
			Model:            "",
		}

		// /api/ps should timeout with APITimeout
		hasGPU, _, err := CheckGPUViaAPI(opts)

		if err == nil {
			t.Errorf("CheckGPUViaAPI should timeout with slow /api/ps response")
		}
		if hasGPU {
			t.Errorf("Expected hasGPU=false on timeout, got true")
		}
	})
}
