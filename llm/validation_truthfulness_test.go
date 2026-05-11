package llm

import (
	"testing"
	"time"
)

// TestValidationTruthfulnessPhase1 tests the validation logic to ensure it
// reports truthfully about GPU vs CPU execution and response presence.
//
// PHASE 1 OBJECTIVE: Make validation truthful by:
// 1. Removing false GPU success signals for CPU-only fallback (0/N layers)
// 2. Failing validation when response output is empty
// 3. NOT inferring GPU acceleration from elapsed time alone
// 4. Avoiding false positives from generic text matching (GPU, offload, etc)

// TestValidation_CPUFallbackNotReportedAsGPUSuccess tests that validation
// correctly identifies CPU-only execution and does NOT report it as GPU success.
func TestValidation_CPUFallbackNotReportedAsGPUSuccess(t *testing.T) {
	t.Run("cpu_fallback_0_layers_should_fail_gpu_check", func(t *testing.T) {
		// Scenario: Gemma4 attempted partial Vulkan offload but fell back to CPU (0/36 layers)
		// The validation should correctly identify this as CPU-only, NOT GPU success
		logOutput := `
INFO: Gemma4 partial Vulkan offload failed
INFO: GPU offloaded 0/36 layers to GPU
INFO: fallback to CPU execution
`

		// This validation check should FAIL (return false) because GPU layers = 0
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to return false for 0/36 layers, but got true")
		}
	})

	t.Run("actual_gpu_offload_should_pass", func(t *testing.T) {
		// Scenario: Full Vulkan offload succeeded (36/36 layers)
		// The validation should correctly identify this as GPU success
		logOutput := `
INFO: Attempting Gemma4 Vulkan offload
INFO: GPU offloaded 36/36 layers to GPU
INFO: GPU memory allocated successfully
`

		// This validation check should PASS (return true) because GPU layers = 36
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if !gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to return true for 36/36 layers, but got false")
		}
	})

	t.Run("partial_gpu_offload_should_pass", func(t *testing.T) {
		// Scenario: Partial Vulkan offload succeeded (35/36 layers)
		// The validation should correctly identify this as GPU success (not full, but not zero)
		logOutput := `
INFO: Attempting Gemma4 Vulkan offload
INFO: GPU offloaded 35/36 layers to GPU
INFO: GPU memory allocated successfully
`

		// This validation check should PASS (return true) because GPU layers > 0
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if !gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to return true for 35/36 layers, but got false")
		}
	})
}

// TestValidation_EmptyResponseFailsValidation tests that validation
// correctly fails when response output is empty.
func TestValidation_EmptyResponseFailsValidation(t *testing.T) {
	t.Run("empty_response_should_fail_response_check", func(t *testing.T) {
		// Scenario: Model loaded and responded, but output is empty/whitespace
		// Validation should FAIL because there's no actual response content
		responseContent := "" // Empty response
		hasValidResponse := ValidateResponsePresence(responseContent)
		if hasValidResponse {
			t.Errorf("Expected ValidateResponsePresence to return false for empty response, but got true")
		}
	})

	t.Run("whitespace_only_response_should_fail", func(t *testing.T) {
		// Scenario: Response contains only whitespace (effectively empty)
		// Validation should FAIL
		responseContent := "   \n\t  \n" // Only whitespace
		hasValidResponse := ValidateResponsePresence(responseContent)
		if hasValidResponse {
			t.Errorf("Expected ValidateResponsePresence to return false for whitespace-only response, but got true")
		}
	})

	t.Run("non_empty_response_should_pass", func(t *testing.T) {
		// Scenario: Model returned actual text content
		// Validation should PASS
		responseContent := "Reykjavik is the capital of Iceland"
		hasValidResponse := ValidateResponsePresence(responseContent)
		if !hasValidResponse {
			t.Errorf("Expected ValidateResponsePresence to return true for non-empty response, but got false")
		}
	})
}

// TestValidation_TimeAloneDoesNotIndicateGPU tests that validation
// does NOT infer GPU acceleration from elapsed time alone.
func TestValidation_TimeAloneDoesNotIndicateGPU(t *testing.T) {
	t.Run("fast_response_with_cpu_only_should_not_report_gpu", func(t *testing.T) {
		// Scenario: Response came back fast (~25s), BUT logs show 0/36 GPU layers
		// Validation should NOT report GPU success just because response was fast
		// It must check actual GPU layer allocation, not just time
		logOutput := `
INFO: GPU offloaded 0/36 layers to GPU
INFO: CPU-only execution mode
`
		_ = time.Second * 25 // Fast response - just noting it exists

		// Even with fast response time, GPU check should still fail because 0 GPU layers
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to return false for 0 GPU layers, regardless of response time")
		}
	})

	t.Run("slow_response_with_gpu_should_still_pass", func(t *testing.T) {
		// Scenario: Response took ~45s, but logs show 36/36 GPU layers
		// Validation should report GPU success because actual GPU layers are allocated
		logOutput := `
INFO: GPU offloaded 36/36 layers to GPU
INFO: GPU acceleration active
`
		_ = time.Second * 45 // Slow response - just noting it exists

		// Should pass because we have GPU layers, regardless of total time
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if !gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to return true for 36 GPU layers, regardless of response time")
		}
	})
}

// TestValidation_NoFalsePositivesFromGenericGrep tests that validation
// doesn't get fooled by generic text matches like "GPU", "offload", or model names.
func TestValidation_NoFalsePositivesFromGenericGrep(t *testing.T) {
	t.Run("gpu_in_error_message_not_counted_as_success", func(t *testing.T) {
		// Scenario: Log contains the word "GPU" but in a negative context
		// e.g., "GPU memory allocation failed"
		// Generic grep for "GPU" would incorrectly report success
		logOutput := `
ERROR: GPU memory allocation failed, insufficient VRAM for offload
INFO: GPU offloaded 0/36 layers to GPU
INFO: Falling back to CPU-only mode
`

		// Must parse correctly despite "GPU" appearing in error message
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to return false despite 'GPU' appearing in error message")
		}
	})

	t.Run("offload_in_error_not_counted_as_success", func(t *testing.T) {
		// Scenario: Log contains "offload" but in context of failure
		logOutput := `
WARN: offload initialization failed due to backend mismatch
INFO: GPU offloaded 0/36 layers to GPU
INFO: CPU fallback active
`

		// Must correctly identify 0 GPU layers despite "offload" in log
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to return false despite 'offload' appearing in log")
		}
	})

	t.Run("model_name_in_success_message_not_false_match", func(t *testing.T) {
		// Scenario: Log mentions "Gemma4" but context is about CPU fallback
		logOutput := `
INFO: Gemma4 attempted GPU acceleration but fell back to CPU
INFO: GPU offloaded 0/36 layers to GPU
`

		// Must correctly parse layer count, not get confused by model name
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to return false despite 'Gemma4' appearing in log")
		}
	})

	t.Run("generic_grep_for_gpu_would_fail_on_multiple_lines", func(t *testing.T) {
		// Scenario: Log has multiple lines with GPU mentioned
		// A naive grep for "GPU" would match on the first occurrence
		// even though the actual status is CPU-only
		logOutput := `
INFO: Checking GPU availability
INFO: GPU memory check complete
INFO: GPU offloaded 0/36 layers to GPU
INFO: CPU-only fallback in effect
`

		// Must parse to find ACTUAL GPU layer count (0), not just presence of "GPU"
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if gpuSuccess {
			t.Errorf("Expected ValidateGPUOffloadSuccess to correctly identify 0 GPU layers, not be fooled by 'GPU' text")
		}
	})
}

// TestValidation_CombinedChecksRejectFalsePositives tests that validation
// correctly rejects scenarios where older logic would have incorrectly reported success.
func TestValidation_CombinedChecksRejectFalsePositives(t *testing.T) {
	t.Run("false_positive_case_fast_response_empty_output_cpu_only", func(t *testing.T) {
		// THE ORIGINAL BUG: Live validation output showed:
		// - ✗ PROBLEM: Gemma4 is CPU-only (0/36 layers on GPU)
		// - ✗ ERROR: No response received
		// - BUT ALSO printed: ✓ FAST: Response in < 30s (likely GPU-accelerated)
		//
		// This is the exact false-positive case that Phase 1 must fix.
		// Validation should FAIL on all three checks:
		// 1. GPU layers = 0 (CPU-only)
		// 2. Response is empty or very short
		// 3. Fast time should NOT be counted as GPU success

		logs := `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="disabling partial Vulkan offload for model" architecture=gemma4
time=2026-05-10T15:30:01Z level=INFO source=server.go msg="gemma4 vulkan full offload fallback not possible, falling back to CPU"
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 0/36 layers to GPU"
`
		responseOutput := "" // Empty response (the error message said "No response received")
		_ = time.Second * 28 // Fast response (under 30s threshold) - just noting it exists

		// Combined validation result should be FAILED
		// at least one of these checks should fail:
		gpuSuccess := ValidateGPUOffloadSuccess(logs)
		hasResponse := ValidateResponsePresence(responseOutput)
		allChecksPassed := gpuSuccess && hasResponse

		if allChecksPassed {
			t.Errorf("FALSE POSITIVE BUG: Combined validation should FAIL for CPU-only + empty response scenario, but all checks passed")
		}

		// Verify specific failures
		if gpuSuccess {
			t.Errorf("GPU check incorrectly passed for 0/36 layers")
		}
		if hasResponse {
			t.Errorf("Response check incorrectly passed for empty response")
		}
	})
}

// TestValidation_EdgeCasesAndFormats tests that validation correctly handles
// various log formats and edge cases.
func TestValidation_EdgeCasesAndFormats(t *testing.T) {
	t.Run("multiline_logs_with_multiple_gpu_mentions", func(t *testing.T) {
		// Logs with multiple occurrences of GPU-related text
		logOutput := `
INFO: Checking GPU compatibility
INFO: GPU memory: 8GB available
INFO: Attempting GPU offload
INFO: GPU offloaded 0/36 layers to GPU
INFO: GPU acceleration failed, falling back to CPU
`
		gpuSuccess := ValidateGPUOffloadSuccess(logOutput)
		if gpuSuccess {
			t.Errorf("Should identify 0/36 layers despite multiple GPU mentions")
		}
	})

	t.Run("different_layer_count_patterns", func(t *testing.T) {
		tests := []struct {
			name        string
			logOutput   string
			expectPass  bool
			description string
		}{
			{
				name:        "format_offloaded_N_of_M_layers",
				logOutput:   "offloaded 15/36 layers",
				expectPass:  true,
				description: "Standard format: offloaded N/M layers",
			},
			{
				name:        "format_gpu_offloaded_N_of_M",
				logOutput:   "GPU offloaded 36/36 layers to GPU",
				expectPass:  true,
				description: "Full format: GPU offloaded N/M layers to GPU",
			},
			{
				name:        "format_offloaded_layers_N_M",
				logOutput:   "offloaded_layers: 20/36",
				expectPass:  true,
				description: "Key-value format: offloaded_layers: N/M",
			},
			{
				name:        "format_gpu_layers_N_M",
				logOutput:   "GPU layers: 0/36",
				expectPass:  false,
				description: "Key-value format with GPU layers: 0/M means CPU-only",
			},
			{
				name:        "zero_GPU_layers_offloaded_format",
				logOutput:   "offloaded 0/36 layers",
				expectPass:  false,
				description: "Explicitly states 0 GPU layers",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := ValidateGPUOffloadSuccess(tt.logOutput)
				if result != tt.expectPass {
					t.Errorf("%s: expected %v, got %v", tt.description, tt.expectPass, result)
				}
			})
		}
	})

	t.Run("response_with_various_whitespace", func(t *testing.T) {
		tests := []struct {
			name       string
			content    string
			expectPass bool
		}{
			{
				name:       "single_space",
				content:    " ",
				expectPass: false,
			},
			{
				name:       "multiple_spaces",
				content:    "     ",
				expectPass: false,
			},
			{
				name:       "tabs_and_newlines",
				content:    "\t\n\t\n",
				expectPass: false,
			},
			{
				name:       "single_character",
				content:    "a",
				expectPass: true,
			},
			{
				name:       "single_word",
				content:    "response",
				expectPass: true,
			},
			{
				name:       "text_with_surrounding_whitespace",
				content:    "  \n  response content  \n  ",
				expectPass: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := ValidateResponsePresence(tt.content)
				if result != tt.expectPass {
					t.Errorf("ValidateResponsePresence(%q): expected %v, got %v", tt.content, tt.expectPass, result)
				}
			})
		}
	})
}
