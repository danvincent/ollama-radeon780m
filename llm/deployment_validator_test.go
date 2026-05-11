package llm

import (
	"testing"
)

// TestDeploymentValidator_Phase3 tests deployment-level validation to ensure
// the deployment path rejects false-positive claims about GPU execution.
//
// PHASE 3 OBJECTIVE: Ensure deployment claims match the actual running service behavior.
// The deployment validator should:
// 1. Reject false claims of GPU success from logs that show CPU-only fallback
// 2. Reject empty or meaningless responses as valid model execution
// 3. Reject stale/misconfigured service states
// 4. Not be fooled by generic grep patterns or timing heuristics

func TestDeploymentValidator_RejectsCPUFallbackAsGPUSuccess(t *testing.T) {
	t.Run("cpu_fallback_logs_rejected_as_gpu_success", func(t *testing.T) {
		// Scenario: Service logs show CPU fallback (0/36 layers)
		// Deployment validator should REJECT this as GPU success
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="disabling partial Vulkan offload for model" architecture=gemma4
time=2026-05-10T15:30:01Z level=INFO source=server.go msg="gemma4 vulkan full offload fallback not possible, falling back to CPU"
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 0/36 layers to GPU"
`,
			LastResponseContent: "Reykjavik",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to reject CPU fallback (0/36 layers), but got true")
		}
	})

	t.Run("actual_gpu_success_logs_accepted", func(t *testing.T) {
		// Scenario: Service logs show successful GPU offload
		// Deployment validator should ACCEPT this
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="attempting GPU offload"
time=2026-05-10T15:30:01Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
time=2026-05-10T15:30:02Z level=INFO source=server.go msg="model loaded successfully with GPU acceleration"
`,
			LastResponseContent: "Reykjavik is the capital of Iceland",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if !isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to accept GPU success (36/36 layers), but got false")
		}
	})

	t.Run("partial_gpu_offload_accepted", func(t *testing.T) {
		// Scenario: Service logs show partial GPU offload (non-zero but not full)
		// Deployment validator should ACCEPT this (partial GPU is still GPU)
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="attempting GPU offload"
time=2026-05-10T15:30:01Z level=INFO source=ggml.go msg="offloaded 25/36 layers to GPU"
time=2026-05-10T15:30:02Z level=INFO source=server.go msg="model loaded successfully"
`,
			LastResponseContent: "The capital of Iceland is Reykjavik",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if !isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to accept partial GPU (25/36 layers), but got false")
		}
	})
}

func TestDeploymentValidator_RejectsEmptyResponses(t *testing.T) {
	t.Run("empty_response_rejected", func(t *testing.T) {
		// Scenario: Service is running but model returns empty response
		// Deployment validator should REJECT this
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
time=2026-05-10T15:30:02Z level=INFO source=server.go msg="completion processed"
`,
			LastResponseContent: "", // Empty response
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to reject empty response, but got true")
		}
	})

	t.Run("whitespace_only_response_rejected", func(t *testing.T) {
		// Scenario: Service returns whitespace-only response
		// Deployment validator should REJECT this
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
`,
			LastResponseContent: "   \n\t  \n",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to reject whitespace-only response, but got true")
		}
	})

	t.Run("valid_response_accepted", func(t *testing.T) {
		// Scenario: Service returns real response with GPU success
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
`,
			LastResponseContent: "Real model response content",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if !isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to accept valid response with GPU, but got false")
		}
	})
}

func TestDeploymentValidator_RejectsStaleServiceStates(t *testing.T) {
	t.Run("inactive_service_rejected", func(t *testing.T) {
		// Scenario: Service is not running
		// Deployment validator should REJECT this
		serviceState := DeploymentState{
			ServiceActive: false, // Service not running
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
`,
			LastResponseContent: "Response from before service stopped",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to reject inactive service, but got true")
		}
	})

	t.Run("active_service_with_gpu_accepted", func(t *testing.T) {
		// Scenario: Service is active and shows GPU success
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
`,
			LastResponseContent: "Valid response",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if !isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to accept active service with GPU, but got false")
		}
	})
}

func TestDeploymentValidator_OriginalFalsePositiveCase(t *testing.T) {
	t.Run("false_positive_case_from_issue_rejected", func(t *testing.T) {
		// THE ORIGINAL BUG from deployment issue:
		// - Service logs say "offloaded 0/36 layers" (CPU-only)
		// - No response received (empty response)
		// - BUT deployment validation incorrectly reported SUCCESS
		//
		// Phase 3 fix: deployment validator should REJECT this case
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="disabling partial Vulkan offload for model" architecture=gemma4
time=2026-05-10T15:30:01Z level=INFO source=server.go msg="gemma4 vulkan full offload fallback not possible, falling back to CPU"
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 0/36 layers to GPU"
`,
			LastResponseContent: "", // No response (the error case)
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if isValidDeployment {
			t.Errorf("FALSE POSITIVE BUG: Deployment validator should REJECT CPU-only + empty response, but accepted it")
		}

		// Verify both checks fail individually:
		if ValidateGPUOffloadSuccess(serviceState.LastLogLines) {
			t.Errorf("GPU check should fail for 0/36 layers")
		}
		if ValidateResponsePresence(serviceState.LastResponseContent) {
			t.Errorf("Response check should fail for empty response")
		}
	})
}

func TestDeploymentValidator_GenericGrepPatternsNotFooling(t *testing.T) {
	t.Run("gpu_mentioned_in_error_logs_not_accepted", func(t *testing.T) {
		// Scenario: Logs mention "GPU" but context shows failure
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
ERROR: GPU memory allocation failed
ERROR: GPU offload not possible
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 0/36 layers to GPU"
`,
			LastResponseContent: "error message",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to reject GPU mention in error context, but got true")
		}
	})

	t.Run("offload_mentioned_in_error_logs_not_accepted", func(t *testing.T) {
		// Scenario: Logs mention "offload" but context shows fallback
		serviceState := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
WARN: offload initialization failed
INFO: GPU offloaded 0/36 layers
INFO: falling back to CPU
`,
			LastResponseContent: "CPU fallback response",
		}

		isValidDeployment := ValidateDeploymentState(serviceState)
		if isValidDeployment {
			t.Errorf("Expected ValidateDeploymentState to reject offload mention in error context, but got true")
		}
	})
}

func TestDeploymentValidator_CombinedValidationChecks(t *testing.T) {
	t.Run("all_checks_must_pass_for_valid_deployment", func(t *testing.T) {
		tests := []struct {
			name              string
			state             DeploymentState
			expectValid       bool
			description       string
		}{
			{
				name: "gpu_success_with_response",
				state: DeploymentState{
					ServiceActive:       true,
					LastLogLines:        "offloaded 36/36 layers to GPU",
					LastResponseContent: "Valid response",
				},
				expectValid: true,
				description: "Should accept: active service + GPU + response",
			},
			{
				name: "cpu_fallback_with_response",
				state: DeploymentState{
					ServiceActive:       true,
					LastLogLines:        "offloaded 0/36 layers to GPU",
					LastResponseContent: "Some response",
				},
				expectValid: false,
				description: "Should reject: CPU-only fallback (0/36)",
			},
			{
				name: "gpu_success_no_response",
				state: DeploymentState{
					ServiceActive:       true,
					LastLogLines:        "offloaded 36/36 layers to GPU",
					LastResponseContent: "",
				},
				expectValid: false,
				description: "Should reject: no response content",
			},
			{
				name: "service_inactive",
				state: DeploymentState{
					ServiceActive:       false,
					LastLogLines:        "offloaded 36/36 layers to GPU",
					LastResponseContent: "Stale response",
				},
				expectValid: false,
				description: "Should reject: service not active",
			},
			{
				name: "partial_gpu_with_response",
				state: DeploymentState{
					ServiceActive:       true,
					LastLogLines:        "offloaded 20/36 layers to GPU",
					LastResponseContent: "Partial GPU response",
				},
				expectValid: true,
				description: "Should accept: partial GPU (>0 layers) + response",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := ValidateDeploymentState(tt.state)
				if result != tt.expectValid {
					t.Errorf("%s: expected %v, got %v", tt.description, tt.expectValid, result)
				}
			})
		}
	})
}
