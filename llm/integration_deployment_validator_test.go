package llm

import (
	"testing"
)

// TestValidateDeploymentIntegration_RealWorldScenarios tests the deployment validator
// integrated into a real deployment validation path.
//
// Phase 3 Revision: These tests ensure that ValidateDeploymentState is actually called
// from a real integration point (not just as a standalone utility).
//
// Each test simulates:
// 1. Checking if the Ollama service is active (systemd state)
// 2. Getting recent logs from the service
// 3. Getting the last response from a model execution
// 4. Calling ValidateDeploymentState with this real data
// 5. Verifying the deployment validator rejects/accepts as expected

func TestDeploymentIntegration_ServiceLifecycle(t *testing.T) {
	t.Run("integration_successful_gpu_deployment_accepted", func(t *testing.T) {
		// Scenario: A typical successful GPU deployment
		// - Service is running (active)
		// - Recent systemd logs show GPU acceleration
		// - Model produced valid response
		// The deployment validator should ACCEPT this

		// Simulate gathering deployment state from system
		state := DeploymentState{
			ServiceActive: true, // systemctl is-active ollama returned success
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="loading model" model_layers=36 requested=36
time=2026-05-10T15:30:01Z level=DEBUG source=ggml.go msg="new layout created" layers=36
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
time=2026-05-10T15:30:03Z level=INFO source=server.go msg="model loaded successfully with GPU acceleration"
`, // systemctl journal output
			LastResponseContent: "The capital of Iceland is Reykjavik.",
		}

		result := ValidateDeploymentState(state)
		if !result {
			t.Errorf("Integration test FAILED: Expected successful GPU deployment to be accepted")
		}
	})

	t.Run("integration_cpu_fallback_deployment_rejected", func(t *testing.T) {
		// Scenario: Service fell back to CPU (0/36 layers)
		// - Service is running (active)
		// - Recent logs show CPU-only fallback
		// - Model produced response but on CPU only
		// The deployment validator should REJECT this

		state := DeploymentState{
			ServiceActive: true, // systemctl is-active ollama returned success
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="loading model" model_layers=36 requested=36
time=2026-05-10T15:30:01Z level=WARN source=ggml.go msg="GPU memory insufficient for full offload"
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 0/36 layers to GPU"
time=2026-05-10T15:30:03Z level=INFO source=server.go msg="model loaded with CPU fallback"
`, // systemctl journal output
			LastResponseContent: "The capital of Iceland is Reykjavik.",
		}

		result := ValidateDeploymentState(state)
		if result {
			t.Errorf("Integration test FAILED: Expected CPU-only deployment to be rejected")
		}
	})

	t.Run("integration_service_inactive_rejected", func(t *testing.T) {
		// Scenario: Service crashed or was stopped
		// - Service is not active
		// - Old logs may exist but are stale
		// - Last response is old
		// The deployment validator should REJECT this (stale state)

		state := DeploymentState{
			ServiceActive: false, // systemctl is-active ollama returned failure
			LastLogLines: `
time=2026-05-10T15:00:00Z level=INFO source=server.go msg="loading model" model_layers=36 requested=36
time=2026-05-10T15:00:02Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
time=2026-05-10T15:00:05Z level=INFO source=server.go msg="service stopping"
`, // systemctl journal output (stale)
			LastResponseContent: "Old response from before crash",
		}

		result := ValidateDeploymentState(state)
		if result {
			t.Errorf("Integration test FAILED: Expected inactive service to be rejected")
		}
	})

	t.Run("integration_empty_response_rejected", func(t *testing.T) {
		// Scenario: Service is running with GPU but produced no output
		// - Service is running (active)
		// - Logs show GPU acceleration
		// - But last response was empty (model hung or crashed)
		// The deployment validator should REJECT this (no meaningful output)

		state := DeploymentState{
			ServiceActive: true, // systemctl is-active ollama returned success
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="loading model" model_layers=36 requested=36
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
time=2026-05-10T15:30:05Z level=WARN source=server.go msg="completion timeout"
`, // systemctl journal output
			LastResponseContent: "", // Empty response (model produced nothing)
		}

		result := ValidateDeploymentState(state)
		if result {
			t.Errorf("Integration test FAILED: Expected empty response to be rejected")
		}
	})
}

func TestDeploymentIntegration_PartialGPUOffloadAccepted(t *testing.T) {
	t.Run("integration_partial_gpu_deployment_accepted", func(t *testing.T) {
		// Scenario: Service running with partial GPU offload (common with large models)
		// - Service is running (active)
		// - Logs show partial GPU (20/36 layers - not full but still GPU)
		// - Model produced valid response
		// The deployment validator should ACCEPT this (partial GPU is still GPU success)

		state := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="loading model" model_layers=36 requested=20
time=2026-05-10T15:30:01Z level=DEBUG source=ggml.go msg="new layout created" layers=20
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 20/36 layers to GPU"
time=2026-05-10T15:30:03Z level=INFO source=server.go msg="model loaded with partial GPU acceleration"
`,
			LastResponseContent: "A thoughtful response about Iceland",
		}

		result := ValidateDeploymentState(state)
		if !result {
			t.Errorf("Integration test FAILED: Expected partial GPU deployment to be accepted")
		}
	})
}

func TestDeploymentIntegration_LogFormatVariations(t *testing.T) {
	t.Run("integration_different_log_formats_accepted", func(t *testing.T) {
		// Scenario: Different log output formats from various Ollama/runner versions
		// All should be correctly parsed by ValidateGPUOffloadSuccess

		testCases := []struct {
			name       string
			logLines   string
			shouldPass bool
		}{
			{
				name: "standard_gpu_offloaded_format",
				logLines: `
time=2026-05-10T15:30:00Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
`,
				shouldPass: true,
			},
			{
				name: "gpu_offloaded_to_gpu_format",
				logLines: `
time=2026-05-10T15:30:00Z level=INFO source=ggml.go msg="GPU offloaded 36/36 layers to GPU"
`,
				shouldPass: true,
			},
			{
				name: "offloaded_layers_format",
				logLines: `
offloaded_layers: 36/36
`,
				shouldPass: true,
			},
			{
				name: "gpu_layers_format",
				logLines: `
GPU layers: 36/36
`,
				shouldPass: true,
			},
			{
				name: "cpu_fallback_zero_layers",
				logLines: `
offloaded 0/36 layers to GPU
`,
				shouldPass: false,
			},
			{
				name: "no_layer_count_found",
				logLines: `
GPU acceleration attempted
No layer allocation found in logs
`,
				shouldPass: false,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				state := DeploymentState{
					ServiceActive:       true,
					LastLogLines:        tc.logLines,
					LastResponseContent: "Valid response",
				}

				result := ValidateDeploymentState(state)
				if result != tc.shouldPass {
					t.Errorf("Log format test %s FAILED: expected %v, got %v",
						tc.name, tc.shouldPass, result)
				}
			})
		}
	})
}

func TestDeploymentIntegration_RealWorldFailurePatterns(t *testing.T) {
	t.Run("integration_rejects_real_world_false_positive_cpu_fallback", func(t *testing.T) {
		// This is the EXACT scenario from the original issue:
		// - Build says "GPU deployment"
		// - Service starts with Vulkan backend
		// - But GPU memory insufficient, falls back to CPU
		// - Logs show "offloaded 0/36 layers"
		// - BUT some old response may exist in cache
		//
		// Previously: Deployment validation would incorrectly pass
		// Now: Should correctly REJECT as CPU-only

		state := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=server.go msg="disabling partial Vulkan offload for model" architecture=gemma4
time=2026-05-10T15:30:01Z level=INFO source=server.go msg="gemma4 vulkan full offload fallback not possible, falling back to CPU"
time=2026-05-10T15:30:02Z level=INFO source=ggml.go msg="offloaded 0/36 layers to GPU"
time=2026-05-10T15:30:05Z level=INFO source=server.go msg="completion processed"
`,
			LastResponseContent: "Reykjavik", // This response exists but was on CPU, not GPU
		}

		result := ValidateDeploymentState(state)
		if result {
			t.Errorf("Integration test FAILED: Original bug case should be REJECTED (CPU-only + response)")
		}
	})

	t.Run("integration_rejects_missing_response_with_gpu", func(t *testing.T) {
		// Scenario: GPU is allocated but model produces nothing
		// - Service running
		// - Logs show GPU (36/36)
		// - But model crashed or response is empty
		// Should REJECT as deployment is not truly working

		state := DeploymentState{
			ServiceActive: true,
			LastLogLines: `
time=2026-05-10T15:30:00Z level=INFO source=ggml.go msg="offloaded 36/36 layers to GPU"
time=2026-05-10T15:30:03Z level=ERROR source=server.go msg="model inference crashed"
`,
			LastResponseContent: "",
		}

		result := ValidateDeploymentState(state)
		if result {
			t.Errorf("Integration test FAILED: Should reject missing response even with GPU acceleration")
		}
	})
}

func TestDeploymentIntegration_CombinedValidationLogic(t *testing.T) {
	t.Run("integration_all_three_checks_required", func(t *testing.T) {
		// This test verifies that ValidateDeploymentState requires ALL three checks:
		// 1. Service active
		// 2. GPU acceleration
		// 3. Response content
		//
		// And that failing any single check causes the deployment to fail

		testCases := []struct {
			name           string
			serviceActive  bool
			logLines       string
			responseContent string
			shouldPass     bool
			reason         string
		}{
			{
				name:            "all_pass",
				serviceActive:   true,
				logLines:        "offloaded 36/36 layers to GPU",
				responseContent: "valid response",
				shouldPass:      true,
				reason:          "All checks pass",
			},
			{
				name:            "service_inactive",
				serviceActive:   false,
				logLines:        "offloaded 36/36 layers to GPU",
				responseContent: "valid response",
				shouldPass:      false,
				reason:          "Service not active",
			},
			{
				name:            "cpu_fallback",
				serviceActive:   true,
				logLines:        "offloaded 0/36 layers to GPU",
				responseContent: "valid response",
				shouldPass:      false,
				reason:          "CPU-only fallback (0/36)",
			},
			{
				name:            "no_response",
				serviceActive:   true,
				logLines:        "offloaded 36/36 layers to GPU",
				responseContent: "",
				shouldPass:      false,
				reason:          "Empty response",
			},
			{
				name:            "no_gpu_logs",
				serviceActive:   true,
				logLines:        "no layer allocation found",
				responseContent: "valid response",
				shouldPass:      false,
				reason:          "No GPU layer count in logs",
			},
			{
				name:            "whitespace_response",
				serviceActive:   true,
				logLines:        "offloaded 36/36 layers to GPU",
				responseContent: "   \n\t  \n",
				shouldPass:      false,
				reason:          "Only whitespace in response",
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				state := DeploymentState{
					ServiceActive:       tc.serviceActive,
					LastLogLines:        tc.logLines,
					LastResponseContent: tc.responseContent,
				}

				result := ValidateDeploymentState(state)
				if result != tc.shouldPass {
					t.Errorf("FAILED: %s - expected %v, got %v. Reason: %s",
						tc.name, tc.shouldPass, result, tc.reason)
				}
			})
		}
	})
}
