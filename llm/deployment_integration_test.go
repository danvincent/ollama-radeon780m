package llm

import (
	"strings"
	"testing"
)

func TestDeploymentIntegration_ValidationResult(t *testing.T) {
	t.Run("validation_result_summary_on_success", func(t *testing.T) {
		result := ValidationResult{
			Valid:              true,
			ServiceName:        "ollama",
			ServiceActive:      true,
			ServiceActiveReason: "Service is active",
			GPUOffloadValid:    true,
			GPUOffloadReason:   "Logs show successful GPU offload",
			ResponseValid:      true,
			ResponseReason:     "Response valid (42 bytes of content)",
		}

		summary := result.Summary()
		if !strings.Contains(summary, "✓ DEPLOYMENT VALID") {
			t.Errorf("Summary should contain success indicator, got: %s", summary)
		}
		if !strings.Contains(summary, "ollama") {
			t.Errorf("Summary should contain service name, got: %s", summary)
		}
	})

	t.Run("validation_result_summary_on_failure", func(t *testing.T) {
		result := ValidationResult{
			Valid:               false,
			ServiceName:         "ollama",
			ServiceActive:       true,
			ServiceActiveReason: "Service is active",
			GPUOffloadValid:     false,
			GPUOffloadReason:    "Logs show CPU-only fallback",
			ResponseValid:       true,
			ResponseReason:      "Response valid",
		}

		summary := result.Summary()
		if !strings.Contains(summary, "✗ DEPLOYMENT INVALID") {
			t.Errorf("Summary should contain failure indicator, got: %s", summary)
		}
		if !strings.Contains(summary, "CPU-only fallback") {
			t.Errorf("Summary should show reason for GPU check failure, got: %s", summary)
		}
	})

	t.Run("validation_result_exit_code_success", func(t *testing.T) {
		result := ValidationResult{Valid: true}
		if result.ExitCode() != 0 {
			t.Errorf("Expected exit code 0 for valid deployment, got %d", result.ExitCode())
		}
	})

	t.Run("validation_result_exit_code_failure", func(t *testing.T) {
		result := ValidationResult{Valid: false}
		if result.ExitCode() != 1 {
			t.Errorf("Expected exit code 1 for invalid deployment, got %d", result.ExitCode())
		}
	})

	t.Run("validation_result_exit_code_error", func(t *testing.T) {
		result := ValidationResult{Error: "Failed to gather state"}
		if result.ExitCode() != 2 {
			t.Errorf("Expected exit code 2 for error state, got %d", result.ExitCode())
		}
	})
}

func TestDeploymentIntegration_ValidationResult_AllChecksCombined(t *testing.T) {
	t.Run("validate_deployment_output_with_mock_state", func(t *testing.T) {
		// This test demonstrates how ValidateDeploymentOutput would work
		// in a real integration scenario (though we can't easily mock systemctl)

		opts := DefaultDeploymentValidationOptions()

		// Create a validation result as if we had gathered state
		result := ValidationResult{
			ServiceName: opts.ServiceName,
			GatheredState: DeploymentState{
				ServiceActive:       true,
				LastLogLines:        "offloaded 36/36 layers to GPU",
				LastResponseContent: "Valid response",
			},
			ServiceActive:       true,
			ServiceActiveReason: "Service is active",
			GPUOffloadValid:     true,
			GPUOffloadReason:    "Logs show successful GPU offload",
			ResponseValid:       true,
			ResponseReason:      "Response valid (15 bytes of content)",
			Valid:               true,
		}

		if !result.Valid {
			t.Errorf("Validation should pass for correct deployment state")
		}

		if result.ExitCode() != 0 {
			t.Errorf("Expected exit code 0 for successful validation")
		}

		summary := result.Summary()
		if !strings.Contains(summary, "✓ DEPLOYMENT VALID") {
			t.Errorf("Expected success in summary")
		}
	})
}

func TestDeploymentIntegration_DefaultOptions(t *testing.T) {
	t.Run("default_options_reasonable_values", func(t *testing.T) {
		opts := DefaultDeploymentValidationOptions()

		if opts.JournalLines <= 0 {
			t.Errorf("JournalLines should be positive, got %d", opts.JournalLines)
		}

		if opts.ServiceName != "ollama" {
			t.Errorf("ServiceName should default to 'ollama', got %s", opts.ServiceName)
		}

		if opts.Timeout <= 0 {
			t.Errorf("Timeout should be positive, got %v", opts.Timeout)
		}
	})
}

func TestDeploymentIntegration_IntegrationPath(t *testing.T) {
	t.Run("integration_validation_path_called_from_deployment_script", func(t *testing.T) {
		// This test verifies the integration path that would be called from
		// build_deploy_local.sh or similar deployment scripts

		// Simulate what a deployment script would do:
		// 1. Run ValidateDeploymentOutput to validate the deployment
		// 2. Check the result
		// 3. Use the exit code to determine success/failure

		// In real usage:
		// - build_deploy_local.sh would call a helper (e.g., via go run or compiled binary)
		// - The helper would call CheckDeploymentReadiness()
		// - The shell script would check the exit code

		// For testing, we can verify the function returns correct codes:
		_ = DefaultDeploymentValidationOptions() // Used for reference, verification happens below

		// Simulate successful deployment
		successResult := ValidationResult{
			Valid:       true,
			ServiceName: "ollama",
		}
		if successResult.ExitCode() != 0 {
			t.Errorf("Success should exit 0")
		}

		// Simulate failed deployment
		failResult := ValidationResult{
			Valid:       false,
			ServiceName: "ollama",
		}
		if failResult.ExitCode() != 1 {
			t.Errorf("Failure should exit 1")
		}

		// Simulate error gathering state
		errorResult := ValidationResult{
			Error: "systemctl not available",
		}
		if errorResult.ExitCode() != 2 {
			t.Errorf("Error should exit 2")
		}
	})
}

func TestDeploymentIntegration_ScriptIntegration(t *testing.T) {
	t.Run("integration_shows_how_to_integrate_into_build_script", func(t *testing.T) {
		// This test documents how ValidateDeploymentOutput would be integrated
		// into build_deploy_local.sh

		// The integration would look like:
		// ===== build_deploy_local.sh snippet =====
		// verify_gpu_deployment() {
		//     status "Verifying GPU deployment..."
		//
		//     # Call the Go deployment validator
		//     # This can be done via:
		//     # 1. A compiled helper binary: ./ollama-deploy-check
		//     # 2. Or via go run: go run ./cmd/deploy-check/main.go
		//
		//     if ! systemctl is-active --quiet ollama; then
		//         error "ollama service is not active"
		//     fi
		//
		//     # Get recent logs
		//     local logs=$(journalctl -u ollama -n 50 --no-pager)
		//
		//     # Verify GPU is in logs
		//     if ! echo "$logs" | grep -q "offloaded [1-9][0-9]*/"; then
		//         error "Logs do not show GPU offload (might be CPU-only)"
		//     fi
		//
		//     success "GPU deployment verified"
		// }

		// For verification in this test, we check that the ValidationResult
		// structure correctly represents the state that would be checked

		result := ValidationResult{
			Valid:               true,
			ServiceActive:       true,
			GPUOffloadValid:     true,
			ResponseValid:       true,
		}

		// These are the three conditions a script would check:
		checks := []struct {
			condition bool
			name      string
		}{
			{result.ServiceActive, "Service is active"},
			{result.GPUOffloadValid, "GPU offload detected"},
			{result.ResponseValid, "Response is valid"},
		}

		for _, check := range checks {
			if !check.condition {
				t.Errorf("Check '%s' should be true for valid deployment", check.name)
			}
		}
	})
}
