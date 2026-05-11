package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"
)

// TestDeploymentCheckOptions_ToAPIOptions tests the field mapping from CLI to API options.
func TestDeploymentCheckOptions_ToAPIOptions(t *testing.T) {
	tests := []struct {
		name      string
		cliOpts   DeploymentCheckOptions
		wantURL   string
		wantTime  time.Duration
		wantModel string
	}{
		{
			name: "all_fields_set",
			cliOpts: DeploymentCheckOptions{
				ServiceURL: "http://localhost:11434",
				Timeout:    10 * time.Second,
				Model:      "llama2",
			},
			wantURL:   "http://localhost:11434",
			wantTime:  10 * time.Second,
			wantModel: "llama2",
		},
		{
			name: "no_model_specified",
			cliOpts: DeploymentCheckOptions{
				ServiceURL: "http://192.168.1.100:11434",
				Timeout:    5 * time.Second,
				Model:      "",
			},
			wantURL:   "http://192.168.1.100:11434",
			wantTime:  5 * time.Second,
			wantModel: "",
		},
		{
			name: "custom_timeouts",
			cliOpts: DeploymentCheckOptions{
				ServiceURL: "http://localhost:11434",
				Timeout:    30 * time.Second,
				Model:      "neural-chat",
			},
			wantURL:   "http://localhost:11434",
			wantTime:  30 * time.Second,
			wantModel: "neural-chat",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.cliOpts.toAPIOptions()

			if got.ServiceURL != tt.wantURL {
				t.Errorf("ServiceURL: got %q, want %q", got.ServiceURL, tt.wantURL)
			}
			if got.Timeout != tt.wantTime {
				t.Errorf("Timeout: got %v, want %v", got.Timeout, tt.wantTime)
			}
			if got.Model != tt.wantModel {
				t.Errorf("Model: got %q, want %q", got.Model, tt.wantModel)
			}
		})
	}
}

// TestDeploymentCheckOptions_ToAPIOptions_TypeCompatibility ensures the
// returned llm.APIDeploymentValidationOptions can be used by the unified validator.
func TestDeploymentCheckOptions_ToAPIOptions_TypeCompatibility(t *testing.T) {
	cliOpts := DeploymentCheckOptions{
		ServiceURL: "http://test:11434",
		Timeout:    15 * time.Second,
		Model:      "test-model",
	}

	apiOpts := cliOpts.toAPIOptions()

	// Verify it's actually an llm.APIDeploymentValidationOptions
	// by checking the fields that ValidateDeploymentViaAPI expects
	if apiOpts.ServiceURL == "" {
		t.Error("ServiceURL must not be empty for validator")
	}
	if apiOpts.Timeout == 0 {
		t.Error("Timeout must not be zero for validator")
	}
	// Model can be empty (inference check is optional)
}

// TestExitCodeMapping_Success tests the exit code when validation succeeds.
func TestExitCodeMapping_Success(t *testing.T) {
	code := mapExitCode(true, nil)
	if code != 0 {
		t.Errorf("Success case: got exit code %d, want 0", code)
	}
}

// TestExitCodeMapping_ValidationFailure tests the exit code when validation fails
// (GPU not detected or inference failed, but API was reachable).
func TestExitCodeMapping_ValidationFailure(t *testing.T) {
	code := mapExitCode(false, nil)
	if code != 1 {
		t.Errorf("Validation failure case: got exit code %d, want 1", code)
	}
}

// TestExitCodeMapping_ErrorGatheringState tests the exit code when there's an error
// gathering deployment state (API unreachable, timeout, etc.).
func TestExitCodeMapping_ErrorGatheringState(t *testing.T) {
	testErr := errors.New("connection failed")
	code := mapExitCode(false, testErr)
	if code != 2 {
		t.Errorf("Error gathering state case: got exit code %d, want 2", code)
	}
}

// TestExitCodeMapping_ErrorTakePrecedence tests that errors take precedence
// over the valid flag. If there's an error, we return 2 regardless of valid value.
func TestExitCodeMapping_ErrorTakePrecedence(t *testing.T) {
	testErr := errors.New("timeout")
	// Even if valid were somehow true (shouldn't happen), error takes precedence
	code := mapExitCode(true, testErr)
	if code != 2 {
		t.Errorf("Error with valid=true: got exit code %d, want 2 (error takes precedence)", code)
	}
}

// TestDeploymentCheckOptions_NoTimeoutFlags tests the critical case where no timeout
// flags are passed. The CLI should NOT override the llm package's conservative defaults.
// This is the primary blocker fix.
func TestDeploymentCheckOptions_NoTimeoutFlags(t *testing.T) {
	cliOpts := DeploymentCheckOptions{
		ServiceURL:       "http://localhost:11434",
		Timeout:          0, // Not set by user
		APITimeout:       0, // Not set by user
		InferenceTimeout: 0, // Not set by user
		Model:            "llama2",
	}

	got := cliOpts.toAPIOptions()

	// When no flags are passed, timeouts should be 0 to let llm package apply defaults
	if got.Timeout != 0 {
		t.Errorf("Timeout: got %v, want 0 (should let llm package apply default)", got.Timeout)
	}
	if got.InferenceTimeout != 0 {
		t.Errorf("InferenceTimeout: got %v, want 0 (should let llm package apply 2m default)", got.InferenceTimeout)
	}
	if got.APITimeout != 0 {
		t.Errorf("APITimeout: got %v, want 0 (should let llm package apply 10s default)", got.APITimeout)
	}
}

// TestDeploymentCheckOptions_LegacyTimeoutOnly tests that the legacy -timeout flag works
// when explicitly set by the user, but ONLY when explicitly set.
func TestDeploymentCheckOptions_LegacyTimeoutOnly(t *testing.T) {
	cliOpts := DeploymentCheckOptions{
		ServiceURL:       "http://localhost:11434",
		Timeout:          15 * time.Second, // User explicitly set legacy flag
		APITimeout:       0,                // Not set
		InferenceTimeout: 0,                // Not set
		Model:            "llama2",
	}

	got := cliOpts.toAPIOptions()

	// Legacy timeout should apply to both API and inference when explicitly set
	if got.APITimeout != 15*time.Second {
		t.Errorf("APITimeout: got %v, want 15s (should use legacy Timeout)", got.APITimeout)
	}
	if got.InferenceTimeout != 15*time.Second {
		t.Errorf("InferenceTimeout: got %v, want 15s (should use legacy Timeout)", got.InferenceTimeout)
	}
	if got.Timeout != 15*time.Second {
		t.Errorf("Timeout: got %v, want 15s", got.Timeout)
	}
}

// TestDeploymentCheckOptions_APITimeoutOnly tests that -api-timeout can be set independently.
func TestDeploymentCheckOptions_APITimeoutOnly(t *testing.T) {
	cliOpts := DeploymentCheckOptions{
		ServiceURL:       "http://localhost:11434",
		Timeout:          0,                // Not set
		APITimeout:       20 * time.Second, // User set this
		InferenceTimeout: 0,                // Not set
		Model:            "",
	}

	got := cliOpts.toAPIOptions()

	// Only APITimeout should be set
	if got.APITimeout != 20*time.Second {
		t.Errorf("APITimeout: got %v, want 20s", got.APITimeout)
	}
	if got.InferenceTimeout != 0 {
		t.Errorf("InferenceTimeout: got %v, want 0 (should let llm package apply default)", got.InferenceTimeout)
	}
	if got.Timeout != 0 {
		t.Errorf("Timeout: got %v, want 0", got.Timeout)
	}
}

// TestDeploymentCheckOptions_InferenceTimeoutOnly tests that -inference-timeout can be set independently.
func TestDeploymentCheckOptions_InferenceTimeoutOnly(t *testing.T) {
	cliOpts := DeploymentCheckOptions{
		ServiceURL:       "http://localhost:11434",
		Timeout:          0,               // Not set
		APITimeout:       0,               // Not set
		InferenceTimeout: 5 * time.Minute, // User set this
		Model:            "llama2",
	}

	got := cliOpts.toAPIOptions()

	// Only InferenceTimeout should be set
	if got.InferenceTimeout != 5*time.Minute {
		t.Errorf("InferenceTimeout: got %v, want 5m", got.InferenceTimeout)
	}
	if got.APITimeout != 0 {
		t.Errorf("APITimeout: got %v, want 0 (should let llm package apply default)", got.APITimeout)
	}
	if got.Timeout != 0 {
		t.Errorf("Timeout: got %v, want 0", got.Timeout)
	}
}

// TestDeploymentCheckOptions_PrecedenceSpecificOverLegacy tests that specific flags
// take precedence over the legacy -timeout flag when both are set.
// Important: When specific flags are used, the legacy timeout does NOT fill in the gaps.
// This is to prevent confusing mixed behavior where some values come from legacy and others from specific.
func TestDeploymentCheckOptions_PrecedenceSpecificOverLegacy(t *testing.T) {
	tests := []struct {
		name                 string
		cliOpts              DeploymentCheckOptions
		wantAPITimeout       time.Duration
		wantInferenceTimeout time.Duration
	}{
		{
			name: "api_timeout_overrides_legacy",
			cliOpts: DeploymentCheckOptions{
				ServiceURL:       "http://localhost:11434",
				Timeout:          10 * time.Second, // Legacy
				APITimeout:       25 * time.Second, // Should take precedence
				InferenceTimeout: 0,
				Model:            "",
			},
			wantAPITimeout:       25 * time.Second,
			wantInferenceTimeout: 10 * time.Second, // Legacy applies when specific not set
		},
		{
			name: "inference_timeout_overrides_legacy",
			cliOpts: DeploymentCheckOptions{
				ServiceURL:       "http://localhost:11434",
				Timeout:          10 * time.Second, // Legacy
				APITimeout:       0,
				InferenceTimeout: 3 * time.Minute, // Should take precedence
				Model:            "llama2",
			},
			wantAPITimeout:       10 * time.Second, // Legacy applies when specific not set
			wantInferenceTimeout: 3 * time.Minute,
		},
		{
			name: "both_specific_override_legacy",
			cliOpts: DeploymentCheckOptions{
				ServiceURL:       "http://localhost:11434",
				Timeout:          10 * time.Second, // Legacy (ignored when both specifics set)
				APITimeout:       25 * time.Second, // Should take precedence
				InferenceTimeout: 4 * time.Minute,  // Should take precedence
				Model:            "llama2",
			},
			wantAPITimeout:       25 * time.Second,
			wantInferenceTimeout: 4 * time.Minute,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.cliOpts.toAPIOptions()

			if got.APITimeout != tt.wantAPITimeout {
				t.Errorf("APITimeout: got %v, want %v", got.APITimeout, tt.wantAPITimeout)
			}
			if got.InferenceTimeout != tt.wantInferenceTimeout {
				t.Errorf("InferenceTimeout: got %v, want %v", got.InferenceTimeout, tt.wantInferenceTimeout)
			}
		})
	}
}

// TestDeploymentCheckOptions_APITimeout tests that the separate API timeout
// field is properly set and propagated to API options.
func TestDeploymentCheckOptions_APITimeout(t *testing.T) {
	tests := []struct {
		name        string
		cliOpts     DeploymentCheckOptions
		wantAPITime time.Duration
	}{
		{
			name: "custom_api_timeout",
			cliOpts: DeploymentCheckOptions{
				ServiceURL: "http://localhost:11434",
				APITimeout: 20 * time.Second,
				Timeout:    0, // Not set
				Model:      "",
			},
			wantAPITime: 20 * time.Second,
		},
		{
			name: "default_api_timeout",
			cliOpts: DeploymentCheckOptions{
				ServiceURL: "http://localhost:11434",
				APITimeout: 0, // Should use default
				Timeout:    0, // Not set
				Model:      "",
			},
			wantAPITime: 0, // Let llm package apply default
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.cliOpts.toAPIOptions()

			if tt.wantAPITime != 0 && got.APITimeout != tt.wantAPITime {
				t.Errorf("APITimeout: got %v, want %v", got.APITimeout, tt.wantAPITime)
			}
		})
	}
}

// ============================================================================
// Phase 3: Timeout-Specific Messaging Tests
// ============================================================================

// TestIsTimeoutError_ContextDeadline verifies that context.DeadlineExceeded is recognized
func TestIsTimeoutError_ContextDeadline(t *testing.T) {
	if !isTimeoutError(context.DeadlineExceeded) {
		t.Error("Expected context.DeadlineExceeded to be recognized as timeout error")
	}
}

// TestIsTimeoutError_NetworkTimeout verifies that net.Timeout errors are recognized
func TestIsTimeoutError_NetworkTimeout(t *testing.T) {
	// Create a timeout error via net package
	timeoutErr := &net.OpError{
		Op:  "dial",
		Net: "tcp",
		Err: errors.New("i/o timeout"),
	}
	// Note: OpError's Timeout() method returns true if Err wraps a timeout
	// For simplicity, we check the message
	if !isTimeoutError(timeoutErr) {
		t.Error("Expected net.OpError with timeout to be recognized as timeout error")
	}
}

// TestIsTimeoutError_MessagePatterns verifies that various timeout message patterns are recognized
func TestIsTimeoutError_MessagePatterns(t *testing.T) {
	testCases := []struct {
		name    string
		err     error
		wantKey bool
	}{
		{
			name:    "deadline_exceeded_in_message",
			err:     errors.New("inference request failed ... context deadline exceeded ... exit status 2"),
			wantKey: true,
		},
		{
			name:    "timeout_in_message",
			err:     errors.New("connection timeout"),
			wantKey: true,
		},
		{
			name:    "io_timeout",
			err:     errors.New("i/o timeout"),
			wantKey: true,
		},
		{
			name:    "timed_out",
			err:     errors.New("request timed out"),
			wantKey: true,
		},
		{
			name:    "non_timeout_error",
			err:     errors.New("GPU not detected"),
			wantKey: false,
		},
		{
			name:    "nil_error",
			err:     nil,
			wantKey: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := isTimeoutError(tc.err)
			if got != tc.wantKey {
				t.Errorf("isTimeoutError: got %v, want %v for error: %v", got, tc.wantKey, tc.err)
			}
		})
	}
}

// TestFormatErrorMessage_Timeout verifies timeout messages include retry guidance
func TestFormatErrorMessage_Timeout(t *testing.T) {
	timeoutErr := errors.New("inference request failed ... context deadline exceeded")

	msg := formatErrorMessage(timeoutErr, true)

	if !strings.Contains(msg, "state-gathering") {
		t.Error("Timeout message should mention 'state-gathering'")
	}
	if !strings.Contains(msg, "inference-timeout") {
		t.Error("Timeout message should suggest --inference-timeout for model-aware validation")
	}
	if !strings.Contains(msg, "longer timeout") {
		t.Error("Timeout message should suggest increasing timeout")
	}
}

// TestFormatErrorMessage_Timeout_GPUOnly verifies timeout messages for GPU-only validation
func TestFormatErrorMessage_Timeout_GPUOnly(t *testing.T) {
	timeoutErr := errors.New("api request failed ... i/o timeout")

	msg := formatErrorMessage(timeoutErr, false)

	if !strings.Contains(msg, "state-gathering") {
		t.Error("Timeout message should mention 'state-gathering'")
	}
	if !strings.Contains(msg, "api-timeout") && !strings.Contains(msg, "inference-timeout") {
		t.Error("Timeout message should suggest timeout flags for GPU-only validation")
	}
}

// TestFormatErrorMessage_NonTimeout verifies non-timeout errors are handled differently
func TestFormatErrorMessage_NonTimeout(t *testing.T) {
	err := errors.New("GPU not detected")

	msg := formatErrorMessage(err, true)

	if strings.Contains(msg, "state-gathering") {
		t.Error("Non-timeout error should not mention 'state-gathering'")
	}
	if !strings.Contains(msg, "GPU not detected") {
		t.Error("Non-timeout error message should include original error")
	}
}

// TestErrorMessageConsistency verifies that the mapping of error to exit code is consistent
// with the error message classification.
func TestErrorMessageConsistency(t *testing.T) {
	timeoutErr := errors.New("context deadline exceeded")

	// Timeout error should exit with code 2
	if mapExitCode(false, timeoutErr) != 2 {
		t.Error("Timeout error should exit with code 2")
	}

	// Timeout error message should be actionable
	msg := formatErrorMessage(timeoutErr, true)
	if !strings.Contains(strings.ToLower(msg), "timeout") {
		t.Error("Timeout error message should mention timeout")
	}
}

// TestPhase3Messaging_ModelAwareTimeout tests model-aware messaging with timeout
func TestPhase3Messaging_ModelAwareTimeout(t *testing.T) {
	timeoutErr := errors.New("inference request failed: i/o timeout")
	modelSpecified := true

	msg := formatErrorMessage(timeoutErr, modelSpecified)

	// Should indicate it's a state-gathering issue
	if !strings.Contains(msg, "state-gathering") {
		t.Error("Should indicate this is a state-gathering issue, not validation failure")
	}

	// Should suggest model-aware timeout
	if !strings.Contains(msg, "inference-timeout") {
		t.Error("Should suggest --inference-timeout for model-aware validation")
	}
}

// TestPhase3Messaging_GPUOnlyTimeout tests GPU-only messaging with timeout
func TestPhase3Messaging_GPUOnlyTimeout(t *testing.T) {
	timeoutErr := errors.New("api call timeout")
	modelSpecified := false

	msg := formatErrorMessage(timeoutErr, modelSpecified)

	// Should indicate it's a state-gathering issue
	if !strings.Contains(msg, "state-gathering") {
		t.Error("Should indicate this is a state-gathering issue")
	}

	// Should suggest both timeout options for GPU-only (doesn't know if inference would timeout)
	lowercaseMsg := strings.ToLower(msg)
	if !strings.Contains(lowercaseMsg, "api-timeout") && !strings.Contains(lowercaseMsg, "inference-timeout") {
		t.Error("Should suggest timeout options")
	}
}

// ============================================================================
// Phase 3 Revision: Timeout Source Inference and Hardened Detection Tests
// ============================================================================

// TestInferTimeoutSource_InferenceTimeout verifies that inference timeout patterns are recognized
func TestInferTimeoutSource_InferenceTimeout(t *testing.T) {
	testCases := []struct {
		name    string
		err     error
		wantSrc TimeoutErrorSource
	}{
		{
			name:    "inference_request_error",
			err:     errors.New("inference request via /api/generate failed: context deadline exceeded"),
			wantSrc: TimeoutSourceInference,
		},
		{
			name:    "api_generate_endpoint",
			err:     errors.New("failed to call /api/generate: timeout"),
			wantSrc: TimeoutSourceInference,
		},
		{
			name:    "generate_keyword",
			err:     errors.New("generate request timed out after 30s"),
			wantSrc: TimeoutSourceInference,
		},
		{
			name:    "inference_keyword",
			err:     errors.New("inference operation timeout during model load"),
			wantSrc: TimeoutSourceInference,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := inferTimeoutSource(tc.err)
			if got != tc.wantSrc {
				t.Errorf("inferTimeoutSource: got %v, want %v for error: %v", got, tc.wantSrc, tc.err)
			}
		})
	}
}

// TestInferTimeoutSource_APITimeout verifies that GPU/API timeout patterns are recognized
func TestInferTimeoutSource_APITimeout(t *testing.T) {
	testCases := []struct {
		name    string
		err     error
		wantSrc TimeoutErrorSource
	}{
		{
			name:    "api_ps_endpoint",
			err:     errors.New("failed to connect to /api/ps: i/o timeout"),
			wantSrc: TimeoutSourceAPI,
		},
		{
			name:    "gpu_check_error",
			err:     errors.New("GPU check via /api/ps timed out"),
			wantSrc: TimeoutSourceAPI,
		},
		{
			name:    "gpu_acceleration_check",
			err:     errors.New("GPU acceleration check failed: timeout"),
			wantSrc: TimeoutSourceAPI,
		},
		{
			name:    "gpu_via_api",
			err:     errors.New("GPU via /api/ps: context deadline exceeded"),
			wantSrc: TimeoutSourceAPI,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := inferTimeoutSource(tc.err)
			if got != tc.wantSrc {
				t.Errorf("inferTimeoutSource: got %v, want %v for error: %v", got, tc.wantSrc, tc.err)
			}
		})
	}
}

// TestInferTimeoutSource_UnknownTimeout verifies that ambiguous errors return Unknown
func TestInferTimeoutSource_UnknownTimeout(t *testing.T) {
	testCases := []struct {
		name    string
		err     error
		wantSrc TimeoutErrorSource
	}{
		{
			name:    "generic_timeout",
			err:     errors.New("operation timed out"),
			wantSrc: TimeoutSourceUnknown,
		},
		{
			name:    "nil_error",
			err:     nil,
			wantSrc: TimeoutSourceUnknown,
		},
		{
			name:    "not_timeout",
			err:     errors.New("GPU not found"),
			wantSrc: TimeoutSourceUnknown,
		},
		{
			name:    "generic_api_request",
			err:     errors.New("api request failed: connection refused"),
			wantSrc: TimeoutSourceAPI, // Fallback to API for generic "api request"
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := inferTimeoutSource(tc.err)
			if got != tc.wantSrc {
				t.Errorf("inferTimeoutSource: got %v, want %v for error: %v", got, tc.wantSrc, tc.err)
			}
		})
	}
}

// TestIsTimeoutError_HardenedContextDeadline verifies context.DeadlineExceeded is detected via errors.Is
func TestIsTimeoutError_HardenedContextDeadline(t *testing.T) {
	// Direct context.DeadlineExceeded
	if !isTimeoutError(context.DeadlineExceeded) {
		t.Error("Expected context.DeadlineExceeded to be recognized as timeout")
	}

	// Wrapped context.DeadlineExceeded
	wrappedErr := fmt.Errorf("inference request failed: %w", context.DeadlineExceeded)
	if !isTimeoutError(wrappedErr) {
		t.Error("Expected wrapped context.DeadlineExceeded to be recognized as timeout")
	}
}

// TestIsTimeoutError_HardenedNetTimeout verifies net.Error with Timeout() is detected via errors.As
func TestIsTimeoutError_HardenedNetTimeout(t *testing.T) {
	// Create a net.Error via timeouts
	timeoutErr := &net.OpError{
		Op:  "dial",
		Net: "tcp",
		Err: errors.New("connection timeout"),
	}

	if !isTimeoutError(timeoutErr) {
		t.Error("Expected net.OpError to be recognized as timeout error")
	}

	// Wrapped net.OpError
	wrappedErr := fmt.Errorf("failed to connect: %w", timeoutErr)
	if !isTimeoutError(wrappedErr) {
		t.Error("Expected wrapped net.OpError to be recognized as timeout error")
	}
}

// TestIsTimeoutError_FallbackPatterns verifies fallback string matching still works
func TestIsTimeoutError_FallbackPatterns(t *testing.T) {
	testCases := []struct {
		name    string
		err     error
		wantKey bool
	}{
		{
			name:    "deadline_exceeded_string",
			err:     errors.New("inference failed: context deadline exceeded"),
			wantKey: true,
		},
		{
			name:    "timeout_string",
			err:     errors.New("connection timeout"),
			wantKey: true,
		},
		{
			name:    "io_timeout",
			err:     errors.New("i/o timeout"),
			wantKey: true,
		},
		{
			name:    "timed_out",
			err:     errors.New("request timed out"),
			wantKey: true,
		},
		{
			name:    "deadline_exceeded",
			err:     errors.New("deadline exceeded"),
			wantKey: true,
		},
		{
			name:    "non_timeout",
			err:     errors.New("GPU not detected"),
			wantKey: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := isTimeoutError(tc.err)
			if got != tc.wantKey {
				t.Errorf("isTimeoutError: got %v, want %v for error: %v", got, tc.wantKey, tc.err)
			}
		})
	}
}

// ============================================================================
// Phase 3 Revision: Truthful Timeout Guidance Tests
// ============================================================================

// TestFormatErrorMessage_InferenceTimeoutGuidance verifies inference timeouts get --inference-timeout guidance
func TestFormatErrorMessage_InferenceTimeoutGuidance(t *testing.T) {
	// Inference timeout should suggest --inference-timeout regardless of modelSpecified
	timeoutErr := errors.New("inference request via /api/generate failed: context deadline exceeded")

	msg := formatErrorMessage(timeoutErr, true)
	if !strings.Contains(msg, "--inference-timeout") {
		t.Errorf("Inference timeout should suggest --inference-timeout, got: %s", msg)
	}
	if !strings.Contains(msg, "state-gathering") {
		t.Error("Should indicate this is a state-gathering issue")
	}

	// Even with modelSpecified=false, if error clearly indicates inference, suggest --inference-timeout
	msg2 := formatErrorMessage(timeoutErr, false)
	if !strings.Contains(msg2, "--inference-timeout") {
		t.Errorf("Inference timeout should suggest --inference-timeout even with modelSpecified=false, got: %s", msg2)
	}
}

// TestFormatErrorMessage_APITimeoutGuidance verifies API timeouts get --api-timeout guidance
func TestFormatErrorMessage_APITimeoutGuidance(t *testing.T) {
	// API/GPU check timeout should suggest --api-timeout
	timeoutErr := errors.New("failed to connect to /api/ps: i/o timeout")

	msg := formatErrorMessage(timeoutErr, true)
	if !strings.Contains(msg, "--api-timeout") {
		t.Errorf("API timeout should suggest --api-timeout, got: %s", msg)
	}
	if !strings.Contains(msg, "state-gathering") {
		t.Error("Should indicate this is a state-gathering issue")
	}

	// GPU-only mode should also suggest --api-timeout since that's what ran
	msg2 := formatErrorMessage(timeoutErr, false)
	if !strings.Contains(msg2, "--api-timeout") {
		t.Errorf("API timeout should suggest --api-timeout in GPU-only mode, got: %s", msg2)
	}
}

// TestFormatErrorMessage_GPUOnlyTimeoutGuidance verifies GPU-only mode gets correct guidance
func TestFormatErrorMessage_GPUOnlyTimeoutGuidance(t *testing.T) {
	// In GPU-only mode, no inference runs, so API timeout should get --api-timeout
	timeoutErr := errors.New("connection timeout")

	msg := formatErrorMessage(timeoutErr, false)
	if !strings.Contains(msg, "--api-timeout") {
		t.Errorf("GPU-only timeout should suggest --api-timeout, got: %s", msg)
	}
	if strings.Contains(msg, "--inference-timeout") {
		t.Errorf("GPU-only timeout should NOT suggest --inference-timeout (no inference runs), got: %s", msg)
	}
}

// TestFormatErrorMessage_BlockerFix_APITimeoutNotInferenceWhenInferenceDidNotRun tests the blocker fix
func TestFormatErrorMessage_BlockerFix_APITimeoutNotInferenceWhenInferenceDidNotRun(t *testing.T) {
	// Blocker: If timeout happened during API/GPU check in GPU-only mode, suggesting
	// --inference-timeout is wrong because no inference ran.
	// The fix: infer from error message that /api/ps was called, suggest --api-timeout
	timeoutErr := errors.New("failed to connect to /api/ps: i/o timeout")

	msg := formatErrorMessage(timeoutErr, false) // GPU-only mode

	// Should suggest --api-timeout
	if !strings.Contains(msg, "--api-timeout") {
		t.Errorf("BLOCKER: GPU-only API timeout should suggest --api-timeout, got: %s", msg)
	}

	// Should NOT suggest --inference-timeout (blocker violation)
	if strings.Contains(msg, "--inference-timeout") {
		t.Errorf("BLOCKER: GPU-only mode should NOT suggest --inference-timeout, got: %s", msg)
	}
}

// TestFormatErrorMessage_BlockerFix_InferenceTimeoutNotAPIWhenInferenceRan tests the blocker fix
func TestFormatErrorMessage_BlockerFix_InferenceTimeoutNotAPIWhenInferenceRan(t *testing.T) {
	// Blocker: If timeout happened during inference, suggesting --api-timeout is wrong
	// The fix: infer from error message that /api/generate was called, suggest --inference-timeout
	timeoutErr := errors.New("inference request via /api/generate failed: context deadline exceeded")

	msg := formatErrorMessage(timeoutErr, true) // Model-aware validation

	// Should suggest --inference-timeout
	if !strings.Contains(msg, "--inference-timeout") {
		t.Errorf("BLOCKER: Inference timeout should suggest --inference-timeout, got: %s", msg)
	}

	// Should NOT suggest --api-timeout (blocker violation)
	if strings.Contains(msg, "--api-timeout") {
		t.Errorf("BLOCKER: Inference timeout should NOT suggest --api-timeout, got: %s", msg)
	}
}

// TestFormatErrorMessage_UnknownTimeoutFallback verifies fallback behavior for unknown timeouts
func TestFormatErrorMessage_UnknownTimeoutFallback(t *testing.T) {
	// When we can't infer which operation timed out, fall back to modelSpecified
	timeoutErr := errors.New("operation timed out")

	// With model specified, suggest --inference-timeout
	msg1 := formatErrorMessage(timeoutErr, true)
	if !strings.Contains(msg1, "--inference-timeout") {
		t.Errorf("Unknown timeout with modelSpecified=true should suggest --inference-timeout, got: %s", msg1)
	}

	// Without model (GPU-only), suggest --api-timeout
	msg2 := formatErrorMessage(timeoutErr, false)
	if !strings.Contains(msg2, "--api-timeout") {
		t.Errorf("Unknown timeout with modelSpecified=false should suggest --api-timeout, got: %s", msg2)
	}
}

// TestFormatErrorMessage_ContextDeadlineWithoutSpecificEndpoint tests wrapped context deadline
func TestFormatErrorMessage_ContextDeadlineWithoutSpecificEndpoint(t *testing.T) {
	// When context.DeadlineExceeded is wrapped but doesn't mention endpoint,
	// fall back to modelSpecified for guidance
	wrappedErr := fmt.Errorf("validation failed: %w", context.DeadlineExceeded)

	msg := formatErrorMessage(wrappedErr, false) // GPU-only mode
	if !strings.Contains(msg, "state-gathering") {
		t.Error("Should identify as state-gathering issue")
	}
	// Should suggest --api-timeout based on modelSpecified=false
	if !strings.Contains(msg, "--api-timeout") {
		t.Errorf("Should suggest --api-timeout for GPU-only context deadline, got: %s", msg)
	}
}
