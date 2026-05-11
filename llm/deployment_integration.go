package llm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// DeploymentValidationOptions controls how deployment validation is performed
type DeploymentValidationOptions struct {
	// JournalLines: number of recent lines to fetch from systemd journal
	JournalLines int

	// ServiceName: name of the systemd service to validate
	ServiceName string

	// Timeout: maximum time to wait for systemctl operations
	Timeout time.Duration
}

// DefaultDeploymentValidationOptions returns reasonable defaults
func DefaultDeploymentValidationOptions() DeploymentValidationOptions {
	return DeploymentValidationOptions{
		JournalLines: 100,
		ServiceName:  "ollama",
		Timeout:      5 * time.Second,
	}
}

// GatherDeploymentState gathers the current deployment state from the system
// by checking service status and recent logs. This is used by deployment verification
// scripts to validate that a deployment is actually working with GPU acceleration.
//
// Returns an error if systemctl/journalctl are not available or cannot be queried.
// The returned DeploymentState can then be passed to ValidateDeploymentState to
// determine if deployment was successful.
func GatherDeploymentState(opts DeploymentValidationOptions) (DeploymentState, error) {
	state := DeploymentState{}

	// Check if service is active
	cmd := exec.Command("systemctl", "is-active", "--quiet", opts.ServiceName)
	err := cmd.Run()
	state.ServiceActive = err == nil

	// Get recent logs from systemd journal
	// Use journalctl to get the last N lines from the service
	journalCmd := exec.Command(
		"journalctl",
		"-u", opts.ServiceName,
		"-n", fmt.Sprintf("%d", opts.JournalLines),
		"--no-pager",
	)

	journalOutput, err := journalCmd.Output()
	if err != nil {
		// If journalctl fails, use empty logs (deployment still might be valid if response exists)
		state.LastLogLines = ""
	} else {
		state.LastLogLines = string(journalOutput)
	}

	// LastResponseContent is not populated here - it should be set by the caller
	// based on actual model inference results
	state.LastResponseContent = ""

	return state, nil
}

// ValidateDeploymentOutput validates deployment by checking service state and logs.
// This is intended to be called from deployment scripts to verify GPU acceleration
// worked as expected.
//
// Returns a structured validation result that describes what passed/failed.
func ValidateDeploymentOutput(opts DeploymentValidationOptions) ValidationResult {
	result := ValidationResult{
		Timestamp:   time.Now(),
		ServiceName: opts.ServiceName,
	}

	// Gather state
	state, err := GatherDeploymentState(opts)
	if err != nil {
		result.Error = fmt.Sprintf("Failed to gather deployment state: %v", err)
		return result
	}

	// Store the state we gathered
	result.GatheredState = state

	// Check service active
	if !state.ServiceActive {
		result.ServiceActive = false
		result.ServiceActiveReason = "Service is not currently active (check 'systemctl status ollama')"
		result.Valid = false
		return result
	}
	result.ServiceActive = true
	result.ServiceActiveReason = "Service is active"

	// Check GPU offload
	gpuValid := ValidateGPUOffloadSuccess(state.LastLogLines)
	result.GPUOffloadValid = gpuValid

	if !gpuValid {
		result.GPUOffloadReason = "Logs show CPU-only fallback or no GPU layer allocation found"
	} else {
		result.GPUOffloadReason = "Logs show successful GPU offload"
	}

	// Check response
	responseValid := ValidateResponsePresence(state.LastResponseContent)
	result.ResponseValid = responseValid

	if !responseValid {
		if state.LastResponseContent == "" {
			result.ResponseReason = "No response content available"
		} else {
			result.ResponseReason = "Response contains only whitespace"
		}
	} else {
		result.ResponseReason = fmt.Sprintf("Response valid (%d bytes of content)", len(strings.TrimSpace(state.LastResponseContent)))
	}

	// Final validation
	result.Valid = ValidateDeploymentState(state)

	return result
}

// ValidationResult represents the result of a deployment validation
type ValidationResult struct {
	// Timestamp when validation was performed
	Timestamp time.Time

	// ServiceName that was validated
	ServiceName string

	// GatheredState is the deployment state that was collected and validated
	GatheredState DeploymentState

	// Valid is the final validation result
	Valid bool

	// ServiceActive indicates if the service is currently running
	ServiceActive       bool
	ServiceActiveReason string

	// GPUOffloadValid indicates if logs show GPU acceleration
	GPUOffloadValid  bool
	GPUOffloadReason string

	// ResponseValid indicates if the last response has content
	ResponseValid  bool
	ResponseReason string

	// Error if validation failed to gather state
	Error string
}

// Summary returns a human-readable validation summary
func (r ValidationResult) Summary() string {
	var lines []string

	if r.Error != "" {
		lines = append(lines, fmt.Sprintf("ERROR: %s", r.Error))
		return strings.Join(lines, "\n")
	}

	lines = append(lines, "Deployment Validation Results")
	lines = append(lines, "==============================")
	lines = append(lines, "")
	lines = append(lines, fmt.Sprintf("Service: %s", r.ServiceName))
	lines = append(lines, fmt.Sprintf("Timestamp: %s", r.Timestamp.Format(time.RFC3339)))
	lines = append(lines, "")

	lines = append(lines, "Checks:")
	lines = append(lines, fmt.Sprintf("  Service Active: %v - %s", r.ServiceActive, r.ServiceActiveReason))
	lines = append(lines, fmt.Sprintf("  GPU Offload: %v - %s", r.GPUOffloadValid, r.GPUOffloadReason))
	lines = append(lines, fmt.Sprintf("  Response Valid: %v - %s", r.ResponseValid, r.ResponseReason))
	lines = append(lines, "")

	if r.Valid {
		lines = append(lines, "RESULT: ✓ DEPLOYMENT VALID")
		lines = append(lines, "The service is running GPU-accelerated with valid output")
	} else {
		lines = append(lines, "RESULT: ✗ DEPLOYMENT INVALID")
		lines = append(lines, "The deployment does not meet success criteria")
	}

	return strings.Join(lines, "\n")
}

// ExitCode returns appropriate process exit code for a validation result
// 0 = success, 1 = validation failed, 2 = error gathering state
func (r ValidationResult) ExitCode() int {
	if r.Error != "" {
		return 2 // Error state
	}
	if r.Valid {
		return 0 // Success
	}
	return 1 // Validation failed
}

// CheckDeploymentReadiness is a convenience function for deployment scripts.
// It gathers deployment state, validates it, and writes a summary to stderr.
// Returns exit code suitable for shell scripts (0 = success, 1 = failed validation, 2 = error).
//
// Usage in shell scripts:
//
//	go run ./cmd/deploy-check/main.go
//	if [ $? -ne 0 ]; then echo "Deployment validation failed"; exit 1; fi
func CheckDeploymentReadiness(opts DeploymentValidationOptions) int {
	result := ValidateDeploymentOutput(opts)

	// Print summary to stderr
	fmt.Fprintf(os.Stderr, "%s\n", result.Summary())

	return result.ExitCode()
}

// ============================================================================
// API-Based Deployment Validation (Phase 3 Revised)
// ============================================================================
//
// This section provides the UNIFIED deployment validation implementation
// used by all deployment paths:
// - cmd/deploy-check/main.go (CLI wrapper)
// - scripts/verify-gpu-deployment.sh (shell script wrapper)
// - deployment integration tests
//
// Architecture: A single ValidateDeploymentViaAPI function is the entry point.
// All validation logic is centralized here to avoid duplication and ensure
// consistent behavior across all deployment paths.

// APIDeploymentValidationOptions controls API-based validation behavior
type APIDeploymentValidationOptions struct {
	// ServiceURL is the base URL of the Ollama service (e.g., http://localhost:11434)
	ServiceURL string

	// Timeout is the maximum time to wait for API calls (DEPRECATED: use APITimeout or InferenceTimeout)
	// Kept for backward compatibility.
	Timeout time.Duration

	// APITimeout is the maximum time to wait for fast API calls like /api/ps
	// This is used for quick health checks and should be short (5-10 seconds)
	APITimeout time.Duration

	// InferenceTimeout is the maximum time to wait for inference requests via /api/generate
	// This is longer to tolerate first-load latency from GPU kernel compilation.
	// A conservative default is important here.
	InferenceTimeout time.Duration

	// Model is optional - if provided, tests inference response; otherwise just checks /api/ps
	Model string
}

// DefaultAPIDeploymentValidationOptions returns reasonable defaults
func DefaultAPIDeploymentValidationOptions() APIDeploymentValidationOptions {
	return APIDeploymentValidationOptions{
		ServiceURL:       "http://localhost:11434",
		Timeout:          10 * time.Second, // Legacy default
		APITimeout:       10 * time.Second, // Fast API timeout for /api/ps
		InferenceTimeout: 2 * time.Minute,  // Conservative for first-load latency
		Model:            "",
	}
}

// CheckGPUViaAPI validates GPU acceleration using the /api/ps endpoint.
// This is called by ValidateDeploymentViaAPI as part of unified validation.
//
// IMPORTANT MODEL-AWARE BEHAVIOR:
// When opts.Model is set (model-aware validation):
//   - The specified model MUST be present in /api/ps AND have SizeVRAM > 0
//   - If the model is absent, returns specific "model not found" error
//   - If the model is present but SizeVRAM == 0, returns specific "CPU-only" error
//   - This prevents false positives where unrelated models are on GPU
//
// When opts.Model is empty (generic GPU-only validation):
//   - Any model with SizeVRAM > 0 passes (backward compatible)
//
// TIMEOUT BEHAVIOR:
// Uses opts.APITimeout for fast /api/ps health checks.
// Falls back to opts.Timeout for backward compatibility.
//
// Returns (hasGPU, description, error)
func CheckGPUViaAPI(opts APIDeploymentValidationOptions) (bool, string, error) {
	// Determine which timeout to use
	timeout := opts.APITimeout
	if timeout == 0 && opts.Timeout != 0 {
		// Fallback to legacy Timeout for backward compatibility
		timeout = opts.Timeout
	}
	if timeout == 0 {
		// Final fallback if both are zero
		timeout = 10 * time.Second
	}

	client := &http.Client{Timeout: timeout}

	resp, err := client.Get(fmt.Sprintf("%s/api/ps", opts.ServiceURL))
	if err != nil {
		return false, "", fmt.Errorf("failed to connect to /api/ps: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, "", fmt.Errorf("/api/ps returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, "", fmt.Errorf("failed to read /api/ps response: %w", err)
	}

	var psResp api.ProcessResponse
	if err := json.Unmarshal(body, &psResp); err != nil {
		return false, "", fmt.Errorf("failed to parse /api/ps response: %w", err)
	}

	// MODEL-AWARE VALIDATION: Check for specific requested model
	if opts.Model != "" {
		// Find the specific requested model in the loaded models
		var foundModel *api.ProcessModelResponse
		for i := range psResp.Models {
			if psResp.Models[i].Model == opts.Model || psResp.Models[i].Name == opts.Model {
				foundModel = &psResp.Models[i]
				break
			}
		}

		if foundModel == nil {
			// Requested model not found in /api/ps
			return false, fmt.Sprintf("Requested model %q not found in loaded models (GPU check requires model to be loaded)", opts.Model), nil
		}

		if foundModel.SizeVRAM == 0 {
			// Model is loaded but CPU-only
			return false, fmt.Sprintf("Model %q is loaded but running CPU-only (SizeVRAM = 0, no GPU acceleration)", foundModel.Model), nil
		}

		// Model is loaded and on GPU
		return true, fmt.Sprintf("GPU acceleration active: %s (VRAM: %d bytes)", foundModel.Model, foundModel.SizeVRAM), nil
	}

	// GENERIC GPU VALIDATION: No specific model requested, check for any GPU acceleration
	// Check if any loaded model has GPU VRAM allocated
	for _, model := range psResp.Models {
		if model.SizeVRAM > 0 {
			return true, fmt.Sprintf("GPU acceleration active: %s (VRAM: %d bytes)", model.Name, model.SizeVRAM), nil
		}
	}

	// If models are loaded but none have GPU VRAM, they're CPU-only
	if len(psResp.Models) > 0 {
		return false, fmt.Sprintf("Service running CPU-only (%d models loaded)", len(psResp.Models)), nil
	}

	return false, "No models currently loaded", nil
}

// CheckInferenceViaAPI verifies that inference requests work and complete successfully.
// It makes a simple API request to the service and checks if the model produces output.
//
// Returns:
//   - valid (bool): true if model produced non-empty output
//   - message (string): descriptive message about the result
//   - err (error): non-nil if the request failed (timeout, connection error, etc.)
//
// NOTE: Empty responses return (false, "Model returned empty response", nil).
// This is a validation failure but NOT an error - it indicates the request completed
// but the model produced no content. Callers must check both valid and err.
//
// TIMEOUT BEHAVIOR:
// Uses opts.InferenceTimeout for inference requests to tolerate first-load latency.
// Falls back to opts.Timeout for backward compatibility.
// Inference can take much longer than fast API health checks (e.g., GPU kernel compilation).
func CheckInferenceViaAPI(opts APIDeploymentValidationOptions) (bool, string, error) {
	if opts.Model == "" {
		return true, "Inference check skipped (no model specified)", nil
	}

	// Determine which timeout to use
	timeout := opts.InferenceTimeout
	if timeout == 0 && opts.Timeout != 0 {
		// Fallback to legacy Timeout for backward compatibility
		timeout = opts.Timeout
	}
	if timeout == 0 {
		// Final fallback if both are zero (use conservative default)
		timeout = 2 * time.Minute
	}

	client := &http.Client{Timeout: timeout}

	// Create a simple inference request
	stream := false
	req := api.GenerateRequest{
		Model:  opts.Model,
		Prompt: "What is the capital of France?",
		Stream: &stream,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return false, "", fmt.Errorf("failed to marshal inference request: %w", err)
	}

	httpReq, err := http.NewRequest(
		"POST",
		fmt.Sprintf("%s/api/generate", opts.ServiceURL),
		strings.NewReader(string(reqBody)),
	)
	if err != nil {
		return false, "", fmt.Errorf("failed to create inference request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		return false, "", fmt.Errorf("inference request via /api/generate failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, "", fmt.Errorf("inference returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, "", fmt.Errorf("failed to read inference response: %w", err)
	}

	var genResp api.GenerateResponse
	if err := json.Unmarshal(body, &genResp); err != nil {
		return false, "", fmt.Errorf("failed to parse inference response: %w", err)
	}

	// Check if response has content
	if ValidateResponsePresence(genResp.Response) {
		return true, fmt.Sprintf("Model produced output (%d bytes)", len(strings.TrimSpace(genResp.Response))), nil
	}

	return false, "Model returned empty response", nil
}

// ValidateDeploymentViaAPI performs comprehensive API-based deployment validation.
//
// This is the UNIFIED entry point for all deployment validation paths:
// - cmd/deploy-check/main.go (CLI) calls this via ValidateDeploymentViaAPI
// - scripts/verify-gpu-deployment.sh (shell) calls the compiled CLI, which calls this
// - Integration tests call this or its helper functions
//
// MODEL-AWARE VALIDATION ORDER (truthful deployment validation):
// When a model is specified (opts.Model != ""):
//  1. Check inference response FIRST (loads model if needed)
//  2. Check GPU via /api/ps SECOND (now model is loaded, GPU residency is meaningful)
//
// When no model is specified (opts.Model == ""):
//  1. Check GPU via /api/ps ONLY (lightweight check, no inference required)
//
// This ordering ensures:
// - Fresh service with no model loaded can still be validated (inference loads the model)
// - GPU residency check is meaningful (model is loaded before checking)
// - Inference errors are not hidden by GPU failures
//
// Importantly: Inference errors are NOT silently coerced to success.
// If inference fails with an error AND a model was specified, that's a failure.
//
// IMPORTANT: This function is side-effect free - no direct writes to stderr/stdout.
// The returned message and error should be handled by the caller.
//
// Returns (valid, description, error):
//   - valid=true, err=nil: Deployment is valid
//   - valid=false, err=nil: Deployment is invalid (GPU missing or inference failed)
//   - valid=false, err!=nil: Error gathering state (connection error, timeout, etc.)
func ValidateDeploymentViaAPI(opts APIDeploymentValidationOptions) (bool, string, error) {
	// MODEL-AWARE VALIDATION ORDER
	if opts.Model != "" {
		// When model is specified: infer first (to load model), then check GPU

		// Step 1: Check inference response (loads model if needed)
		inferenceOK, inferenceMsg, err := CheckInferenceViaAPI(opts)
		if err != nil {
			// Inference errors are NOT silently coerced to success
			// If a model was specified and inference failed, return the error
			return false, "", err
		}

		if !inferenceOK {
			// Inference returned empty or failed - report this
			return false, inferenceMsg, nil
		}

		// Step 2: Check GPU via /api/ps (now model is loaded)
		gpuOK, gpuMsg, err := CheckGPUViaAPI(opts)
		if err != nil {
			return false, "", err
		}

		if !gpuOK {
			// GPU not active even after inference
			return false, gpuMsg, nil
		}

		// Both inference and GPU checks passed
		return true, "Model inference successful and GPU acceleration confirmed", nil
	} else {
		// When no model is specified: just check GPU (lightweight check)

		// Check GPU via /api/ps (real-time API state)
		gpuOK, gpuMsg, err := CheckGPUViaAPI(opts)
		if err != nil {
			return false, "", err
		}

		if !gpuOK {
			// Return the specific GPU failure reason
			return false, gpuMsg, nil
		}

		return true, "GPU acceleration confirmed (no model inference performed)", nil
	}
}
