package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
)

// DeploymentCheckOptions controls deployment validation behavior.
//
// This is a thin CLI wrapper struct that maps command-line flags to the unified
// llm.APIDeploymentValidationOptions for API-based deployment validation.
//
// The validation strategy:
// 1. Check GPU acceleration is active via /api/ps (real-time service state)
// 2. Check inference response if model specified
// 3. Exit with code 0 (success), 1 (validation failed), or 2 (error)
//
// Architecture: This CLI is intentionally thin - all validation logic is in the
// llm package. The CLI only handles command-line parsing and maps to the unified
// ValidateDeploymentViaAPI function in the llm package.
type DeploymentCheckOptions struct {
	ServiceURL       string        // URL of Ollama service
	Timeout          time.Duration // Legacy timeout field (DEPRECATED: use APITimeout or InferenceTimeout)
	APITimeout       time.Duration // Fast API timeout for /api/ps (overrides Timeout if set)
	InferenceTimeout time.Duration // Slow inference timeout for /api/generate (overrides Timeout if set)
	Model            string        // Model to test with for inference validation
}

// toAPIOptions converts CLI options to llm API options for the unified validator.
// This is the bridge between CLI flags and the unified llm package validation.
//
// Timeout handling (CRITICAL FIX for Phase 2):
// - The CLI should NOT override llm package's conservative defaults.
// - Only when the user explicitly sets a timeout should we apply it.
// - If specific timeouts (-api-timeout, -inference-timeout) are set, use them (they take precedence).
// - Otherwise, if legacy -timeout is set, use it for both (backward compatibility).
// - If nothing is set (all zeros), pass zeros to llm package so it applies its own defaults:
//   - API calls default to 10s (CheckGPUViaAPI)
//   - Inference calls default to 2m (CheckInferenceViaAPI)
func (opts DeploymentCheckOptions) toAPIOptions() llm.APIDeploymentValidationOptions {
	apiOpts := llm.APIDeploymentValidationOptions{
		ServiceURL: opts.ServiceURL,
		Model:      opts.Model,
	}

	// Determine InferenceTimeout: specific > legacy > 0 (let llm package apply default)
	if opts.InferenceTimeout > 0 {
		// User explicitly set -inference-timeout, use it
		apiOpts.InferenceTimeout = opts.InferenceTimeout
	} else if opts.Timeout > 0 {
		// User explicitly set legacy -timeout and didn't set -inference-timeout, use legacy
		apiOpts.InferenceTimeout = opts.Timeout
	}
	// Otherwise: leave as 0, let llm package apply its conservative default (2m)

	// Determine APITimeout: specific > legacy > 0 (let llm package apply default)
	if opts.APITimeout > 0 {
		// User explicitly set -api-timeout, use it
		apiOpts.APITimeout = opts.APITimeout
	} else if opts.Timeout > 0 {
		// User explicitly set legacy -timeout and didn't set -api-timeout, use legacy
		apiOpts.APITimeout = opts.Timeout
	}
	// Otherwise: leave as 0, let llm package apply its conservative default (10s)

	// Keep Timeout for backward compatibility
	if opts.Timeout > 0 {
		apiOpts.Timeout = opts.Timeout
	}

	return apiOpts
}

// mapExitCode determines the exit code based on validation result and error state.
// This is the logic that determines how to convert (valid, err) to an exit code.
// Returns: 0 (success), 1 (validation failed), 2 (error gathering state)
func mapExitCode(valid bool, err error) int {
	if err != nil {
		// Error gathering state (connection failed, timeout, parsing error, etc.)
		// Exit with code 2 to distinguish from validation failure
		return 2
	}
	if valid {
		// Validation succeeded (GPU detected and inference passed if specified)
		return 0
	}
	// Validation failed (GPU not detected or inference failed)
	return 1
}

// ============================================================================
// Phase 3: Timeout-Specific Messaging (REVISED)
// ============================================================================
//
// BLOCKER FIX:
// The original code chose timeout guidance from `modelSpecified`, not from
// the operation that actually failed. This led to incorrect suggestions:
// - API timeout -> suggests --inference-timeout (wrong, inference didn't run)
// - GPU-only timeout -> suggests --inference-timeout (wrong, no inference)
//
// REVISED APPROACH:
// 1. Try to infer which operation failed from the error wrapping/chaining
// 2. Use typed error checks: context.DeadlineExceeded, net.Error with Timeout()
// 3. Fall back to string matching only if typed checks don't work
// 4. Provide guidance only for flags that could actually help

// TimeoutErrorSource represents which operation timed out, inferred from error.
type TimeoutErrorSource int

const (
	// TimeoutSourceUnknown: Cannot determine which operation timed out
	TimeoutSourceUnknown TimeoutErrorSource = iota
	// TimeoutSourceInference: /api/generate timed out (model inference)
	TimeoutSourceInference
	// TimeoutSourceAPI: /api/ps timed out (GPU check)
	TimeoutSourceAPI
)

// inferTimeoutSource attempts to determine which operation timed out by examining
// the error chain and message. Returns the most specific source, or Unknown if
// we can't infer which operation failed.
//
// Inference heuristics:
// - If error message contains "generate" or "inference", it's inference timeout
// - If error message contains "/api/ps" or "gpu check", it's API timeout
// - If error message contains "api request" without specific endpoint, it's likely API
// - Otherwise, return Unknown (be conservative)
func inferTimeoutSource(err error) TimeoutErrorSource {
	if err == nil {
		return TimeoutSourceUnknown
	}

	errMsg := strings.ToLower(err.Error())

	// Check for inference-specific patterns
	inferencePatterns := []string{
		"generate",
		"inference",
		"/api/generate",
		"api/generate",
	}
	for _, pattern := range inferencePatterns {
		if strings.Contains(errMsg, pattern) {
			return TimeoutSourceInference
		}
	}

	// Check for API/GPU check patterns
	apiPatterns := []string{
		"/api/ps",
		"api/ps",
		"gpu check",
		"gpu acceleration",
		"gpu via",
	}
	for _, pattern := range apiPatterns {
		if strings.Contains(errMsg, pattern) {
			return TimeoutSourceAPI
		}
	}

	// Generic patterns
	if strings.Contains(errMsg, "api request") || strings.Contains(errMsg, "failed to connect") {
		// Could be either, but lean toward API since that's the initial check
		return TimeoutSourceAPI
	}

	return TimeoutSourceUnknown
}

// isTimeoutError checks if an error represents a timeout or deadline-exceeded condition.
// Uses hardened type checking with typed error comparisons, falling back to string matching.
//
// Returns true if the error is:
// - context.DeadlineExceeded (typed check using errors.Is)
// - net.Error with Timeout() returning true (typed check using errors.As)
// - Error message containing timeout-like patterns (fallback)
func isTimeoutError(err error) bool {
	if err == nil {
		return false
	}

	// Hardened check 1: context.DeadlineExceeded using errors.Is
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	// Hardened check 2: net.Error with Timeout() method using errors.As
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}

	// Fallback: check error message for timeout-like patterns
	errMsg := strings.ToLower(err.Error())
	timeoutPatterns := []string{
		"context deadline exceeded",
		"timeout",
		"i/o timeout",
		"connection timeout",
		"timed out",
		"deadline exceeded",
	}

	for _, pattern := range timeoutPatterns {
		if strings.Contains(errMsg, pattern) {
			return true
		}
	}

	return false
}

// formatErrorMessage generates user-facing error messages that distinguish between
// timeout/state-gathering errors and validation failures.
//
// BLOCKER FIX: Now infers which operation failed from the error itself, rather than
// using modelSpecified flag. This ensures guidance is truthful.
//
// For timeout errors: Returns message with suggestion for the operation that timed out
// For validation failures: Returns the specific failure reason
// For other errors: Returns generic error message
//
// The modelSpecified parameter is used as a fallback hint when inference source
// cannot be determined from the error message.
func formatErrorMessage(err error, modelSpecified bool) string {
	if err == nil {
		return ""
	}

	if isTimeoutError(err) {
		// Infer which operation timed out from the error message
		source := inferTimeoutSource(err)

		// Build timeout suggestion based on inferred source
		var timeoutSuggestion string
		switch source {
		case TimeoutSourceInference:
			// Inference definitely timed out, suggest inference-timeout
			timeoutSuggestion = "--inference-timeout"
		case TimeoutSourceAPI:
			// API check definitely timed out, suggest api-timeout
			timeoutSuggestion = "--api-timeout"
		case TimeoutSourceUnknown:
			// Can't infer from error, use modelSpecified as hint
			if modelSpecified {
				// If model is specified, inference might have run
				timeoutSuggestion = "--inference-timeout"
			} else {
				// GPU-only mode: only API check runs, use api-timeout
				timeoutSuggestion = "--api-timeout"
			}
		}

		return fmt.Sprintf(
			"Deployment validation timed out. This is a state-gathering issue, not a validation failure.\n"+
				"Try again with a longer timeout (e.g., %s 5m).\n"+
				"Original error: %v",
			timeoutSuggestion, err,
		)
	}

	// Other errors: validation failure or real system error
	return fmt.Sprintf("Deployment validation error: %v", err)
}

// main runs the unified deployment validation from the llm package.
//
// Exit codes:
//
//	0 = Deployment is valid (GPU-accelerated with valid responses)
//	1 = Deployment is invalid (CPU-only, no response, or inference failed)
//	2 = Error gathering state (e.g., API unreachable, connection timeout)
//
// The CLI is responsible for:
// - Calling the unified llm.ValidateDeploymentViaAPI
// - Displaying validation details to stderr (GPU check, inference check)
// - Displaying the final success/failure banner
// - Setting appropriate exit code
//
// This ensures consistent user experience whether called directly or via shell wrapper.
func main() {
	opts := DeploymentCheckOptions{
		ServiceURL:       "http://localhost:11434",
		Timeout:          0, // Initialize to 0: only use if user explicitly sets -timeout
		APITimeout:       0, // 0 means use llm package default
		InferenceTimeout: 0, // 0 means use llm package default
		Model:            "",
	}

	flag.StringVar(&opts.ServiceURL, "url", opts.ServiceURL, "URL of Ollama service")
	flag.DurationVar(&opts.Timeout, "timeout", opts.Timeout, "Legacy timeout for both API and inference calls (use -api-timeout or -inference-timeout for granular control)")
	flag.DurationVar(&opts.APITimeout, "api-timeout", opts.APITimeout, "Timeout for fast API calls like /api/ps (overrides -timeout if set)")
	flag.DurationVar(&opts.InferenceTimeout, "inference-timeout", opts.InferenceTimeout, "Timeout for slow inference calls via /api/generate (overrides -timeout if set)")
	flag.StringVar(&opts.Model, "model", opts.Model, "Model to test for inference (optional)")
	flag.Parse()

	// Convert to llm API options and use unified validation.
	// This ensures all validation logic is in one place (llm package) and not duplicated
	// in the CLI or shell scripts.
	apiOpts := opts.toAPIOptions()

	// Use the unified ValidateDeploymentViaAPI function from llm package.
	// This function:
	// - Checks GPU via /api/ps
	// - Checks inference if model specified
	// - Returns (valid, description, error)
	// - NO SIDE EFFECTS (no writes to stderr)
	valid, msg, err := llm.ValidateDeploymentViaAPI(apiOpts)

	// Map the validation result to an exit code and exit
	exitCode := mapExitCode(valid, err)

	// Display validation details and final result
	// The CLI is responsible for all output to ensure single source of truth
	if exitCode == 0 {
		// Success: show final banner
		fmt.Fprintf(os.Stderr, "\n✓ Deployment validation passed\n")
		if msg != "" {
			fmt.Fprintf(os.Stderr, "%s\n", msg)
		}
	} else if exitCode == 2 {
		// Error gathering state: use formatted message that distinguishes timeout from other errors
		formattedErr := formatErrorMessage(err, opts.Model != "")
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", formattedErr)
	} else {
		// Validation failed: show specific reason
		fmt.Fprintf(os.Stderr, "\n✗ Deployment validation failed\n")
		if msg != "" {
			fmt.Fprintf(os.Stderr, "Reason: %s\n", msg)
		}
	}

	os.Exit(exitCode)
}
