package llm

import (
	"regexp"
	"strings"
)

// validation.go provides truthful validation of GPU vs CPU execution and response presence.
// This module ensures that validation logic cannot be fooled by:
// - Generic text matching (grep for "GPU", "offload", etc)
// - Time-based heuristics alone
// - Missing response content
//
// PHASE 1 FIXES:
// - Regexes are precompiled at module scope for efficiency
// - GPU log parser evaluates the MOST RELEVANT (final) layer count, not just the first match
// - Handles multiline logs from retry/fallback scenarios correctly
// - Works with real Ollama structured slog-style output
//
// PHASE 3 ADDITIONS:
// - DeploymentState represents the observable state of a running service
// - ValidateDeploymentState performs comprehensive validation of deployment success
// - Ensures deployment claims match actual GPU/CPU execution and response presence

// Precompiled regexes at module scope for efficiency
// These patterns match GPU layer allocation statements from the runner logs
var (
	gpuLayersPattern1 = regexp.MustCompile(`offloaded\s+(\d+)/(\d+)\s+layers`)       // "offloaded N/M layers"
	gpuLayersPattern2 = regexp.MustCompile(`GPU\s+offloaded\s+(\d+)/(\d+)\s+layers`) // "GPU offloaded N/M layers to GPU"
	gpuLayersPattern3 = regexp.MustCompile(`offloaded_layers:\s*(\d+)/(\d+)`)        // "offloaded_layers: N/M"
	gpuLayersPattern4 = regexp.MustCompile(`GPU layers:\s*(\d+)/(\d+)`)              // "GPU layers: N/M"
)

// ValidateGPUOffloadSuccess checks if logs show actual GPU layer allocation.
// It returns true only if the logs contain explicit evidence of GPU layers being allocated.
// It does NOT rely on generic grep matches or time heuristics.
//
// IMPORTANT: When logs contain multiple layer allocation statements (e.g., from retry scenarios),
// this function evaluates the FINAL (most relevant) state, not the first match.
// This ensures that CPU fallback after a failed retry is correctly identified.
func ValidateGPUOffloadSuccess(logOutput string) bool {
	// Find ALL matches across all patterns and track the most recent one
	// This handles multiline logs where the runner may attempt multiple allocations
	// and we need to evaluate the final result, not intermediate attempts.

	patterns := []*regexp.Regexp{
		gpuLayersPattern1,
		gpuLayersPattern2,
		gpuLayersPattern3,
		gpuLayersPattern4,
	}

	var lastGPULayers string
	lastIndex := -1

	// Find all occurrences of layer counts and keep track of the LAST one
	for _, pattern := range patterns {
		allMatches := pattern.FindAllStringSubmatchIndex(logOutput, -1)
		for _, match := range allMatches {
			// match[0] and match[1] are the full match bounds
			// match[2] and match[3] are the first capture group (GPU layers) bounds
			// match[4] and match[5] are the second capture group (total layers) bounds
			matchStart := match[0]
			if matchStart > lastIndex {
				lastIndex = matchStart
				gpuLayersStr := logOutput[match[2]:match[3]]
				lastGPULayers = gpuLayersStr
			}
		}
	}

	if lastGPULayers == "" {
		// No explicit layer count found - be conservative and return false
		return false
	}

	// Check if GPU layers > 0
	// Return true only if we have non-zero GPU layers
	return lastGPULayers != "0"
}

// ValidateResponsePresence checks if response contains meaningful content.
// It returns true only if the response has non-whitespace content.
// Empty or whitespace-only responses return false.
func ValidateResponsePresence(responseContent string) bool {
	// Trim whitespace and check if anything remains
	trimmed := strings.TrimSpace(responseContent)
	return len(trimmed) > 0
}

// hasValidCompletionActivity checks if a CompletionResponse represents meaningful activity.
// This is used internally by the completion streaming logic to distinguish between
// actual generated content vs whitespace padding. Whitespace tokens still reach the callback
// for streaming protocol compliance, but this function helps track whether any "real"
// content has been generated for final validation purposes.
// Valid activity includes:
//   - Non-whitespace text content (text generation)
//   - Image data (image generation)
//   - Step/TotalSteps progress updates (image generation progress)
func hasValidCompletionActivity(resp CompletionResponse) bool {
	// Check for non-whitespace text content
	trimmed := strings.TrimSpace(resp.Content)
	if len(trimmed) > 0 {
		return true
	}

	// Check for image generation activity (Image field or Step/TotalSteps progress)
	// Image field indicates we have generated image data
	if resp.Image != "" {
		return true
	}

	// Step > 0 indicates progress has been made in image generation
	// (Step=0 with TotalSteps>0 just means "about to start", not real progress)
	if resp.Step > 0 {
		return true
	}

	return false
}

// ============================================================================
// PHASE 3: Deployment Validation
// ============================================================================

// DeploymentState represents the observable state of a running Ollama service.
// This is used to validate that deployment has actually succeeded and the service
// is truly using GPU (not CPU-only or misconfigured).
type DeploymentState struct {
	// ServiceActive indicates whether the Ollama service is currently running
	ServiceActive bool

	// LastLogLines contains recent service logs (typically from systemd journal)
	// Used to determine GPU vs CPU execution state
	LastLogLines string

	// LastResponseContent contains the most recent model response
	// Used to verify the model is actually producing output
	LastResponseContent string
}

// ValidateDeploymentState performs comprehensive validation of deployment success.
// It returns true only if ALL of the following conditions are met:
// 1. The service is currently active/running
// 2. Recent logs show actual GPU acceleration (not CPU-only fallback)
// 3. The last model response contains meaningful content (not empty/whitespace)
//
// This function prevents false-positive deployment claims by:
// - Using ValidateGPUOffloadSuccess to truthfully evaluate GPU state from logs
// - Using ValidateResponsePresence to confirm model output is real
// - Checking ServiceActive to ensure the state is current (not stale)
//
// Returns false if any validation check fails, ensuring deployment is only
// reported as successful when the service is actively running GPU-accelerated
// operations and producing real responses.
func ValidateDeploymentState(state DeploymentState) bool {
	// Check 1: Service must be active
	if !state.ServiceActive {
		return false
	}

	// Check 2: Logs must show GPU success (not CPU fallback)
	if !ValidateGPUOffloadSuccess(state.LastLogLines) {
		return false
	}

	// Check 3: Response must have meaningful content
	if !ValidateResponsePresence(state.LastResponseContent) {
		return false
	}

	// All checks passed
	return true
}
