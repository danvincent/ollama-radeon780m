package llm

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

// TestGemma4FAGuardVulkan tests that FA is disabled for Gemma4 on Vulkan
func TestGemma4FAGuardVulkan(t *testing.T) {
	gpus := []ml.DeviceInfo{
		{
			DeviceID: ml.DeviceID{Library: "Vulkan", ID: "0"},
			Name:     "Vulkan GPU",
		},
	}

	shouldDisable := gemma4FAGuardShouldDisable("gemma4", gpus)
	if !shouldDisable {
		t.Errorf("Expected gemma4FAGuardShouldDisable to return true for Gemma4 on Vulkan, but got false")
	}
}

// TestGemma4FAGuardCUDAPreTuring tests that FA is disabled for Gemma4 on pre-Turing CUDA
func TestGemma4FAGuardCUDAPreTuring(t *testing.T) {
	tests := []struct {
		name              string
		computeMajor      int
		computeMinor      int
		expectedDisabled  bool
	}{
		{
			name:             "Maxwell (compute 5.2)",
			computeMajor:     5,
			computeMinor:     2,
			expectedDisabled: true,
		},
		{
			name:             "Pascal (compute 6.1)",
			computeMajor:     6,
			computeMinor:     1,
			expectedDisabled: true,
		},
		{
			name:             "Volta (compute 7.0)",
			computeMajor:     7,
			computeMinor:     0,
			expectedDisabled: true,
		},
		{
			name:             "Turing edge case (compute 7.4)",
			computeMajor:     7,
			computeMinor:     4,
			expectedDisabled: true,
		},
		{
			name:             "Turing (compute 7.5)",
			computeMajor:     7,
			computeMinor:     5,
			expectedDisabled: false,
		},
		{
			name:             "Ampere (compute 8.0)",
			computeMajor:     8,
			computeMinor:     0,
			expectedDisabled: false,
		},
		{
			name:             "Ada (compute 8.9)",
			computeMajor:     8,
			computeMinor:     9,
			expectedDisabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gpus := []ml.DeviceInfo{
				{
					DeviceID:     ml.DeviceID{Library: "CUDA", ID: "0"},
					ComputeMajor: tt.computeMajor,
					ComputeMinor: tt.computeMinor,
				},
			}

			shouldDisable := gemma4FAGuardShouldDisable("gemma4", gpus)
			if tt.expectedDisabled && !shouldDisable {
				t.Errorf("Expected FA to be disabled for compute %d.%d, but got shouldDisable=%v", tt.computeMajor, tt.computeMinor, shouldDisable)
			}
			if !tt.expectedDisabled && shouldDisable {
				t.Errorf("Expected FA to be enabled for compute %d.%d, but got shouldDisable=%v", tt.computeMajor, tt.computeMinor, shouldDisable)
			}
		})
	}
}

// TestGemma4FAGuardNonGemma4Model tests that non-Gemma4 models are not affected by the guard
func TestGemma4FAGuardNonGemma4Model(t *testing.T) {
	tests := []struct {
		name         string
		architecture string
		library      string
	}{
		{
			name:         "Llama2 on Vulkan",
			architecture: "llama",
			library:      "Vulkan",
		},
		{
			name:         "Mistral on Vulkan",
			architecture: "mistral",
			library:      "Vulkan",
		},
		{
			name:         "Llama3 on Vulkan",
			architecture: "llama3",
			library:      "Vulkan",
		},
		{
			name:         "Gemma2 on Vulkan",
			architecture: "gemma2",
			library:      "Vulkan",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gpus := []ml.DeviceInfo{
				{
					DeviceID: ml.DeviceID{Library: tt.library, ID: "0"},
					Name:     tt.library + " GPU",
				},
			}

			shouldDisable := gemma4FAGuardShouldDisable(tt.architecture, gpus)
			if shouldDisable {
				t.Errorf("Expected FA guard to not affect %s, but got shouldDisable=%v", tt.name, shouldDisable)
			}
		})
	}
}

// TestGemma4FAGuardMultiGPU tests the guard with multiple GPUs (should disable if ANY GPU doesn't support it)
func TestGemma4FAGuardMultiGPU(t *testing.T) {
	tests := []struct {
		name             string
		gpus             []ml.DeviceInfo
		expectedDisabled bool
	}{
		{
			name: "One Vulkan, one Turing CUDA",
			gpus: []ml.DeviceInfo{
				{
					DeviceID: ml.DeviceID{Library: "Vulkan", ID: "0"},
				},
				{
					DeviceID:     ml.DeviceID{Library: "CUDA", ID: "1"},
					ComputeMajor: 7,
					ComputeMinor: 5,
				},
			},
			expectedDisabled: true, // Should disable because of Vulkan
		},
		{
			name: "Two Turing CUDA GPUs",
			gpus: []ml.DeviceInfo{
				{
					DeviceID:     ml.DeviceID{Library: "CUDA", ID: "0"},
					ComputeMajor: 7,
					ComputeMinor: 5,
				},
				{
					DeviceID:     ml.DeviceID{Library: "CUDA", ID: "1"},
					ComputeMajor: 8,
					ComputeMinor: 0,
				},
			},
			expectedDisabled: false, // Should not disable
		},
		{
			name: "Pre-Turing and Turing CUDA",
			gpus: []ml.DeviceInfo{
				{
					DeviceID:     ml.DeviceID{Library: "CUDA", ID: "0"},
					ComputeMajor: 6,
					ComputeMinor: 1,
				},
				{
					DeviceID:     ml.DeviceID{Library: "CUDA", ID: "1"},
					ComputeMajor: 7,
					ComputeMinor: 5,
				},
			},
			expectedDisabled: true, // Should disable because of pre-Turing
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldDisable := gemma4FAGuardShouldDisable("gemma4", tt.gpus)
			if tt.expectedDisabled && !shouldDisable {
				t.Errorf("Expected FA to be disabled, but got shouldDisable=%v", shouldDisable)
			}
			if !tt.expectedDisabled && shouldDisable {
				t.Errorf("Expected FA to be enabled, but got shouldDisable=%v", shouldDisable)
			}
		})
	}
}
