package ml

import (
	"runtime"
	"strings"
	"testing"
)

// TestROCmNeedsInitValidation validates that ROCm and CUDA devices are correctly
// identified as requiring init validation, while Vulkan does not.
func TestROCmNeedsInitValidation(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	rocmDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "ROCm"},
		Name:     "Radeon 780M",
	}

	if !rocmDevice.NeedsInitValidation() {
		t.Fatal("ROCm device should require init validation")
	}

	cudaDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "CUDA"},
		Name:     "NVIDIA GPU",
	}

	if !cudaDevice.NeedsInitValidation() {
		t.Fatal("CUDA device should require init validation")
	}

	vulkanDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "Vulkan"},
		Name:     "Vulkan Device",
	}

	if vulkanDevice.NeedsInitValidation() {
		t.Fatal("Vulkan device should not require init validation")
	}

	t.Log("✓ Init validation requirements correctly differentiated by library")
}

// TestROCmComputeString validates the gfx format for ROCm compute versions.
func TestROCmComputeString(t *testing.T) {
	tests := []struct {
		name         string
		major        int
		minor        int
		expectedGfx  string
	}{
		{
			name:        "gfx1103",
			major:       0x11,
			minor:       0x03,
			expectedGfx: "gfx1103",
		},
		{
			name:        "gfx942",
			major:       0x09,
			minor:       0x42,
			expectedGfx: "gfx942",
		},
		{
			name:        "gfx1100",
			major:       0x11,
			minor:       0x00,
			expectedGfx: "gfx1100",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			device := DeviceInfo{
				DeviceID: DeviceID{ID: "0", Library: "ROCm"},
				ComputeMajor: tc.major,
				ComputeMinor: tc.minor,
			}

			compute := device.Compute()
			if !strings.EqualFold(compute, tc.expectedGfx) {
				t.Fatalf("Expected %s, got %s", tc.expectedGfx, compute)
			}
		})
	}

	t.Log("✓ ROCm device compute strings correctly formatted")
}

// TestROCmPreferredOverVulkan validates that ROCm is preferred over Vulkan.
func TestROCmPreferredOverVulkan(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	rocmDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "ROCm"},
	}

	vulkanDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "Vulkan"},
	}

	if !rocmDevice.PreferredLibrary(vulkanDevice) {
		t.Fatal("ROCm device should be preferred over Vulkan")
	}

	if vulkanDevice.PreferredLibrary(rocmDevice) {
		t.Fatal("Vulkan device should not be preferred over ROCm")
	}

	t.Log("✓ ROCm correctly preferred over Vulkan")
}

// TestROCmAddInitValidation verifies that AddInitValidation correctly sets the
// GGML_CUDA_INIT environment variable for ROCm devices.
func TestROCmAddInitValidation(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	device := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "ROCm"},
		Name:     "Radeon 780M",
	}

	env := make(map[string]string)
	device.AddInitValidation(env)

	if env["GGML_CUDA_INIT"] != "1" {
		t.Fatalf("Expected GGML_CUDA_INIT=1, got: %v", env)
	}

	t.Log("✓ AddInitValidation correctly sets GGML_CUDA_INIT=1")
}

// TestRunnerEnvOverridesPropagation validates that RunnerEnvOverrides are
// correctly propagated by GetDevicesEnv.
func TestRunnerEnvOverridesPropagation(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	device := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "ROCm"},
		RunnerEnvOverrides: map[string]string{
			"HSA_OVERRIDE_GFX_VERSION": "gfx1103",
		},
	}

	env := GetDevicesEnv([]DeviceInfo{device}, false)

	if env["HSA_OVERRIDE_GFX_VERSION"] != "gfx1103" {
		t.Fatalf("RunnerEnvOverrides not propagated: got %v", env)
	}

	t.Log("✓ RunnerEnvOverrides correctly propagated")
}

