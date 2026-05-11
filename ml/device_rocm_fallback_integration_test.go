package ml

import (
	"runtime"
	"testing"
)

// TestROCmDiscoveryFallbackPath provides end-to-end validation of the device
// discovery and environment setup flow for ROCm devices. This complements the
// boundary test (device_rocm_boundary_test.go) which tests actual production
// code paths via GetDevicesFromRunner().
//
// This test validates the helper-level functions that operate on discovered devices,
// and should be kept minimal to avoid duplication with boundary tests.
func TestROCmDiscoveryFallbackPath(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	// Simulate discovered ROCm device
	rocmDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "ROCm"},
		Name:     "Radeon 780M",
		Description: "RDNA3 - gfx1103",
		ComputeMajor: 0x11,
		ComputeMinor: 0x03,
		TotalMemory:  uint64(512 * 1024 * 1024),
		FreeMemory:   uint64(512 * 1024 * 1024),
	}

	// Validate device identification
	if rocmDevice.Library != "ROCm" {
		t.Fatalf("Device library mismatch: %s", rocmDevice.Library)
	}

	if rocmDevice.Compute() != "gfx1103" {
		t.Fatalf("Device compute string incorrect: %s", rocmDevice.Compute())
	}

	// Validate device requires init validation
	if !rocmDevice.NeedsInitValidation() {
		t.Fatal("ROCm device should require init validation")
	}

	// Validate environment filtering
	env := GetDevicesEnv([]DeviceInfo{rocmDevice}, true)
	
	expectedVar := "ROCR_VISIBLE_DEVICES"
	if runtime.GOOS != "linux" {
		expectedVar = "HIP_VISIBLE_DEVICES"
	}

	if env[expectedVar] != "0" {
		t.Fatalf("Device filtering failed: expected %s=0, got %s", expectedVar, env[expectedVar])
	}

	// Validate RunnerEnvOverrides propagation
	rocmDevice.RunnerEnvOverrides = map[string]string{
		"HSA_OVERRIDE_GFX_VERSION": "gfx1103",
	}
	
	envWithOverrides := GetDevicesEnv([]DeviceInfo{rocmDevice}, false)
	if envWithOverrides["HSA_OVERRIDE_GFX_VERSION"] != "gfx1103" {
		t.Fatalf("RunnerEnvOverrides not propagated: %v", envWithOverrides)
	}
	
	t.Log("✓ ROCm device discovery and environment setup validation complete")
}

// TestROCmFallbackWithMultipleDevices validates device filtering when both
// CUDA and ROCm devices are present. This ensures the fallback path correctly
// handles mixed device environments.
func TestROCmFallbackWithMultipleDevices(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	devices := []DeviceInfo{
		{
			DeviceID: DeviceID{ID: "0", Library: "CUDA"},
			Name:     "NVIDIA GPU",
		},
		{
			DeviceID: DeviceID{ID: "0", Library: "ROCm"},
			Name:     "Radeon 780M",
		},
	}

	// Discovery phase: both should be filtered
	discoveryEnv := GetDevicesEnv(devices, true)
	if discoveryEnv["CUDA_VISIBLE_DEVICES"] != "0" {
		t.Fatalf("CUDA not filtered during discovery: %v", discoveryEnv)
	}

	expectedVar := "ROCR_VISIBLE_DEVICES"
	if runtime.GOOS != "linux" {
		expectedVar = "HIP_VISIBLE_DEVICES"
	}
	if discoveryEnv[expectedVar] != "0" {
		t.Fatalf("ROCm not filtered during discovery: %v", discoveryEnv)
	}

	// Model load phase: only ROCm filtered (avoid confusing CUDA)
	modelLoadEnv := GetDevicesEnv(devices, false)
	if modelLoadEnv["CUDA_VISIBLE_DEVICES"] != "" {
		t.Fatalf("CUDA should not be filtered during model load (mixed env): %v", modelLoadEnv)
	}
	if modelLoadEnv[expectedVar] != "0" {
		t.Fatalf("ROCm should be filtered during model load: %v", modelLoadEnv)
	}

	t.Log("✓ Mixed device environment correctly handled")
}

