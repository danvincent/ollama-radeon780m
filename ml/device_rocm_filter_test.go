package ml

import (
	"runtime"
	"testing"
)

// TestROCmGetDevicesEnvMustFilterParameter validates the difference between
// mustFilter=true (used during discovery validation) and mustFilter=false (used during model load).
// This is important because ROCm handling differs based on the context.
func TestROCmGetDevicesEnvMustFilterParameter(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	rocmDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "ROCm"},
		Name:     "Radeon 780M",
	}

	cudaDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "CUDA"},
		Name:     "NVIDIA GPU",
	}

	// Test with ROCm device in both cases
	rocmOnlyEnvMustFilter := GetDevicesEnv([]DeviceInfo{rocmDevice}, true)
	rocmOnlyEnvNoFilter := GetDevicesEnv([]DeviceInfo{rocmDevice}, false)

	expectedVar := "ROCR_VISIBLE_DEVICES"
	if runtime.GOOS != "linux" {
		expectedVar = "HIP_VISIBLE_DEVICES"
	}

	// ROCm should always be filtered, regardless of mustFilter parameter
	if rocmOnlyEnvMustFilter[expectedVar] != "0" {
		t.Fatalf("ROCm not filtered with mustFilter=true: got %v", rocmOnlyEnvMustFilter)
	}

	if rocmOnlyEnvNoFilter[expectedVar] != "0" {
		t.Fatalf("ROCm not filtered with mustFilter=false: got %v", rocmOnlyEnvNoFilter)
	}

	t.Log("✓ ROCm consistently filtered regardless of mustFilter")

	// Test with CUDA device - should NOT be filtered when mustFilter=false
	cudaEnvMustFilter := GetDevicesEnv([]DeviceInfo{cudaDevice}, true)
	cudaEnvNoFilter := GetDevicesEnv([]DeviceInfo{cudaDevice}, false)

	// When mustFilter=true, CUDA should be filtered
	if cudaEnvMustFilter["CUDA_VISIBLE_DEVICES"] != "0" {
		t.Fatalf("CUDA should be filtered with mustFilter=true, got: %v", cudaEnvMustFilter)
	}

	// When mustFilter=false, CUDA should NOT be filtered (to avoid confusing ROCm)
	if cudaEnvNoFilter["CUDA_VISIBLE_DEVICES"] != "" {
		t.Fatalf("CUDA should NOT be filtered with mustFilter=false, got: %v", cudaEnvNoFilter)
	}

	t.Log("✓ CUDA filtering correctly depends on mustFilter parameter")

	// Test with both ROCm and CUDA - the actual mixed environment case
	mixedEnvMustFilter := GetDevicesEnv([]DeviceInfo{rocmDevice, cudaDevice}, true)
	mixedEnvNoFilter := GetDevicesEnv([]DeviceInfo{rocmDevice, cudaDevice}, false)

	// With mustFilter=true, both should be filtered
	if mixedEnvMustFilter[expectedVar] != "0" {
		t.Fatalf("ROCm should be filtered with mustFilter=true, got: %v", mixedEnvMustFilter)
	}
	if mixedEnvMustFilter["CUDA_VISIBLE_DEVICES"] != "0" {
		t.Fatalf("CUDA should be filtered with mustFilter=true, got: %v", mixedEnvMustFilter)
	}

	// With mustFilter=false, ROCm should be filtered but CUDA should NOT
	if mixedEnvNoFilter[expectedVar] != "0" {
		t.Fatalf("ROCm should be filtered with mustFilter=false, got: %v", mixedEnvNoFilter)
	}
	if mixedEnvNoFilter["CUDA_VISIBLE_DEVICES"] != "" {
		t.Fatalf("CUDA should NOT be filtered with mustFilter=false (mixed env), got: %v", mixedEnvNoFilter)
	}

	t.Log("✓ Mixed ROCm/CUDA environment correctly handled")
}

// TestROCmMultipleDeviceFiltering validates that multiple ROCm devices are
// properly filtered and their IDs correctly accumulated in the environment variable.
func TestROCmMultipleDeviceFiltering(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	devices := []DeviceInfo{
		{
			DeviceID: DeviceID{ID: "0", Library: "ROCm"},
			Name:     "GPU0",
		},
		{
			DeviceID: DeviceID{ID: "1", Library: "ROCm"},
			Name:     "GPU1",
		},
	}

	env := GetDevicesEnv(devices, false)

	expectedVar := "ROCR_VISIBLE_DEVICES"
	if runtime.GOOS != "linux" {
		expectedVar = "HIP_VISIBLE_DEVICES"
	}

	// Both devices should be included in the filter
	expected := "0,1"
	if env[expectedVar] != expected {
		t.Fatalf("Expected %s=%s, got %s=%s", expectedVar, expected, expectedVar, env[expectedVar])
	}

	t.Logf("✓ Multiple ROCm devices correctly accumulated: %s=%s", expectedVar, env[expectedVar])
}
