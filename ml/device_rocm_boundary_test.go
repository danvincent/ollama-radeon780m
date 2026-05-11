package ml

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"runtime"
	"testing"
	"time"
)

// mockRunner implements BaseRunner for testing GetDevicesFromRunner
type mockRunner struct {
	port   int
	exited bool
}

func (m *mockRunner) GetPort() int {
	return m.port
}

func (m *mockRunner) HasExited() bool {
	return m.exited
}

// TestGetDevicesFromRunnerWithROCmDevice exercises the real production boundary
// code path from GetDevicesFromRunner() through device discovery and environment setup.
//
// This test:
// 1. Uses httptest to mock the runner's /info endpoint
// 2. Invokes GetDevicesFromRunner() (actual production code)
// 3. Validates device discovery yields expected ROCm information
// 4. Exercises environment setup and validation
// 5. Proves the repo-controlled boundary completes successfully before runtime execution
func TestGetDevicesFromRunnerWithROCmDevice(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	// Mock device returned by the runner's /info endpoint
	rocmDevice := DeviceInfo{
		DeviceID: DeviceID{ID: "0", Library: "ROCm"},
		Name:     "Radeon 780M",
		Description: "RDNA3 - gfx1103",
		ComputeMajor: 0x11,
		ComputeMinor: 0x03,
		TotalMemory:  uint64(512 * 1024 * 1024), // 512 MB
		FreeMemory:   uint64(512 * 1024 * 1024),
	}

	// Create httptest server that serves the /info endpoint
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/info" && r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode([]DeviceInfo{rocmDevice})
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	// Extract port from server URL (e.g., "http://127.0.0.1:xxxxx" -> xxxxx)
	listener := server.Listener.(*net.TCPListener)
	port := listener.Addr().(*net.TCPAddr).Port

	// Create mock runner with the httptest server port
	runner := &mockRunner{port: port, exited: false}

	// BOUNDARY TEST: Invoke actual production code GetDevicesFromRunner()
	// This exercises the real HTTP request logic and unmarshalling
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	devices, err := GetDevicesFromRunner(ctx, runner)
	if err != nil {
		t.Fatalf("GetDevicesFromRunner failed: %v", err)
	}

	// Validate discovery returned expected device
	if len(devices) != 1 {
		t.Fatalf("Expected 1 device, got %d", len(devices))
	}

	device := devices[0]
	if device.Library != "ROCm" {
		t.Fatalf("Expected Library=ROCm, got %s", device.Library)
	}

	if device.ID != "0" {
		t.Fatalf("Expected ID=0, got %s", device.ID)
	}

	if device.ComputeMajor != 0x11 || device.ComputeMinor != 0x03 {
		t.Fatalf("Expected gfx1103, got gfx%x%02x", device.ComputeMajor, device.ComputeMinor)
	}

	// VALIDATION: Verify device requires init validation (repo-controlled step)
	if !device.NeedsInitValidation() {
		t.Fatal("ROCm device should require init validation")
	}

	// ENVIRONMENT SETUP: Verify environment variables are correctly configured
	// This is the last Ollama-controlled step before runtime execution
	// NOTE: This mirrors the production flow in discover/runner.go where GetDevicesEnv()
	// is called first, then AddInitValidation() is applied
	discoveryEnv := GetDevicesEnv(devices, true) // mustFilter=true for discovery phase
	devices[0].AddInitValidation(discoveryEnv)   // Apply init validation flag
	
	expectedVar := "ROCR_VISIBLE_DEVICES"
	if runtime.GOOS != "linux" {
		expectedVar = "HIP_VISIBLE_DEVICES"
	}

	if discoveryEnv[expectedVar] != "0" {
		t.Fatalf("Expected %s=0 for device filtering, got %s", expectedVar, discoveryEnv[expectedVar])
	}

	// Verify init validation flag is set
	if discoveryEnv["GGML_CUDA_INIT"] != "1" {
		t.Fatalf("Expected GGML_CUDA_INIT=1 for init validation, got: %v", discoveryEnv)
	}

	// MODEL LOAD PHASE: Verify production environment (mustFilter=false)
	modelLoadEnv := GetDevicesEnv(devices, false)
	if modelLoadEnv[expectedVar] != "0" {
		t.Fatalf("Expected %s=0 during model load, got %s", expectedVar, modelLoadEnv[expectedVar])
	}

	// BOUNDARY PROOF: At this point, all repo-controlled device discovery,
	// validation, and environment setup is complete. The next step would be
	// runner exec(), which is external to Ollama's direct control.
	t.Log("✓ Boundary test passed: GetDevicesFromRunner() successfully completed")
	t.Log("  - Device discovery: gfx1103 identified and unmarshalled")
	t.Log("  - Device validation: ROCm rocblas init validation enabled")
	t.Log("  - Environment setup: ROCR_VISIBLE_DEVICES=0, GGML_CUDA_INIT=1")
	t.Log("  - Last Ollama-controlled step complete; runtime execution begins next")
}

// TestGetDevicesFromRunnerBoundaryWithTimeout validates the boundary behavior
// when the runner is slow to respond (timeout scenario)
func TestGetDevicesFromRunnerBoundaryWithTimeout(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	// Create a server that delays response indefinitely
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Don't respond, let the context timeout
		<-r.Context().Done()
	}))
	defer server.Close()

	listener := server.Listener.(*net.TCPListener)
	port := listener.Addr().(*net.TCPAddr).Port
	runner := &mockRunner{port: port, exited: false}

	// Use a very short timeout to trigger timeout error
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := GetDevicesFromRunner(ctx, runner)
	if err == nil {
		t.Fatal("Expected timeout error")
	}

	// Boundary behavior: timeout error is clear and actionable
	if err.Error() != "failed to finish discovery before timeout" {
		t.Fatalf("Expected timeout error message, got: %v", err)
	}

	t.Log("✓ Boundary test passed: Timeout handled cleanly by GetDevicesFromRunner()")
}

// TestGetDevicesFromRunnerBoundaryWithRunnerExit validates behavior when
// runner crashes during discovery
func TestGetDevicesFromRunnerBoundaryWithRunnerExit(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("ROCm is not available on macOS")
	}

	// Create a server that immediately closes
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create listener: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close() // Close immediately so connections fail

	runner := &mockRunner{port: port, exited: true}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err = GetDevicesFromRunner(ctx, runner)
	if err == nil {
		t.Fatal("Expected error when runner exits")
	}

	// Boundary behavior: runner exit is detected
	if err.Error() != "runner crashed" {
		t.Fatalf("Expected 'runner crashed' error, got: %v", err)
	}

	t.Log("✓ Boundary test passed: Runner exit detected by GetDevicesFromRunner()")
}
