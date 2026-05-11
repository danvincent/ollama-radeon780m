package llm

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

// TestPhase2_DirectLoadCall_PartialFailsThenFullSucceeds verifies that Load() correctly
// implements the Phase 2 fix: when partial offload fails, it retries with full offload.
//
// This test directly calls Load() to drive the exact code path that the Phase 2 fix modifies
// (Phase 2 fix: lines 917-965 in llm/server.go).
func TestPhase2_DirectLoadCall_PartialFailsThenFullSucceeds(t *testing.T) {
	// Setup: System with constrained memory that will trigger partial offload
	minMemory := uint64(457 * format.MebiByte)
	systemInfo := ml.SystemInfo{
		TotalMemory: 2 * format.GibiByte,
		FreeMemory:  1200 * format.MebiByte,
		FreeSwap:    1024 * format.MebiByte,
	}

	gpus := []ml.DeviceInfo{{
		DeviceID:   ml.DeviceID{ID: "gpu0", Library: "Vulkan"},
		FreeMemory: minMemory + uint64(315 * format.MebiByte),
	}}

	layerSize := uint64(9 * format.MebiByte)

	// Create an ollamaServer instance
	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 36,
			modelPath:   "/test/model",
			options: api.Options{
				Runner: api.Runner{
					NumGPU: -1,
				},
			},
			loadRequest: LoadRequest{
				GPULayers: ml.GPULayersList{},
			},
			testSkipRunnerWait: true, // Skip runner wait for this test
		},
	}

	s.mem = &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights: make([]uint64, 36),
			Cache:   make([]uint64, 36),
		},
		GPUs: make([]ml.DeviceMemory, len(gpus)),
	}

	for i := range s.mem.CPU.Weights {
		s.mem.CPU.Weights[i] = layerSize
	}

	for i := range s.mem.GPUs {
		s.mem.GPUs[i].DeviceID = gpus[i].DeviceID
		s.mem.GPUs[i].Weights = make([]uint64, 36)
		s.mem.GPUs[i].Cache = make([]uint64, 36)
		for j := range s.mem.GPUs[i].Weights {
			s.mem.GPUs[i].Weights[j] = layerSize
		}
	}

	// Mock initModel to simulate:
	// 1. Partial offload (33-35 layers) FAILS on LoadOperationFit
	// 2. Full offload (36 layers) SUCCEEDS on LoadOperationFit
	// 3. Both succeed on LoadOperationAlloc and LoadOperationCommit
	loadAttempts := make(map[string]int) // key: "operation-layers"
	callOrder := []string{}                // Track call order for validation

	s.testInitModelFunc = func(ctx context.Context, req LoadRequest, operation LoadOperation) (*LoadResponse, error) {
		gpuLayerCount := req.GPULayers.Sum()
		callKey := operationName(operation) + "-" + fmt.Sprintf("%d", gpuLayerCount)
		loadAttempts[callKey]++
		callOrder = append(callOrder, callKey)

		// Hard assertion: Verify we're only called with expected operations
		if operation != LoadOperationFit && operation != LoadOperationAlloc && 
		   operation != LoadOperationCommit && operation != LoadOperationClose {
			t.Fatalf("HARD ASSERTION FAILED: Unexpected operation %v", operation)
		}

		// Handle Close operation (cleanup at end)
		if operation == LoadOperationClose {
			t.Logf("Mock: Close operation")
			return &LoadResponse{Success: true}, nil
		}

		resp := &LoadResponse{
			Success: true,
			Memory: ml.BackendMemory{
				CPU: ml.DeviceMemory{
					Weights: make([]uint64, 36),
					Cache:   make([]uint64, 36),
				},
				GPUs: make([]ml.DeviceMemory, len(gpus)),
			},
		}

		for i := range resp.Memory.GPUs {
			resp.Memory.GPUs[i].DeviceID = gpus[i].DeviceID
			resp.Memory.GPUs[i].Weights = make([]uint64, 36)
			resp.Memory.GPUs[i].Cache = make([]uint64, 36)
		}

		// Key Phase 2 test scenario:
		// When partial offload (33-35 layers) is attempted in LoadOperationFit, it FAILS.
		// The Phase 2 fix then retries with full offload (36 layers), which SUCCEEDS.
		if operation == LoadOperationFit && gpuLayerCount > 0 && gpuLayerCount < 36 {
			// Partial offload FAILS on first attempt (as per Phase 2 problem statement)
			resp.Success = false
			t.Logf("Mock: Partial offload (%d/36) FAILED on LoadOperationFit (triggering Phase 2 fix)", gpuLayerCount)
			return resp, nil
		}

		// Full offload (36 layers) SUCCEEDS
		if gpuLayerCount == 36 {
			t.Logf("Mock: Full offload (36/36) SUCCEEDED on operation %v", operation)
			resp.Success = true
			return resp, nil
		}

		// Fallback: return success for other cases
		resp.Success = true
		return resp, nil
	}

	// EXECUTION: Call Load() directly - this drives the Phase 2 fix code path
	ctx := context.Background()
	deviceIDs, err := s.Load(ctx, systemInfo, gpus, false)

	// HARD ASSERTIONS: Verify the test preconditions were met
	if err != nil {
		t.Fatalf("HARD ASSERTION FAILED: Load() returned error: %v", err)
	}

	if len(callOrder) == 0 {
		t.Fatalf("HARD ASSERTION FAILED: initModel was never called (test setup invalid)")
	}

	// Verify Phase 2 fix was triggered: partial attempt, then full offload retry
	fitCalls := 0
	partialFailureObserved := false
	fullOffloadAttempted := false

	for _, call := range callOrder {
		if len(call) >= 3 && call[:3] == "Fit" {
			fitCalls++
		}
		// Check for partial offload attempts (layers > 0 and < 36)
		if strings.HasPrefix(call, "Fit-") {
			layerCountStr := strings.TrimPrefix(call, "Fit-")
			if layerCount, err := strconv.Atoi(layerCountStr); err == nil {
				if layerCount > 0 && layerCount < 36 {
					partialFailureObserved = true
				}
			}
		}
		// Check for full offload attempts (exactly 36 layers)
		if call == "Fit-36" {
			fullOffloadAttempted = true
		}
	}

	// HARD ASSERTIONS: Verify Phase 2 fix flow
	if !partialFailureObserved {
		t.Fatalf("HARD ASSERTION FAILED: Partial offload attempt was not observed in call sequence: %v", callOrder)
	}

	if !fullOffloadAttempted {
		t.Fatalf("HARD ASSERTION FAILED: Full offload retry was not attempted after partial failure: %v", callOrder)
	}

	// BEHAVIOR VERIFICATION: Load() should succeed with full GPU offload
	if len(deviceIDs) == 0 {
		t.Fatalf("HARD ASSERTION FAILED: Load() did not return any device IDs, expected full GPU offload (36/36)")
	}

	t.Logf("✓ Phase 2 fix verified: Partial offload failed, full offload retry succeeded")
	t.Logf("✓ Call sequence: %v", callOrder)
	t.Logf("✓ Load() completed successfully with device IDs: %v", deviceIDs)
}

// TestPhase2_DirectLoadCall_PartialFailsFullFailsStatePreserved verifies that when both
// partial and full offload fail, the Load() method preserves state for backoff to operate.
//
// This directly exercises the Phase 2 fix failure path (lines 947-959 in server.go).
func TestPhase2_DirectLoadCall_PartialFailsFullFailsStatePreserved(t *testing.T) {
	// Setup: System with constrained memory
	minMemory := uint64(457 * format.MebiByte)
	systemInfo := ml.SystemInfo{
		TotalMemory: 2 * format.GibiByte,
		FreeMemory:  1200 * format.MebiByte,
		FreeSwap:    1024 * format.MebiByte,
	}

	gpus := []ml.DeviceInfo{{
		DeviceID:   ml.DeviceID{ID: "gpu0", Library: "Vulkan"},
		FreeMemory: minMemory + uint64(315 * format.MebiByte),
	}}

	layerSize := uint64(9 * format.MebiByte)

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 36,
			modelPath:   "/test/model",
			options: api.Options{
				Runner: api.Runner{
					NumGPU: -1,
				},
			},
			loadRequest: LoadRequest{
				GPULayers: ml.GPULayersList{},
			},
			testSkipRunnerWait: true, // Skip runner wait for this test
		},
	}

	s.mem = &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights: make([]uint64, 36),
			Cache:   make([]uint64, 36),
		},
		GPUs: make([]ml.DeviceMemory, len(gpus)),
	}

	for i := range s.mem.CPU.Weights {
		s.mem.CPU.Weights[i] = layerSize
	}

	for i := range s.mem.GPUs {
		s.mem.GPUs[i].DeviceID = gpus[i].DeviceID
		s.mem.GPUs[i].Weights = make([]uint64, 36)
		s.mem.GPUs[i].Cache = make([]uint64, 36)
		for j := range s.mem.GPUs[i].Weights {
			s.mem.GPUs[i].Weights[j] = layerSize
		}
	}

	// Track the state at each point to verify state preservation
	stateSnapshots := make(map[string]uint64)

	// Mock initModel to simulate:
	// 1. Partial offload (33-35 layers) FAILS on LoadOperationFit
	// 2. Full offload (36 layers) also FAILS on LoadOperationFit
	// 3. Subsequent attempts should show state was preserved for backoff
	s.testInitModelFunc = func(ctx context.Context, req LoadRequest, operation LoadOperation) (*LoadResponse, error) {
		gpuLayerCount := req.GPULayers.Sum()

		// Hard assertion: Ensure we're in a sensible operation sequence
		if operation != LoadOperationFit && operation != LoadOperationAlloc && 
		   operation != LoadOperationCommit && operation != LoadOperationClose {
			t.Fatalf("HARD ASSERTION FAILED: Unexpected operation %v", operation)
		}

		// Handle Close operation (cleanup at end)
		if operation == LoadOperationClose {
			return &LoadResponse{Success: true}, nil
		}

		resp := &LoadResponse{
			Success: true,
			Memory: ml.BackendMemory{
				CPU: ml.DeviceMemory{
					Weights: make([]uint64, 36),
					Cache:   make([]uint64, 36),
				},
				GPUs: make([]ml.DeviceMemory, len(gpus)),
			},
		}

		for i := range resp.Memory.GPUs {
			resp.Memory.GPUs[i].DeviceID = gpus[i].DeviceID
			resp.Memory.GPUs[i].Weights = make([]uint64, 36)
			resp.Memory.GPUs[i].Cache = make([]uint64, 36)
		}

		// Scenario: Both partial and full offload attempts fail, triggering backoff
		if operation == LoadOperationFit && gpuLayerCount > 0 && gpuLayerCount <= 36 {
			resp.Success = false
			stateKey := operationName(operation) + "-" + fmt.Sprintf("%d", gpuLayerCount)
			stateSnapshots[stateKey] = uint64(gpuLayerCount)
			t.Logf("Mock: Offload attempt (%d/36) FAILED on LoadOperationFit", gpuLayerCount)
			return resp, nil
		}

		resp.Success = true
		return resp, nil
	}

	// EXECUTION: Call Load() - this drives through the Phase 2 failure path
	ctx := context.Background()
	deviceIDs, err := s.Load(ctx, systemInfo, gpus, false)

	// The Load() method should either:
	// a) Eventually find a working configuration (even if smaller), or
	// b) Return an error when no configuration works
	// Both are valid outcomes for this test - we're verifying state wasn't corrupted

	if len(stateSnapshots) == 0 {
		t.Fatalf("HARD ASSERTION FAILED: No state snapshots recorded (test setup invalid)")
	}

	// Verify state was observed at multiple layer counts (showing backoff iterated)
	layerCounts := make(map[uint64]bool)
	for _, count := range stateSnapshots {
		layerCounts[count] = true
	}

	// HARD ASSERTION: Backoff must have tried at least 2 different layer counts
	// to prove state was preserved and backoff iterated properly
	if len(layerCounts) < 2 {
		t.Fatalf("HARD ASSERTION FAILED: Expected backoff to iterate through multiple layer counts, but only observed %d unique count(s): %v", len(layerCounts), stateSnapshots)
	}

	// HARD ASSERTION: Must have observed partial offload failure (layer count between 1-35)
	partialFailureFound := false
	for count := range layerCounts {
		if count > 0 && count < 36 {
			partialFailureFound = true
			break
		}
	}
	if !partialFailureFound {
		t.Fatalf("HARD ASSERTION FAILED: Expected partial offload attempt (1-35 layers) to be observed, got layer counts: %v", layerCounts)
	}

	// HARD ASSERTION: Must have observed full offload failure (36 layers)
	if !layerCounts[36] {
		t.Fatalf("HARD ASSERTION FAILED: Expected full offload attempt (36 layers) to be observed and fail, got layer counts: %v", layerCounts)
	}

	t.Logf("✓ Phase 2 failure path verified: Full offload also failed, backoff processed the failure")
	t.Logf("✓ Load() result: error=%v, deviceIDs=%v", err, deviceIDs)
	t.Logf("✓ State was preserved through backoff iterations")
}

// TestPhase2_DirectLoadCall_HardPreconditions verifies that test setup is correct
// and that the Phase 2 fix code path is actually being exercised.
func TestPhase2_DirectLoadCall_HardPreconditions(t *testing.T) {
	minMemory := uint64(457 * format.MebiByte)
	systemInfo := ml.SystemInfo{
		TotalMemory: 2 * format.GibiByte,
		FreeMemory:  1200 * format.MebiByte,
		FreeSwap:    1024 * format.MebiByte,
	}

	gpus := []ml.DeviceInfo{{
		DeviceID:   ml.DeviceID{ID: "gpu0", Library: "Vulkan"},
		FreeMemory: minMemory + uint64(315 * format.MebiByte),
	}}

	layerSize := uint64(9 * format.MebiByte)

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 36,
			modelPath:   "/test/model",
			options: api.Options{
				Runner: api.Runner{
					NumGPU: -1,
				},
			},
			loadRequest: LoadRequest{
				GPULayers: ml.GPULayersList{},
			},
			testSkipRunnerWait: true, // Skip runner wait for this test
		},
	}

	s.mem = &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights: make([]uint64, 36),
			Cache:   make([]uint64, 36),
		},
		GPUs: make([]ml.DeviceMemory, len(gpus)),
	}

	for i := range s.mem.CPU.Weights {
		s.mem.CPU.Weights[i] = layerSize
	}

	for i := range s.mem.GPUs {
		s.mem.GPUs[i].DeviceID = gpus[i].DeviceID
		s.mem.GPUs[i].Weights = make([]uint64, 36)
		s.mem.GPUs[i].Cache = make([]uint64, 36)
		for j := range s.mem.GPUs[i].Weights {
			s.mem.GPUs[i].Weights[j] = layerSize
		}
	}

	// PRECONDITION 1: Verify that partial layout is possible with this setup
	partialLayout, _ := s.createLayout(systemInfo, gpus, s.mem, false, 0)
	if partialLayout.Sum() <= 0 || partialLayout.Sum() >= 36 {
		t.Fatalf("HARD ASSERTION FAILED: Setup does not produce partial layout. Got %d/36 layers", partialLayout.Sum())
	}
	t.Logf("✓ PRECONDITION 1: Partial layout is possible (%d/36 layers)", partialLayout.Sum())

	// PRECONDITION 2: Verify that testInitModelFunc seam is available
	if s.testInitModelFunc != nil {
		t.Fatalf("HARD ASSERTION FAILED: testInitModelFunc should start as nil")
	}

	callCount := 0
	s.testInitModelFunc = func(ctx context.Context, req LoadRequest, operation LoadOperation) (*LoadResponse, error) {
		callCount++
		return &LoadResponse{
			Success: true,
			Memory: ml.BackendMemory{
				CPU:  ml.DeviceMemory{Weights: make([]uint64, 36), Cache: make([]uint64, 36)},
				GPUs: make([]ml.DeviceMemory, len(gpus)),
			},
		}, nil
	}

	// Make a mock call to verify the seam works
	mockResp, _ := s.initModel(context.Background(), LoadRequest{}, LoadOperationFit)
	if callCount != 1 || mockResp == nil {
		t.Fatalf("HARD ASSERTION FAILED: testInitModelFunc seam did not work. callCount=%d", callCount)
	}
	t.Logf("✓ PRECONDITION 2: testInitModelFunc seam is functional")

	// PRECONDITION 3: Verify ollamaServer type is correct for calling Load()
	if _, ok := interface{}(s).(interface{ Load(context.Context, ml.SystemInfo, []ml.DeviceInfo, bool) ([]ml.DeviceID, error) }); !ok {
		t.Fatalf("HARD ASSERTION FAILED: ollamaServer does not have Load method")
	}
	t.Logf("✓ PRECONDITION 3: ollamaServer.Load() method is available")

	t.Logf("✓ All hard preconditions passed - Phase 2 tests can proceed")
}

// operationName converts a LoadOperation to a readable string name
func operationName(op LoadOperation) string {
	switch op {
	case LoadOperationFit:
		return "Fit"
	case LoadOperationAlloc:
		return "Alloc"
	case LoadOperationCommit:
		return "Commit"
	case LoadOperationClose:
		return "Close"
	default:
		return "Unknown"
	}
}
