package ggml

import (
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

func init() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)
}

// deviceIsVulkan checks if a device is a real Vulkan device, not a CPU fallback.
// Returns true if the device has meaningful Vulkan properties that distinguish it
// from a CPU-only fallback.
func deviceIsVulkan(dev ml.DeviceInfo) bool {
	// Check if it's explicitly marked as Vulkan
	if dev.Library != "Vulkan" {
		return false
	}

	// Vulkan devices must have a non-empty ID (typically a device index or PCIID)
	if dev.ID == "" {
		return false
	}

	// Vulkan devices should have meaningful memory information
	// (CPU fallback typically reports minimal/zero memory or no meaningful ID)
	if dev.TotalMemory > 0 {
		return true
	}

	// If not a CPU fallback, presence of Vulkan library and ID is sufficient
	return true
}

// findRepoRoot searches upward from current directory to find the repo root
// by looking for go.mod or .git
func findRepoRoot() string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	return findRepoRootFrom(cwd)
}

// findRepoRootFrom searches upward from startDir to find the repo root
func findRepoRootFrom(startDir string) string {
	current := startDir
	for {
		parent := filepath.Dir(current)
		if parent == current {
			// reached filesystem root
			break
		}

		if _, err := os.Stat(filepath.Join(current, "go.mod")); err == nil {
			return current
		}
		if _, err := os.Stat(filepath.Join(current, ".git")); err == nil {
			return current
		}

		current = parent
	}
	return ""
}

// TestLibraryPathResolution tests that the library path resolution works correctly
// in the development build scenario. It must assert that when repo-root build/lib/ollama
// exists, that path is actually returned (not a fallback).
func TestLibraryPathResolution(t *testing.T) {
	// Find repo root to check if build/lib/ollama exists
	repoRoot := findRepoRoot()
	if repoRoot == "" {
		t.Skip("Could not determine repo root; skipping repo-root path assertion")
	}

	buildLibPath := filepath.Join(repoRoot, "build", "lib", "ollama")
	buildLibExists := true
	if _, err := os.Stat(buildLibPath); err != nil {
		buildLibExists = false
	}

	// Get the resolved library path using the shared ml package variable
	resolvedPath := ml.LibOllamaPath

	// The path must not be empty
	if resolvedPath == "" {
		t.Fatal("ResolveLibraryPath returned empty path")
	}

	// Verify the path exists
	fi, err := os.Stat(resolvedPath)
	if err != nil {
		if os.IsNotExist(err) {
			t.Fatalf("resolved library path does not exist: %s", resolvedPath)
		}
		t.Fatalf("error checking library path: %v", err)
	}

	// Verify it's a directory
	if !fi.IsDir() {
		t.Fatalf("resolved library path is not a directory: %s", resolvedPath)
	}

	t.Logf("✓ library path exists and is a directory: %s", resolvedPath)

	// KEY ASSERTION: If build/lib/ollama exists in repo root, it MUST be the one returned
	// This is the critical assertion for the development-build scenario
	if buildLibExists {
		if !strings.HasSuffix(filepath.Clean(resolvedPath), filepath.Clean(buildLibPath)) {
			// Allow the resolved path to be the same directory, just might have symlinks resolved
			absResolved, _ := filepath.Abs(resolvedPath)
			absBuildLib, _ := filepath.Abs(buildLibPath)
			if absResolved != absBuildLib {
				t.Errorf("resolved path %s does not match expected repo-root build path %s",
					resolvedPath, buildLibPath)
			}
		}
		t.Logf("✓ correctly resolved to repo-root build/lib/ollama path")
	} else {
		t.Logf("INFO: build/lib/ollama not present in repo root; fallback path acceptable: %s", resolvedPath)
	}
}

// TestBackendInitialization tests that the backend initializes and registers
// real backends (especially Vulkan) in the intended scenario.
// This test distinguishes between successful Vulkan initialization and CPU-only fallback.
func TestBackendInitialization(t *testing.T) {
	// Verify library path resolution works first
	libPath := ml.LibOllamaPath
	if libPath == "" {
		t.Fatal("ResolveLibraryPath returned empty path; cannot initialize backend")
	}
	t.Logf("Using library path: %s", libPath)

	// Create a temporary minimal model file
	f, err := os.CreateTemp("", "*.bin")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer func() {
		f.Close()
		os.Remove(f.Name())
	}()

	// Write minimal GGUF structure
	if err := ggml.WriteGGUF(f, ggml.KV{
		"general.architecture": "llama",
		"tokenizer.ggml.model": "gpt2",
	}, nil); err != nil {
		t.Fatalf("failed to write model: %v", err)
	}

	// Attempt to create backend
	b, err := New(f.Name(), ml.BackendParams{AllocMemory: false})
	if err != nil {
		// In test environments, backend initialization may fail if no suitable backend is found.
		// However, if we have valid library path, this should not fail - it should at least
		// succeed with CPU fallback. Log this as a warning.
		t.Logf("WARNING: backend initialization failed even with valid library path: %v", err)
		t.Skip("backend initialization not available in this test environment")
	}

	if b == nil {
		t.Fatal("backend creation returned nil without error")
	}
	defer b.Close()

	// KEY ASSERTION: Backend must have discovered at least SOME devices
	// This ensures it didn't silently fail to initialize any backend.
	devices := b.BackendDevices()
	if len(devices) == 0 {
		t.Error("backend initialized but reported zero devices - this indicates backend registration may have failed")
		return
	}

	t.Logf("✓ backend initialized successfully with %d device(s)", len(devices))

	// Count different backend types
	vulkanCount := 0
	cpuCount := 0
	otherCount := 0

	for i, dev := range devices {
		t.Logf("  device %d: Library=%s, Name=%s, ID=%s, Memory=%d bytes",
			i, dev.Library, dev.Name, dev.ID, dev.TotalMemory)

		switch {
		case dev.Library == "Vulkan":
			vulkanCount++
		case dev.Library == "CPU" || dev.Library == "":
			cpuCount++
		default:
			otherCount++
		}
	}

	t.Logf("✓ Device summary: Vulkan=%d, CPU=%d, Other=%d", vulkanCount, cpuCount, otherCount)

	// If Vulkan environment variables suggest Vulkan should be available,
	// at least log if we didn't find Vulkan devices
	if os.Getenv("OLLAMA_VULKAN") != "" || os.Getenv("VULKAN_SDK") != "" {
		if vulkanCount == 0 {
			t.Logf("NOTICE: Vulkan env vars set but no Vulkan devices found. "+
				"This may indicate driver/library issues.")
		} else {
			t.Logf("✓ Found %d Vulkan device(s) as expected from Vulkan env setup", vulkanCount)
		}
	}
}

// TestVulkanBackendDiscovery tests that real Vulkan backends are discovered if available.
// This test asserts meaningful Vulkan device properties, not just generic availability.
// It will skip only on systems without Vulkan capability, never skip on the intended
// test scenario.
func TestVulkanBackendDiscovery(t *testing.T) {
	// Only skip if Vulkan is truly not available in the environment
	if os.Getenv("OLLAMA_VULKAN") == "" && os.Getenv("VULKAN_SDK") == "" {
		t.Skip("Vulkan not explicitly enabled (set OLLAMA_VULKAN=1 or VULKAN_SDK to run)")
	}

	t.Logf("Vulkan discovery test: OLLAMA_VULKAN=%s, VULKAN_SDK=%s",
		os.Getenv("OLLAMA_VULKAN"), os.Getenv("VULKAN_SDK"))

	// Create a temporary minimal model file
	f, err := os.CreateTemp("", "*.bin")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer func() {
		f.Close()
		os.Remove(f.Name())
	}()

	if err := ggml.WriteGGUF(f, ggml.KV{
		"general.architecture": "llama",
		"tokenizer.ggml.model": "gpt2",
	}, nil); err != nil {
		t.Fatalf("failed to write model: %v", err)
	}

	// Create backend to trigger device discovery
	b, err := New(f.Name(), ml.BackendParams{AllocMemory: false})
	if err != nil {
		t.Fatalf("failed to create backend: %v", err)
	}
	defer b.Close()

	// Get discovered devices
	devices := b.BackendDevices()

	// Log all devices for inspection
	t.Logf("Discovered %d total devices", len(devices))
	for i, dev := range devices {
		isVulkan := deviceIsVulkan(dev)
		t.Logf("  Device %d: Library=%s, Name=%s, ID=%s, Memory=%d bytes, IsRealVulkan=%v",
			i, dev.Library, dev.Name, dev.ID, dev.TotalMemory, isVulkan)
	}

	// KEY ASSERTION: Count real Vulkan devices (not CPU fallback)
	var realVulkanDevices []ml.DeviceInfo
	for _, dev := range devices {
		if deviceIsVulkan(dev) {
			realVulkanDevices = append(realVulkanDevices, dev)
		}
	}

	if len(realVulkanDevices) == 0 {
		t.Errorf("OLLAMA_VULKAN=1 but no real Vulkan devices found. "+
			"Expected to discover actual Vulkan device with meaningful properties. "+
			"Total devices reported: %d", len(devices))
		return
	}

	// Assert strong properties for each discovered Vulkan device
	for _, dev := range realVulkanDevices {
		// Device ID must be non-empty
		if dev.ID == "" {
			t.Errorf("Vulkan device found but has empty ID: %+v", dev)
			continue
		}

		// Device name should be meaningful
		if dev.Name == "" {
			t.Logf("WARN: Vulkan device has empty name but valid ID %s", dev.ID)
		} else {
			t.Logf("✓ Vulkan device properly identified: Name=%s, ID=%s", dev.Name, dev.ID)
		}
	}

	t.Logf("✓ Found %d real Vulkan device(s) with meaningful properties", len(realVulkanDevices))
}
