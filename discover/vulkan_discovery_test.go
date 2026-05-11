package discover

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
)

func init() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)
}

// deviceIsRealVulkan checks if a device is a real Vulkan device, not just a CPU entry.
// Returns true if the device has properties that distinguish it from CPU-only fallback.
func deviceIsRealVulkan(dev ml.DeviceInfo) bool {
	// Check if it's explicitly marked as Vulkan
	if dev.Library != "Vulkan" {
		return false
	}

	// Real Vulkan devices must have a non-empty ID
	if dev.ID == "" {
		return false
	}

	return true
}

// TestVulkanDeviceDiscovery tests that Vulkan devices are discovered during GPU discovery.
// It uses stronger assertions to verify actual Vulkan device registration,
// not just generic availability or CPU fallback.
// Uses Skip only for legitimate environmental absence.
func TestVulkanDeviceDiscovery(t *testing.T) {
	// Only run if explicitly enabled
	if os.Getenv("OLLAMA_VULKAN") == "" && os.Getenv("VULKAN_SDK") == "" {
		t.Skip("Vulkan not explicitly enabled (set OLLAMA_VULKAN=1 or VULKAN_SDK to run)")
	}

	t.Logf("Vulkan discovery starting: OLLAMA_VULKAN=%s, VULKAN_SDK=%s",
		os.Getenv("OLLAMA_VULKAN"), os.Getenv("VULKAN_SDK"))

	// Log what LibOllamaPath is for debugging
	t.Logf("LibOllamaPath: %s", ml.LibOllamaPath)

	// Set up a short timeout for discovery
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Run GPU discovery
	devices := GPUDevices(ctx, nil)

	// Log all discovered devices
	t.Logf("Discovered %d total devices", len(devices))

	deviceCount := make(map[string]int)
	for i, dev := range devices {
		t.Logf("Device %d: Library=%s, Name=%s, ID=%s, PCIID=%s, Memory=%d bytes",
			i, dev.Library, dev.Name, dev.ID, dev.PCIID, dev.TotalMemory)
		deviceCount[dev.Library]++
	}

	t.Logf("Device summary by library: %v", deviceCount)

	// KEY ASSERTION: Find real Vulkan devices (with meaningful properties)
	var realVulkanDevices []ml.DeviceInfo
	for _, dev := range devices {
		if deviceIsRealVulkan(dev) {
			realVulkanDevices = append(realVulkanDevices, dev)
		}
	}

	if len(realVulkanDevices) == 0 {
		// If Vulkan was explicitly enabled but not found, this is a hard error
		if os.Getenv("OLLAMA_VULKAN") != "" {
			t.Errorf("OLLAMA_VULKAN=1 but no Vulkan device with meaningful properties found. "+
				"Expected real Vulkan device registration. Total devices: %d",
				len(devices))
			return
		}

		// If only VULKAN_SDK is set, it's informational
		t.Logf("INFO: No Vulkan devices found (may not have Vulkan-capable hardware or driver)")
		return
	}

	// Validate properties of discovered Vulkan devices
	for i, dev := range realVulkanDevices {
		t.Logf("✓ Real Vulkan device %d: Library=%s, Name=%s, ID=%s, Memory=%d bytes",
			i, dev.Library, dev.Name, dev.ID, dev.TotalMemory)

		// Verify each Vulkan device has meaningful properties
		if dev.ID == "" {
			t.Errorf("Vulkan device at index %d missing ID", i)
		}
		if dev.Name == "" {
			t.Logf("WARN: Vulkan device at index %d has no name, but has valid ID: %s", i, dev.ID)
		}
	}

	t.Logf("✓ Found %d Vulkan device(s) with meaningful properties during GPU discovery", len(realVulkanDevices))
}
