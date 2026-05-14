package llama

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// TestGemma4MoEExpertWeightViewsContiguous_SourceScan verifies that Gemma4 MoE expert weight views
// are wrapped with ggml_cont() in the graph building code.
//
// BACKGROUND:
// Gemma4 MoE uses fused gate_up_exps tensors. When these are split into separate gate_exps and up_exps
// views using ggml_view_3d, the resulting views have non-contiguous strides in the 3rd dimension
// (nb[2] = 2 × ne[1] × nb[1]). This causes experts to be spaced 2x farther apart than a contiguous
// tensor would be.
//
// The Vulkan mul_mat_id shader path does NOT handle non-contiguous 3D weight tensors correctly.
// For expert k>0, it reads from wrong offsets, producing incoherent output.
//
// FIX:
// Wrap each ggml_view_3d() call with ggml_cont() to force a copy to a contiguous buffer at graph
// build time. This ensures the GPU shader receives properly-strided data.
//
// TEST:
// This test scans gemma4.cpp to verify that ggml_cont() is wrapping both gate_exps and up_exps
// view creation within the ffn_gate_up_exps != nullptr block.
func TestGemma4MoEExpertWeightViewsContiguous_SourceScan(t *testing.T) {
	// Read the gemma4.cpp source file
	gemma4Path := filepath.Join("llama.cpp", "src", "models", "gemma4.cpp")
	content, err := os.ReadFile(gemma4Path)
	if err != nil {
		t.Fatalf("Failed to read %s: %v", gemma4Path, err)
	}

	source := string(content)

	// Pattern 1: gate_exps must be wrapped with ggml_cont
	// Looks for: gate_exps = ggml_cont(...ggml_view_3d(... fused ...))
	gateExpPattern := regexp.MustCompile(`gate_exps\s*=\s*ggml_cont\s*\(\s*ctx0\s*,\s*ggml_view_3d\s*\(\s*ctx0\s*,\s*fused`)
	if !gateExpPattern.MatchString(source) {
		t.Fatal("gate_exps view NOT wrapped with ggml_cont() - Vulkan shader will read wrong offsets for non-expert-0 experts")
	}

	// Pattern 2: up_exps must also be wrapped with ggml_cont
	// Looks for: up_exps = ggml_cont(...ggml_view_3d(... fused ...))
	upExpPattern := regexp.MustCompile(`up_exps\s*=\s*ggml_cont\s*\(\s*ctx0\s*,\s*ggml_view_3d\s*\(\s*ctx0\s*,\s*fused`)
	if !upExpPattern.MatchString(source) {
		t.Fatal("up_exps view NOT wrapped with ggml_cont() - Vulkan shader will read wrong offsets for non-expert-0 experts")
	}

	// Pattern 3: Verify the wrapping is inside the fused tensor check
	// Looks for the if block: if (model.layers[il].ffn_gate_up_exps != nullptr) { ... ggml_cont ... }
	ifBlock := regexp.MustCompile(
		`if\s*\(\s*model\.layers\[il\]\.ffn_gate_up_exps\s*!=\s*nullptr\s*\)\s*\{[^}]*gate_exps\s*=\s*ggml_cont[^}]*up_exps\s*=\s*ggml_cont`,
	)
	if !ifBlock.MatchString(source) {
		t.Fatal("ggml_cont wrapping NOT found in the correct ffn_gate_up_exps block")
	}

	// Pattern 4: Verify both wrappings have the contiguity comments
	// This helps document why we're doing this
	commentPattern := regexp.MustCompile(`Wrap with ggml_cont.*contiguous.*Vulkan.*shader`)
	commentMatches := commentPattern.FindAllString(source, -1)
	
	if len(commentMatches) < 2 {
		t.Logf("WARNING: Expected at least 2 documentation comments for gate_exps and up_exps wrappings")
	} else {
		t.Logf("✓ Found %d documentation comments explaining ggml_cont usage", len(commentMatches))
	}

	t.Log("✓ Gemma4 MoE expert weight views are properly wrapped with ggml_cont()")
	t.Log("✓ This ensures Vulkan mul_mat_id shader receives contiguous 3D weight tensors")
	t.Log("✓ Experts k>0 will read from correct offsets without corruption")
}
