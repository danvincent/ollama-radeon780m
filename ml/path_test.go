package ml

import (
	"os"
	"path/filepath"
	"testing"
)

// TestResolveLibraryPathResolvesToRepoBuild tests that ResolveLibraryPath correctly
// identifies and resolves to repo-root build/lib/ollama in development scenarios.
func TestResolveLibraryPathResolvesToRepoBuild(t *testing.T) {
	// Save current directory
	originalCwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get current directory: %v", err)
	}
	defer os.Chdir(originalCwd)

	// Find repo root from current directory
	repoRoot := findRepoRoot(originalCwd)
	if repoRoot == "" {
		t.Skip("Could not find repo root; skipping development scenario test")
	}

	expectedPath := filepath.Join(repoRoot, "build", "lib", "ollama")

	// Check if the path exists
	if _, err := os.Stat(expectedPath); err != nil {
		if os.IsNotExist(err) {
			t.Skipf("Expected development build path does not exist: %s", expectedPath)
		}
		t.Fatalf("Error checking expected path: %v", err)
	}

	// Test from original directory
	resolved := ResolveLibraryPath()
	if resolved != expectedPath {
		t.Errorf("ResolveLibraryPath from original cwd: expected %s, got %s",
			expectedPath, resolved)
	}
	t.Logf("✓ ResolveLibraryPath correctly resolved to repo-root build path from original cwd")

	// Test from subdirectory - this is important for go test which often changes to test directories
	testSubdir := filepath.Join(repoRoot, "ml", "backend", "ggml")
	if err := os.Chdir(testSubdir); err != nil {
		t.Skipf("Could not change to test subdirectory %s: %v", testSubdir, err)
	}

	resolved = ResolveLibraryPath()
	if resolved != expectedPath {
		t.Errorf("ResolveLibraryPath from ml/backend/ggml subdir: expected %s, got %s",
			expectedPath, resolved)
	}
	t.Logf("✓ ResolveLibraryPath correctly resolved to repo-root build path from subdirectory")

	// Test that test override works
	testOverridePath := "/test/override/path"
	SetLibraryPathForTest(testOverridePath)
	defer ClearLibraryPathOverride()

	resolved = ResolveLibraryPath()
	if resolved != testOverridePath {
		t.Errorf("ResolveLibraryPath with test override: expected %s, got %s",
			testOverridePath, resolved)
	}
	t.Logf("✓ ResolveLibraryPath respects test override")
}

// TestFindRepoRoot tests that findRepoRoot correctly locates repo root
func TestFindRepoRoot(t *testing.T) {
	// Get current directory
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get current directory: %v", err)
	}

	// Find repo root
	repoRoot := findRepoRoot(cwd)
	if repoRoot == "" {
		t.Fatal("findRepoRoot returned empty string")
	}

	// Verify it has go.mod or .git
	goModPath := filepath.Join(repoRoot, "go.mod")
	gitPath := filepath.Join(repoRoot, ".git")

	if _, err := os.Stat(goModPath); err == nil {
		t.Logf("✓ Found go.mod at repo root: %s", repoRoot)
	} else if _, err := os.Stat(gitPath); err == nil {
		t.Logf("✓ Found .git at repo root: %s", repoRoot)
	} else {
		t.Fatalf("Repo root %s has neither go.mod nor .git", repoRoot)
	}
}

// TestLibOllamaPathInitializes tests that LibOllamaPath is properly initialized
// This is the module-level variable that should be set correctly on module init
func TestLibOllamaPathInitializes(t *testing.T) {
	if LibOllamaPath == "" {
		t.Fatal("LibOllamaPath is empty; module initialization may have failed")
	}

	// Verify it exists
	fi, err := os.Stat(LibOllamaPath)
	if err != nil {
		t.Fatalf("LibOllamaPath does not exist: %s", LibOllamaPath)
	}

	if !fi.IsDir() {
		t.Fatalf("LibOllamaPath is not a directory: %s", LibOllamaPath)
	}

	t.Logf("✓ LibOllamaPath initialized correctly: %s", LibOllamaPath)
}
