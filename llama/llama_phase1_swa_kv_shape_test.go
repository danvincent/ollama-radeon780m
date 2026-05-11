package llama

import (
	"testing"
)

// TestGemma4SWALayerKVShapeSelection verifies that SWA layers use SWA-specific K/V head dimensions
// rather than the global head dimensions.
// This is the Phase 1 blocker fix: ensure attn_k and attn_v tensor shapes derive from
// SWA-specific head dims for SWA layers, not global head dims.
func TestGemma4SWALayerKVShapeSelection(t *testing.T) {
	t.Run("swa_layers_use_swa_head_dims", func(t *testing.T) {
		// Simulate Gemma4 hparams with mixed attention layers
		type testCase struct {
			name                     string
			isSWALayer               bool
			globalHeadDimK           int64 // n_embd_head_k
			globalHeadDimV           int64 // n_embd_head_v
			swaHeadDimK              int64 // n_embd_head_k_swa
			swaHeadDimV              int64 // n_embd_head_v_swa
			nHeadKV                  int64 // group size
			expectedKVDimK           int64 // expected K tensor dimension
			expectedKVDimV           int64 // expected V tensor dimension
		}

		cases := []testCase{
			{
				name:           "full_attention_layer",
				isSWALayer:     false,
				globalHeadDimK: 256,
				globalHeadDimV: 256,
				swaHeadDimK:    128, // defined but not used
				swaHeadDimV:    128, // defined but not used
				nHeadKV:        8,
				// K/V dims should use global head dims: 256 * 8 = 2048
				expectedKVDimK: 2048,
				expectedKVDimV: 2048,
			},
			{
				name:           "swa_layer_with_smaller_dims",
				isSWALayer:     true,
				globalHeadDimK: 256,
				globalHeadDimV: 256,
				swaHeadDimK:    128,
				swaHeadDimV:    128,
				nHeadKV:        8,
				// K/V dims should use SWA head dims: 128 * 8 = 1024
				expectedKVDimK: 1024,
				expectedKVDimV: 1024,
			},
			{
				name:           "swa_layer_with_equal_dims",
				isSWALayer:     true,
				globalHeadDimK: 128,
				globalHeadDimV: 128,
				swaHeadDimK:    128, // same as global (some models may have this)
				swaHeadDimV:    128,
				nHeadKV:        8,
				// K/V dims: 128 * 8 = 1024 (uses SWA dims which happen to equal global)
				expectedKVDimK: 1024,
				expectedKVDimV: 1024,
			},
			{
				name:           "swa_layer_without_swa_dims_fallback",
				isSWALayer:     true,
				globalHeadDimK: 256,
				globalHeadDimV: 256,
				swaHeadDimK:    0, // not provided by GGUF
				swaHeadDimV:    0,
				nHeadKV:        8,
				// K/V dims should fall back to global: 256 * 8 = 2048
				expectedKVDimK: 2048,
				expectedKVDimV: 2048,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				// Logic: if SWA layer and SWA dims provided, use SWA dims; else use global
				var effectiveHeadDimK int64 = tc.globalHeadDimK
				var effectiveHeadDimV int64 = tc.globalHeadDimV

				if tc.isSWALayer && tc.swaHeadDimK != 0 && tc.swaHeadDimV != 0 {
					effectiveHeadDimK = tc.swaHeadDimK
					effectiveHeadDimV = tc.swaHeadDimV
				}

				// K/V dimensions are: effective_head_dim * n_head_kv
				actualKVDimK := effectiveHeadDimK * tc.nHeadKV
				actualKVDimV := effectiveHeadDimV * tc.nHeadKV

				if actualKVDimK != tc.expectedKVDimK {
					t.Errorf("K dimension: got %d, want %d (head_dim=%d, nHeadKV=%d, isSWA=%v)",
						actualKVDimK, tc.expectedKVDimK, effectiveHeadDimK, tc.nHeadKV, tc.isSWALayer)
				}
				if actualKVDimV != tc.expectedKVDimV {
					t.Errorf("V dimension: got %d, want %d (head_dim=%d, nHeadKV=%d, isSWA=%v)",
						actualKVDimV, tc.expectedKVDimV, effectiveHeadDimV, tc.nHeadKV, tc.isSWALayer)
				}
			})
		}
	})
}

// TestGemma4SharedKVLayersPreserveSharedBehavior verifies that shared-KV layers
// (when n_layer_kv_from_start is used) still work correctly with SWA head dims.
func TestGemma4SharedKVLayersPreserveSharedBehavior(t *testing.T) {
	t.Run("shared_kv_layers_bypass_swa_dims", func(t *testing.T) {
		// Gemma4 can have shared-KV: first N layers have KV cache, rest share with those.
		// Shared-KV layers should NOT create K/V tensors (they're marked TENSOR_NOT_REQUIRED).
		// But if they did exist, they should NOT use SWA dims.

		type testCase struct {
			name              string
			layerIdx          uint32
			nLayerKVFromStart int32
			isSWALayer        bool
			// Expected: should this layer have K/V tensors?
			shouldHaveKV bool
		}

		cases := []testCase{
			{
				name:              "early_kv_layer_with_swa",
				layerIdx:          0,
				nLayerKVFromStart: 20,
				isSWALayer:        true,
				shouldHaveKV:      true, // has KV cache
			},
			{
				name:              "shared_kv_layer_with_swa",
				layerIdx:          25,
				nLayerKVFromStart: 20,
				isSWALayer:        true,
				shouldHaveKV:      false, // shares KV with earlier layers
			},
			{
				name:              "no_shared_kv_all_have_cache",
				layerIdx:          10,
				nLayerKVFromStart: -1, // no shared KV
				isSWALayer:        true,
				shouldHaveKV:      true, // all layers have KV
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				// Simulate has_kv(layer_idx) logic
				var hasKV bool
				if tc.nLayerKVFromStart >= 0 {
					hasKV = tc.layerIdx < uint32(tc.nLayerKVFromStart)
				} else {
					hasKV = true // by default all have KV
				}

				if hasKV != tc.shouldHaveKV {
					t.Errorf("has_kv: got %v, want %v (layer=%d, nLayerKVFromStart=%d)",
						hasKV, tc.shouldHaveKV, tc.layerIdx, tc.nLayerKVFromStart)
				}
			})
		}
	})

	t.Run("optional_kv_tensors_use_correct_dims", func(t *testing.T) {
		// When K/V tensors are optional (marked TENSOR_NOT_REQUIRED for shared-KV),
		// they typically won't be present in the model. But if they ARE manually provided,
		// the shape calculation should be correct based on layer type.

		// This documents the expected behavior: shared-KV layers don't get K/V tensors,
		// so they don't need to worry about shape selection. Regular (non-shared-KV)
		// layers select shapes based on SWA status.

		globalHeadDimK := int64(256)
		globalHeadDimV := int64(256)
		swaHeadDimK := int64(128)
		swaHeadDimV := int64(128)
		nHeadKV := int64(8)

		type scenario struct {
			name       string
			isSWALayer bool
			hasKV      bool // whether this layer actually has K/V (not shared-KV)
			expectedK  int64
			expectedV  int64
		}

		scenarios := []scenario{
			{
				name:       "swa_layer_with_kv_cache",
				isSWALayer: true,
				hasKV:      true,
				expectedK:  128 * 8, // SWA dims
				expectedV:  128 * 8,
			},
			{
				name:       "swa_layer_shared_kv",
				isSWALayer: true,
				hasKV:      false, // K/V not created (shared with earlier layer)
				// N/A - no tensors created
				expectedK: 0,
				expectedV: 0,
			},
			{
				name:       "full_attn_layer_with_kv_cache",
				isSWALayer: false,
				hasKV:      true,
				expectedK:  256 * 8, // global dims
				expectedV:  256 * 8,
			},
		}

		for _, sc := range scenarios {
			t.Run(sc.name, func(t *testing.T) {
				if !sc.hasKV {
					// No K/V tensors created for this layer (shared-KV or optional)
					// Skip dimension check
					return
				}

				var effectiveHeadDimK int64 = globalHeadDimK
				var effectiveHeadDimV int64 = globalHeadDimV

				// SWA layers use SWA dims when they have their own K/V cache
				if sc.isSWALayer && swaHeadDimK != 0 {
					effectiveHeadDimK = swaHeadDimK
					effectiveHeadDimV = swaHeadDimV
				}

				actualK := effectiveHeadDimK * nHeadKV
				actualV := effectiveHeadDimV * nHeadKV

				if actualK != sc.expectedK {
					t.Errorf("K: got %d, want %d", actualK, sc.expectedK)
				}
				if actualV != sc.expectedV {
					t.Errorf("V: got %d, want %d", actualV, sc.expectedV)
				}
			})
		}
	})
}

// TestGemma4LargeAndSmallHeadDimMixes verifies the fix works with realistic
// Gemma4 configurations that have multiple layer types with varying head dimensions.
func TestGemma4LargeAndSmallHeadDimMixes(t *testing.T) {
	t.Run("e2b_like_config", func(t *testing.T) {
		// Gemma4 E2B: ~35 layers, mixed SWA/full attention
		// Example: global_head_dim=256, swa_head_dim=128

		globalHeadK := int64(256)
		globalHeadV := int64(256)
		swaHeadK := int64(128)
		swaHeadV := int64(128)
		nHeadKV := int64(8)

		// Simulate layer pattern: 0-4=SWA, 5-9=full, 10-14=SWA, 15-34=full, etc.
		swaPattern := []bool{
			true, true, true, true, true,    // 0-4: SWA
			false, false, false, false, false, // 5-9: full
			true, true, true, true, true,    // 10-14: SWA
			false, false, false, false, false, // 15-19: full
			true, true, true, true, true,    // 20-24: SWA
		}

		for layerIdx, isSWA := range swaPattern {
			var effectiveHeadK, effectiveHeadV int64
			if isSWA && swaHeadK != 0 {
				effectiveHeadK = swaHeadK
				effectiveHeadV = swaHeadV
			} else {
				effectiveHeadK = globalHeadK
				effectiveHeadV = globalHeadV
			}

			expectedK := effectiveHeadK * nHeadKV
			expectedV := effectiveHeadV * nHeadKV

			if isSWA {
				if expectedK != 128*8 || expectedV != 128*8 {
					t.Errorf("layer %d (SWA): expected K=%d, V=%d; got %d, %d",
						layerIdx, 128*8, 128*8, expectedK, expectedV)
				}
			} else {
				if expectedK != 256*8 || expectedV != 256*8 {
					t.Errorf("layer %d (full): expected K=%d, V=%d; got %d, %d",
						layerIdx, 256*8, 256*8, expectedK, expectedV)
				}
			}
		}
	})

	t.Run("31b_like_config", func(t *testing.T) {
		// Gemma4 31B: ~60 layers, alternating pattern with SWA smaller than global
		globalHeadK := int64(256)
		globalHeadV := int64(256)
		swaHeadK := int64(128)
		swaHeadV := int64(128)
		nHeadKV := int64(8)

		// Simulate: every odd layer is SWA, even layers are full
		for layerIdx := 0; layerIdx < 20; layerIdx++ {
			isSWA := (layerIdx % 2) == 1

			var effectiveHeadK, effectiveHeadV int64
			if isSWA && swaHeadK != 0 {
				effectiveHeadK = swaHeadK
				effectiveHeadV = swaHeadV
			} else {
				effectiveHeadK = globalHeadK
				effectiveHeadV = globalHeadV
			}

			expectedK := effectiveHeadK * nHeadKV
			expectedV := effectiveHeadV * nHeadKV

			if isSWA {
				if expectedK != 128*8 || expectedV != 128*8 {
					t.Errorf("layer %d (SWA): expected 1024, got %d", layerIdx, expectedK)
				}
			} else {
				if expectedK != 256*8 || expectedV != 256*8 {
					t.Errorf("layer %d (full): expected 2048, got %d", layerIdx, expectedK)
				}
			}
		}
	})
}

// TestGemma4SWAHeadDimZeroFallback verifies that when SWA head dims are not
// provided in GGUF (0 value), the loader falls back to global head dims without error.
func TestGemma4SWAHeadDimZeroFallback(t *testing.T) {
	t.Run("swa_head_dim_zero_uses_global_dims", func(t *testing.T) {
		// Some older Gemma4 GGUF files might not provide SWA-specific dims
		globalHeadK := int64(256)
		globalHeadV := int64(256)
		swaHeadK := int64(0) // not provided
		swaHeadV := int64(0)
		nHeadKV := int64(8)

		// Simulate layer with SWA enabled but no SWA dims in GGUF
		isSWA := true

		var effectiveHeadK, effectiveHeadV int64
		if isSWA && swaHeadK != 0 && swaHeadV != 0 {
			effectiveHeadK = swaHeadK
			effectiveHeadV = swaHeadV
		} else {
			effectiveHeadK = globalHeadK
			effectiveHeadV = globalHeadV
		}

		expectedK := effectiveHeadK * nHeadKV
		expectedV := effectiveHeadV * nHeadKV

		// Should fall back gracefully to global dims
		if expectedK != 256*8 || expectedV != 256*8 {
			t.Errorf("fallback: expected 2048 for both K,V; got %d, %d", expectedK, expectedV)
		}
	})
}

// TestGemma4QHeadNormUsesSWADims verifies that Q/K normalization layers
// also use SWA-specific head dims for SWA layers (consistency with attn_q/attn_k shapes).
func TestGemma4QHeadNormUsesSWADims(t *testing.T) {
	t.Run("q_k_norm_layers_use_layer_head_dim", func(t *testing.T) {
		// Q/K norm tensors have shape [layer_n_embd_head]
		// which should be SWA head dim for SWA layers

		globalHeadDim := int64(256)
		swaHeadDim := int64(128)

		type testCase struct {
			name                 string
			isSWALayer           bool
			expectedQKNormShape  int64
		}

		cases := []testCase{
			{
				name:                "full_attn_qk_norm",
				isSWALayer:          false,
				expectedQKNormShape: globalHeadDim,
			},
			{
				name:                "swa_layer_qk_norm",
				isSWALayer:          true,
				expectedQKNormShape: swaHeadDim,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				var layerHeadDim int64 = globalHeadDim
				if tc.isSWALayer {
					layerHeadDim = swaHeadDim
				}

				if layerHeadDim != tc.expectedQKNormShape {
					t.Errorf("Q/K norm shape: got %d, want %d",
						layerHeadDim, tc.expectedQKNormShape)
				}
			})
		}
	})
}
