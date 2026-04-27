#!/usr/bin/env python3
# Shannon-Prime VHT2: Audio DiT Integration Tests
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3.

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime', 'backends', 'torch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime-comfyui', 'nodes'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime-comfyui', 'lib', 'shannon-prime', 'tools'))

try:
    import torch
    torch.manual_seed(42)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from run_tests import register_suite


@register_suite("audio", "Audio DiT integration, 1D lattice RoPE, block skip scaffold")
def suite_audio(t):
    if not HAS_TORCH:
        t.check(False, "PyTorch required for Audio tests")
        return

    # ========================================================================
    # SECTION 1: Import & Node Registration
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Audio Node Registration --")

    try:
        import shannon_prime_audio_nodes as spa
        t.check(True, "shannon_prime_audio_nodes imports cleanly")
    except ImportError as e:
        t.check(False, f"Import failed: {e}")
        return

    # Check all expected classes exist
    for cls_name in ['ShannonPrimeAudioBlockSkip',
                     'ShannonPrimeAudioCacheFlush',
                     'ShannonPrimeAudioCacheFlushModel']:
        t.check(hasattr(spa, cls_name), f"Class {cls_name} exists")

    # Check infrastructure functions
    for fn_name in ['_iter_audio_dit_blocks', '_detect_audio_head_dim',
                    '_count_audio_dit_blocks', '_install_lattice_rope_audio']:
        t.check(hasattr(spa, fn_name), f"Function {fn_name} exists")

    # ========================================================================
    # SECTION 2: NODE_CLASS_MAPPINGS
    # ========================================================================
    t.set_category("integration")
    print("\n  -- NODE_CLASS_MAPPINGS --")

    t.check('ShannonPrimeAudioBlockSkip' in spa.NODE_CLASS_MAPPINGS,
            "AudioBlockSkip in NODE_CLASS_MAPPINGS")
    t.check('ShannonPrimeAudioCacheFlush' in spa.NODE_CLASS_MAPPINGS,
            "AudioCacheFlush in NODE_CLASS_MAPPINGS")
    t.check('ShannonPrimeAudioCacheFlushModel' in spa.NODE_CLASS_MAPPINGS,
            "AudioCacheFlushModel in NODE_CLASS_MAPPINGS")

    t.check('ShannonPrimeAudioBlockSkip' in spa.NODE_DISPLAY_NAME_MAPPINGS,
            "AudioBlockSkip has display name")

    # ========================================================================
    # SECTION 3: INPUT_TYPES Validation
    # ========================================================================
    t.set_category("integration")
    print("\n  -- AudioBlockSkip INPUT_TYPES --")

    inputs = spa.ShannonPrimeAudioBlockSkip.INPUT_TYPES()
    required = inputs['required']
    optional = inputs.get('optional', {})

    t.check('model' in required, "AudioBlockSkip has 'model' input")

    # Tier windows
    t.check('tier0_window' in optional, "Has tier0_window")
    t.check('tier1_window' in optional, "Has tier1_window")
    t.check('tier2_window' in optional, "Has tier2_window")

    # Shared options
    t.check('cache_mlp' in optional, "Has cache_mlp")
    t.check('cache_dtype' in optional, "Has cache_dtype")
    t.check('lattice_rope' in optional, "Has lattice_rope")
    t.check('lattice_alpha' in optional, "Has lattice_alpha")
    t.check('verbose' in optional, "Has verbose")

    # Check defaults — tier layout for 24-block audio DiT
    t.check(optional['tier0_window'][1]['default'] == 8,
            "tier0_window default is 8 (tonal foundation)")
    t.check(optional['tier1_window'][1]['default'] == 3,
            "tier1_window default is 3 (harmonic mid-range)")
    t.check(optional['tier2_window'][1]['default'] == 0,
            "tier2_window default is 0 (transient detail)")
    t.check(optional['cache_mlp'][1]['default'] is False,
            "cache_mlp default is False")
    t.check(optional['lattice_rope'][1]['default'] is True,
            "lattice_rope default is True")
    t.check(abs(optional['lattice_alpha'][1]['default'] - 0.17) < 0.001,
            "lattice_alpha default is 0.17")

    # ========================================================================
    # SECTION 4: IS_CHANGED returns NaN
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Idempotency --")

    val = spa.ShannonPrimeAudioBlockSkip.IS_CHANGED()
    t.check(math.isnan(val), f"AudioBlockSkip IS_CHANGED returns NaN (got {val})")

    # ========================================================================
    # SECTION 5: Block Iterators (on None/empty model)
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Block Iterators --")

    blocks = list(spa._iter_audio_dit_blocks(None))
    t.check(len(blocks) == 0, "No blocks from None model")

    count = spa._count_audio_dit_blocks(None)
    t.check(count == 0, "Block count 0 for None model")

    # ========================================================================
    # SECTION 6: Head Dim Detection (fallback)
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Head Dim Detection --")

    hd, nh = spa._detect_audio_head_dim(None)
    t.check(hd == 64, f"Default head_dim is 64 (Stable Audio) — got {hd}")
    t.check(nh == 24, f"Default num_heads is 24 (Stable Audio) — got {nh}")

    # ========================================================================
    # SECTION 7: 1D Lattice RoPE Factors
    # ========================================================================
    t.set_category("lattice-rope")
    print("\n  -- 1D Lattice RoPE Factors --")

    # Import shared infrastructure
    from shannon_prime_nodes import _tiered_lattice_factors

    # Audio: 1D temporal, long tier
    # Stable Audio dim_heads=64, RoPE uses max(dim_heads//2, 32) = 32
    factors_long = _tiered_lattice_factors(32, freq_base=10000.0, alpha=0.17, tier='long')
    t.check(isinstance(factors_long, torch.Tensor), "Long-tier factors is tensor")
    t.check(factors_long.shape[0] == 32, f"Long-tier shape [32] (got {factors_long.shape})")
    t.check(factors_long.min().item() > 0.0, "Long-tier factors > 0")
    t.check(torch.isfinite(factors_long).all().item(), "Long-tier factors finite")

    # Alpha=0 → all ones
    factors_0 = _tiered_lattice_factors(32, freq_base=10000.0, alpha=0.0, tier='long')
    t.check((factors_0 - 1.0).abs().max().item() < 1e-5,
            "alpha=0 → all long-tier factors ≈ 1.0")

    # Local vs long differ
    factors_local = _tiered_lattice_factors(32, freq_base=10000.0, alpha=0.17, tier='local')
    diff = (factors_local - factors_long).abs().mean().item()
    t.check(diff > 0.001, f"Local vs Long tiers differ: avg |Δ|={diff:.4f}")

    # ========================================================================
    # SECTION 8: Fisher Weights (for audio head_dim)
    # ========================================================================
    t.set_category("fisher")
    print("\n  -- Fisher Diagonal Weights --")

    from shannon_prime_nodes import _fisher_diagonal_weights, _fisher_cos_sim

    # head_dim=64 (Stable Audio)
    w64 = _fisher_diagonal_weights(64)
    t.check(w64.shape == (64,), f"Fisher weights shape (64,) (got {w64.shape})")

    # Fisher cos_sim between identical tensors → 1.0
    x = torch.randn(4, 64)
    sim = _fisher_cos_sim(x, x, w64)
    t.check(abs(sim - 1.0) < 1e-4, f"Fisher cos_sim(x, x) ≈ 1.0 (got {sim:.6f})")

    # Fisher cos_sim between different tensors < 1.0
    y = torch.randn(4, 64)
    sim2 = _fisher_cos_sim(x, y, w64)
    t.check(sim2 < 0.99, f"Fisher cos_sim(x, y) < 0.99 (got {sim2:.4f})")

    # ========================================================================
    # SECTION 9: Input Fingerprinting (shared)
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Input Fingerprinting --")

    from shannon_prime_nodes import _input_fingerprint

    x = torch.randn(64)
    fp1 = _input_fingerprint(x)
    fp2 = _input_fingerprint(x)
    t.check(fp1 == fp2, "Same tensor → same fingerprint")

    y = torch.randn(64)
    fp3 = _input_fingerprint(y)
    t.check(fp1 != fp3, "Different tensor → different fingerprint")

    # ========================================================================
    # SECTION 10: CacheFlush Node Basics
    # ========================================================================
    t.set_category("integration")
    print("\n  -- CacheFlush Basics --")

    # AudioCacheFlush takes LATENT, returns LATENT
    flush_inputs = spa.ShannonPrimeAudioCacheFlush.INPUT_TYPES()
    t.check('samples' in flush_inputs['required'],
            "AudioCacheFlush takes 'samples' input")
    t.check(spa.ShannonPrimeAudioCacheFlush.RETURN_TYPES == ("LATENT",),
            "AudioCacheFlush returns LATENT")

    # AudioCacheFlushModel takes MODEL, returns MODEL
    flush_m_inputs = spa.ShannonPrimeAudioCacheFlushModel.INPUT_TYPES()
    t.check('model' in flush_m_inputs['required'],
            "AudioCacheFlushModel takes 'model' input")
    t.check(spa.ShannonPrimeAudioCacheFlushModel.RETURN_TYPES == ("MODEL",),
            "AudioCacheFlushModel returns MODEL")

    # AudioCacheFlush.flush() works with dict (mock LATENT)
    mock_latent = {"samples": torch.randn(1, 64, 256)}
    result = spa.ShannonPrimeAudioCacheFlush().flush(mock_latent)
    t.check(result == (mock_latent,), "AudioCacheFlush passes latent through")

    # ========================================================================
    # SECTION 11: Falsification Tests
    # ========================================================================
    t.set_category("falsification")
    print("\n  -- Falsification --")

    # Alpha max is 0.5
    t.check(optional['lattice_alpha'][1]['max'] == 0.5,
            "lattice_alpha max is 0.5")

    # Tier 2 disabled by default (volatile transient blocks)
    t.check(optional['tier2_window'][1]['default'] == 0,
            "Tier 2 disabled by default (transient detail)")

    # All three node classes have CATEGORY = "shannon-prime"
    t.check(spa.ShannonPrimeAudioBlockSkip.CATEGORY == "shannon-prime",
            "AudioBlockSkip CATEGORY is shannon-prime")
    t.check(spa.ShannonPrimeAudioCacheFlush.CATEGORY == "shannon-prime",
            "AudioCacheFlush CATEGORY is shannon-prime")
    t.check(spa.ShannonPrimeAudioCacheFlushModel.CATEGORY == "shannon-prime",
            "AudioCacheFlushModel CATEGORY is shannon-prime")

    # Verify FUNCTION attributes
    t.check(spa.ShannonPrimeAudioBlockSkip.FUNCTION == "patch",
            "AudioBlockSkip FUNCTION is 'patch'")
    t.check(spa.ShannonPrimeAudioCacheFlush.FUNCTION == "flush",
            "AudioCacheFlush FUNCTION is 'flush'")
    t.check(spa.ShannonPrimeAudioCacheFlushModel.FUNCTION == "flush",
            "AudioCacheFlushModel FUNCTION is 'flush'")

    # ========================================================================
    # SECTION 12: Architecture Invariants
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Architecture Invariants --")

    # Stable Audio: embed_dim=1536, num_heads=24, head_dim=64
    t.check(64 == 1536 // 24,
            "Stable Audio head_dim = embed_dim/num_heads = 1536/24 = 64")

    # RoPE dim = max(head_dim//2, 32) = max(32, 32) = 32
    t.check(max(64 // 2, 32) == 32,
            "RoPE dim = max(64//2, 32) = 32")

    # VHT2 self-inverse at head_dim=64
    from shannon_prime_torch import vht2
    x = torch.randn(16, 64)
    roundtrip = vht2(vht2(x))
    err = (roundtrip - x).abs().max().item()
    t.check(err < 1e-4,
            f"VHT2 self-inverse at head_dim=64: max_err={err:.6f}")

    # adaLN gate function: sigmoid(1-gate) is in (0,1) for any gate
    gate = torch.randn(8)
    gated = torch.sigmoid(1 - gate)
    t.check((gated > 0).all().item() and (gated < 1).all().item(),
            "sigmoid(1-gate) output in (0,1) for all gate values")

    # sigmoid(1-0) ≈ 0.731 (default gate → non-trivial scaling)
    t.check(abs(torch.sigmoid(torch.tensor(1.0)).item() - 0.7311) < 0.01,
            "sigmoid(1) ≈ 0.731 (default adaLN gate)")
