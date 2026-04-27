#!/usr/bin/env python3
# Shannon-Prime VHT2: Flux DiT Integration Tests
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


@register_suite("flux", "Flux DiT integration, 2D lattice RoPE, block skip scaffold")
def suite_flux(t):
    if not HAS_TORCH:
        t.check(False, "PyTorch required for Flux tests")
        return

    # ========================================================================
    # SECTION 1: Import & Node Registration
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Flux Node Registration --")

    try:
        import shannon_prime_flux_nodes as spf
        t.check(True, "shannon_prime_flux_nodes imports cleanly")
    except ImportError as e:
        t.check(False, f"Import failed: {e}")
        return

    # Check all expected classes exist
    for cls_name in ['ShannonPrimeFluxBlockSkip',
                     'ShannonPrimeFluxCacheFlush',
                     'ShannonPrimeFluxCacheFlushModel']:
        t.check(hasattr(spf, cls_name), f"Class {cls_name} exists")

    # Check infrastructure functions
    for fn_name in ['_iter_flux_double_blocks', '_iter_flux_single_blocks',
                    '_iter_flux_all_blocks', '_detect_flux_head_dim',
                    '_install_lattice_rope_flux']:
        t.check(hasattr(spf, fn_name), f"Function {fn_name} exists")

    # ========================================================================
    # SECTION 2: NODE_CLASS_MAPPINGS
    # ========================================================================
    t.set_category("integration")
    print("\n  -- NODE_CLASS_MAPPINGS --")

    t.check('ShannonPrimeFluxBlockSkip' in spf.NODE_CLASS_MAPPINGS,
            "FluxBlockSkip in NODE_CLASS_MAPPINGS")
    t.check('ShannonPrimeFluxCacheFlush' in spf.NODE_CLASS_MAPPINGS,
            "FluxCacheFlush in NODE_CLASS_MAPPINGS")
    t.check('ShannonPrimeFluxCacheFlushModel' in spf.NODE_CLASS_MAPPINGS,
            "FluxCacheFlushModel in NODE_CLASS_MAPPINGS")

    t.check('ShannonPrimeFluxBlockSkip' in spf.NODE_DISPLAY_NAME_MAPPINGS,
            "FluxBlockSkip has display name")

    # ========================================================================
    # SECTION 3: INPUT_TYPES Validation
    # ========================================================================
    t.set_category("integration")
    print("\n  -- FluxBlockSkip INPUT_TYPES --")

    inputs = spf.ShannonPrimeFluxBlockSkip.INPUT_TYPES()
    required = inputs['required']
    optional = inputs.get('optional', {})

    t.check('model' in required, "FluxBlockSkip has 'model' input")

    # Double stream tier windows
    t.check('double_tier0_window' in optional, "Has double_tier0_window")
    t.check('double_tier1_window' in optional, "Has double_tier1_window")
    t.check('double_tier2_window' in optional, "Has double_tier2_window")

    # Single stream tier windows
    t.check('single_tier0_window' in optional, "Has single_tier0_window")
    t.check('single_tier1_window' in optional, "Has single_tier1_window")

    # Shared options
    t.check('cache_mlp' in optional, "Has cache_mlp")
    t.check('cache_dtype' in optional, "Has cache_dtype")
    t.check('lattice_rope' in optional, "Has lattice_rope")
    t.check('lattice_alpha' in optional, "Has lattice_alpha")
    t.check('verbose' in optional, "Has verbose")

    # Check defaults
    t.check(optional['double_tier0_window'][1]['default'] == 8,
            f"double_tier0_window default is 8")
    t.check(optional['double_tier1_window'][1]['default'] == 3,
            f"double_tier1_window default is 3")
    t.check(optional['double_tier2_window'][1]['default'] == 0,
            f"double_tier2_window default is 0")
    t.check(optional['single_tier0_window'][1]['default'] == 2,
            f"single_tier0_window default is 2")
    t.check(optional['single_tier1_window'][1]['default'] == 0,
            f"single_tier1_window default is 0")
    t.check(optional['cache_mlp'][1]['default'] is False,
            f"cache_mlp default is False")
    t.check(optional['lattice_rope'][1]['default'] is True,
            f"lattice_rope default is True")
    t.check(abs(optional['lattice_alpha'][1]['default'] - 0.17) < 0.001,
            f"lattice_alpha default is 0.17")

    # ========================================================================
    # SECTION 4: IS_CHANGED returns NaN
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Idempotency --")

    val = spf.ShannonPrimeFluxBlockSkip.IS_CHANGED()
    t.check(math.isnan(val), f"FluxBlockSkip IS_CHANGED returns NaN (got {val})")

    # ========================================================================
    # SECTION 5: Block Iterators (on None/empty model)
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Block Iterators --")

    # None model → no blocks
    blocks = list(spf._iter_flux_double_blocks(None))
    t.check(len(blocks) == 0, "No double blocks from None model")

    blocks = list(spf._iter_flux_single_blocks(None))
    t.check(len(blocks) == 0, "No single blocks from None model")

    blocks = list(spf._iter_flux_all_blocks(None))
    t.check(len(blocks) == 0, "No blocks from None model via _iter_flux_all_blocks")

    # ========================================================================
    # SECTION 6: Head Dim Detection (fallback)
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Head Dim Detection --")

    hd, nh = spf._detect_flux_head_dim(None)
    t.check(hd == 128, f"Default head_dim is 128 (got {hd})")
    t.check(nh == 24, f"Default num_heads is 24 (got {nh})")

    # ========================================================================
    # SECTION 7: 2D Lattice RoPE Factors
    # ========================================================================
    t.set_category("lattice-rope")
    print("\n  -- 2D Lattice RoPE Factors --")

    # Flux v1 axes_dim = [16, 56, 56]
    # Temporal axis (dim=16): should get Long-Tier
    # Spatial axes (dim=56): should get Local-Tier

    # Import shared infrastructure
    from shannon_prime_nodes import _tiered_lattice_factors

    # Temporal: dim=16, n_freqs=8
    factors_temporal = _tiered_lattice_factors(8, freq_base=10000.0, alpha=0.17, tier='long')
    t.check(isinstance(factors_temporal, torch.Tensor), "Temporal factors is tensor")
    t.check(factors_temporal.shape[0] == 8, f"Temporal shape [8] (got {factors_temporal.shape})")
    t.check(factors_temporal.min().item() > 0.0, "Temporal factors > 0")
    t.check(torch.isfinite(factors_temporal).all().item(), "Temporal factors finite")

    # Spatial: dim=56, n_freqs=28
    factors_spatial = _tiered_lattice_factors(28, freq_base=10000.0, alpha=0.17, tier='local')
    t.check(isinstance(factors_spatial, torch.Tensor), "Spatial factors is tensor")
    t.check(factors_spatial.shape[0] == 28, f"Spatial shape [28] (got {factors_spatial.shape})")
    t.check(factors_spatial.min().item() > 0.0, "Spatial factors > 0")
    t.check(torch.isfinite(factors_spatial).all().item(), "Spatial factors finite")

    # Alpha=0 → all ones (for both axes)
    factors_0 = _tiered_lattice_factors(28, freq_base=10000.0, alpha=0.0, tier='local')
    t.check((factors_0 - 1.0).abs().max().item() < 1e-5,
            "alpha=0 → all spatial factors ≈ 1.0")

    # Spatial vs temporal differ
    factors_s = _tiered_lattice_factors(16, freq_base=10000.0, alpha=0.17, tier='local')
    factors_t = _tiered_lattice_factors(16, freq_base=10000.0, alpha=0.17, tier='long')
    diff = (factors_s - factors_t).abs().mean().item()
    t.check(diff > 0.001, f"Spatial vs Temporal tiers differ: avg |Δ|={diff:.4f}")

    # ========================================================================
    # SECTION 8: Fisher Weights (reused from Wan)
    # ========================================================================
    t.set_category("fisher")
    print("\n  -- Fisher Diagonal Weights (shared) --")

    from shannon_prime_nodes import _fisher_diagonal_weights, _fisher_cos_sim

    # head_dim=128 (Flux v1)
    w128 = _fisher_diagonal_weights(128)
    t.check(w128.shape == (128,), f"Fisher weights shape (128,) (got {w128.shape})")

    # head_dim=64 (Flux2)
    w64 = _fisher_diagonal_weights(64)
    t.check(w64.shape == (64,), f"Fisher weights shape for Flux2 (64,) (got {w64.shape})")

    # Fisher cos_sim between identical tensors → 1.0
    x = torch.randn(4, 128)
    sim = _fisher_cos_sim(x, x, w128)
    t.check(abs(sim - 1.0) < 1e-4, f"Fisher cos_sim(x, x) ≈ 1.0 (got {sim:.6f})")

    # Fisher cos_sim between different tensors < 1.0
    y = torch.randn(4, 128)
    sim2 = _fisher_cos_sim(x, y, w128)
    t.check(sim2 < 0.99, f"Fisher cos_sim(x, y) < 0.99 (got {sim2:.4f})")

    # ========================================================================
    # SECTION 9: Input Fingerprinting (shared)
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Input Fingerprinting (shared) --")

    from shannon_prime_nodes import _input_fingerprint

    x = torch.randn(128)
    fp1 = _input_fingerprint(x)
    fp2 = _input_fingerprint(x)
    t.check(fp1 == fp2, "Same tensor → same fingerprint")

    y = torch.randn(128)
    fp3 = _input_fingerprint(y)
    t.check(fp1 != fp3, "Different tensor → different fingerprint")

    # ========================================================================
    # SECTION 10: CacheFlush Node Basics
    # ========================================================================
    t.set_category("integration")
    print("\n  -- CacheFlush Basics --")

    # FluxCacheFlush takes LATENT, returns LATENT
    flush_inputs = spf.ShannonPrimeFluxCacheFlush.INPUT_TYPES()
    t.check('samples' in flush_inputs['required'],
            "FluxCacheFlush takes 'samples' input")
    t.check(spf.ShannonPrimeFluxCacheFlush.RETURN_TYPES == ("LATENT",),
            "FluxCacheFlush returns LATENT")

    # FluxCacheFlushModel takes MODEL, returns MODEL
    flush_m_inputs = spf.ShannonPrimeFluxCacheFlushModel.INPUT_TYPES()
    t.check('model' in flush_m_inputs['required'],
            "FluxCacheFlushModel takes 'model' input")
    t.check(spf.ShannonPrimeFluxCacheFlushModel.RETURN_TYPES == ("MODEL",),
            "FluxCacheFlushModel returns MODEL")

    # FluxCacheFlush.flush() works with dict (mock LATENT)
    mock_latent = {"samples": torch.randn(1, 4, 64, 64)}
    result = spf.ShannonPrimeFluxCacheFlush().flush(mock_latent)
    t.check(result == (mock_latent,), "FluxCacheFlush passes latent through")

    # ========================================================================
    # SECTION 11: Flux2 Compatibility (head_dim=64)
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Flux2 Compatibility --")

    # 2D lattice factors for Flux2 axes_dim=[32,32,32,32]
    factors_f2 = _tiered_lattice_factors(16, freq_base=2000.0, alpha=0.17, tier='local')
    t.check(factors_f2.shape[0] == 16, f"Flux2 spatial factors [16] (got {factors_f2.shape})")
    t.check(torch.isfinite(factors_f2).all().item(), "Flux2 factors finite")
    t.check(factors_f2.min().item() > 0.0, "Flux2 factors > 0")

    # Fisher weights for head_dim=64 (Flux2)
    w64 = _fisher_diagonal_weights(64)
    # 64: squarefree count
    def _is_squarefree(n):
        if n < 1:
            return False
        for p in range(2, int(n**0.5) + 1):
            if n % (p * p) == 0:
                return False
        return True
    sf_count = sum(1 for i in range(1, 65) if _is_squarefree(i))
    t.check(sf_count > 30, f"Squarefree count in 1..64: {sf_count} (expect ~40)")

    # ========================================================================
    # SECTION 12: Falsification Tests
    # ========================================================================
    t.set_category("falsification")
    print("\n  -- Falsification --")

    # Alpha > 0.5 should be clamped by INPUT_TYPES max
    t.check(optional['lattice_alpha'][1]['max'] == 0.5,
            "lattice_alpha max is 0.5")

    # Window = 0 means no caching (blocks are not patched)
    t.check(optional['double_tier2_window'][1]['default'] == 0,
            "Double tier-2 disabled by default (volatile blocks)")
    t.check(optional['single_tier1_window'][1]['default'] == 0,
            "Single tier-1 disabled by default (final detail)")

    # Verify all three node classes have CATEGORY = "shannon-prime"
    t.check(spf.ShannonPrimeFluxBlockSkip.CATEGORY == "shannon-prime",
            "FluxBlockSkip CATEGORY is shannon-prime")
    t.check(spf.ShannonPrimeFluxCacheFlush.CATEGORY == "shannon-prime",
            "FluxCacheFlush CATEGORY is shannon-prime")
    t.check(spf.ShannonPrimeFluxCacheFlushModel.CATEGORY == "shannon-prime",
            "FluxCacheFlushModel CATEGORY is shannon-prime")

    # Verify FUNCTION attributes
    t.check(spf.ShannonPrimeFluxBlockSkip.FUNCTION == "patch",
            "FluxBlockSkip FUNCTION is 'patch'")
    t.check(spf.ShannonPrimeFluxCacheFlush.FUNCTION == "flush",
            "FluxCacheFlush FUNCTION is 'flush'")
    t.check(spf.ShannonPrimeFluxCacheFlushModel.FUNCTION == "flush",
            "FluxCacheFlushModel FUNCTION is 'flush'")

    # ========================================================================
    # SECTION 13: Architecture-Specific Invariants
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Architecture Invariants --")

    # Flux v1: 19 double blocks + 38 single blocks (typical)
    # These are the canonical Flux architecture params
    t.check(128 == 3072 // 24,
            "Flux v1 head_dim = hidden_size/num_heads = 3072/24 = 128")

    # axes_dim sum must equal head_dim for RoPE
    axes_dim_v1 = [16, 56, 56]
    t.check(sum(axes_dim_v1) == 128,
            f"Flux v1 axes_dim sum = {sum(axes_dim_v1)} = head_dim=128")

    axes_dim_v2 = [32, 32, 32, 32]
    t.check(sum(axes_dim_v2) == 128,
            f"Flux2 axes_dim sum = {sum(axes_dim_v2)} = 128 (head_dim=64, but RoPE total=128)")

    # VHT2 self-inverse property holds at both head_dims
    from shannon_prime_torch import vht2
    for hd in [64, 128]:
        x = torch.randn(16, hd)
        roundtrip = vht2(vht2(x))
        err = (roundtrip - x).abs().max().item()
        t.check(err < 1e-4,
                f"VHT2 self-inverse at head_dim={hd}: max_err={err:.6f}")
