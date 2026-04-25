#!/usr/bin/env python3
# Shannon-Prime VHT2: ComfyUI Integration + Video Bench Tests
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


@register_suite("comfyui", "ComfyUI integration, caching, lattice RoPE")
def suite_comfyui(t):
    if not HAS_TORCH:
        t.check(False, "PyTorch required for ComfyUI tests")
        return

    # ========================================================================
    # SECTION 1: Import & Node Registration
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Node Registration --")

    try:
        import shannon_prime_nodes as spn
        t.check(True, "shannon_prime_nodes imports cleanly")
    except ImportError as e:
        t.check(False, f"Import failed: {e}")
        return

    # Check all expected classes exist
    for cls_name in ['ShannonPrimeWanCache', 'ShannonPrimeWanCacheStats',
                     'ShannonPrimeWanCacheSqfree', 'ShannonPrimeWanCacheFlush']:
        t.check(hasattr(spn, cls_name), f"Class {cls_name} exists")

    # Check BlockSkip if present
    has_blockskip = hasattr(spn, 'ShannonPrimeWanBlockSkip')
    t.check(has_blockskip, "ShannonPrimeWanBlockSkip exists")

    # ========================================================================
    # SECTION 2: INPUT_TYPES Validation
    # ========================================================================
    t.set_category("integration")
    print("\n  -- INPUT_TYPES Validation --")

    wan_inputs = spn.ShannonPrimeWanCache.INPUT_TYPES()
    required = wan_inputs['required']
    t.check('model' in required, "WanCache has 'model' input")
    t.check('k_bits' in required, "WanCache has 'k_bits' input")
    t.check('v_bits' in required, "WanCache has 'v_bits' input")
    t.check('use_mobius' in required, "WanCache has 'use_mobius' input")
    t.check('lattice_rope' in required, "WanCache has 'lattice_rope' input")
    t.check('lattice_alpha' in required, "WanCache has 'lattice_alpha' input")

    # Check defaults
    t.check(required['k_bits'][1]['default'] == "4,3,3,3",
            f"k_bits default is '4,3,3,3' (got '{required['k_bits'][1]['default']}')")
    t.check(required['v_bits'][1]['default'] == "4,3,3,3",
            f"v_bits default is '4,3,3,3'")
    t.check(required['use_mobius'][1]['default'] is False,
            f"use_mobius default is False")
    t.check(required['lattice_rope'][1]['default'] is True,
            f"lattice_rope default is True")
    t.check(abs(required['lattice_alpha'][1]['default'] - 0.17) < 0.001,
            f"lattice_alpha default is 0.17")

    # ========================================================================
    # SECTION 3: Lattice RoPE Helper Functions
    # ========================================================================
    t.set_category("lattice-rope")
    print("\n  -- Lattice RoPE Helpers --")

    # Test sieve
    t.check(hasattr(spn, '_sieve_primes'), "_sieve_primes exists")
    primes = spn._sieve_primes(1000)
    t.check(len(primes) == 168, f"_sieve_primes(1000) returns 168 primes (got {len(primes)})")
    t.check(primes[0] == 2 and primes[-1] == 997, "First=2, last=997")

    # Test pick_evenly
    t.check(hasattr(spn, '_pick_evenly'), "_pick_evenly exists")
    pool = list(range(100))
    picked = spn._pick_evenly(pool, 10)
    t.check(len(picked) == 10, f"_pick_evenly returns 10 items (got {len(picked)})")
    t.check(picked[0] == 0 and picked[-1] == pool[90], "Evenly spaced from pool")

    # Test tiered lattice factors
    t.check(hasattr(spn, '_tiered_lattice_factors'), "_tiered_lattice_factors exists")

    factors_auto = spn._tiered_lattice_factors(64, freq_base=10000.0, alpha=0.17, tier='auto')
    t.check(isinstance(factors_auto, torch.Tensor), "Returns a tensor")
    t.check(factors_auto.shape[0] == 64, f"Shape is [64] (got {factors_auto.shape})")
    t.check(factors_auto.min().item() > 0.0, "All factors > 0")
    t.check(torch.isfinite(factors_auto).all().item(), "All factors are finite")

    # Alpha=0 → all ones
    factors_0 = spn._tiered_lattice_factors(64, freq_base=10000.0, alpha=0.0, tier='auto')
    t.check((factors_0 - 1.0).abs().max().item() < 1e-5, "alpha=0 → all factors ≈ 1.0")

    # Tier-specific: long tier should have different distribution than local
    factors_long = spn._tiered_lattice_factors(64, freq_base=10000.0, alpha=0.17, tier='long')
    factors_local = spn._tiered_lattice_factors(64, freq_base=10000.0, alpha=0.17, tier='local')
    diff = (factors_long - factors_local).abs().mean().item()
    t.check(diff > 0.001, f"Long vs Local tiers differ: avg |Δ|={diff:.4f}")

    # ========================================================================
    # SECTION 4: Lattice RoPE Monkey-Patch
    # ========================================================================
    t.set_category("lattice-rope")
    print("\n  -- Lattice RoPE Monkey-Patch --")

    # Test _install_lattice_rope without ComfyUI (should return False or True
    # depending on whether comfy.ldm.wan.model is importable)
    t.check(hasattr(spn, '_install_lattice_rope'), "_install_lattice_rope exists")

    # ========================================================================
    # SECTION 5: Fisher Diagonal Weighting
    # ========================================================================
    t.set_category("fisher")
    print("\n  -- Fisher Diagonal Weights --")

    t.check(hasattr(spn, '_fisher_diagonal_weights'), "_fisher_diagonal_weights exists")
    w = spn._fisher_diagonal_weights(128)
    t.check(w.shape == (128,), f"Fisher weights shape (128,) (got {w.shape})")

    # Squarefree positions should have HIGHER weight than non-squarefree
    # Index 0 (n=1) is squarefree, index 3 (n=4=2^2) is not
    w_sf = w[0].item()
    w_nsf = w[3].item()
    t.check(w_sf > w_nsf, f"Squarefree weight ({w_sf:.4f}) > non-squarefree ({w_nsf:.4f})")
    t.check(w_sf > 0 and w_nsf > 0, "All weights are positive")
    ratio = w_sf / w_nsf
    t.check(ratio > 2.0, f"Squarefree/non-squarefree ratio > 2 (got {ratio:.2f})")

    # Verify relative ordering: mean of squarefree weights > mean of non-squarefree
    def _is_squarefree(n):
        if n < 1:
            return False
        for p in range(2, int(n**0.5) + 1):
            if n % (p * p) == 0:
                return False
        return True

    sf_weights = [w[i].item() for i in range(128) if _is_squarefree(i + 1)]
    nsf_weights = [w[i].item() for i in range(128) if not _is_squarefree(i + 1)]
    mean_sf = sum(sf_weights) / len(sf_weights)
    mean_nsf = sum(nsf_weights) / len(nsf_weights)
    t.check(mean_sf > mean_nsf, f"Mean squarefree ({mean_sf:.4f}) > mean non-squarefree ({mean_nsf:.4f})")

    # Caching works
    w2 = spn._fisher_diagonal_weights(128)
    t.check(w.data_ptr() == w2.data_ptr(), "Fisher weights are cached (same tensor)")

    # ========================================================================
    # SECTION 6: CachingLinear Wrapper
    # ========================================================================
    t.set_category("integration")
    print("\n  -- CachingLinear Wrapper --")

    t.check(hasattr(spn, '_SPCachingLinear'), "_SPCachingLinear class exists")

    # Create a simple linear and wrap it
    lin = torch.nn.Linear(128, 64, bias=False)
    wrapped = spn._SPCachingLinear(lin, "test_k")
    t.check(hasattr(wrapped, '_sp_cached'), "Wrapped linear has _sp_cached attr")
    t.check(wrapped._sp_cached is None, "Cache starts empty")

    # First call: compute + cache
    x = torch.randn(1, 10, 128)
    y1 = wrapped(x)
    t.check(y1.shape == (1, 10, 64), f"Output shape correct: {y1.shape}")
    t.check(wrapped._sp_cached is not None, "Cache populated after first call")

    # Second call with same input (same data_ptr): cache hit
    y2 = wrapped(x)
    t.check(torch.equal(y1, y2), "Cache hit returns identical output")

    # New input: cache miss → recompute
    x2 = torch.randn(1, 10, 128)
    y3 = wrapped(x2)
    t.check(not torch.equal(y1, y3), "New input triggers cache miss")

    # ========================================================================
    # SECTION 7: Bits Parsing
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Bits CSV Parsing --")

    t.check(hasattr(spn, '_parse_bits_csv'), "_parse_bits_csv exists")
    parsed = spn._parse_bits_csv("4,3,3,3", [5, 5, 5, 5], 4)
    t.check(parsed == [4, 3, 3, 3], f"'4,3,3,3' → {parsed}")

    parsed_5 = spn._parse_bits_csv("3,3,3,3,3", [5]*5, 5)
    t.check(parsed_5 == [3, 3, 3, 3, 3], f"5-band parse: {parsed_5}")

    # Empty/invalid fallback to default
    parsed_bad = spn._parse_bits_csv("", [5, 4, 4, 3], 4)
    t.check(parsed_bad == [5, 4, 4, 3], f"Empty string → default: {parsed_bad}")

    # ========================================================================
    # SECTION 8: Block Iterator
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Block Iterator --")

    t.check(hasattr(spn, '_iter_wan_blocks'), "_iter_wan_blocks exists")
    # Can't test with real model, but verify it doesn't crash on None
    blocks = list(spn._iter_wan_blocks(None))
    t.check(len(blocks) == 0, "No blocks from None model")

    # ========================================================================
    # SECTION 9: Idempotency
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Idempotency --")

    # WanCache IS_CHANGED returns NaN (forces re-eval)
    val = spn.ShannonPrimeWanCache.IS_CHANGED()
    t.check(math.isnan(val), f"IS_CHANGED returns NaN (got {val})")

    # ========================================================================
    # SECTION 10: Memory / VRAM Tracking Helpers
    # ========================================================================
    t.set_category("integration")
    print("\n  -- Memory Tracking --")

    # input_fingerprint should return consistent values for same tensor
    t.check(hasattr(spn, '_input_fingerprint'), "_input_fingerprint exists")
    x = torch.randn(128)
    fp1 = spn._input_fingerprint(x)
    fp2 = spn._input_fingerprint(x)
    t.check(fp1 == fp2, "Same tensor → same fingerprint")

    # Different tensor → different fingerprint
    y = torch.randn(128)
    fp3 = spn._input_fingerprint(y)
    t.check(fp1 != fp3, "Different tensor → different fingerprint")
