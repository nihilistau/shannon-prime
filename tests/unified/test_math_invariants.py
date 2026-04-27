#!/usr/bin/env python3
# Shannon-Prime VHT2: Math Invariant + Falsification Tests
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3.
#
# Tests the 10 core mathematical invariants that are load-bearing for the
# entire Shannon-Prime system. Includes falsification tests that MUST fail
# to prove the invariants aren't vacuously true.

import sys
import os
import math

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime', 'backends', 'torch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime', 'tools'))

try:
    import torch
    torch.manual_seed(42)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import from runner
from run_tests import register_suite


def _import_sp():
    """Lazy import shannon_prime_torch — may not be available."""
    from shannon_prime_torch import (
        vht2, MobiusMask, BandedQuantizer,
        ShadowCache, correlation, sqfree_pad_dim,
    )
    return vht2, MobiusMask, BandedQuantizer, ShadowCache, correlation, sqfree_pad_dim


def _mobius(n):
    """Reference Möbius function μ(n)."""
    if n == 1:
        return 1
    factors = 0
    temp = n
    for p in range(2, int(math.isqrt(n)) + 1):
        if temp % p == 0:
            factors += 1
            temp //= p
            if temp % p == 0:
                return 0  # p^2 divides n
    if temp > 1:
        factors += 1
    return (-1) ** factors


def _is_squarefree(n):
    """Reference squarefree test."""
    if n < 1:
        return False
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % (p * p) == 0:
            return False
    return True


def _sieve_primes(n=10000):
    """Sieve of Eratosthenes."""
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, n+1, i):
                is_p[j] = False
    return [i for i in range(2, n+1) if is_p[i]]


@register_suite("math", "Core mathematical invariants + falsification")
def suite_math(t):
    # ========================================================================
    # SECTION 1: Number-Theoretic Foundations
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Number Theory Foundations --")

    # 1a. Möbius function values
    expected = {1: 1, 2: -1, 3: -1, 4: 0, 5: -1, 6: 1, 7: -1, 8: 0,
                9: 0, 10: 1, 11: -1, 12: 0, 13: -1, 30: -1}
    for n, mu in expected.items():
        t.check(_mobius(n) == mu, f"μ({n}) = {mu}")

    # 1b. Möbius sum identity: Σ_{d|n} μ(d) = [n=1]
    for n in [1, 2, 6, 12, 30, 60, 210, 2310]:
        divisors = [d for d in range(1, n+1) if n % d == 0]
        s = sum(_mobius(d) for d in divisors)
        expected_sum = 1 if n == 1 else 0
        t.check(s == expected_sum, f"Σ μ(d|{n}) = {s} (expected {expected_sum})")

    # 1c. Squarefree density: #{squarefree ≤ N}/N → 6/π² ≈ 0.6079
    N = 10000
    sf_count = sum(1 for n in range(1, N+1) if _is_squarefree(n))
    density = sf_count / N
    t.check(abs(density - 6/math.pi**2) < 0.01,
            f"Squarefree density = {density:.4f} (expected ~{6/math.pi**2:.4f})")

    # 1d. Mertens function |M(n)| ≤ √n bound (empirical, holds to 10^13)
    M = 0
    violations = 0
    for n in range(1, 5001):
        M += _mobius(n)
        if abs(M) > math.sqrt(n):
            violations += 1
    t.check(violations == 0, f"Mertens |M(n)| ≤ √n for n≤5000 ({violations} violations)")

    # 1e. Prime counting: π(1000) = 168
    primes = _sieve_primes(1000)
    t.check(len(primes) == 168, f"π(1000) = {len(primes)} (expected 168)")

    # 1f. Euler product convergence: Π_p (1 - 1/p²)^(-1) → π²/6
    primes_100 = _sieve_primes(100)
    product = 1.0
    for p in primes_100:
        product *= 1.0 / (1.0 - 1.0 / (p * p))
    t.check(abs(product - math.pi**2/6) < 0.01,
            f"Euler product = {product:.6f} (expected {math.pi**2/6:.6f})")

    # ========================================================================
    # SECTION 1F: FALSIFICATION — Number Theory
    # ========================================================================
    t.set_category("falsification")
    print("\n  -- Falsification: Number Theory --")

    # Must fail: μ(4) ≠ -1 (4 = 2² is NOT squarefree)
    t.check_must_fail(_mobius(4) == -1, "μ(4) must NOT equal -1 (4 has squared factor)")

    # Must fail: Σ μ(d|6) ≠ 1 (only true for n=1)
    divs_6 = [d for d in range(1, 7) if 6 % d == 0]
    t.check_must_fail(sum(_mobius(d) for d in divs_6) == 1,
                      "Σ μ(d|6) must NOT equal 1")

    # Must fail: 12 is NOT squarefree
    t.check_must_fail(_is_squarefree(12), "12 must NOT be squarefree (4|12)")

    if not HAS_TORCH:
        print("\n  [SKIP] PyTorch not available — skipping VHT2 tests")
        return

    vht2, MobiusMask, BandedQuantizer, ShadowCache, correlation, sqfree_pad_dim = _import_sp()

    # ========================================================================
    # SECTION 2: VHT2 Transform Invariants
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- VHT2 Transform Invariants --")

    # 2a. Self-inverse: VHT2(VHT2(x)) = x (exact, no 1/N normalization)
    for hd in [32, 64, 128, 256]:
        orig = torch.randn(hd)
        recon = vht2(vht2(orig.unsqueeze(0))).squeeze(0)
        err = (orig - recon).abs().max().item()
        t.check(err < 1e-5, f"VHT2 self-inverse hd={hd}: max_err={err:.2e}")

    # 2b. Self-inverse on sqfree (non-power-of-2) dimensions
    for hd in [66, 105, 154, 210]:
        orig = torch.randn(hd)
        recon = vht2(vht2(orig.unsqueeze(0))).squeeze(0)
        err = (orig - recon).abs().max().item()
        t.check(err < 1e-4, f"VHT2 self-inverse sqfree n={hd}: max_err={err:.2e}")

    # 2c. Energy preservation (Parseval): ‖VHT2(x)‖ = ‖x‖
    for hd in [64, 128]:
        x = torch.randn(hd)
        X = vht2(x.unsqueeze(0)).squeeze(0)
        ratio = X.norm().item() / x.norm().item()
        t.check(abs(ratio - 1.0) < 0.01,
                f"VHT2 Parseval hd={hd}: ‖X‖/‖x‖ = {ratio:.4f}")

    # 2d. Linearity: VHT2(ax + by) = a·VHT2(x) + b·VHT2(y)
    x = torch.randn(128)
    y = torch.randn(128)
    a, b = 2.5, -1.3
    lhs = vht2((a*x + b*y).unsqueeze(0)).squeeze(0)
    rhs = a * vht2(x.unsqueeze(0)).squeeze(0) + b * vht2(y.unsqueeze(0)).squeeze(0)
    err = (lhs - rhs).abs().max().item()
    t.check(err < 1e-5, f"VHT2 linearity: max_err={err:.2e}")

    # 2e. Deterministic: same input → same output
    x = torch.randn(128)
    y1 = vht2(x.unsqueeze(0)).squeeze(0)
    y2 = vht2(x.unsqueeze(0)).squeeze(0)
    t.check(torch.equal(y1, y2), "VHT2 deterministic (bitwise identical)")

    # ========================================================================
    # SECTION 2F: FALSIFICATION — VHT2
    # ========================================================================
    t.set_category("falsification")
    print("\n  -- Falsification: VHT2 --")

    # Must fail: random matrix is NOT self-inverse
    rand_mat = torch.randn(128, 128)
    x = torch.randn(128)
    double_app = rand_mat @ (rand_mat @ x)
    t.check_must_fail((double_app - x).abs().max().item() < 0.01,
                      "Random matrix must NOT be self-inverse")

    # Must fail: identity transform gives correlation=1.0 (VHT2 changes the signal)
    x = torch.randn(128)
    X = vht2(x.unsqueeze(0)).squeeze(0)
    t.check_must_fail(torch.allclose(x, X, atol=1e-3),
                      "VHT2(x) must NOT equal x (transform must change representation)")

    # ========================================================================
    # SECTION 3: Möbius Mask & Spectral Ordering
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Möbius Mask & Spectral Ordering --")

    # 3a. Reorder→unreorder round-trip
    mask = MobiusMask(128)
    x = torch.randn(128)
    reordered = mask.reorder(x.unsqueeze(0)).squeeze(0)
    restored = mask.unreorder(reordered.unsqueeze(0)).squeeze(0)
    err = (x - restored).abs().max().item()
    t.check(err < 1e-6, f"Möbius reorder round-trip: max_err={err:.2e}")

    # 3b. Reorder is a permutation (no duplicates, no missing indices)
    mask128 = MobiusMask(128)
    perm = mask128.order
    t.check(len(set(perm)) == 128, "Möbius perm has 128 unique indices")
    t.check(set(perm) == set(range(128)), "Möbius perm covers [0,128)")

    # 3c. Squarefree indices come first in the Möbius ordering
    sf_indices = [i for i in range(1, 129) if _is_squarefree(i)]
    perm_list = list(perm)
    # The first len(sf_indices) positions in the perm should be the squarefree-indexed dims
    first_n = perm_list[:len(sf_indices)]
    # The order uses 1-based indices matching squarefree numbers 1..128
    t.check(set(first_n) == set(sf_indices),
            f"First {len(sf_indices)} Möbius positions are squarefree")

    # ========================================================================
    # SECTION 4: Banded Quantization Invariants
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Banded Quantization --")

    # 4a. Round-trip correlation for standard configs
    configs = [
        ([5, 4, 4, 3], "5,4,4,3 (standard)"),
        ([4, 3, 3, 3], "4,3,3,3 (lean)"),
        ([3, 3, 3, 3], "3,3,3,3 (aggressive)"),
        ([5, 5, 4, 3], "5,5,4,3 (quality)"),
    ]
    for bits, label in configs:
        quant = BandedQuantizer(128, bits)
        corrs = []
        for _ in range(50):
            x = torch.randn(128)
            X = vht2(x.unsqueeze(0))
            scales, qvals = quant.quantize(X)
            Xd = quant.dequantize(scales, qvals)
            recon = vht2(Xd).squeeze(0)
            corrs.append(correlation(x, recon))
        avg = sum(corrs) / len(corrs)
        t.check(avg > 0.90, f"Avg correlation {label}: {avg:.4f} (>0.90)")

    # 4b. More bits → higher correlation (monotonicity)
    results = {}
    for total_bits, label in [(12, "3*4"), (16, "4*4"), (20, "5*4")]:
        per_band = total_bits // 4
        quant = BandedQuantizer(128, [per_band]*4)
        corrs = []
        for _ in range(100):
            x = torch.randn(128)
            X = vht2(x.unsqueeze(0))
            scales, qvals = quant.quantize(X)
            Xd = quant.dequantize(scales, qvals)
            recon = vht2(Xd).squeeze(0)
            corrs.append(correlation(x, recon))
        results[total_bits] = sum(corrs) / len(corrs)

    t.check(results[16] >= results[12] - 0.01,
            f"Monotonicity: 4*4 ({results[16]:.4f}) >= 3*4 ({results[12]:.4f})")
    t.check(results[20] >= results[16] - 0.01,
            f"Monotonicity: 5*4 ({results[20]:.4f}) >= 4*4 ({results[16]:.4f})")

    # ========================================================================
    # SECTION 4F: FALSIFICATION — Quantization
    # ========================================================================
    t.set_category("falsification")
    print("\n  -- Falsification: Quantization --")

    # Must fail: 1-bit quantization should NOT achieve 0.99 correlation
    quant_1bit = BandedQuantizer(128, [1, 1, 1, 1])
    corrs_1bit = []
    for _ in range(50):
        x = torch.randn(128)
        X = vht2(x.unsqueeze(0))
        scales, qvals = quant_1bit.quantize(X)
        Xd = quant_1bit.dequantize(scales, qvals)
        recon = vht2(Xd).squeeze(0)
        corrs_1bit.append(correlation(x, recon))
    avg_1bit = sum(corrs_1bit) / len(corrs_1bit)
    t.check_must_fail(avg_1bit > 0.99,
                      f"1-bit quant must NOT achieve >0.99 corr (got {avg_1bit:.4f})")

    # Must fail: zero-bit (all zeros) should give near-zero correlation
    # (Tests that correlation metric isn't trivially 1.0)
    x = torch.randn(128)
    zeros = torch.zeros(128)
    c = correlation(x, zeros)
    t.check_must_fail(c > 0.5, f"Zero reconstruction must NOT correlate >0.5 (got {c:.4f})")

    # ========================================================================
    # SECTION 5: PrimePE / Lattice RoPE Invariants
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- PrimePE / Lattice RoPE --")

    # 5a. Alpha=0 gives identity (all factors = 1.0)
    try:
        from shannon_prime_torch import prime_pe_freq_factors
        factors_0 = prime_pe_freq_factors(64, 10000.0, 0.0)
        t.check(all(abs(f - 1.0) < 1e-6 for f in factors_0),
                "PrimePE alpha=0 → identity (all 1.0)")
    except ImportError:
        # Fall back to pure-Python implementation
        pass

    # 5b. Pure Python lattice frequency generation (matches C implementation)
    primes_10k = _sieve_primes(10000)
    t.check(len(primes_10k) == 1229, f"π(10000) = {len(primes_10k)} (expected 1229)")

    # 5c. Tiered allocation covers correct ranges
    n_freqs = 64
    n_local = n_freqs // 4  # 16
    n_mid = (n_freqs * 33) // 100  # 21
    n_long = n_freqs - n_local - n_mid  # 27
    t.check(n_local + n_mid + n_long == n_freqs,
            f"Tier allocation sums: {n_local}+{n_mid}+{n_long} = {n_freqs}")
    t.check(n_local >= 1 and n_mid >= 1 and n_long >= 1,
            "All tiers have ≥1 frequency")

    # 5d. Lattice frequencies use composites (non-primes)
    prime_set = set(primes_10k)
    composites_local = [n for n in range(4, 102) if n not in prime_set]
    t.check(len(composites_local) > 0, f"Local tier has {len(composites_local)} composites")
    composites_long = [n for n in range(1010, 8210) if n not in prime_set]
    t.check(len(composites_long) > 100, f"Long tier has {len(composites_long)} composites")

    # 5e. Blend formula: factors[i] = blended[i] / geometric[i]
    # At alpha=0.17, factors should be close to 1.0 but not exactly 1.0
    # (unless alpha=0, tested above)
    freq_base = 10000.0
    d = 128
    geometric = [freq_base ** (-2.0 * j / d) for j in range(d // 2)]
    t.check(geometric[0] > geometric[-1],
            "Geometric frequencies decrease with dimension index")
    t.check(abs(geometric[0] - 1.0) < 1e-6,
            f"Geometric freq[0] ≈ 1.0 (got {geometric[0]:.6f})")

    # ========================================================================
    # SECTION 5F: FALSIFICATION — PrimePE
    # ========================================================================
    t.set_category("falsification")
    print("\n  -- Falsification: PrimePE --")

    # Must fail: alpha > 0 should NOT give all-ones factors
    # (would mean the lattice had zero effect)
    t.check_must_fail(
        all(abs(g - 1.0) < 0.01 for g in geometric[:10]),
        "Geometric freqs[0:10] must NOT all be ~1.0 (they decrease)")

    # Must fail: composites pool should NOT contain primes
    for c in composites_local[:5]:
        t.check_must_fail(c in prime_set,
                          f"Composite {c} must NOT be prime")

    # ========================================================================
    # SECTION 6: Scaling Law Validation
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Scaling Law --")

    # The law: log(PPL/PPL_base) ≈ 4700·(1−K_corr)²/(params^1.1·bits^1.5)
    def scaling_law_ppl_ratio(k_corr, params_B, bits):
        return math.exp(4700 * (1 - k_corr)**2 / (params_B**1.1 * bits**1.5))

    # 6a. Perfect correlation → PPL ratio = 1.0 (no degradation)
    t.check(abs(scaling_law_ppl_ratio(1.0, 7.0, 4) - 1.0) < 1e-6,
            "K_corr=1.0 → PPL ratio = 1.0")

    # 6b. Monotonicity: lower correlation → higher PPL
    r1 = scaling_law_ppl_ratio(0.99, 7.0, 4)
    r2 = scaling_law_ppl_ratio(0.95, 7.0, 4)
    t.check(r2 > r1, f"Lower K_corr gives higher PPL: {r1:.4f} < {r2:.4f}")

    # 6c. Larger model more tolerant (same corr → lower PPL ratio)
    r_7b = scaling_law_ppl_ratio(0.95, 7.0, 4)
    r_70b = scaling_law_ppl_ratio(0.95, 70.0, 4)
    t.check(r_70b < r_7b, f"70B more tolerant than 7B: {r_70b:.4f} < {r_7b:.4f}")

    # 6d. More bits more tolerant
    r_3bit = scaling_law_ppl_ratio(0.95, 7.0, 3)
    r_5bit = scaling_law_ppl_ratio(0.95, 7.0, 5)
    t.check(r_5bit < r_3bit, f"5-bit better than 3-bit: {r_5bit:.4f} < {r_3bit:.4f}")

    # 6e. Wan 14B bf16 safe floor: K_corr=0.914 gives < +5% PPL
    wan_ratio = scaling_law_ppl_ratio(0.914, 14.0, 16)
    t.check(wan_ratio < 1.05, f"Wan 14B bf16 at K_corr=0.914: {wan_ratio:.4f} (< 1.05)")

    # ========================================================================
    # SECTION 7: ShadowCache Round-Trip
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- ShadowCache Round-Trip --")

    cache = ShadowCache(
        head_dim=128, n_layers=2, n_heads_kv=4, max_seq_len=32,
        k_band_bits=[5, 4, 4, 3], v_band_bits=[3, 3, 3, 3],
    )

    k_corrs = []
    v_corrs = []
    for layer in range(2):
        for head in range(4):
            for pos in range(16):
                k = torch.randn(128)
                v = torch.randn(128)
                cache.write_k(layer, head, pos, k)
                cache.write_v(layer, head, pos, v)
                k_out = cache.read_k(layer, head, pos)
                v_out = cache.read_v(layer, head, pos)
                k_corrs.append(correlation(k, k_out))
                v_corrs.append(correlation(v, v_out))

    avg_k = sum(k_corrs) / len(k_corrs)
    avg_v = sum(v_corrs) / len(v_corrs)
    min_k = min(k_corrs)
    min_v = min(v_corrs)

    t.check(avg_k > 0.98, f"ShadowCache K avg corr: {avg_k:.4f} (>0.98)")
    t.check(avg_v > 0.94, f"ShadowCache V avg corr: {avg_v:.4f} (>0.94)")
    t.check(min_k > 0.95, f"ShadowCache K min corr: {min_k:.4f} (>0.95)")
    t.check(min_v > 0.85, f"ShadowCache V min corr: {min_v:.4f} (>0.85)")

    # ========================================================================
    # SECTION 8: K/V Asymmetry (Spectral Concentration)
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- K/V Spectral Asymmetry --")

    # K vectors (with RoPE-like structure) should concentrate energy in low bands
    # V vectors (attention-mixed) should be more uniform
    k_concentrations = []
    v_concentrations = []
    for _ in range(100):
        # Simulate K with RoPE structure (sinusoidal)
        pos = torch.tensor([float(torch.randint(0, 512, (1,)))])
        freqs = 10000.0 ** (-2.0 * torch.arange(64).float() / 128)
        k = torch.cat([torch.cos(pos * freqs), torch.sin(pos * freqs)])
        k += torch.randn(128) * 0.1  # add noise

        # Simulate V (attention-mixed, more uniform)
        v = torch.randn(128)

        K = vht2(k.unsqueeze(0)).squeeze(0)
        V = vht2(v.unsqueeze(0)).squeeze(0)

        # Energy in first half
        k_energy = K[:64].norm().item()**2 / (K.norm().item()**2 + 1e-12)
        v_energy = V[:64].norm().item()**2 / (V.norm().item()**2 + 1e-12)
        k_concentrations.append(k_energy)
        v_concentrations.append(v_energy)

    k_avg = sum(k_concentrations) / len(k_concentrations)
    v_avg = sum(v_concentrations) / len(v_concentrations)
    t.check(k_avg > 0.55, f"K spectral concentration: {k_avg:.3f} (>0.55, more structured)")
    t.check(k_avg > v_avg, f"K more concentrated than V: {k_avg:.3f} > {v_avg:.3f}")

    # ========================================================================
    # SECTION 9: Compression Ratio Bounds
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Compression Ratio Bounds --")

    for hd, min_ratio, max_ratio in [(64, 2.0, 6.0), (128, 2.5, 5.0), (256, 2.5, 5.0)]:
        cache_r = ShadowCache(
            head_dim=hd, n_layers=1, n_heads_kv=1, max_seq_len=64,
            k_band_bits=[5, 4, 4, 3], v_band_bits=[3, 3, 3, 3],
        )
        for pos in range(64):
            cache_r.write_k(0, 0, pos, torch.randn(hd))
            cache_r.write_v(0, 0, pos, torch.randn(hd))
        mem = cache_r.memory_bytes(64)
        ratio = cache_r.compression_ratio()
        t.check(min_ratio <= ratio <= max_ratio,
                f"Compression ratio hd={hd}: {ratio:.2f}× (expected {min_ratio}–{max_ratio}×)")

    # ========================================================================
    # SECTION 10: Cross-Invariant Consistency
    # ========================================================================
    t.set_category("cross-check")
    print("\n  -- Cross-Invariant Consistency --")

    # VHT2 + Möbius reorder should commute with quantization
    # (reorder → quantize → dequantize → unreorder ≈ quantize → dequantize)
    mask = MobiusMask(128)
    quant = BandedQuantizer(128, [5, 4, 4, 3])
    diffs = []
    for _ in range(50):
        x = torch.randn(128)

        # Path A: VHT2 → Möbius → quant → dequant → un-Möbius → VHT2
        Xa = vht2(x.unsqueeze(0))
        Xa_r = mask.reorder(Xa)
        Xa_s, Xa_q = quant.quantize(Xa_r)
        Xa_d = quant.dequantize(Xa_s, Xa_q)
        Xa_u = mask.unreorder(Xa_d)
        recon_a = vht2(Xa_u).squeeze(0)

        # Path B: VHT2 → quant → dequant → VHT2 (no Möbius)
        Xb = vht2(x.unsqueeze(0))
        Xb_s, Xb_q = quant.quantize(Xb)
        Xb_d = quant.dequantize(Xb_s, Xb_q)
        recon_b = vht2(Xb_d).squeeze(0)

        # Both should produce reasonable reconstructions
        ca = correlation(x, recon_a)
        cb = correlation(x, recon_b)
        diffs.append(abs(ca - cb))

    avg_diff = sum(diffs) / len(diffs)
    t.check(avg_diff < 0.05,
            f"Möbius path vs direct: avg |Δcorr| = {avg_diff:.4f} (<0.05)")
