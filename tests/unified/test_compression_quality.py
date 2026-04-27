#!/usr/bin/env python3
# Shannon-Prime VHT2: Compression Quality + Scaling Law Tests
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3.

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime', 'backends', 'torch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime', 'tools'))

try:
    import torch
    torch.manual_seed(42)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from run_tests import register_suite


def _scaling_law(k_corr, params_B, bits):
    """PPL/PPL_base = exp(4700·(1−K_corr)²/(params^1.1·bits^1.5))"""
    return math.exp(4700 * (1 - k_corr)**2 / (params_B**1.1 * bits**1.5))


@register_suite("compression", "Compression quality vs scaling law predictions")
def suite_compression(t):
    if not HAS_TORCH:
        t.check(False, "PyTorch required for compression tests")
        return

    from shannon_prime_torch import (
        vht2, MobiusMask, BandedQuantizer, ShadowCache, correlation, sqfree_pad_dim,
    )

    # ========================================================================
    # SECTION 1: Bit Allocation Grid Search
    # ========================================================================
    t.set_category("quality")
    print("\n  -- Bit Allocation Grid --")

    # Every config we ship or might ship, tested against 200 random vectors
    CONFIGS = [
        # (k_bits, v_bits, label, min_k_corr, min_v_corr)
        ([5, 5, 4, 3], [5, 4, 4, 3], "quality K=5543 V=5443", 0.985, 0.980),
        ([5, 4, 4, 3], [5, 4, 4, 3], "standard K=V=5443",     0.985, 0.975),
        ([4, 3, 3, 3], [4, 3, 3, 3], "lean K=V=4333",         0.965, 0.960),
        ([3, 3, 3, 3], [3, 3, 3, 3], "aggressive K=V=3333",   0.955, 0.940),
        ([4, 3, 2, 2], [3, 2, 2, 2], "ultra K=4322 V=3222",   0.880, 0.850),
        ([2, 2, 2, 2], [2, 2, 2, 2], "extreme K=V=2222",      0.740, 0.820),
    ]

    results_table = []

    for k_bits, v_bits, label, min_k, min_v in CONFIGS:
        k_quant = BandedQuantizer(128, k_bits)
        v_quant = BandedQuantizer(128, v_bits)

        k_corrs = []
        v_corrs = []
        for _ in range(200):
            # K with RoPE structure
            pos = float(torch.randint(0, 2048, (1,)))
            freqs = 10000.0 ** (-2.0 * torch.arange(64).float() / 128)
            k = torch.cat([torch.cos(torch.tensor([pos]) * freqs),
                           torch.sin(torch.tensor([pos]) * freqs)])
            k += torch.randn(128) * 0.05

            # V from attention mixing
            v = torch.randn(128)

            K = vht2(k.unsqueeze(0))
            Ks, Kq = k_quant.quantize(K)
            Kd = k_quant.dequantize(Ks, Kq)
            k_recon = vht2(Kd).squeeze(0)
            k_corrs.append(correlation(k, k_recon))

            V = vht2(v.unsqueeze(0))
            Vs, Vq = v_quant.quantize(V)
            Vd = v_quant.dequantize(Vs, Vq)
            v_recon = vht2(Vd).squeeze(0)
            v_corrs.append(correlation(v, v_recon))

        k_avg = sum(k_corrs) / len(k_corrs)
        v_avg = sum(v_corrs) / len(v_corrs)
        k_min = min(k_corrs)
        v_min = min(v_corrs)
        k_std = (sum((c - k_avg)**2 for c in k_corrs) / len(k_corrs))**0.5
        v_std = (sum((c - v_avg)**2 for c in v_corrs) / len(v_corrs))**0.5

        results_table.append({
            'label': label, 'k_bits': k_bits, 'v_bits': v_bits,
            'k_avg': k_avg, 'v_avg': v_avg, 'k_min': k_min, 'v_min': v_min,
            'k_std': k_std, 'v_std': v_std,
        })

        t.check(k_avg >= min_k,
                f"{label} K_avg={k_avg:.4f} (>={min_k})", f"min={k_min:.4f} std={k_std:.4f}")
        t.check(v_avg >= min_v,
                f"{label} V_avg={v_avg:.4f} (>={min_v})", f"min={v_min:.4f} std={v_std:.4f}")

    # Print table summary
    print(f"\n  {'Config':<35} {'K_avg':>7} {'K_min':>7} {'K_σ':>7} {'V_avg':>7} {'V_min':>7} {'V_σ':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in results_table:
        print(f"  {r['label']:<35} {r['k_avg']:>7.4f} {r['k_min']:>7.4f} {r['k_std']:>7.4f} "
              f"{r['v_avg']:>7.4f} {r['v_min']:>7.4f} {r['v_std']:>7.4f}")

    # ========================================================================
    # SECTION 2: Scaling Law Accuracy
    # ========================================================================
    t.set_category("scaling-law")
    print("\n  -- Scaling Law Predictions --")

    # For each config, check that measured K_corr → predicted PPL ratio makes sense
    MODEL_PACKS = [
        # (name, params_B, quant_bits, expected_max_ppl_ratio)
        ("phi3-mini Q4_K_M",   3.8,  4, 1.15),
        ("qwen3 7B Q6_K",      7.0,  6, 1.05),
        ("llama3 8B Q8_0",     8.0,  8, 1.03),
        ("qwen3 14B bf16",    14.0, 16, 1.02),
        ("llama3 70B Q4_K_M", 70.0,  4, 1.03),
    ]

    for name, params, bits, max_ratio in MODEL_PACKS:
        # Use the "lean" config K_corr as representative
        k_corr = results_table[2]['k_avg']  # lean 4,3,3,3
        pred = _scaling_law(k_corr, params, bits)
        t.check(pred < max_ratio,
                f"{name}: predicted PPL ratio = {pred:.4f} (< {max_ratio})",
                f"K_corr={k_corr:.4f}")

    # ========================================================================
    # SECTION 3: Head Dimension Sweep
    # ========================================================================
    t.set_category("quality")
    print("\n  -- Head Dimension Sweep --")

    for hd in [32, 64, 96, 128, 192, 256]:
        n_bands = min(4, max(2, hd // 32))
        bits = [4] * n_bands
        quant = BandedQuantizer(hd, bits)
        corrs = []
        for _ in range(100):
            x = torch.randn(hd)
            X = vht2(x.unsqueeze(0))
            scales, qvals = quant.quantize(X)
            Xd = quant.dequantize(scales, qvals)
            recon = vht2(Xd).squeeze(0)
            corrs.append(correlation(x, recon))
        avg = sum(corrs) / len(corrs)
        t.check(avg > 0.92, f"hd={hd} n_bands={n_bands}: avg_corr={avg:.4f} (>0.92)")

    # ========================================================================
    # SECTION 4: Sequence Length Stress
    # ========================================================================
    t.set_category("stress")
    print("\n  -- Sequence Length Stress --")

    for seq_len in [1, 16, 128, 512, 2048]:
        cache = ShadowCache(
            head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=seq_len,
            k_band_bits=[4, 3, 3, 3], v_band_bits=[3, 3, 3, 3],
        )
        corrs = []
        for pos in range(min(seq_len, 256)):
            k = torch.randn(128)
            cache.write_k(0, 0, pos, k)
            cache.write_v(0, 0, pos, torch.randn(128))
            k_out = cache.read_k(0, 0, pos)
            corrs.append(correlation(k, k_out))
        avg = sum(corrs) / len(corrs) if corrs else 0
        t.check(avg > 0.97 or seq_len <= 1,
                f"seq_len={seq_len}: avg_k_corr={avg:.4f}")

    # ========================================================================
    # SECTION 5: Band Count Exploration (3, 4, 5, 6 bands)
    # ========================================================================
    t.set_category("quality")
    print("\n  -- Band Count Exploration --")

    band_results = {}
    total_bits_budget = 16  # same total budget
    for n_bands in [3, 4, 5, 6]:
        per_band = total_bits_budget // n_bands
        remainder = total_bits_budget % n_bands
        bits = [per_band + (1 if i < remainder else 0) for i in range(n_bands)]
        quant = BandedQuantizer(128, bits)
        corrs = []
        for _ in range(200):
            x = torch.randn(128)
            X = vht2(x.unsqueeze(0))
            scales, qvals = quant.quantize(X)
            Xd = quant.dequantize(scales, qvals)
            recon = vht2(Xd).squeeze(0)
            corrs.append(correlation(x, recon))
        avg = sum(corrs) / len(corrs)
        band_results[n_bands] = avg
        t.check(avg > 0.92,
                f"{n_bands} bands ({bits}): avg_corr={avg:.4f} (>0.92)")

    # Report which band count is best (informational, not pass/fail)
    best = max(band_results, key=band_results.get)
    print(f"\n  Best band count at {total_bits_budget}-bit budget: {best} bands "
          f"(corr={band_results[best]:.4f})")

    # ========================================================================
    # SECTION 6: Determinism + Reproducibility
    # ========================================================================
    t.set_category("invariant")
    print("\n  -- Determinism --")

    torch.manual_seed(12345)
    x = torch.randn(128)
    quant = BandedQuantizer(128, [4, 3, 3, 3])
    X = vht2(x.unsqueeze(0))
    scales1, qvals1 = quant.quantize(X)
    Xd1 = quant.dequantize(scales1, qvals1)

    # Run again — must be identical
    scales2, qvals2 = quant.quantize(X)
    Xd2 = quant.dequantize(scales2, qvals2)
    t.check(torch.equal(Xd1, Xd2), "Quantization is deterministic (bitwise)")

    torch.manual_seed(42)  # restore for other suites
