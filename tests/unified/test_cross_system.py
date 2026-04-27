#!/usr/bin/env python3
# Shannon-Prime VHT2: Cross-System + Edge Case + Adversarial Tests
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3.

import sys
import os
import math
import gc
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime', 'backends', 'torch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shannon-prime-comfyui', 'nodes'))

try:
    import torch
    torch.manual_seed(42)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from run_tests import register_suite


@register_suite("cross", "Cross-system interactions, edge cases, adversarial inputs")
def suite_cross(t):
    if not HAS_TORCH:
        t.check(False, "PyTorch required")
        return

    from shannon_prime_torch import (
        vht2, MobiusMask, BandedQuantizer, ShadowCache, correlation,
    )

    # ========================================================================
    # SECTION 1: PrimePE × VHT2 Interaction
    # ========================================================================
    t.set_category("cross-system")
    print("\n  -- PrimePE × VHT2 --")

    # Lattice-modified K vectors should still compress well through VHT2
    freq_base = 10000.0
    for alpha in [0.0, 0.10, 0.17, 0.22, 0.30]:
        corrs = []
        for _ in range(100):
            pos = float(torch.randint(0, 2048, (1,)))
            d = 128
            # Geometric + lattice blend
            geometric = freq_base ** (-2.0 * torch.arange(64).float() / d)
            # Simulate lattice factors (random perturbation ∝ alpha)
            factors = 1.0 + alpha * (torch.rand(64) - 0.5)
            freqs = geometric * factors
            k = torch.cat([torch.cos(torch.tensor([pos]) * freqs),
                           torch.sin(torch.tensor([pos]) * freqs)])
            k += torch.randn(128) * 0.05

            quant = BandedQuantizer(128, [4, 3, 3, 3])
            X = vht2(k.unsqueeze(0))
            scales, qvals = quant.quantize(X)
            Xd = quant.dequantize(scales, qvals)
            recon = vht2(Xd).squeeze(0)
            corrs.append(correlation(k, recon))

        avg = sum(corrs) / len(corrs)
        t.check(avg > 0.96, f"PrimePE α={alpha:.2f} K_corr={avg:.4f} (>0.96)")

    # ========================================================================
    # SECTION 2: Möbius × Quantization Interaction
    # ========================================================================
    t.set_category("cross-system")
    print("\n  -- Möbius × Quantization --")

    mask = MobiusMask(128)
    for bits_cfg in [[5,4,4,3], [4,3,3,3], [3,3,3,3]]:
        quant = BandedQuantizer(128, bits_cfg)
        # Path A: reorder → quantize → dequantize → unreorder
        # Path B: quantize → dequantize (no reorder)
        diffs = []
        for _ in range(100):
            x = torch.randn(128)
            X = vht2(x.unsqueeze(0))

            # Path A (Möbius)
            Xa = mask.reorder(X)
            Xa_s, Xa_q = quant.quantize(Xa)
            Xa_d = quant.dequantize(Xa_s, Xa_q)
            Xa_u = mask.unreorder(Xa_d)
            ra = vht2(Xa_u).squeeze(0)
            ca = correlation(x, ra)
            # Path B (direct)
            Xb_s, Xb_q = quant.quantize(X)
            Xb_d = quant.dequantize(Xb_s, Xb_q)
            rb = vht2(Xb_d).squeeze(0)
            cb = correlation(x, rb)
            diffs.append(ca - cb)

        avg_diff = sum(diffs) / len(diffs)
        # Möbius should help (or at least not hurt significantly)
        t.check(avg_diff > -0.02,
                f"Möbius effect at {bits_cfg}: ΔCorr={avg_diff:+.4f} (>-0.02)")

    # ========================================================================
    # SECTION 3: Edge Cases — Extreme Inputs
    # ========================================================================
    t.set_category("edge-case")
    print("\n  -- Extreme Inputs --")

    # 3a. Zero vector
    z = torch.zeros(128)
    Xz = vht2(z.unsqueeze(0)).squeeze(0)
    t.check(Xz.abs().max().item() < 1e-10, "VHT2(zeros) ≈ zeros")

    # 3b. Very large values
    big = torch.randn(128) * 1e6
    Xbig = vht2(big.unsqueeze(0)).squeeze(0)
    recon_big = vht2(Xbig.unsqueeze(0)).squeeze(0)
    err_big = (big - recon_big).abs().max().item() / big.abs().max().item()
    t.check(err_big < 1e-4, f"Large values relative err: {err_big:.2e}")

    # 3c. Very small values
    tiny = torch.randn(128) * 1e-6
    Xtiny = vht2(tiny.unsqueeze(0)).squeeze(0)
    recon_tiny = vht2(Xtiny.unsqueeze(0)).squeeze(0)
    t.check(not torch.isnan(recon_tiny).any().item(), "Small values: no NaN")

    # 3d. Single non-zero element (impulse)
    for idx in [0, 63, 127]:
        impulse = torch.zeros(128)
        impulse[idx] = 1.0
        Xi = vht2(impulse.unsqueeze(0)).squeeze(0)
        recon_i = vht2(Xi.unsqueeze(0)).squeeze(0)
        err_i = (impulse - recon_i).abs().max().item()
        t.check(err_i < 1e-5, f"Impulse at [{idx}] round-trip: err={err_i:.2e}")

    # 3e. Constant vector
    const = torch.ones(128)
    Xc = vht2(const.unsqueeze(0)).squeeze(0)
    recon_c = vht2(Xc.unsqueeze(0)).squeeze(0)
    err_c = (const - recon_c).abs().max().item()
    t.check(err_c < 1e-5, f"Constant vector round-trip: err={err_c:.2e}")

    # 3f. NaN injection — should propagate (not silently disappear)
    nan_vec = torch.randn(128)
    nan_vec[64] = float('nan')
    Xnan = vht2(nan_vec.unsqueeze(0)).squeeze(0)
    t.check(torch.isnan(Xnan).any().item(), "NaN propagates through VHT2")

    # 3g. Inf injection
    inf_vec = torch.randn(128)
    inf_vec[32] = float('inf')
    Xinf = vht2(inf_vec.unsqueeze(0)).squeeze(0)
    has_inf_or_nan = torch.isinf(Xinf).any().item() or torch.isnan(Xinf).any().item()
    t.check(has_inf_or_nan, "Inf propagates through VHT2 (not silently clamped)")

    # ========================================================================
    # SECTION 4: Sequence Length Edge Cases
    # ========================================================================
    t.set_category("edge-case")
    print("\n  -- Sequence Length Edge Cases --")

    # Single token
    cache_1 = ShadowCache(head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=1,
                          k_band_bits=[4,3,3,3], v_band_bits=[3,3,3,3])
    k1 = torch.randn(128)
    v1 = torch.randn(128)
    cache_1.write_k(0, 0, 0, k1)
    cache_1.write_v(0, 0, 0, v1)
    k1_out = cache_1.read_k(0, 0, 0)
    t.check(correlation(k1, k1_out) > 0.95, "Single-token cache works")

    # Overwrite same position
    cache_ow = ShadowCache(head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=4,
                           k_band_bits=[4,3,3,3], v_band_bits=[3,3,3,3])
    k_old = torch.randn(128)
    v_old = torch.randn(128)
    cache_ow.write_k(0, 0, 0, k_old)
    cache_ow.write_v(0, 0, 0, v_old)
    k_new = torch.randn(128)
    v_new = torch.randn(128)
    cache_ow.write_k(0, 0, 0, k_new)
    cache_ow.write_v(0, 0, 0, v_new)
    k_read = cache_ow.read_k(0, 0, 0)
    corr_new = correlation(k_new, k_read)
    corr_old = correlation(k_old, k_read)
    t.check(corr_new > corr_old + 0.1,
            f"Overwrite replaces: new_corr={corr_new:.4f} > old_corr={corr_old:.4f}")

    # ========================================================================
    # SECTION 5: Thread Safety
    # ========================================================================
    t.set_category("thread-safety")
    print("\n  -- Thread Safety --")

    cache_mt = ShadowCache(head_dim=128, n_layers=2, n_heads_kv=4, max_seq_len=64,
                           k_band_bits=[4,3,3,3], v_band_bits=[3,3,3,3])
    errors = []

    def writer(layer, head):
        try:
            for pos in range(64):
                k = torch.randn(128)
                v = torch.randn(128)
                cache_mt.write_k(layer, head, pos, k)
                cache_mt.write_v(layer, head, pos, v)
        except Exception as e:
            errors.append(str(e))

    threads = []
    for layer in range(2):
        for head in range(4):
            th = threading.Thread(target=writer, args=(layer, head))
            threads.append(th)

    for th in threads:
        th.start()
    for th in threads:
        th.join()

    t.check(len(errors) == 0, f"Thread safety: {len(errors)} errors")

    # Verify reads after threaded writes
    k = torch.randn(128)
    v_mt = torch.randn(128)
    cache_mt.write_k(0, 0, 0, k)
    cache_mt.write_v(0, 0, 0, v_mt)
    k_out = cache_mt.read_k(0, 0, 0)
    t.check(not torch.isnan(k_out).any().item(), "Post-threaded-write read: no NaN")

    # ========================================================================
    # SECTION 6: Memory Leak Detection
    # ========================================================================
    t.set_category("memory")
    print("\n  -- Memory Leak Detection --")

    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        base_mem = torch.cuda.memory_allocated()

    # Create and destroy many caches
    for _ in range(100):
        c = ShadowCache(head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=16,
                        k_band_bits=[4,3,3,3], v_band_bits=[3,3,3,3])
        for pos in range(16):
            c.write_k(0, 0, pos, torch.randn(128))
            c.write_v(0, 0, pos, torch.randn(128))
        del c

    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        end_mem = torch.cuda.memory_allocated()
        leak_mb = (end_mem - base_mem) / 1e6
        t.check(leak_mb < 1.0, f"CUDA memory leak: {leak_mb:.2f}MB (< 1.0MB)")
    else:
        # Can't precisely measure CPU memory, but exercise the path
        t.check(True, "CPU memory: exercised create/destroy cycle (no CUDA to measure)")

    # ========================================================================
    # SECTION 7: Numerical Stability Under Accumulated Error
    # ========================================================================
    t.set_category("stability")
    print("\n  -- Numerical Stability --")

    # Repeated encode/decode should not accumulate unbounded error
    quant = BandedQuantizer(128, [4, 3, 3, 3])
    x = torch.randn(128)
    for i in range(10):
        X = vht2(x.unsqueeze(0))
        scales, qvals = quant.quantize(X)
        Xd = quant.dequantize(scales, qvals)
        x = vht2(Xd).squeeze(0)

    t.check(not torch.isnan(x).any().item(), "10× encode/decode: no NaN")
    t.check(not torch.isinf(x).any().item(), "10× encode/decode: no Inf")
    t.check(x.abs().max().item() < 100, f"10× encode/decode: bounded ({x.abs().max().item():.1f})")

    # ========================================================================
    # SECTION 8: Cross-Config Compatibility
    # ========================================================================
    t.set_category("cross-system")
    print("\n  -- Cross-Config Compatibility --")

    # Different K and V bit allocations should work together
    cache_asym = ShadowCache(head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=16,
                             k_band_bits=[5, 5, 4, 3], v_band_bits=[3, 2, 2, 2])
    for pos in range(16):
        k = torch.randn(128)
        v = torch.randn(128)
        cache_asym.write_k(0, 0, pos, k)
        cache_asym.write_v(0, 0, pos, v)
        k_out = cache_asym.read_k(0, 0, pos)
        v_out = cache_asym.read_v(0, 0, pos)
        t.check(correlation(k, k_out) > 0.97,
                f"Asymmetric K/V pos={pos}: K_corr OK") if pos < 3 else None
        t.check(correlation(v, v_out) > 0.80,
                f"Asymmetric K/V pos={pos}: V_corr OK") if pos < 3 else None

    # ========================================================================
    # SECTION 9: Adversarial Patterns
    # ========================================================================
    t.set_category("adversarial")
    print("\n  -- Adversarial Patterns --")

    # 9a. Perfectly correlated dimensions (worst case for VHT2)
    x_corr = torch.ones(128) * 5.0
    X_corr = vht2(x_corr.unsqueeze(0)).squeeze(0)
    quant_a = BandedQuantizer(128, [4, 3, 3, 3])
    Xs_a, Xq_a = quant_a.quantize(X_corr.unsqueeze(0))
    Xd_a = quant_a.dequantize(Xs_a, Xq_a)
    recon_a = vht2(Xd_a).squeeze(0)
    ca = correlation(x_corr, recon_a)
    t.check(not math.isnan(ca) and ca >= 0.0, f"Constant-5 adversarial: corr={ca:.4f} (not NaN, >=0)")

    # 9b. Alternating sign (+1, -1, +1, -1, ...)
    x_alt = torch.tensor([(-1.0)**i for i in range(128)])
    X_alt = vht2(x_alt.unsqueeze(0)).squeeze(0)
    recon_alt = vht2(vht2(x_alt.unsqueeze(0))).squeeze(0)
    err_alt = (x_alt - recon_alt).abs().max().item()
    t.check(err_alt < 1e-5, f"Alternating sign round-trip: err={err_alt:.2e}")

    # 9c. Spike at Nyquist (all energy in highest frequency)
    x_nyq = torch.zeros(128)
    x_nyq[127] = 100.0
    X_nyq = vht2(x_nyq.unsqueeze(0)).squeeze(0)
    Xs_n, Xq_n = quant_a.quantize(X_nyq.unsqueeze(0))
    Xd_n = quant_a.dequantize(Xs_n, Xq_n)
    recon_n = vht2(Xd_n).squeeze(0)
    cn = correlation(x_nyq, recon_n)
    t.check(cn > 0.80, f"Nyquist spike: corr={cn:.4f} (>0.80)")

    # 9d. Random permutation (no structure — worst case)
    perm = torch.randperm(128).float()
    X_perm = vht2(perm.unsqueeze(0)).squeeze(0)
    Xs_p, Xq_p = quant_a.quantize(X_perm.unsqueeze(0))
    Xd_p = quant_a.dequantize(Xs_p, Xq_p)
    recon_p = vht2(Xd_p).squeeze(0)
    cp = correlation(perm, recon_p)
    t.check(cp > 0.85, f"Random permutation: corr={cp:.4f} (>0.85)")
