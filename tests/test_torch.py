# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
Test suite for PyTorch backend.
Validates against the same invariants as the C core tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backends', 'torch'))

import torch
torch.manual_seed(42)

from shannon_prime_torch import (
    vht2, MobiusMask, BandedQuantizer,
    ShadowCache, correlation,
    sqfree_pad_dim,
)

passed = 0
total = 0

def check(cond, msg):
    global passed, total
    total += 1
    if cond:
        passed += 1
        print(f"  [\033[32mPASS\033[0m] {msg}")
    else:
        print(f"  [\033[31mFAIL\033[0m] {msg}")


# ============================================================================
# Test 1: VHT2 round-trip (self-inverse, no 1/N)
# ============================================================================
print("\n== VHT2 Round-Trip ==")
for hd in [32, 64, 128, 256]:
    orig = torch.randn(hd)
    recon = vht2(vht2(orig.unsqueeze(0))).squeeze(0)
    err = (orig - recon).abs().max().item()
    check(err < 1e-5, f"VHT2 round-trip hd={hd}: max_err={err:.2e}")

# VHT2 also works on sqfree (non-power-of-2) dimensions
for hd, pad in [(64, 66), (128, 154), (210, 210)]:
    assert sqfree_pad_dim(hd) == pad or hd == pad
    orig = torch.randn(pad)
    recon = vht2(vht2(orig.unsqueeze(0))).squeeze(0)
    err = (orig - recon).abs().max().item()
    check(err < 1e-4, f"VHT2 sqfree round-trip n={pad}: max_err={err:.2e}")

# ============================================================================
# Test 2: Möbius values
# ============================================================================
print("\n== Möbius Function ==")
mask = MobiusMask(32)
check(mask.mu[1] ==  1, "μ(1) = 1")
check(mask.mu[2] == -1, "μ(2) = -1")
check(mask.mu[3] == -1, "μ(3) = -1")
check(mask.mu[4] ==  0, "μ(4) = 0")
check(mask.mu[5] == -1, "μ(5) = -1")
check(mask.mu[6] ==  1, "μ(6) = 1")
check(mask.mu[30] == -1, "μ(30) = -1")

# ============================================================================
# Test 3: Möbius reorder/unreorder roundtrip
# ============================================================================
print("\n== Möbius Reorder Round-Trip ==")
for hd in [64, 128]:
    mask = MobiusMask(hd)
    orig = torch.randn(hd)
    work = mask.reorder(orig)
    work = mask.unreorder(work)
    err = (orig - work).abs().max().item()
    check(err < 1e-7, f"Möbius roundtrip hd={hd}: max_err={err:.2e}")

# ============================================================================
# Test 4: Banded quantization
# ============================================================================
print("\n== Banded Quantization ==")
hd = 128
configs = [
    ([5,5,4,3], 0.01, "5/5/4/3 (ship default)"),
    ([5,4,4,3], 0.02, "5/4/4/3"),
    ([4,4,4,3], 0.02, "4/4/4/3"),
    ([4,3,3,3], 0.05, "4/3/3/3"),
    ([3,3,3,3], 0.08, "3/3/3/3 (floor)"),
]
for bits, max_loss, name in configs:
    bq = BandedQuantizer(hd, bits)
    total_corr = 0.0
    n_trials = 100
    for _ in range(n_trials):
        orig = torch.randn(hd)
        # Forward VHT2
        work = vht2(orig.unsqueeze(0)).squeeze(0)
        scales, quants = bq.quantize(work.unsqueeze(0))
        recon = bq.dequantize(
            [s.unsqueeze(0) for s in [ss.squeeze(0) for ss in scales]],
            [q.unsqueeze(0) for q in [qq.squeeze(0) for qq in quants]]
        ).squeeze(0)
        # Inverse VHT2 (= forward, self-inverse)
        recon = vht2(recon.unsqueeze(0)).squeeze(0)
        total_corr += correlation(orig, recon)
    avg_corr = total_corr / n_trials
    comp = hd * 2 / bq.compressed_bytes_per_vec()
    check(avg_corr > 1.0 - max_loss,
          f"{name}: avg_corr={avg_corr:.4f}, compression={comp:.1f}×")

# ============================================================================
# Test 5: K/V spectral asymmetry
# ============================================================================
print("\n== K/V Spectral Asymmetry ==")
hd = 128
band_size = hd // 4

# K: structured periodic signal (simulates RoPE)
t = torch.arange(hd, dtype=torch.float32)
k_vec = (torch.cos(2 * 3.14159 * t / 7) +
         0.5 * torch.cos(2 * 3.14159 * t / 13) +
         0.3 * torch.cos(2 * 3.14159 * t / 3))
k_vec = vht2(k_vec.unsqueeze(0)).squeeze(0)

# V: random
v_vec = torch.randn(hd)
v_vec = vht2(v_vec.unsqueeze(0)).squeeze(0)

k_energy = [(k_vec[b*band_size:(b+1)*band_size]**2).sum().item() for b in range(4)]
v_energy = [(v_vec[b*band_size:(b+1)*band_size]**2).sum().item() for b in range(4)]
k_total = sum(k_energy)
v_total = sum(v_energy)

k_first = (k_energy[0] + k_energy[1]) / k_total
v_first = (v_energy[0] + v_energy[1]) / v_total
check(k_first > 0.6, f"K first-half energy: {k_first*100:.1f}% (expect >60%)")
check(0.3 < v_first < 0.7, f"V first-half energy: {v_first*100:.1f}% (expect ~50%)")

# ============================================================================
# Test 6: VHT2 on multi-prime dimensions (replaces the old Vilenkin test)
# ============================================================================
print("\n== VHT2 Multi-Prime Round-Trip ==")
for n in [6, 30, 210]:  # 2·3, 2·3·5, 2·3·5·7
    orig = torch.randn(n)
    recon = vht2(vht2(orig.unsqueeze(0))).squeeze(0)
    err = (orig - recon).abs().max().item()
    check(err < 1e-4, f"VHT2 n={n} (multi-prime): max_err={err:.2e}")

# ============================================================================
# Test 7: Full VHT2 pipeline
# ============================================================================
print("\n== Full VHT2 Pipeline ==")
cache = ShadowCache(head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=8)

k_orig = torch.randn(128)
cache.write_k(0, 0, 0, k_orig)
k_recon = cache.read_k(0, 0, 0)
k_corr = correlation(k_orig, k_recon)
check(k_corr > 0.990, f"K full pipeline: correlation={k_corr:.4f} (need >0.990)")

v_orig = torch.randn(128)
cache.write_v(0, 0, 0, v_orig)
v_recon = cache.read_v(0, 0, 0)
v_corr = correlation(v_orig, v_recon)
check(v_corr > 0.950, f"V full pipeline: correlation={v_corr:.4f} (need >0.950)")

ratio = cache.compression_ratio()
print(f"  Compression ratio: {ratio:.1f}×")

# ============================================================================
# Test 8: Möbius quality improvement
# ============================================================================
print("\n== Möbius Quality Improvement ==")
hd = 128
bq = BandedQuantizer(hd, [5,5,4,3])
mask = MobiusMask(hd)
n_trials = 200

total_plain = 0.0
total_mobius = 0.0
for _ in range(n_trials):
    t = torch.arange(hd, dtype=torch.float32)
    orig = (torch.cos(2*3.14159*t/5) * (1 + 0.3*torch.sin(t*0.1))
            + 0.2*(torch.rand(hd) - 0.5))

    # Without Möbius
    work = vht2(orig.unsqueeze(0)).squeeze(0)
    s, q = bq.quantize(work.unsqueeze(0))
    recon = bq.dequantize(s, q).squeeze(0)
    recon = vht2(recon.unsqueeze(0)).squeeze(0)
    total_plain += correlation(orig, recon)

    # With Möbius
    work = vht2(orig.unsqueeze(0)).squeeze(0)
    work = mask.reorder(work)
    s, q = bq.quantize(work.unsqueeze(0))
    recon = bq.dequantize(s, q).squeeze(0)
    recon = mask.unreorder(recon)
    recon = vht2(recon.unsqueeze(0)).squeeze(0)
    total_mobius += correlation(orig, recon)

avg_plain = total_plain / n_trials
avg_mobius = total_mobius / n_trials
check(avg_mobius >= avg_plain - 0.001,
      f"Plain: {avg_plain:.4f}, Möbius: {avg_mobius:.4f} (Δ={avg_mobius-avg_plain:+.4f})")

# ============================================================================
# Test 9: Compression ratios
# ============================================================================
print("\n== Compression Ratios ==")
cache128 = ShadowCache(head_dim=128, n_layers=1, n_heads_kv=1)
r = cache128.compression_ratio()
check(3.0 < r < 4.5, f"hd=128 compression: {r:.1f}× (paper: 3.4–3.8×)")

cache64 = ShadowCache(head_dim=64, n_layers=1, n_heads_kv=1)
r = cache64.compression_ratio()
check(2.5 < r < 5.0, f"hd=64 compression: {r:.1f}×")

# ============================================================================
# Test 10: Memory estimation
# ============================================================================
print("\n== Memory Estimation ==")
cache = ShadowCache(head_dim=128, n_layers=32, n_heads_kv=8, max_seq_len=32768)
mem = cache.memory_bytes(32768)
print(f"  32K context: {mem['total_bytes']/1024/1024:.1f} MB "
      f"(vs {mem['baseline_bytes']/1024/1024:.1f} MB fp16, "
      f"{mem['ratio']:.1f}× compression)")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
sys.exit(0 if passed == total else 1)
