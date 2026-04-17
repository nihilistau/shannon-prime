# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
Tests for the squarefree prime-Hartley + Möbius CSR + spinor path.

Run: python tests/test_sqfree.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

# Import from the new module
from backends.torch.shannon_prime_sqfree import (
    mobius, is_squarefree, sqfree_pad_dim, sqfree_pad, sqfree_unpad,
    vilenkin_forward, vilenkin_inverse, _prime_factorize,
    build_knight_mask, quantize_residual, dequantize_residual,
    SqfreeShadowCache,
)

# Import from the scaling law
from tools.sp_scaling_law import (
    predicted_ppl_ratio, predicted_ppl_pct, is_pareto_viable,
    min_k_corr_for_budget, safe_k_corr_table,
)

# Import from original backend for correlation
from backends.torch.shannon_prime_torch import correlation

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")


def make_rope_like(n: int, head_dim: int, seed: int = 42) -> torch.Tensor:
    """Generate RoPE-like K vectors with spectral concentration."""
    torch.manual_seed(seed)
    idx = torch.arange(1, head_dim + 1, dtype=torch.float32)
    decay = idx ** -1.5
    signs = torch.randint(0, 2, (n, head_dim), dtype=torch.float32) * 2 - 1
    spectrum = signs * decay.unsqueeze(0) + torch.randn(n, head_dim) * 1e-3
    return spectrum  # Already in spectral domain; VHT2 would give time-domain


# ============================================================================
# Möbius function tests
# ============================================================================

print("\n── Möbius function ──")

check("μ(1)=1",    mobius(1) == 1)
check("μ(2)=-1",   mobius(2) == -1)
check("μ(3)=-1",   mobius(3) == -1)
check("μ(4)=0",    mobius(4) == 0,  f"got {mobius(4)}")
check("μ(5)=-1",   mobius(5) == -1)
check("μ(6)=1",    mobius(6) == 1)
check("μ(7)=-1",   mobius(7) == -1)
check("μ(8)=0",    mobius(8) == 0)
check("μ(9)=0",    mobius(9) == 0,  f"got {mobius(9)}")
check("μ(10)=1",   mobius(10) == 1)
check("μ(30)=-1",  mobius(30) == -1)

check("sqfree(1)",  is_squarefree(1))
check("sqfree(6)",  is_squarefree(6))    # 2·3
check("!sqfree(4)", not is_squarefree(4))  # 2²
check("!sqfree(9)", not is_squarefree(9))  # 3²
check("sqfree(30)", is_squarefree(30))    # 2·3·5

# ============================================================================
# Sqfree padding tests
# ============================================================================

print("\n── Sqfree padding ──")

pd64 = sqfree_pad_dim(64)
pd128 = sqfree_pad_dim(128)
pd256 = sqfree_pad_dim(256)

check(f"pad(64)={pd64} is sqfree",  pd64 >= 64 and is_squarefree(pd64))
check(f"pad(128)={pd128} is sqfree", pd128 >= 128 and is_squarefree(pd128))
check(f"pad(256)={pd256} is sqfree", pd256 >= 256 and is_squarefree(pd256))

# Known values from the papers
check("pad(64)=66",  pd64 == 66,  f"got {pd64}")  # 2·3·11
check("pad(128)=154", pd128 == 154, f"got {pd128}")  # 2·7·11

# Pad/unpad roundtrip
x = torch.randn(4, 64)
padded = sqfree_pad(x, pd64)
check("pad shape", padded.shape == (4, pd64))
unpadded = sqfree_unpad(padded, 64)
check("pad/unpad lossless", torch.allclose(x, unpadded, atol=1e-6))

# ============================================================================
# Prime-Hartley (Vilenkin) transform tests
# ============================================================================

print("\n── Vilenkin transform ──")

# Roundtrip on various sqfree dimensions
for dim in [6, 30, 66, 154, 210]:
    try:
        factors = _prime_factorize(dim)
        v = torch.randn(dim, dtype=torch.float32)
        fwd = vilenkin_forward(v.unsqueeze(0), factors).squeeze(0)
        back = vilenkin_inverse(fwd.unsqueeze(0), factors).squeeze(0)
        err = (v - back).abs().max().item()
        check(f"Vilenkin roundtrip dim={dim}", err < 1e-4, f"err={err:.2e}")
    except Exception as e:
        check(f"Vilenkin roundtrip dim={dim}", False, str(e))

# Batch roundtrip
batch = torch.randn(8, 30)
factors_30 = _prime_factorize(30)
fwd_batch = vilenkin_forward(batch, factors_30)
back_batch = vilenkin_inverse(fwd_batch, factors_30)
batch_err = (batch - back_batch).abs().max().item()
check(f"Vilenkin batch roundtrip", batch_err < 1e-4, f"err={batch_err:.2e}")

# ============================================================================
# Knight mask + Möbius CSR tests
# ============================================================================

print("\n── Knight mask + CSR ──")

mask = build_knight_mask(66, 50)
check("mask dim", mask.dim == 66)
check("mask sk_k ≤ 50", mask.sk_k <= 50)
check("mask skeleton sorted", torch.all(mask.skeleton_idx[:-1] <= mask.skeleton_idx[1:]).item())
check("mask residual sorted", mask.residual_idx.numel() == 0 or
      torch.all(mask.residual_idx[:-1] <= mask.residual_idx[1:]).item())

# No overlap between skeleton and residual
skel_set = set(mask.skeleton_idx.tolist())
res_set = set(mask.residual_idx.tolist())
check("skel ∩ res = ∅", len(skel_set & res_set) == 0)

# CSR offsets are valid
n_res = mask.residual_idx.numel()
check("CSR offsets length", mask.csr_offsets.numel() == n_res + 1)
check("CSR offsets monotone", torch.all(mask.csr_offsets[:-1] <= mask.csr_offsets[1:]).item())
check("CSR offsets start=0", mask.csr_offsets[0].item() == 0)
check("CSR offsets end matches",
      mask.csr_offsets[-1].item() == mask.csr_skel_slot.numel())

# Larger dimension
mask128 = build_knight_mask(154, 100)
check(f"mask(154) n_res={mask128.residual_idx.numel()}", mask128.residual_idx.numel() > 0)

# ============================================================================
# Residual quantization tests
# ============================================================================

print("\n── Residual quantization ──")

for nbits in [1, 2, 3, 4]:
    vals = torch.randn(100)
    mag = vals.abs().mean()
    levels = quantize_residual(vals, nbits, mag)
    recon = dequantize_residual(levels, nbits, mag)

    L = 1 << nbits
    check(f"{nbits}-bit levels in [0, {L-1}]",
          levels.min().item() >= 0 and levels.max().item() <= L - 1)

    # SNR should improve with more bits
    noise = (vals - recon).pow(2).mean().sqrt()
    signal = vals.pow(2).mean().sqrt()
    snr = 20 * math.log10(signal / noise.clamp(min=1e-12))
    check(f"{nbits}-bit SNR ≥ {3*nbits} dB", snr >= 3 * nbits,
          f"got {snr:.1f} dB")

# ============================================================================
# Full sqfree shadow cache roundtrip
# ============================================================================

print("\n── Sqfree shadow cache ──")

cache = SqfreeShadowCache(
    head_dim=128, n_layers=2, n_heads_kv=4, max_seq_len=16,
    band_bits=[5, 4, 4, 4, 5], residual_bits=3, use_spinor=True,
)

# Write and read K vectors
torch.manual_seed(123)
for pos in range(4):
    k = torch.randn(128)
    cache.write_k(layer=0, head=0, pos=pos, k_vec=k)
    k_recon = cache.read_k(layer=0, head=0, pos=pos)
    corr = correlation(k, k_recon)
    check(f"K roundtrip pos={pos} corr={corr:.4f} ≥ 0.95",
          corr >= 0.95, f"got {corr:.4f}")

# Write and read V vectors
for pos in range(4):
    v = torch.randn(128)
    cache.write_v(layer=0, head=0, pos=pos, v_vec=v)
    v_recon = cache.read_v(layer=0, head=0, pos=pos)
    corr = correlation(v, v_recon)
    check(f"V roundtrip pos={pos} corr={corr:.4f} ≥ 0.95",
          corr >= 0.95, f"got {corr:.4f}")

# Compression ratio
ratio = cache.compression_ratio()
check(f"compression ratio {ratio:.2f}× ≥ 2.0", ratio >= 2.0,
      f"got {ratio:.2f}×")

# Spinor vs no-spinor: spinor should not regress on random data
cache_nospin = SqfreeShadowCache(
    head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=4,
    band_bits=[5, 4, 4, 4, 5], residual_bits=3, use_spinor=False,
)
cache_spin = SqfreeShadowCache(
    head_dim=128, n_layers=1, n_heads_kv=1, max_seq_len=4,
    band_bits=[5, 4, 4, 4, 5], residual_bits=3, use_spinor=True,
)

torch.manual_seed(456)
corrs_nospin = []
corrs_spin = []
for i in range(4):
    k = torch.randn(128)
    cache_nospin.write_k(0, 0, i, k)
    cache_spin.write_k(0, 0, i, k)
    corrs_nospin.append(correlation(k, cache_nospin.read_k(0, 0, i)))
    corrs_spin.append(correlation(k, cache_spin.read_k(0, 0, i)))

mean_nospin = sum(corrs_nospin) / len(corrs_nospin)
mean_spin = sum(corrs_spin) / len(corrs_spin)
check(f"spinor no regression (spin={mean_spin:.4f} vs nospin={mean_nospin:.4f})",
      mean_spin >= mean_nospin - 0.01,
      f"spinor regressed by {mean_nospin - mean_spin:.4f}")

# hd=64 also works
cache64 = SqfreeShadowCache(
    head_dim=64, n_layers=1, n_heads_kv=1, max_seq_len=4,
    band_bits=[5, 4, 4, 4, 5], residual_bits=3, use_spinor=True,
)
k64 = torch.randn(64)
cache64.write_k(0, 0, 0, k64)
k64_recon = cache64.read_k(0, 0, 0)
corr64 = correlation(k64, k64_recon)
check(f"hd=64 roundtrip corr={corr64:.4f} ≥ 0.92", corr64 >= 0.92,
      f"got {corr64:.4f}")

# ============================================================================
# Scaling law tests
# ============================================================================

print("\n── Scaling law ──")

# Sanity: perfect K_corr → ratio 1.0
check("k_corr=1.0 → ratio=1.0", predicted_ppl_ratio(1.0, 8.0, 8) == 1.0)

# Sanity: lower K_corr → worse ratio
r1 = predicted_ppl_ratio(0.99, 8.0, 8)
r2 = predicted_ppl_ratio(0.95, 8.0, 8)
check("lower k_corr → worse ratio", r2 > r1, f"{r2:.4f} vs {r1:.4f}")

# Sanity: bigger model → better tolerance
r_small = predicted_ppl_ratio(0.97, 1.0, 8)
r_big = predicted_ppl_ratio(0.97, 70.0, 8)
check("bigger model → less PPL impact", r_big < r_small,
      f"70B={r_big:.4f} vs 1B={r_small:.4f}")

# Sanity: higher bits → better tolerance
r_q8 = predicted_ppl_ratio(0.97, 8.0, 8)
r_q3 = predicted_ppl_ratio(0.97, 8.0, 3)
check("Q8 better than Q3", r_q8 < r_q3, f"Q8={r_q8:.4f} vs Q3={r_q3:.4f}")

# Pareto viability
check("k_corr=0.999 always viable", is_pareto_viable(0.999, 1.0, 8))
check("k_corr=0.5 never viable",   not is_pareto_viable(0.5, 70.0, 16))

# Min K_corr floor
floor = min_k_corr_for_budget(1.0, 8, 0.03)
check(f"Dolphin 1B Q8 floor={floor:.4f} ≈ 0.988",
      abs(floor - 0.988) < 0.005, f"got {floor:.4f}")

floor_70b = min_k_corr_for_budget(70.0, 8, 0.03)
check(f"70B Q8 floor={floor_70b:.4f} < 0.95", floor_70b < 0.95,
      f"got {floor_70b:.4f}")

# Table sanity
table = safe_k_corr_table()
check("table has 6 entries", len(table) == 6)
check("Wan floor < Dolphin floor",
      table["Wan 2.2 14B bf16"] < table["Dolphin 1B Q8"])

# Percentage convenience
pct = predicted_ppl_pct(0.988, 1.0, 8)
check(f"pct format ({pct:.2f}%) positive", pct > 0)

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*50}")
total = passed + failed
print(f" {passed}/{total} tests passed, {failed} failed")
if failed > 0:
    print(f" FAILURES: {failed}")
    sys.exit(1)
else:
    print(f" All tests passed.")
    sys.exit(0)