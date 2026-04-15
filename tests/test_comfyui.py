# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
ComfyUI cross-attention cache integration test.
Tests Wan 2.1 (dense), Wan 2.2 MoE, and the linear-layer caching wrapper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backends', 'torch'))

import torch
torch.manual_seed(42)

from shannon_prime_comfyui import (
    VHT2CrossAttentionCache, WanVHT2Wrapper, WanCrossAttnCachingLinear
)
from shannon_prime_torch import correlation

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
# Test 1: Basic cache put/get (cross-attn K/V, no RoPE)
# ============================================================================
print("\n== Cross-Attention Cache (no RoPE) ==")

cache = VHT2CrossAttentionCache(head_dim=128)

# Cross-attn shape in Wan 14B: (batch=1, n_heads=40, seq_len=77, head_dim=128)
k_orig = torch.randn(1, 40, 77, 128)
v_orig = torch.randn(1, 40, 77, 128)

cache.put("block_0", k_orig, v_orig)
k_recon, v_recon = cache.get("block_0")

k_corr = correlation(k_orig, k_recon)
v_corr = correlation(v_orig, v_recon)
check(k_corr > 0.985, f"K correlation: {k_corr:.4f} (need >0.985)")
check(v_corr > 0.985, f"V correlation: {v_corr:.4f} (need >0.985)")
check(k_recon.shape == k_orig.shape, f"K shape preserved: {k_recon.shape}")
check(k_recon.dtype == k_orig.dtype, f"K dtype preserved: {k_recon.dtype}")

# ============================================================================
# Test 2: Wan 2.1 dense model — 40 blocks × 50 timesteps
# ============================================================================
print("\n== Wan 2.1 Dense (40 blocks × 50 steps) ==")

wrapper = WanVHT2Wrapper(head_dim=128, model_type='wan21')
n_timesteps = 50
n_blocks = 40

text_k = torch.randn(1, 40, 77, 128)
text_v = torch.randn(1, 40, 77, 128)

for t in range(n_timesteps):
    for b in range(n_blocks):
        k, v = wrapper.get_or_compute(
            block_id=f"block_{b}",
            compute_fn=lambda: (text_k.clone(), text_v.clone())
        )

stats = wrapper.stats()
check(stats['misses'] == n_blocks,
      f"Misses: {stats['misses']} (expect {n_blocks})")
check(stats['hits'] == n_timesteps * n_blocks - n_blocks,
      f"Hits: {stats['hits']} (expect {n_timesteps * n_blocks - n_blocks})")
check(stats['hit_rate'] > 0.97,
      f"Hit rate: {stats['hit_rate']:.3f} (expect >0.97)")
check(stats['current_expert'] is None,
      f"No expert set for dense model")

# ============================================================================
# Test 3: Wan 2.2 MoE — expert switching at boundary
# ============================================================================
print("\n== Wan 2.2 MoE (expert switching) ==")

wrapper_moe = WanVHT2Wrapper(
    head_dim=128, model_type='wan22_moe', task_type='t2v'
)
check(wrapper_moe.boundary == 0.875, f"T2V boundary: {wrapper_moe.boundary}")

n_blocks = 40
# Simulate 50 timesteps with sigma going from 1.0 → 0.0
sigmas = [1.0 - i/49 for i in range(50)]
boundary_step = None

# Different text embeddings per expert (they have different projection weights)
text_k_high = torch.randn(1, 40, 77, 128)
text_v_high = torch.randn(1, 40, 77, 128)
text_k_low = torch.randn(1, 40, 77, 128)  # Different weights!
text_v_low = torch.randn(1, 40, 77, 128)

for step, sigma in enumerate(sigmas):
    wrapper_moe.set_expert_from_sigma(sigma)

    if sigma <= 0.875 and boundary_step is None:
        boundary_step = step

    for b in range(n_blocks):
        if sigma > 0.875:
            compute = lambda: (text_k_high.clone(), text_v_high.clone())
        else:
            compute = lambda: (text_k_low.clone(), text_v_low.clone())

        k, v = wrapper_moe.get_or_compute(f"block_{b}", compute)

stats_moe = wrapper_moe.stats()
# Should have misses for first high-noise step AND first low-noise step
expected_misses = n_blocks * 2  # 40 blocks × 2 experts
check(stats_moe['misses'] == expected_misses,
      f"MoE misses: {stats_moe['misses']} (expect {expected_misses} = {n_blocks}×2 experts)")

total_calls = 50 * n_blocks
expected_hits = total_calls - expected_misses
check(stats_moe['hits'] == expected_hits,
      f"MoE hits: {stats_moe['hits']} (expect {expected_hits})")

check(stats_moe['hit_rate'] > 0.95,
      f"MoE hit rate: {stats_moe['hit_rate']:.3f} (expect >0.95)")

# Verify different experts produce different cache entries
check(stats_moe['n_entries_cached'] == n_blocks * 2,
      f"Cache entries: {stats_moe['n_entries_cached']} (expect {n_blocks * 2} = {n_blocks}×2)")

# Verify the cached values for each expert are actually different
wrapper_moe.cache.set_expert('high_noise')
k_high, _ = wrapper_moe.cache.get("block_0")
wrapper_moe.cache.set_expert('low_noise')
k_low, _ = wrapper_moe.cache.get("block_0")
expert_diff = correlation(k_high, k_low)
check(abs(expert_diff) < 0.3,
      f"Experts cached different K (cross-corr={expert_diff:.3f}, expect ~0)")

# ============================================================================
# Test 4: I2V boundary is different from T2V
# ============================================================================
print("\n== Wan 2.2 I2V vs T2V boundary ==")
wrapper_i2v = WanVHT2Wrapper(head_dim=128, model_type='wan22_moe', task_type='i2v')
check(wrapper_i2v.boundary == 0.900, f"I2V boundary: {wrapper_i2v.boundary}")

# ============================================================================
# Test 5: TI2V-5B (dense, no MoE) behaves like Wan 2.1
# ============================================================================
print("\n== Wan 2.2 TI2V-5B (dense) ==")
wrapper_5b = WanVHT2Wrapper(head_dim=128, model_type='wan22_5b')
check(wrapper_5b.boundary is None, "5B has no MoE boundary")

k_5b = torch.randn(1, 24, 77, 128)  # 5B has 24 heads
v_5b = torch.randn(1, 24, 77, 128)
for t in range(10):
    for b in range(30):  # 5B has 30 layers
        wrapper_5b.get_or_compute(f"b_{b}", lambda: (k_5b.clone(), v_5b.clone()))

stats_5b = wrapper_5b.stats()
check(stats_5b['misses'] == 30, f"5B misses: {stats_5b['misses']} (expect 30)")

# ============================================================================
# Test 6: WanCrossAttnCachingLinear (linear layer replacement)
# ============================================================================
print("\n== Linear Layer Caching Wrapper ==")

cache2 = VHT2CrossAttentionCache(head_dim=128)

# Simulate a cross_attn_k linear: (dim=5120) → (dim=5120), then reshaped
original_linear = torch.nn.Linear(5120, 5120, bias=False)
wrapped = WanCrossAttnCachingLinear(original_linear, cache2, "block_0_k")

# First call: computes and caches
context = torch.randn(1, 77, 5120)
out1 = wrapped(context)
check(cache2.has("block_0_k"), "First call caches result")

# Second call: returns from cache
out2 = wrapped(context)
out_corr = correlation(out1, out2)
check(out_corr > 0.985, f"Cached linear output correlation: {out_corr:.4f}")

# ============================================================================
# Test 7: Both K and V get Möbius reorder (cross-attn specific)
# ============================================================================
print("\n== Cross-Attn Möbius on Both K and V ==")
cache_m = VHT2CrossAttentionCache(head_dim=128, use_mobius=True)
cache_no = VHT2CrossAttentionCache(head_dim=128, use_mobius=False)

k_test = torch.randn(1, 8, 32, 128)
v_test = torch.randn(1, 8, 32, 128)

cache_m.put("test", k_test, v_test)
cache_no.put("test", k_test, v_test)

km, vm = cache_m.get("test")
kn, vn = cache_no.get("test")

# Both should work well — Möbius may help on V too since no RoPE asymmetry
mk = correlation(k_test, km)
nk = correlation(k_test, kn)
mv = correlation(v_test, vm)
nv = correlation(v_test, vn)

check(mk > 0.98 and nk > 0.98, f"K: möbius={mk:.4f}, no_möbius={nk:.4f}")
check(mv > 0.98 and nv > 0.98, f"V: möbius={mv:.4f}, no_möbius={nv:.4f}")

# ============================================================================
# Test 8: Compression ratio
# ============================================================================
print("\n== Compression ==")
ratio = cache.compression_ratio()
check(ratio > 2.5, f"Compression ratio: {ratio:.1f}× (need >2.5×)")

# ============================================================================
# Test 9: Expert cache clearing
# ============================================================================
print("\n== Expert Cache Clearing ==")
cache_ec = VHT2CrossAttentionCache(head_dim=128)
cache_ec.set_expert('high_noise')
cache_ec.put("b0", torch.randn(128), torch.randn(128))
cache_ec.put("b1", torch.randn(128), torch.randn(128))
cache_ec.set_expert('low_noise')
cache_ec.put("b0", torch.randn(128), torch.randn(128))

check(len(cache_ec._cache) == 3, f"3 entries before clear")
cache_ec.clear_expert('high_noise')
check(len(cache_ec._cache) == 1, f"1 entry after clearing high_noise")
check(cache_ec.has("b0"), "low_noise:b0 still cached")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
sys.exit(0 if passed == total else 1)
