"""
verify_kdiff.py — Validate K_diff energy hypothesis for Differential Phase Encoding.

For each (layer, head, position), compute:
  K_even = K[0::2]   (cos-RoPE components)
  K_odd  = K[1::2]   (sin-RoPE components)
  K_avg  = (K_even + K_odd) / 2
  K_diff = K_even - K_odd

Measure:
  energy_ratio   = mean(||K_diff||²) / mean(||K_avg||²)
  pearson_r      = correlation between K_even and K_odd vectors directly
  compression_gain = 1 / energy_ratio  (how many bits we save by dropping K_diff)

If energy_ratio << 1: K_diff is low-energy → differential encoding directly viable
If energy_ratio ~ 1:  K vectors not intrinsically correlated → need residual approach
"""
import sys, os
sys.path.insert(0, r'D:\F\shannon-prime-repos\shannon-prime\tools')
import numpy as np

MODELS = [
    ('qwen36_35b_moe', 4, 3),   # 40L, 2KV heads, period=4, global_offset=3
    ('qwen36_27b',     4, 3),   # 64L, 4KV heads
    ('gemma4_31b',     4, 3),   # 60L, 16KV heads
    ('phi3mini',       None, None),  # 32L, 32KV heads — dense, no split
]
BASE = r'D:\F\shannon-prime-repos\shannon-prime\logs\phase12'


def pearson_r_vectors(X, Y):
    """Mean Pearson r between corresponding row pairs of X and Y (n_samples, dim)."""
    dx = X - X.mean(axis=1, keepdims=True)
    dy = Y - Y.mean(axis=1, keepdims=True)
    num   = (dx * dy).sum(axis=1)
    denom = np.sqrt((dx**2).sum(axis=1) * (dy**2).sum(axis=1))
    valid = denom > 1e-10
    return float(num[valid].sum() / denom[valid].sum())


def analyse(name, period, offset):
    path = os.path.join(BASE, f'kv_{name}.npz')
    if not os.path.exists(path):
        print(f'  {name}: FILE NOT FOUND')
        return

    data = np.load(path)
    kv   = data['k_vectors'].astype(np.float32)   # (L, H, P, D)
    n_layers, n_heads, n_pos, head_dim = kv.shape
    n_s = min(64, n_pos)

    # Layer split
    if period:
        global_idx = [L for L in range(n_layers) if L % period == offset]
        local_idx  = [L for L in range(n_layers) if L % period != offset]
        depth_half = len(global_idx) // 2
        late_global = global_idx[depth_half:]
    else:
        global_idx = list(range(n_layers))
        local_idx  = []
        depth_half = n_layers // 2
        late_global = list(range(depth_half, n_layers))

    print(f'\n  {name}  ({n_layers}L, {n_heads}H, {head_dim}D)')
    print(f'  {"Layer subset":20s}  {"energy_ratio":>12s}  {"pearson_r":>10s}  {"gain_est":>9s}')
    print(f'  {"─"*20}  {"─"*12}  {"─"*10}  {"─"*9}')

    for label, idx_list in [
        ('all layers',    list(range(n_layers))),
        ('global layers', global_idx),
        ('late global',   late_global),
        ('local layers',  local_idx),
    ]:
        if not idx_list:
            continue
        kv_sub = kv[idx_list, :, :n_s, :]   # (N, H, P, D)
        # Flatten to (N*H*P, D)
        flat = kv_sub.reshape(-1, head_dim)

        # Handle odd head_dim: truncate to even number of pairs
        D_pairs = (head_dim // 2) * 2
        K_even = flat[:, 0:D_pairs:2]   # (samples, D//2)
        K_odd  = flat[:, 1:D_pairs:2]

        K_avg  = (K_even + K_odd) / 2.0
        K_diff =  K_even - K_odd

        e_avg  = float((K_avg  ** 2).sum(axis=1).mean())
        e_diff = float((K_diff ** 2).sum(axis=1).mean())

        ratio = e_diff / e_avg if e_avg > 1e-12 else float('nan')
        gain  = 1.0 / ratio    if ratio > 1e-12 else float('inf')
        r_kv  = pearson_r_vectors(K_even, K_odd)

        print(f'  {label:20s}  {ratio:12.4f}  {r_kv:10.4f}  {gain:8.1f}x')


for name, period, offset in MODELS:
    analyse(name, period, offset)

print('\nInterpretation:')
print('  energy_ratio < 0.10  → K_diff is low-energy; 1-bit or drop viable')
print('  energy_ratio < 0.50  → K_diff compressible to 2-3 bits')
print('  energy_ratio > 0.80  → K vectors not intrinsically correlated; use residual approach')
print('  pearson_r    > 0.70  → K_even and K_odd share significant structure')
