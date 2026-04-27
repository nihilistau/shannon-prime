"""
wan_rope_split.py — 3D RoPE dimension analysis for Wan self-attention K vectors.

Wan uses EmbedND with axes_dim=[d - 4*(d//6), 2*(d//6), 2*(d//6)] where d = head_dim // 2.
At head_dim=128: d=64, axes: temporal=24, x=20, y=20 (each doubled for cos/sin = 48+40+40=128).

For each axis group, computes:
  - Pearson r between even-dim and odd-dim residual norms (the T3 signal)
  - Direct Pearson r between K_even and K_odd (magnitude tracking check)
  - Energy ratio K_diff / K_avg

Across ALL blocks and early/mid/late thirds.
"""
import sys, math
import numpy as np

sys.path.insert(0, r'D:\F\shannon-prime-repos\shannon-prime\tools')

NPZ = r'D:\F\shannon-prime-repos\shannon-prime\logs\phase12\wan_self_attn_step20.npz'
data = np.load(NPZ, allow_pickle=True)
kv   = data['k_vectors'].astype(np.float32)   # (30, 24, 256, 128)

n_blocks, n_heads, n_tokens, head_dim = kv.shape
d = head_dim   # Wan uses full head_dim in EmbedND: dim=d where d=dim//num_heads

# 3D RoPE axis dims from: axes_dim=[d - 4*(d//6), 2*(d//6), 2*(d//6)]
# Each axis gets this many scalar dimensions (applied as cos/sin pairs within each axis)
ax_t = d - 4 * (d // 6)   # temporal  (d=128 → 128-84=44) → dims   0.. 43
ax_x = 2 * (d // 6)        # spatial x (→ 42)               → dims  44.. 85
ax_y = 2 * (d // 6)        # spatial y (→ 42)               → dims  86..127
assert ax_t + ax_x + ax_y == head_dim, f"axis sum {ax_t+ax_x+ax_y} != {head_dim}"

# Dim slices (contiguous blocks within head_dim)
slices = {
    'temporal':  (0,           ax_t),
    'spatial_x': (ax_t,        ax_t + ax_x),
    'spatial_y': (ax_t + ax_x, head_dim),
}

print(f"Shape: {kv.shape}")
print(f"3D RoPE axes: temporal={ax_t} pairs ({2*ax_t} dims), "
      f"x={ax_x} pairs ({2*ax_x} dims), y={ax_y} pairs ({2*ax_y} dims)")
print()


def pearson_r(a, b):
    """Pearson r between two 1-D arrays."""
    da, db = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((da**2).sum() * (db**2).sum()))
    return float((da * db).sum() / denom) if denom > 1e-10 else 0.0


def analyse_slice(kv_subset, dim_start, dim_end, label, skeleton_frac=0.30):
    """
    For a block×head×token×dim_slice tensor, compute:
      - T3 r: Pearson r between ||residual_even|| and ||residual_odd||
      - Direct r: mean element-wise Pearson r between K_even and K_odd
      - energy_ratio: mean(||K_diff||^2) / mean(||K_avg||^2)

    dim_start/end index into head_dim (the slice for this RoPE axis).
    """
    kslice = kv_subset[:, :, :, dim_start:dim_end]  # (B, H, T, D_slice)
    B, H, T, D = kslice.shape
    D_pairs = (D // 2) * 2

    # Flatten to (samples, D_slice)
    flat = kslice.reshape(-1, D)   # (B*H*T, D)
    K_even = flat[:, 0:D_pairs:2]  # (N, D//2)
    K_odd  = flat[:, 1:D_pairs:2]
    K_avg  = (K_even + K_odd) / 2
    K_diff = K_even - K_odd

    e_avg  = float((K_avg  ** 2).sum(axis=1).mean())
    e_diff = float((K_diff ** 2).sum(axis=1).mean())
    energy_ratio = e_diff / e_avg if e_avg > 1e-12 else float('nan')

    # Direct K_even vs K_odd Pearson r (per-dim mean)
    direct_rs = []
    for d_idx in range(min(D // 2, 16)):   # sample first 16 dim pairs
        r = pearson_r(flat[:, d_idx * 2], flat[:, d_idx * 2 + 1])
        direct_rs.append(r)
    direct_r = float(np.mean(direct_rs))

    # T3-style: residual norm correlation after skeleton reconstruction
    # Use a simple T2 skeleton at skeleton_frac of D_slice
    from sp_diagnostics import algebraic_skeleton, vht2_batch, precompute_spectra
    # Rebuild per-block analysis
    kslice_nb = kslice.reshape(B * H, T, D)  # flatten blocks+heads
    n_sample = min(64, T)
    kslice_sample = kslice_nb[:, :n_sample, :]  # (B*H, n_s, D)

    # Pad to nearest factorable dim if needed (D must factor into {2,3,5,7,11})
    from sp_diagnostics import sqfree_pad_dim
    analysis_dim = sqfree_pad_dim(D)

    # Precompute spectra
    kv4d = kslice_sample.reshape(1, B*H, n_sample, D)
    spectra, originals = precompute_spectra(kv4d, analysis_dim, True, n_sample)
    # spectra: (1, B*H, n_s, analysis_dim)

    skel = algebraic_skeleton(analysis_dim, {2, 3})
    k_cap = max(1, int(skeleton_frac * analysis_dim))
    skel  = skel[:min(k_cap, len(skel))]

    mask = np.zeros(analysis_dim, dtype=np.float32)
    mask[skel[skel < analysis_dim]] = 1.0

    from sp_diagnostics import vht2_batch
    specs_flat = spectra[0].reshape(-1, analysis_dim)   # (B*H*n_s, dim)
    origs_flat = originals[0].reshape(-1, analysis_dim)
    sparse     = specs_flat * mask[None, :]
    recons     = vht2_batch(sparse)
    resids     = origs_flat - recons  # (N, dim)

    ev = np.linalg.norm(resids[:, 0::2], axis=1)
    od = np.linalg.norm(resids[:, 1::2], axis=1)
    t3_r = pearson_r(ev, od) if ev.std() > 1e-10 and od.std() > 1e-10 else 0.0

    return {
        'label':        label,
        'dims':         f'{dim_start}..{dim_end-1}',
        'n_dims':       D,
        'energy_ratio': round(energy_ratio, 4),
        'direct_r':     round(direct_r, 4),
        't3_r':         round(t3_r, 4),
    }


# Run for each axis group + all blocks, early third, late third
depth_splits = {
    'all':   (0, n_blocks),
    'early': (0, n_blocks // 3),
    'late':  (2 * n_blocks // 3, n_blocks),
}

print(f"{'Axis':10s}  {'Depth':6s}  {'energy_ratio':>12s}  {'direct_r':>9s}  {'t3_r':>7s}")
print("-" * 55)

for depth_label, (b_start, b_end) in depth_splits.items():
    kv_sub = kv[b_start:b_end]
    for axis_label, (d_start, d_end) in slices.items():
        res = analyse_slice(kv_sub, d_start, d_end, f"{axis_label}/{depth_label}")
        print(f"{axis_label:10s}  {depth_label:6s}  {res['energy_ratio']:12.4f}  "
              f"{res['direct_r']:9.4f}  {res['t3_r']:7.4f}")
    print()

print("\nKey: energy_ratio = Var(K_diff)/Var(K_avg)  [1.0 = independent, <0.5 = phase-locked]")
print("     direct_r = Pearson r between K_even and K_odd raw vectors")
print("     t3_r = Pearson r of residual norms (what T3 measured globally)")
