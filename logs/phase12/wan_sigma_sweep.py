"""
wan_sigma_sweep.py — Temporal Skip-Geodesic analysis across denoising steps.

Loads Wan self-attention K vectors captured at steps {5, 15, 20, 25, 35, 45}
and measures per-block cosine similarity of the temporal RoPE dimensions across steps.

If the temporal K vectors at step 5 are highly similar to those at step 45 (cosine > 0.95),
the "Temporal Skeleton" is established early and can be frozen for most of the denoising.
If they diverge, the temporal axes are actively updated throughout denoising.

Axes (at head_dim=128, d=128):
  temporal: dims  0..43   (44 dims, from axes_dim[0] = d - 4*(d//6))
  spatial_x: dims 44..85  (42 dims)
  spatial_y: dims 86..127 (42 dims)
"""
import sys, os
import numpy as np

BASE = r'D:\F\shannon-prime-repos\shannon-prime\logs\phase12'
STEPS = [5, 15, 20, 35, 45]

# Load all available step files
step_data = {}
for s in STEPS:
    p = os.path.join(BASE, f'wan_self_attn_step{s}.npz')
    if os.path.exists(p):
        d = np.load(p)
        step_data[s] = d['k_vectors'].astype(np.float32)
        n_blocks, n_heads, n_tokens, head_dim = step_data[s].shape
        print(f"  step {s:2d}: {step_data[s].shape}")
    else:
        print(f"  step {s:2d}: NOT FOUND (still generating?)")

if len(step_data) < 2:
    print("Need at least 2 steps to compare. Run again when more files are ready.")
    sys.exit(0)

print()
available = sorted(step_data.keys())
n_blocks, n_heads, n_tokens, head_dim = step_data[available[0]].shape

# 3D RoPE axis slices
d = head_dim
ax_t = d - 4 * (d // 6)          # temporal
ax_x = 2 * (d // 6)               # spatial_x
ax_y = 2 * (d // 6)               # spatial_y
slices = {
    'temporal':  (0,           ax_t),
    'spatial_x': (ax_t,        ax_t + ax_x),
    'spatial_y': (ax_t + ax_x, head_dim),
}


def cosine_sim_batch(A, B):
    """Mean cosine similarity between corresponding rows of A and B."""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return float((A_n * B_n).sum(axis=1).mean())


def l2_drift(A, B):
    """Mean relative L2 drift: ||A-B|| / ||A||."""
    diff  = np.linalg.norm(A - B, axis=1)
    norms = np.linalg.norm(A, axis=1)
    return float((diff / (norms + 1e-10)).mean())


# ── Pairwise comparison: reference = earliest step ─────────────────────────
ref_step = available[0]
ref_kv   = step_data[ref_step]

print(f"Reference step: {ref_step}")
print(f"{'Step':>5s}  {'Axis':12s}  {'cos_sim':>8s}  {'l2_drift':>9s}  "
      f"{'cos_early':>10s}  {'cos_late':>9s}")
print("-" * 65)

results = {}
for step in available[1:]:
    cmp_kv = step_data[step]
    for axis_label, (d_start, d_end) in slices.items():
        # All blocks, all heads, all tokens — flatten to (N, D_axis)
        A_all = ref_kv[:, :, :, d_start:d_end].reshape(-1, d_end - d_start)
        B_all = cmp_kv[:, :, :, d_start:d_end].reshape(-1, d_end - d_start)

        # Early blocks (first 10) vs late blocks (last 10)
        n_early = min(10, n_blocks // 3)
        n_late  = min(10, n_blocks // 3)

        A_early = ref_kv[:n_early, :, :, d_start:d_end].reshape(-1, d_end - d_start)
        B_early = cmp_kv[:n_early, :, :, d_start:d_end].reshape(-1, d_end - d_start)
        A_late  = ref_kv[-n_late:, :, :, d_start:d_end].reshape(-1, d_end - d_start)
        B_late  = cmp_kv[-n_late:, :, :, d_start:d_end].reshape(-1, d_end - d_start)

        cs_all   = cosine_sim_batch(A_all, B_all)
        l2_all   = l2_drift(A_all, B_all)
        cs_early = cosine_sim_batch(A_early, B_early)
        cs_late  = cosine_sim_batch(A_late, B_late)

        key = (step, axis_label)
        results[key] = dict(cos_sim=cs_all, l2_drift=l2_all,
                            cos_early=cs_early, cos_late=cs_late)

        flag = ""
        if cs_all > 0.95:   flag = "  ** FROZEN **"
        elif cs_all > 0.85: flag = "  * stable"

        print(f"{step:5d}  {axis_label:12s}  {cs_all:8.4f}  {l2_all:9.4f}  "
              f"{cs_early:10.4f}  {cs_late:9.4f}{flag}")
    print()

# ── Block-by-block temporal stability ──────────────────────────────────────
print("\n=== Per-block temporal cosine similarity (ref=step{} vs others) ===\n".format(ref_step))
print(f"{'Block':>6s}", end="")
for step in available[1:]:
    print(f"  step{step:02d}", end="")
print()
print("-" * (8 + 8 * len(available)))

d_start, d_end = slices['temporal']
for b in range(n_blocks):
    A_b = ref_kv[b, :, :, d_start:d_end].reshape(-1, d_end - d_start)
    print(f"  L{b:02d}  ", end="")
    for step in available[1:]:
        B_b = step_data[step][b, :, :, d_start:d_end].reshape(-1, d_end - d_start)
        cs  = cosine_sim_batch(A_b, B_b)
        flag = "*" if cs > 0.95 else " "
        print(f"  {cs:.4f}{flag}", end="")
    print()

print("\n  * = cosine > 0.95 (temporal freeze viable for this block)")
print("\n=== Interpretation ===")
print("  cos_sim > 0.95: Temporal K vectors nearly identical across sigma → cache reusable")
print("  cos_sim < 0.80: Temporal K vectors actively evolving → must recompute each step")
print("  Early/late gap: if cos_early >> cos_late, structure freezes faster in early blocks")
