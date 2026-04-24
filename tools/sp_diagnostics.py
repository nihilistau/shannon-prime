# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
sp_diagnostics.py — Phase 12 Diagnostic Suite

Four empirical tests derived from cross-domain analogies. Each test takes
the same .npz K-vector input and probes a specific structural hypothesis
about the VHT2 spectral manifold. Run all four, read the JSON, decide
which ideas are worth building on.

  Test 1 — Boundary Sharpness   Is the Regime A/B transition a step function
                                 or a smooth gradient? (Superconductivity analogy)

  Test 2 — Ghost Basin Hunt      Do ghost heads cluster into stable attractor
                                 basins via DBSCAN? (Collatz clustering analogy)

  Test 3 — RoPE Pair Correlation Are reconstruction errors in RoPE (cos, sin)
                                 pairs correlated across positions?
                                 (Twin-prime pairing analogy)

  Test 4 — Fractional Lookahead  Does a fractional-order Grünwald-Letnikov
                                 derivative of the p3 curve give earlier
                                 warning of the regime transition than the
                                 standard first difference?
                                 (Fractional operators analogy)

Input:  K vectors .npz (same format as sp_chord_diagnostic.py / sp_regime_analysis.py)
        Shape: k_vectors → (n_layers, n_heads, n_positions, head_dim)

Output: Per-test verdict + JSON report

Usage:
    python sp_diagnostics.py --input kv_phi3.npz --sqfree
    python sp_diagnostics.py --input kv_phi3.npz --sqfree --output diag.json
    python sp_diagnostics.py --input kv_phi3.npz --skeleton-frac 0.30 --sample 64
    python sp_diagnostics.py --input kv_phi3.npz --tests 1,2,3,4
"""

import sys
import os
import argparse
import json
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError with box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7 fallback


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PRIMES_5   = [2, 3, 5, 7, 11]
PRIMES_EXT = [2, 3, 5, 7, 11, 13, 17, 19, 23]

ENTROPY_GHOST = None         # None → auto: log2(analysis_dim) * 0.70 at runtime
SQFREE_PAD    = {64: 66, 96: 110, 128: 154, 256: 330}


# ─────────────────────────────────────────────────────────────────────────────
# VHT2 math — verbatim from sp_chord_diagnostic.py
# ─────────────────────────────────────────────────────────────────────────────

def _hartley_kernel(p: int) -> np.ndarray:
    idx    = np.arange(p)
    angles = (2.0 * np.pi / p) * np.outer(idx, idx)
    return (np.cos(angles) + np.sin(angles)) / np.sqrt(p)


def _factor_small_primes(n: int) -> List[int]:
    d, primes = n, []
    for p in [2, 3, 5, 7, 11]:
        while d % p == 0:
            primes.append(p)
            d //= p
    if d != 1:
        raise ValueError(f"dim {n} has prime factor > 11 (residue {d})")
    return primes


def _vht2_pow2(x: np.ndarray) -> np.ndarray:
    n, out = len(x), x.copy().astype(np.float64)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = out[j], out[j + h]
                out[j], out[j + h] = a + b, a - b
        h *= 2
    out /= np.sqrt(n)
    return out.astype(np.float32)


def _vht2_forward(x: np.ndarray) -> np.ndarray:
    n = len(x)
    primes = _factor_small_primes(n)
    out = x.copy().astype(np.float64)
    stride = 1
    for p in primes:
        H = _hartley_kernel(p)
        for block_start in range(0, n, stride * p):
            for s in range(stride):
                indices = [block_start + s + i * stride for i in range(p)]
                out[indices] = H @ out[indices]
        stride *= p
    return out.astype(np.float32)


def vht2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return _vht2_pow2(x) if (n > 0 and (n & (n - 1)) == 0) else _vht2_forward(x)


# ── Batched VHT2 ─────────────────────────────────────────────────────────────
# Processes a (batch, n) matrix in one numpy call — 50-200x faster than the
# per-vector loop above for large head counts.

def _vht2_pow2_batch(X: np.ndarray) -> np.ndarray:
    """Batch Walsh-Hadamard transform for power-of-2 n. X: (batch, n)."""
    batch, n = X.shape
    out = X.copy().astype(np.float64)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = out[:, i:i + h].copy()
            b = out[:, i + h:i + h * 2].copy()
            out[:, i:i + h]         = a + b
            out[:, i + h:i + h * 2] = a - b
        h *= 2
    out /= np.sqrt(n)
    return out.astype(np.float32)


def _vht2_forward_batch(X: np.ndarray) -> np.ndarray:
    """Batch Vilenkin-Hartley transform for general n. X: (batch, n)."""
    batch, n = X.shape
    primes = _factor_small_primes(n)
    out = X.copy().astype(np.float64)
    stride = 1
    for p in primes:
        H = _hartley_kernel(p)          # (p, p)
        for block_start in range(0, n, stride * p):
            for s in range(stride):
                indices = [block_start + s + i * stride for i in range(p)]
                # (batch, p) @ (p, p).T  →  (batch, p)
                out[:, indices] = out[:, indices] @ H.T
        stride *= p
    return out.astype(np.float32)


def vht2_batch(X: np.ndarray) -> np.ndarray:
    """Batch VHT2: X shape (batch, n) → spectra shape (batch, n)."""
    n = X.shape[1]
    if n > 0 and (n & (n - 1)) == 0:
        return _vht2_pow2_batch(X)
    return _vht2_forward_batch(X)


def precompute_spectra(k_vectors: np.ndarray,
                       analysis_dim: int,
                       use_sqfree: bool,
                       n_sample: int) -> np.ndarray:
    """
    Precompute VHT2 spectra for all (layer, head, pos) triples in one
    batched numpy call.

    Returns:
        spectra  shape (n_layers, n_heads, n_s, analysis_dim)  float32
        originals shape (n_layers, n_heads, n_s, analysis_dim) float32
    """
    n_layers, n_heads, n_pos, head_dim = k_vectors.shape
    n_s = min(n_sample, n_pos)

    # Build originals array (with sqfree padding applied)
    if use_sqfree and analysis_dim != head_dim:
        mean_vals = k_vectors[:, :, :n_s, :].mean(axis=-1, keepdims=True)  # (L,H,P,1)
        originals = np.empty((n_layers, n_heads, n_s, analysis_dim), dtype=np.float32)
        originals[:, :, :, :head_dim] = k_vectors[:, :, :n_s, :]
        originals[:, :, :, head_dim:] = mean_vals
    else:
        originals = k_vectors[:, :, :n_s, :].astype(np.float32)

    # Flatten to (batch, analysis_dim), transform, reshape back
    flat  = originals.reshape(-1, analysis_dim)   # (L*H*P, dim)
    specs = vht2_batch(flat)                       # (L*H*P, dim)
    spectra = specs.reshape(n_layers, n_heads, n_s, analysis_dim)

    return spectra, originals


def _is_sqfree_rich(n: int, min_distinct: int = 3) -> bool:
    distinct, d = 0, n
    for p in [2, 3, 5, 7, 11]:
        if d % p == 0:
            distinct += 1
            d //= p
            if d % p == 0:
                return False
    return d == 1 and distinct >= min_distinct


def sqfree_pad_dim(head_dim: int) -> int:
    if head_dim in SQFREE_PAD:
        return SQFREE_PAD[head_dim]
    n = head_dim
    while n < head_dim * 2:
        if _is_sqfree_rich(n):
            return n
        n += 1
    return head_dim


def sqfree_pad_vector(x: np.ndarray, pad_dim: int) -> np.ndarray:
    hd = len(x)
    if hd >= pad_dim:
        return x[:pad_dim].copy()
    out = np.empty(pad_dim, dtype=x.dtype)
    out[:hd] = x
    out[hd:] = np.mean(x)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spectral helpers
# ─────────────────────────────────────────────────────────────────────────────

def shannon_entropy(energy: np.ndarray) -> float:
    """Shannon entropy in bits from an energy (squared magnitude) vector."""
    total = energy.sum()
    if total < 1e-12:
        return 0.0
    p = energy / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def algebraic_skeleton(dim: int, primes: set) -> np.ndarray:
    """Indices in [0, dim) fully factorable into the given prime set (plus DC=0)."""
    indices = [0]
    for i in range(1, dim):
        d = i
        for p in sorted(primes):
            while d % p == 0:
                d //= p
        if d == 1:
            indices.append(i)
    return np.array(sorted(indices), dtype=np.int64)


def reconstruct_from_skeleton(spectrum: np.ndarray,
                               skeleton: np.ndarray,
                               dim: int) -> np.ndarray:
    sparse = np.zeros(dim, dtype=np.float32)
    for idx in skeleton:
        if idx < dim:
            sparse[idx] = spectrum[idx]
    return vht2(sparse)


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    norm = np.linalg.norm(original)
    if norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(original - reconstructed) / norm)


def head_entropy(k_head: np.ndarray,
                 analysis_dim: int,
                 use_sqfree: bool,
                 n_sample: int = 32) -> float:
    """Mean Shannon entropy (in bits) of a head's VHT2 spectra."""
    head_dim = k_head.shape[-1]
    n_s = min(n_sample, k_head.shape[0])
    entropies = []
    for pos in range(n_s):
        vec = k_head[pos]
        if use_sqfree and analysis_dim != head_dim:
            vec = sqfree_pad_vector(vec, analysis_dim)
        spec = vht2(vec)
        entropies.append(shannon_entropy(spec ** 2))
    return float(np.mean(entropies)) if entropies else 0.0


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient (pure numpy, no scipy dependency)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dx, dy = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((dx**2).sum()) * float((dy**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((dx * dy).sum() / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — verbatim from sp_regime_analysis.py
# ─────────────────────────────────────────────────────────────────────────────

def load_k_vectors(path: str) -> np.ndarray:
    """Load K vectors → (n_layers, n_heads, n_pos, head_dim)."""
    if path.endswith('.npz'):
        data = np.load(path)
        if 'k_vectors' in data:
            return data['k_vectors'].astype(np.float32)
        keys   = sorted(data.files)
        layers, heads = set(), set()
        for key in keys:
            if key.startswith('k_layer'):
                parts = key.replace('k_layer', '').replace('_head', ' ').split()
                if len(parts) == 2:
                    layers.add(int(parts[0]))
                    heads.add(int(parts[1]))
        n_layers, n_heads = max(layers) + 1, max(heads) + 1
        sample  = data[keys[0]]
        n_pos, head_dim = sample.shape
        k = np.zeros((n_layers, n_heads, n_pos, head_dim), dtype=np.float32)
        for key in keys:
            if key.startswith('k_layer'):
                parts = key.replace('k_layer', '').replace('_head', ' ').split()
                L, H  = int(parts[0]), int(parts[1])
                k[L, H] = data[key]
        return k
    elif path.endswith(('.pt', '.pth')):
        import torch
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict) and 'k_vectors' in data:
            return data['k_vectors'].float().numpy()
        return data.float().numpy()
    raise ValueError(f"Unsupported format: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Grünwald-Letnikov fractional derivative
# ─────────────────────────────────────────────────────────────────────────────

def gl_derivative(f: np.ndarray, alpha: float, max_terms: Optional[int] = None) -> np.ndarray:
    """
    Grünwald-Letnikov fractional derivative of order alpha.

    D^α f(n) = Σ_{k=0}^{n} w[k] * f(n-k)

    where w[k] = (-1)^k * C(α,k), C(α,0)=1,
    and C(α,k) = C(α,k-1) * (α - k + 1) / k  (recurrence for binomial coefficients).

    The (-1)^k sign is absorbed into the recurrence:
        w[0] = 1
        w[k] = -w[k-1] * (α - k + 1) / k

    Special cases:
        α = 1 → first difference  D^1 f(n) = f(n) - f(n-1)
        α = 2 → second difference D^2 f(n) = f(n) - 2f(n-1) + f(n-2)
    """
    f  = np.asarray(f, dtype=np.float64)
    n  = len(f)
    mk = n if max_terms is None else min(n, max_terms)

    # Precompute GL weights
    w = np.zeros(mk)
    w[0] = 1.0
    for k in range(1, mk):
        w[k] = -w[k - 1] * (alpha - k + 1) / k

    result = np.zeros(n)
    for i in range(n):
        terms = min(i + 1, mk)
        acc   = 0.0
        for k in range(terms):
            acc += w[k] * f[i - k]
        result[i] = acc

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Boundary Sharpness
# ─────────────────────────────────────────────────────────────────────────────

def test_boundary_sharpness(k_vectors: np.ndarray,
                             analysis_dim: int,
                             use_sqfree: bool,
                             skeleton_frac: float = 0.30,
                             n_sample: int = 64,
                             verbose: bool = True,
                             _spectra=None, _originals=None) -> Dict:
    """
    Test 1 — Superconductivity analogy.

    Computes per-layer reconstruction error (mean and variance) under a fixed
    T2 algebraic skeleton. A superconducting system would show a flat, near-zero
    error floor that breaks sharply at a specific layer — a step function rather
    than a smooth gradient.

    Metrics:
      variance_ratio   σ_early / σ_late  → Win if < 0.20
      step_sharpness   max(Δmean) / (mean_late - mean_early)  → Win if > 0.50
    """
    n_layers, n_heads, n_pos, head_dim = k_vectors.shape
    n_s = min(n_sample, n_pos)

    # Fixed T2 algebraic skeleton
    skel_full = algebraic_skeleton(analysis_dim, {2, 3})
    k_cap     = max(1, int(skeleton_frac * analysis_dim))
    skel      = skel_full[:min(k_cap, len(skel_full))]

    per_layer_mean = []
    per_layer_std  = []

    if verbose:
        print(f"  [T1] Skeleton: T2 algebraic, K={len(skel)}/{analysis_dim} ({skeleton_frac:.0%})")

    # Use precomputed spectra when available (fast path)
    if _spectra is not None and _originals is not None:
        spectra   = _spectra    # (L, H, P, dim)
        originals = _originals  # (L, H, P, dim)

        # Build sparse mask for skeleton
        mask = np.zeros(analysis_dim, dtype=np.float32)
        mask[skel[skel < analysis_dim]] = 1.0

        for L in range(n_layers):
            # (H*P, dim)
            specs_flat = spectra[L].reshape(-1, analysis_dim)
            origs_flat = originals[L].reshape(-1, analysis_dim)
            # Reconstruct: zero non-skeleton, apply VHT2
            sparse = specs_flat * mask[None, :]          # (H*P, dim)
            recons = vht2_batch(sparse)                  # (H*P, dim)
            # Relative L2 error per sample
            orig_norms = np.linalg.norm(origs_flat, axis=1)  # (H*P,)
            diff_norms = np.linalg.norm(origs_flat - recons, axis=1)
            valid = orig_norms > 1e-12
            errs  = np.where(valid, diff_norms / np.where(valid, orig_norms, 1.0), 0.0)
            per_layer_mean.append(float(errs.mean()))
            per_layer_std.append(float(errs.std()))
    else:
        # Slow fallback (per-vector)
        for L in range(n_layers):
            errors = []
            for H in range(n_heads):
                for pos in range(n_s):
                    vec = k_vectors[L, H, pos]
                    if use_sqfree and analysis_dim != head_dim:
                        padded = sqfree_pad_vector(vec, analysis_dim)
                        spec   = vht2(padded)
                        orig   = padded
                    else:
                        spec = vht2(vec)
                        orig = vec.copy()
                    recon = reconstruct_from_skeleton(spec, skel, analysis_dim)
                    errors.append(reconstruction_error(orig, recon))
            per_layer_mean.append(float(np.mean(errors)))
            per_layer_std.append(float(np.std(errors)))

    mean_arr = np.array(per_layer_mean)
    std_arr  = np.array(per_layer_std)

    # Locate transition: first layer where mean error crosses midpoint
    lo, hi   = mean_arr.min(), mean_arr.max()
    mid      = (lo + hi) / 2
    transition = n_layers // 2
    for L in range(n_layers):
        if mean_arr[L] > mid:
            transition = L
            break

    t = transition
    early_mean = float(mean_arr[:t].mean())  if t > 0         else float(mean_arr[0])
    late_mean  = float(mean_arr[t:].mean())  if t < n_layers  else float(mean_arr[-1])
    early_std  = float(std_arr[:t].mean())   if t > 0         else float(std_arr[0])
    late_std   = float(std_arr[t:].mean())   if t < n_layers  else float(std_arr[-1])

    variance_ratio = early_std / late_std if late_std > 1e-8 else 0.0

    deltas       = np.abs(np.diff(mean_arr))
    max_delta    = float(deltas.max()) if len(deltas) > 0 else 0.0
    total_change = late_mean - early_mean
    step_sharpness = max_delta / total_change if total_change > 1e-8 else 0.0

    win_var   = variance_ratio  < 0.20
    win_sharp = step_sharpness  > 0.50

    if win_var and win_sharp:
        verdict = "SHARP BOUNDARY — superconducting analogy holds; harden Regime A kernel"
    elif not win_var and not win_sharp:
        verdict = "GRADIENT — smooth regime shift; no sharp optimisation boundary"
    else:
        verdict = "MIXED — partial step behaviour; investigate the specific transition layer"

    result = {
        "test": "boundary_sharpness",
        "analogy": "superconductivity",
        "skeleton_k": len(skel),
        "analysis_dim": analysis_dim,
        "transition_layer": t,
        "per_layer_mean_error": [round(v, 6) for v in per_layer_mean],
        "per_layer_std_error":  [round(v, 6) for v in per_layer_std],
        "early_mean_error": round(early_mean, 6),
        "late_mean_error":  round(late_mean, 6),
        "early_std_error":  round(early_std, 6),
        "late_std_error":   round(late_std, 6),
        "variance_ratio":   round(variance_ratio, 4),
        "step_sharpness":   round(step_sharpness, 4),
        "win_variance_ratio": win_var,
        "win_step_sharpness": win_sharp,
        "verdict": verdict,
    }

    if verbose:
        print(f"  [T1] Transition at layer {t}")
        print(f"  [T1] Early: mean={early_mean:.4f}, σ={early_std:.4f}")
        print(f"  [T1] Late:  mean={late_mean:.4f}, σ={late_std:.4f}")
        print(f"  [T1] Variance ratio (early/late): {variance_ratio:.4f}  "
              f"{'✓ WIN' if win_var   else '✗ MISS'}  (< 0.20)")
        print(f"  [T1] Step sharpness:              {step_sharpness:.4f}  "
              f"{'✓ WIN' if win_sharp else '✗ MISS'}  (> 0.50)")
        print(f"  [T1] Verdict: {verdict}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Ghost Basin Hunt
# ─────────────────────────────────────────────────────────────────────────────

def test_ghost_basin(k_vectors: np.ndarray,
                     analysis_dim: int,
                     use_sqfree: bool,
                     n_sample: int = 64,
                     eps: Optional[float] = None,
                     min_samples: int = 3,
                     verbose: bool = True,
                     _spectra=None,
                     entropy_threshold: Optional[float] = None) -> Dict:
    """
    Test 2 — Collatz clustering analogy.

    Identifies ghost heads (Shannon entropy > threshold), computes their mean
    VHT2 spectra, then runs DBSCAN to check for stable attractor basins.

    entropy_threshold defaults to log2(analysis_dim) * 0.70, which scales the
    "ghost" definition with the analysis dimension so that sqfree-padded dims
    (110, 154, 231 …) aren't uniformly classified as ghosts.

    If ghost heads cluster into a small number of dense basins, we can replace
    their full spectra with a Basin-ID (a few bits), dramatically cutting
    residual storage for late-layer heads.

    Metrics:
      n_clusters  number of dense DBSCAN clusters found
      coverage    fraction of ghost heads assigned to a cluster (not noise)
      Win if: 2 ≤ n_clusters ≤ 16  AND  coverage ≥ 0.80
    """
    n_layers, n_heads, n_pos, head_dim = k_vectors.shape
    n_s = min(n_sample, n_pos)

    # Adaptive threshold: 70% of maximum possible entropy for this analysis_dim
    ghost_threshold = (entropy_threshold if entropy_threshold is not None
                       else math.log2(analysis_dim) * 0.70)

    # Identify ghost heads (use precomputed spectra when available)
    ghost_entries = []
    if _spectra is not None:
        for L in range(n_layers):
            for H in range(n_heads):
                specs_LH = _spectra[L, H]  # (n_s, analysis_dim)
                energy    = specs_LH ** 2
                # Per-position entropy
                ents = []
                for pos in range(specs_LH.shape[0]):
                    ents.append(shannon_entropy(energy[pos]))
                ent = float(np.mean(ents))
                if ent >= ghost_threshold:
                    ghost_entries.append({
                        "layer": L, "head": H, "entropy": round(ent, 4),
                        "mean_spectrum": specs_LH.mean(axis=0),
                    })
    else:
        for L in range(n_layers):
            for H in range(n_heads):
                ent = head_entropy(k_vectors[L, H], analysis_dim, use_sqfree, n_s)
                if ent >= ghost_threshold:
                    specs = []
                    for pos in range(n_s):
                        vec = k_vectors[L, H, pos]
                        if use_sqfree and analysis_dim != head_dim:
                            vec = sqfree_pad_vector(vec, analysis_dim)
                        specs.append(vht2(vec))
                    mean_spec = np.mean(specs, axis=0)
                    ghost_entries.append({
                        "layer": L, "head": H, "entropy": round(ent, 4),
                        "mean_spectrum": mean_spec,
                    })

    n_ghosts = len(ghost_entries)
    total_heads = n_layers * n_heads

    if verbose:
        print(f"  [T2] Ghost heads: {n_ghosts}/{total_heads} "
              f"({n_ghosts / total_heads:.0%}), entropy threshold={ghost_threshold:.2f} bits "
              f"[log2({analysis_dim})*0.70]")
        if n_ghosts == total_heads:
            print(f"  [T2] NOTE: 100% ghost rate typically means single-prompt data is too uniform.")
            print(f"  [T2]       Re-run with --prompts-file (diverse Prose/Code/Math/Chat inputs)")
            print(f"  [T2]       for meaningful ghost-basin clustering.")

    if n_ghosts < 3:
        result = {
            "test": "ghost_basin",
            "analogy": "collatz_clustering",
            "n_ghost_heads": n_ghosts,
            "n_ghost_fraction": round(n_ghosts / total_heads, 4),
            "n_clusters": 0,
            "coverage": 0.0,
            "win": False,
            "verdict": (
                f"INSUFFICIENT GHOST HEADS ({n_ghosts}) — need ≥ 3 to cluster; "
                f"model may be highly harmonic already"
            ),
        }
        if verbose:
            print(f"  [T2] Verdict: {result['verdict']}")
        return result

    # Build L2-normalised feature matrix from mean spectra
    X = np.array([e["mean_spectrum"] for e in ghost_entries])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    X_norm = X / norms

    # Try DBSCAN via sklearn; fall back to a simple greedy k-means-style scan
    try:
        from sklearn.cluster import DBSCAN as _DBSCAN

        if eps is None:
            # Auto-tune eps: 10th percentile of sampled pairwise L2 distances
            sample = X_norm[np.random.choice(n_ghosts, min(n_ghosts, 200), replace=False)]
            ns = len(sample)
            dists = []
            for i in range(ns):
                for j in range(i + 1, ns):
                    dists.append(float(np.linalg.norm(sample[i] - sample[j])))
            dists.sort()
            eps_use = max(float(np.percentile(dists, 10)), 0.01)
        else:
            eps_use = eps

        labels = _DBSCAN(eps=eps_use, min_samples=min_samples).fit(X_norm).labels_
        backend = "dbscan"

    except ImportError:
        # Manual fallback: agglomerative single-linkage approximation
        # Group points within a fixed distance threshold
        eps_use = eps if eps is not None else 0.20
        labels  = -np.ones(n_ghosts, dtype=int)
        cluster_id = 0
        for i in range(n_ghosts):
            if labels[i] >= 0:
                continue
            labels[i] = cluster_id
            for j in range(i + 1, n_ghosts):
                if labels[j] < 0:
                    d = np.linalg.norm(X_norm[i] - X_norm[j])
                    if d < eps_use:
                        labels[j] = cluster_id
            cluster_id += 1
        backend = "fallback_single_linkage"

    unique_labels = set(labels)
    n_clusters    = len(unique_labels - {-1})
    n_noise       = int(np.sum(labels == -1))
    coverage      = float((n_ghosts - n_noise) / n_ghosts) if n_ghosts > 0 else 0.0

    # Per-cluster summary
    clusters = {}
    for label in sorted(unique_labels - {-1}):
        members = [ghost_entries[i] for i in range(n_ghosts) if labels[i] == label]
        clusters[str(label)] = {
            "size": len(members),
            "mean_entropy": round(float(np.mean([m["entropy"] for m in members])), 3),
            "layers": sorted(set(m["layer"] for m in members)),
            "heads":  sorted(set(m["head"]  for m in members)),
        }

    win = (2 <= n_clusters <= 16) and (coverage >= 0.80)
    if win:
        verdict = (f"WIN — {n_clusters} stable basins, {coverage:.0%} of ghost heads classified; "
                   f"Basin-ID encoding is viable")
    elif n_clusters > 16:
        verdict = (f"TOO MANY CLUSTERS ({n_clusters}) — ghost spectrum is fragmented; "
                   f"try larger eps or more sample positions")
    elif n_clusters < 2:
        verdict = (f"NO CLUSTERS — ghosts are scattered noise; "
                   f"Collatz-basin analogy does not hold here")
    else:
        verdict = (f"LOW COVERAGE ({coverage:.0%}) — {n_clusters} basins found but "
                   f"{n_noise} ghosts unclustered; may need wider eps")

    result = {
        "test": "ghost_basin",
        "analogy": "collatz_clustering",
        "backend": backend,
        "entropy_threshold": round(ghost_threshold, 4),
        "entropy_threshold_formula": f"log2({analysis_dim})*0.70",
        "n_ghost_heads": n_ghosts,
        "n_ghost_fraction": round(n_ghosts / total_heads, 4),
        "eps_used": round(eps_use, 4) if eps_use is not None else None,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "coverage": round(coverage, 4),
        "clusters": clusters,
        "win": win,
        "verdict": verdict,
    }

    if verbose:
        print(f"  [T2] Backend: {backend}, eps={eps_use:.4f}, min_samples={min_samples}")
        print(f"  [T2] Clusters: {n_clusters}  "
              f"{'✓ WIN' if 2 <= n_clusters <= 16 else '✗ MISS'}  (target 2–16)")
        print(f"  [T2] Coverage: {coverage:.0%}  "
              f"{'✓ WIN' if coverage >= 0.80 else '✗ MISS'}  (≥ 80%)")
        if clusters:
            sorted_c = sorted(clusters.items(), key=lambda x: x[1]["size"], reverse=True)
            print(f"  [T2] Top clusters:")
            for label, info in sorted_c[:6]:
                print(f"        Basin {label}: {info['size']:3d} heads, "
                      f"H̄={info['mean_entropy']:.2f} bits, "
                      f"layers={info['layers'][:4]}{'…' if len(info['layers'])>4 else ''}")
        print(f"  [T2] Verdict: {verdict}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: RoPE Pair Correlation
# ─────────────────────────────────────────────────────────────────────────────

def test_rope_pair_correlation(k_vectors: np.ndarray,
                                analysis_dim: int,
                                use_sqfree: bool,
                                skeleton_frac: float = 0.30,
                                n_sample: int = 64,
                                verbose: bool = True,
                                _spectra=None, _originals=None) -> Dict:
    """
    Test 3 — Twin-prime pairing analogy.

    RoPE encodes position by rotating pairs of head dimensions:
        (d_0, d_1), (d_2, d_3), ..., (d_{D-2}, d_{D-1})

    In the reconstructed residual, even-indexed dimensions carry the "cos"
    component and odd-indexed carry the "sin" component of each RoPE pair.

    If RoPE pairing structure survives compression, the residual norms on even
    and odd dimensions should be correlated across token positions — the error
    is a phase shift, not independent noise.

    This test computes Pearson r between ||residual_even||₂ and ||residual_odd||₂
    across positions, for each (layer, head), then averages.

    Metrics:
      overall_r  mean Pearson r across all (layer, head) pairs
      Win if overall_r > 0.70
    """
    n_layers, n_heads, n_pos, head_dim = k_vectors.shape
    n_s = min(n_sample, n_pos)

    skel_full = algebraic_skeleton(analysis_dim, {2, 3})
    k_cap     = max(1, int(skeleton_frac * analysis_dim))
    skel      = skel_full[:min(k_cap, len(skel_full))]

    per_layer_r = []

    # Fast path: use precomputed spectra + batch reconstruction
    if _spectra is not None and _originals is not None:
        mask = np.zeros(analysis_dim, dtype=np.float32)
        mask[skel[skel < analysis_dim]] = 1.0

        for L in range(n_layers):
            layer_r = []
            for H in range(n_heads):
                specs_LH = _spectra[L, H]    # (n_s, dim)
                origs_LH = _originals[L, H]  # (n_s, dim)
                sparse   = specs_LH * mask[None, :]
                recons   = vht2_batch(sparse)
                resids   = origs_LH - recons  # (n_s, dim)
                # Even/odd split
                ev = np.linalg.norm(resids[:, 0::2], axis=1)  # (n_s,)
                od = np.linalg.norm(resids[:, 1::2], axis=1)
                if ev.std() > 1e-10 and od.std() > 1e-10:
                    layer_r.append(pearson_r(ev, od))
            per_layer_r.append(float(np.mean(layer_r)) if layer_r else 0.0)
    else:
        # Slow fallback
        for L in range(n_layers):
            layer_r = []
            for H in range(n_heads):
                even_norms, odd_norms = [], []
                for pos in range(n_s):
                    vec = k_vectors[L, H, pos]
                    if use_sqfree and analysis_dim != head_dim:
                        padded = sqfree_pad_vector(vec, analysis_dim)
                        spec   = vht2(padded)
                        recon  = reconstruct_from_skeleton(spec, skel, analysis_dim)
                        resid  = padded - recon
                    else:
                        spec  = vht2(vec)
                        recon = reconstruct_from_skeleton(spec, skel, analysis_dim)
                        resid = vec - recon
                    even_norms.append(float(np.linalg.norm(resid[0::2])))
                    odd_norms.append(float(np.linalg.norm(resid[1::2])))
                ev, od = np.array(even_norms), np.array(odd_norms)
                if ev.std() > 1e-10 and od.std() > 1e-10:
                    layer_r.append(pearson_r(ev, od))
            per_layer_r.append(float(np.mean(layer_r)) if layer_r else 0.0)

    per_layer_r = np.array(per_layer_r)
    overall_r   = float(per_layer_r.mean())
    early_r     = float(per_layer_r[:n_layers // 2].mean())
    late_r      = float(per_layer_r[n_layers // 2:].mean())
    peak_layer  = int(np.argmax(np.abs(per_layer_r)))

    win = overall_r > 0.70

    if win:
        verdict = (f"WIN — r={overall_r:.3f}: RoPE pairs correlated; "
                   f"differential (cos–sin) residual encoding could halve storage")
    elif overall_r > 0.40:
        verdict = (f"PARTIAL — r={overall_r:.3f}: moderate pairing signal; "
                   f"worth testing differential encoding on early layers only (r_early={early_r:.3f})")
    else:
        verdict = (f"MISS — r={overall_r:.3f}: pairs are independent; "
                   f"differential encoding offers no benefit here")

    result = {
        "test": "rope_pair_correlation",
        "analogy": "twin_prime_symmetry",
        "skeleton_k": len(skel),
        "per_layer_mean_r": [round(v, 4) for v in per_layer_r.tolist()],
        "overall_r": round(overall_r, 4),
        "early_r":   round(early_r,   4),
        "late_r":    round(late_r,    4),
        "peak_layer": peak_layer,
        "win": win,
        "verdict": verdict,
    }

    if verbose:
        print(f"  [T3] Overall Pearson r: {overall_r:.4f}  "
              f"{'✓ WIN' if win else '✗ MISS'}  (> 0.70)")
        print(f"  [T3] Early layers:      {early_r:.4f}")
        print(f"  [T3] Late layers:       {late_r:.4f}")
        print(f"  [T3] Peak correlation at layer {peak_layer}")
        print(f"  [T3] Verdict: {verdict}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Fractional Slope Lookahead
# ─────────────────────────────────────────────────────────────────────────────

def test_fractional_lookahead(k_vectors: np.ndarray,
                               analysis_dim: int,
                               use_sqfree: bool,
                               alphas: Optional[List[float]] = None,
                               n_sample: int = 32,
                               verbose: bool = True,
                               _spectra=None) -> Dict:
    """
    Test 4 — Fractional operators analogy.

    Computes the p3 energy fraction per layer (the "Mertens slope"), then
    applies Grünwald-Letnikov fractional derivatives at various orders α.

    A fractional-order derivative at 0 < α < 1 incorporates a weighted history
    of past values, potentially picking up gradual trends before the sharp drop
    visible in the standard first difference (α = 1).

    For each α, the "detection layer" is the first layer where the GL derivative
    exceeds 2σ below the baseline mean. A lower detection layer = earlier warning.

    Metrics:
      lookahead_layers  (detection of α=1) − (detection of best fractional α)
      Win if lookahead_layers > 0 (any fractional α detects earlier than α=1)
    """
    if alphas is None:
        alphas = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

    n_layers, n_heads, n_pos, head_dim = k_vectors.shape
    n_s = min(n_sample, n_pos)

    # Compute p3 fraction per layer (energy at indices divisible by 3, excl. DC)
    p3_curve = []
    if _spectra is not None:
        # Fast path: use precomputed spectra
        # _spectra shape: (n_layers, n_heads, n_s_full, analysis_dim)
        spec_s = _spectra[:, :, :n_s, :]  # trim to n_s
        energy = spec_s ** 2              # (L, H, n_s, dim)
        total_all = energy.sum(axis=-1)   # (L, H, n_s)
        # p3 indices: 3, 6, 9, ...
        p3_idx = np.arange(3, analysis_dim, 3)
        total_p3 = energy[:, :, :, p3_idx].sum(axis=-1)  # (L, H, n_s)
        p3_frac = np.where(total_all > 0, total_p3 / total_all, 0.0)  # (L, H, n_s)
        p3_curve = p3_frac.mean(axis=(1, 2)).tolist()   # (L,)
    else:
        for L in range(n_layers):
            p3_vals = []
            for H in range(n_heads):
                total_p3, total_all = 0.0, 0.0
                for pos in range(n_s):
                    vec = k_vectors[L, H, pos]
                    if use_sqfree and analysis_dim != head_dim:
                        vec = sqfree_pad_vector(vec, analysis_dim)
                    spec = vht2(vec)
                    e = spec ** 2
                    total_all += float(e.sum())
                    # Indices divisible by 3 (excluding DC=0): 3, 6, 9, ...
                    total_p3 += float(e[3::3].sum())
                p3_vals.append(total_p3 / total_all if total_all > 0 else 0.0)
            p3_curve.append(float(np.mean(p3_vals)))

    p3_arr = np.array(p3_curve)

    # Ground-truth transition: layer after the largest single-step drop in p3
    deltas            = np.diff(p3_arr)
    actual_transition = int(np.argmin(deltas)) + 1 if len(deltas) > 0 else n_layers // 2

    # For each α: compute GL derivative, find detection layer
    n_baseline = max(3, n_layers // 4)

    detection_by_alpha: Dict[float, int] = {}
    gl_curves: Dict[str, List[float]]    = {}

    for alpha in alphas:
        gl = gl_derivative(p3_arr, alpha)
        gl_curves[str(round(alpha, 2))] = [round(v, 6) for v in gl.tolist()]

        baseline      = gl[:n_baseline]
        bl_mean       = float(baseline.mean())
        bl_std        = float(baseline.std())
        threshold     = bl_mean - 2.0 * bl_std   # trigger on downward excursion

        detected = n_layers   # default = no detection
        for L in range(n_baseline, n_layers):
            if gl[L] < threshold:
                detected = L
                break
        detection_by_alpha[alpha] = detected

    std_detection  = detection_by_alpha.get(1.0, actual_transition)
    best_alpha     = min(detection_by_alpha, key=detection_by_alpha.get)
    best_detection = detection_by_alpha[best_alpha]
    lookahead      = std_detection - best_detection

    win = (lookahead > 0) and (best_alpha != 1.0)

    if win:
        verdict = (f"WIN — α={best_alpha} detects transition at layer {best_detection}, "
                   f"{lookahead} layer(s) before standard slope (α=1.0 detects at {std_detection})")
    elif lookahead == 0:
        verdict = (f"TIED — fractional orders match standard slope (all detect at layer {std_detection}); "
                   f"transition is already maximally sharp")
    else:
        verdict = (f"MISS — standard slope (α=1.0) is already optimal "
                   f"(detects at layer {std_detection}); no fractional advantage")

    result = {
        "test": "fractional_lookahead",
        "analogy": "fractional_operators",
        "alphas_tested": alphas,
        "p3_curve": [round(v, 6) for v in p3_curve],
        "actual_transition_layer": actual_transition,
        "standard_detection_layer": std_detection,
        "best_alpha": best_alpha,
        "best_alpha_detection_layer": best_detection,
        "lookahead_layers": lookahead,
        "detection_by_alpha": {str(round(a, 2)): d for a, d in detection_by_alpha.items()},
        "gl_curves": gl_curves,
        "win": win,
        "verdict": verdict,
    }

    if verbose:
        print(f"  [T4] Actual transition:       layer {actual_transition}")
        print(f"  [T4] Standard slope (α=1.00): layer {std_detection}")
        print(f"  [T4] Best fractional (α={best_alpha:.2f}): layer {best_detection}  "
              f"{'✓ WIN' if win else '✗'}")
        print(f"  [T4] Detection by α:")
        for a in sorted(detection_by_alpha):
            d      = detection_by_alpha[a]
            marker = "  ← best" if a == best_alpha else ""
            print(f"        α={a:.2f}: layer {d:3d}{marker}")
        print(f"  [T4] Verdict: {verdict}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shannon-Prime Phase 12 — Diagnostic Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests:
  1  Boundary Sharpness   — Is the regime transition a step function?
  2  Ghost Basin Hunt     — Do ghost heads form stable DBSCAN clusters?
  3  RoPE Pair Corr.      — Are cos/sin residuals correlated across positions?
  4  Fractional Lookahead — Does a fractional-order p3 slope detect earlier?

Examples:
  python sp_diagnostics.py --input kv_phi3.npz --sqfree
  python sp_diagnostics.py --input kv_phi3.npz --sqfree --output diag.json
  python sp_diagnostics.py --input kv_phi3.npz --tests 1,2
  python sp_diagnostics.py --input kv_phi3.npz --skeleton-frac 0.40 --sample 128
        """,
    )
    parser.add_argument("--input",  "-i", required=True, help="K vectors .npz file")
    parser.add_argument("--output", "-o", default=None,  help="Save JSON report to this path")
    parser.add_argument("--sqfree", action="store_true",  help="Pad to sqfree dimensions")
    parser.add_argument("--skeleton-frac", type=float, default=0.30,
                        help="Skeleton fraction of analysis_dim for Tests 1+3 (default: 0.30)")
    parser.add_argument("--sample", type=int, default=64,
                        help="Positions to sample per head (default: 64)")
    parser.add_argument("--tests", type=str, default="1,2,3,4",
                        help="Comma-separated list of tests to run (default: 1,2,3,4)")
    parser.add_argument("--ghost-threshold", type=float, default=None,
                        help="Ghost entropy threshold for Test 2 in bits "
                             "(default: log2(analysis_dim)*0.70 — scales with sqfree dim)")
    parser.add_argument("--dbscan-eps", type=float, default=None,
                        help="DBSCAN epsilon for Test 2 (default: auto-tuned)")
    parser.add_argument("--dbscan-min-samples", type=int, default=3,
                        help="DBSCAN min_samples for Test 2 (default: 3)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    verbose   = not args.quiet
    run_tests = set(int(t.strip()) for t in args.tests.split(","))

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading K vectors from {args.input}...")
    k_vectors = load_k_vectors(args.input)
    n_layers, n_heads, n_pos, head_dim = k_vectors.shape
    print(f"  Shape: {n_layers}L × {n_heads}H × {n_pos}P × {head_dim}D")

    if args.sqfree:
        analysis_dim = sqfree_pad_dim(head_dim)
        print(f"  Sqfree: {head_dim} -> {analysis_dim}")
    else:
        analysis_dim = head_dim
        print(f"  Analysis dim: {analysis_dim} (no sqfree padding)")

    print()

    # ── Precompute all spectra in one batched numpy call ─────────────────────
    print(f"  Precomputing VHT2 spectra (batch mode)...")
    t_pre = time.time()
    _spectra, _originals = precompute_spectra(k_vectors, analysis_dim, args.sqfree, args.sample)
    print(f"  Precompute done in {time.time() - t_pre:.1f}s  "
          f"(shape: {_spectra.shape})\n")

    # ── Run selected tests ───────────────────────────────────────────────────
    report   = {
        "input":        args.input,
        "shape":        {"n_layers": n_layers, "n_heads": n_heads,
                         "n_pos": n_pos, "head_dim": head_dim},
        "analysis_dim": analysis_dim,
        "sqfree":       args.sqfree,
        "results":      {},
    }
    wins     = []
    misses   = []
    t_start  = time.time()

    _sep = "─" * 62

    for test_id in sorted(run_tests):

        if test_id == 1:
            print(_sep)
            print(f"  TEST 1 — Boundary Sharpness  (superconductivity analogy)")
            print(_sep)
            r = test_boundary_sharpness(
                k_vectors, analysis_dim, args.sqfree,
                skeleton_frac=args.skeleton_frac,
                n_sample=args.sample,
                verbose=verbose,
                _spectra=_spectra, _originals=_originals,
            )
            report["results"]["test_1"] = r
            (wins if (r["win_variance_ratio"] or r["win_step_sharpness"]) else misses).append(1)

        elif test_id == 2:
            print(_sep)
            print(f"  TEST 2 — Ghost Basin Hunt  (Collatz clustering analogy)")
            print(_sep)
            r = test_ghost_basin(
                k_vectors, analysis_dim, args.sqfree,
                n_sample=args.sample,
                eps=args.dbscan_eps,
                min_samples=args.dbscan_min_samples,
                verbose=verbose,
                _spectra=_spectra,
                entropy_threshold=args.ghost_threshold,
            )
            report["results"]["test_2"] = r
            (wins if r["win"] else misses).append(2)

        elif test_id == 3:
            print(_sep)
            print(f"  TEST 3 — RoPE Pair Correlation  (twin-prime analogy)")
            print(_sep)
            r = test_rope_pair_correlation(
                k_vectors, analysis_dim, args.sqfree,
                skeleton_frac=args.skeleton_frac,
                n_sample=args.sample,
                verbose=verbose,
                _spectra=_spectra, _originals=_originals,
            )
            report["results"]["test_3"] = r
            (wins if r["win"] else misses).append(3)

        elif test_id == 4:
            print(_sep)
            print(f"  TEST 4 — Fractional Slope Lookahead  (fractional operators analogy)")
            print(_sep)
            r = test_fractional_lookahead(
                k_vectors, analysis_dim, args.sqfree,
                n_sample=min(args.sample, 32),  # p3 curve needs fewer samples
                verbose=verbose,
                _spectra=_spectra,
            )
            report["results"]["test_4"] = r
            (wins if r["win"] else misses).append(4)

        print()

    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 62)
    print(f"  PHASE 12 DIAGNOSTIC SUMMARY  ({elapsed:.1f}s)")
    print("=" * 62)

    labels = {
        1: "Boundary Sharpness",
        2: "Ghost Basin Hunt",
        3: "RoPE Pair Correlation",
        4: "Fractional Lookahead",
    }
    for tid in sorted(run_tests):
        key = f"test_{tid}"
        if key in report["results"]:
            r = report["results"][key]
            # Extract a short verdict line
            v = r.get("verdict", "—")[:70]
            w = r.get("win", r.get("win_variance_ratio", False) or r.get("win_step_sharpness", False))
            marker = "✓ WIN " if w else "  MISS"
            print(f"  [{marker}] T{tid}: {labels.get(tid, '')}  —  {v}")

    print()
    print(f"  Wins: {len(wins)}/{len(run_tests)}  "
          f"(Tests {wins if wins else 'none'} show the analogy has empirical support)")
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to {args.output}")

    print("=" * 62)


if __name__ == "__main__":
    main()
