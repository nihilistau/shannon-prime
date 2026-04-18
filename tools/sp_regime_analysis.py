# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
sp_regime_analysis.py — Two-Regime Reconstruction Quality Analysis

Empirical finding: transformers have two spectral regimes:
  Early (L0 to ~L/2):  Clean T² manifold, p2≈30% and p3≈22% dominant
  Late  (~L/2 to L):   Diffuse spectrum, p3 drops toward p5/p7, near-uniform

This script simulates compression at various skeleton sizes using three
strategies and reports reconstruction quality (relative L2 error) per layer:

  1. T² Algebraic — skeleton = indices divisible by 2 or 3 only
  2. Adaptive Top-K — skeleton = top-K indices by spectral variance
  3. T³ Algebraic — skeleton = indices divisible by 2, 3, or secondary prime

Input:  K vectors .npz file (same as sp_chord_diagnostic.py)
Output: Per-layer reconstruction quality at various compression ratios

Usage:
    python sp_regime_analysis.py --input kv_phi3.npz --sqfree
    python sp_regime_analysis.py --input kv_qwen3.npz --sqfree --output regime.json
"""

import sys
import os
import argparse
import json
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Import VHT2 and sqfree utils from the diagnostic (same math)
# ─────────────────────────────────────────────────────────────────────────────

def _hartley_kernel(p: int) -> np.ndarray:
    idx = np.arange(p)
    angles = (2.0 * np.pi / p) * np.outer(idx, idx)
    return (np.cos(angles) + np.sin(angles)) / np.sqrt(p)


def _factor_small_primes(n: int) -> List[int]:
    d = n
    primes = []
    for p in [2, 3, 5, 7, 11]:
        while d % p == 0:
            primes.append(p)
            d //= p
    if d != 1:
        raise ValueError(f"dim {n} has prime factor > 11 (residue {d})")
    return primes


def vht2_forward(x: np.ndarray) -> np.ndarray:
    n = len(x)
    primes = _factor_small_primes(n)
    out = x.copy().astype(np.float64)
    stride = 1
    for p in primes:
        H = _hartley_kernel(p)
        for block_start in range(0, n, stride * p):
            for s in range(stride):
                indices = [block_start + s + i * stride for i in range(p)]
                vals = out[indices]
                out[indices] = H @ vals
        stride *= p
    return out.astype(np.float32)


def vht2_pow2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    out = x.copy().astype(np.float64)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = out[j], out[j + h]
                out[j] = a + b
                out[j + h] = a - b
        h *= 2
    out /= np.sqrt(n)
    return out.astype(np.float32)


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def vht2(x: np.ndarray) -> np.ndarray:
    if is_power_of_2(len(x)):
        return vht2_pow2(x)
    return vht2_forward(x)


SQFREE_PAD = {64: 66, 96: 110, 128: 154, 256: 330}

def _is_sqfree_rich(n: int, min_distinct: int = 3) -> bool:
    distinct = 0
    d = n
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


def prime_signature(n: int) -> frozenset:
    factors = set()
    d = n
    for p in [2, 3, 5, 7, 11]:
        if d % p == 0:
            factors.add(p)
            while d % p == 0:
                d //= p
    return frozenset(factors)


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton construction strategies
# ─────────────────────────────────────────────────────────────────────────────

def algebraic_skeleton(dim: int, primes: set, max_k: Optional[int] = None) -> np.ndarray:
    """Build a skeleton of indices whose prime factors are subsets of `primes`.

    For T² with primes={2,3}: indices like 1,2,3,4,6,8,9,12,16,...
    (everything factorable into only 2s and 3s, plus index 0 = DC)

    Args:
        dim: analysis dimension
        primes: set of allowed primes
        max_k: optional cap on skeleton size

    Returns:
        sorted array of skeleton indices
    """
    indices = [0]  # DC component always included
    for i in range(1, dim):
        d = i
        for p in sorted(primes):
            while d % p == 0:
                d //= p
        if d == 1:  # Fully factored into allowed primes
            indices.append(i)
    indices = np.array(sorted(indices), dtype=np.int64)
    if max_k and len(indices) > max_k:
        indices = indices[:max_k]
    return indices


def adaptive_skeleton(spectra: np.ndarray, k: int) -> np.ndarray:
    """Build variance-ranked skeleton: top-k indices by spectral variance."""
    var = np.var(spectra, axis=0)
    return np.argsort(var)[::-1][:k]


def hybrid_skeleton(spectra: np.ndarray, dim: int, primes: set,
                    k: int, algebraic_weight: float = 0.7) -> np.ndarray:
    """Hybrid: blend algebraic structure with adaptive variance ranking.

    Scores each index as:
        score = algebraic_weight * is_algebraic + (1-algebraic_weight) * variance_rank
    Then takes top-k by combined score.

    This prefers algebraically clean indices but can pull in high-variance
    non-algebraic indices when the algebra doesn't cover enough energy.
    """
    var = np.var(spectra, axis=0)
    # Normalize variance to [0, 1]
    var_max = var.max()
    var_norm = var / var_max if var_max > 0 else var

    # Algebraic membership
    alg_set = set(algebraic_skeleton(dim, primes).tolist())

    scores = np.zeros(dim, dtype=np.float64)
    for i in range(dim):
        alg_score = 1.0 if i in alg_set else 0.0
        scores[i] = algebraic_weight * alg_score + (1.0 - algebraic_weight) * var_norm[i]

    return np.argsort(scores)[::-1][:k]


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction quality measurement
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_from_skeleton(spectrum: np.ndarray, skeleton_indices: np.ndarray,
                              dim: int) -> np.ndarray:
    """Reconstruct a vector from its skeleton VHT2 coefficients.

    Sets non-skeleton coefficients to zero, then applies inverse VHT2
    (which is the same as forward VHT2 since it's self-inverse).
    """
    sparse = np.zeros(dim, dtype=np.float32)
    for idx in skeleton_indices:
        if idx < dim:
            sparse[idx] = spectrum[idx]
    return vht2(sparse)


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Relative L2 error: ||orig - recon|| / ||orig||."""
    orig_norm = np.linalg.norm(original)
    if orig_norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(original - reconstructed) / orig_norm)


def spectral_energy_captured(spectrum: np.ndarray, skeleton_indices: np.ndarray) -> float:
    """Fraction of total spectral energy captured by the skeleton."""
    total_energy = np.sum(spectrum ** 2)
    if total_energy < 1e-12:
        return 1.0
    skeleton_energy = sum(spectrum[i] ** 2 for i in skeleton_indices if i < len(spectrum))
    return float(skeleton_energy / total_energy)


# ─────────────────────────────────────────────────────────────────────────────
# Transition detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_transition(k_vectors: np.ndarray, analysis_dim: int,
                      use_sqfree: bool) -> Tuple[int, Dict]:
    """Detect the early→late regime transition layer.

    Method: compute per-layer mean p3 fraction. The transition is where
    p3 drops below (early_mean + threshold_mean) / 2.
    """
    n_layers, n_heads = k_vectors.shape[:2]
    head_dim = k_vectors.shape[3]

    p3_by_layer = []
    for L in range(n_layers):
        p3_fracs = []
        for H in range(n_heads):
            kv = k_vectors[L, H]  # (n_pos, head_dim)
            # Transform a few positions to estimate prime fractions
            n_sample = min(32, kv.shape[0])
            total_p3 = 0.0
            total_all = 0.0
            for pos in range(n_sample):
                if use_sqfree and analysis_dim != head_dim:
                    vec = sqfree_pad_vector(kv[pos], analysis_dim)
                    spec = vht2(vec)
                else:
                    spec = vht2(kv[pos])
                var_per_idx = spec ** 2
                for i in range(len(spec)):
                    if i == 0:
                        continue
                    e = var_per_idx[i]
                    total_all += e
                    if i % 3 == 0:
                        total_p3 += e
            p3_fracs.append(total_p3 / total_all if total_all > 0 else 0)
        p3_by_layer.append(float(np.mean(p3_fracs)))

    # Find transition: largest single-layer drop in p3
    max_drop = 0
    transition_layer = n_layers // 2  # Default to midpoint
    for L in range(1, n_layers):
        drop = p3_by_layer[L - 1] - p3_by_layer[L]
        if drop > max_drop:
            max_drop = drop
            transition_layer = L

    # Also check: if the drop is small, use the layer where p3 first
    # crosses below the midpoint between early and late means
    early_mean = np.mean(p3_by_layer[:n_layers // 3])
    late_mean = np.mean(p3_by_layer[2 * n_layers // 3:])
    midpoint = (early_mean + late_mean) / 2

    for L in range(n_layers):
        if p3_by_layer[L] < midpoint:
            transition_layer = min(transition_layer, L)
            break

    info = {
        'transition_layer': transition_layer,
        'p3_curve': p3_by_layer,
        'p3_early_mean': float(early_mean),
        'p3_late_mean': float(late_mean),
        'max_drop': float(max_drop),
    }
    return transition_layer, info


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_regime(k_vectors: np.ndarray, use_sqfree: bool = True,
                   skeleton_fractions: Optional[List[float]] = None,
                   n_sample_positions: int = 64,
                   verbose: bool = True) -> Dict:
    """Run the two-regime reconstruction quality analysis.

    For each layer, simulates compression with three skeleton strategies
    at various skeleton sizes, measuring reconstruction error.
    """
    n_layers, n_heads, n_pos, head_dim = k_vectors.shape

    if use_sqfree:
        analysis_dim = sqfree_pad_dim(head_dim)
    else:
        analysis_dim = head_dim

    if skeleton_fractions is None:
        # Test at 10%, 20%, 30%, 40%, 50%, 60%, 75% of analysis_dim
        skeleton_fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75]

    # Detect sqfree primes for this dimension
    sqfree_primes = set()
    if analysis_dim != head_dim:
        d = analysis_dim
        for p in [2, 3, 5, 7, 11]:
            if d % p == 0:
                sqfree_primes.add(p)
                while d % p == 0:
                    d //= p

    if verbose:
        print(f"{'='*72}")
        print(f"  Two-Regime Reconstruction Analysis")
        print(f"{'='*72}")
        print(f"  Layers: {n_layers}, Heads: {n_heads}, Positions: {n_pos}, Head dim: {head_dim}")
        if use_sqfree:
            primes_str = '×'.join(str(p) for p in sorted(sqfree_primes))
            print(f"  Sqfree: {head_dim} → {analysis_dim} ({primes_str})")
        print(f"  Sample positions: {min(n_sample_positions, n_pos)}")
        print(f"  Skeleton fractions: {skeleton_fractions}")

    # Precompute algebraic skeletons for T²
    t2_primes = {2, 3}
    t2_skeleton_full = algebraic_skeleton(analysis_dim, t2_primes)
    t2_size = len(t2_skeleton_full)

    # T³: add the dominant secondary prime from sqfree padding
    secondary_prime = max(sqfree_primes - {2}, default=3)
    t3_primes = {2, 3, secondary_prime}
    t3_skeleton_full = algebraic_skeleton(analysis_dim, t3_primes)
    t3_size = len(t3_skeleton_full)

    if verbose:
        print(f"\n  T² skeleton ({t2_primes}): {t2_size}/{analysis_dim} indices "
              f"= {t2_size/analysis_dim:.0%} of dim")
        print(f"  T³ skeleton ({t3_primes}): {t3_size}/{analysis_dim} indices "
              f"= {t3_size/analysis_dim:.0%} of dim")

    # Detect transition
    if verbose:
        print(f"\n  Detecting regime transition...")
    transition_layer, transition_info = detect_transition(
        k_vectors, analysis_dim, use_sqfree
    )
    if verbose:
        print(f"  Transition at layer {transition_layer} "
              f"(p3: {transition_info['p3_early_mean']:.3f} → {transition_info['p3_late_mean']:.3f})")

    # ── Per-layer analysis ──────────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*72}")
        # Column headers
        header = f"  {'Layer':>5s} {'Regime':>6s}"
        for frac in skeleton_fractions:
            k = int(frac * analysis_dim)
            header += f"  K={k:3d}"
        print(header)
        print(f"  {'':>5s} {'':>6s}" + "  " + "─" * (len(skeleton_fractions) * 7 - 2))

    n_sample = min(n_sample_positions, n_pos)
    results = {
        'config': {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'head_dim': head_dim,
            'analysis_dim': analysis_dim,
            'sqfree_primes': sorted(sqfree_primes),
            'transition_layer': transition_layer,
            't2_primes': sorted(t2_primes),
            't3_primes': sorted(t3_primes),
            't2_skeleton_size': t2_size,
            't3_skeleton_size': t3_size,
            'skeleton_fractions': skeleton_fractions,
        },
        'transition': transition_info,
        'per_layer': [],
    }

    # Strategy names for output
    strategies = ['T2_alg', 'adaptive', 'hybrid']

    for L in range(n_layers):
        t0 = time.time()
        regime = 'early' if L < transition_layer else 'late'

        # Collect spectra across all heads and sample positions
        all_spectra = []
        all_originals = []
        for H in range(n_heads):
            for pos in range(n_sample):
                kv = k_vectors[L, H, pos]
                if use_sqfree and analysis_dim != head_dim:
                    padded = sqfree_pad_vector(kv, analysis_dim)
                    spec = vht2(padded)
                    all_originals.append(padded)
                else:
                    spec = vht2(kv)
                    all_originals.append(kv.copy())
                all_spectra.append(spec)

        all_spectra = np.array(all_spectra)    # (n_heads*n_sample, analysis_dim)
        all_originals = np.array(all_originals)

        layer_result = {
            'layer': L,
            'regime': regime,
            'strategies': {},
        }

        for strat_name in strategies:
            strat_errors = {}
            strat_energy = {}
            for frac in skeleton_fractions:
                k = max(1, int(frac * analysis_dim))

                # Build skeleton based on strategy
                if strat_name == 'T2_alg':
                    skel = t2_skeleton_full[:min(k, len(t2_skeleton_full))]
                    # If algebraic skeleton is smaller than k, pad with top-variance
                    if len(skel) < k:
                        remaining = adaptive_skeleton(all_spectra, analysis_dim)
                        alg_set = set(skel.tolist())
                        extras = [i for i in remaining if i not in alg_set]
                        skel = np.concatenate([skel, np.array(extras[:k - len(skel)])])
                elif strat_name == 'adaptive':
                    skel = adaptive_skeleton(all_spectra, k)
                elif strat_name == 'hybrid':
                    skel = hybrid_skeleton(all_spectra, analysis_dim, t2_primes, k)

                # Measure reconstruction quality
                errors = []
                energies = []
                for i in range(len(all_spectra)):
                    recon = reconstruct_from_skeleton(all_spectra[i], skel, analysis_dim)
                    err = reconstruction_error(all_originals[i], recon)
                    energy = spectral_energy_captured(all_spectra[i], skel)
                    errors.append(err)
                    energies.append(energy)

                mean_err = float(np.mean(errors))
                mean_energy = float(np.mean(energies))
                strat_errors[f'{frac:.2f}'] = round(mean_err, 6)
                strat_energy[f'{frac:.2f}'] = round(mean_energy, 6)

            layer_result['strategies'][strat_name] = {
                'errors': strat_errors,
                'energy_captured': strat_energy,
            }

        results['per_layer'].append(layer_result)

        elapsed = time.time() - t0

        if verbose:
            # Print adaptive errors (the most representative)
            errs_adaptive = layer_result['strategies']['adaptive']['errors']
            errs_t2 = layer_result['strategies']['T2_alg']['errors']
            line_adapt = f"  L{L:2d}   {regime:>5s}  "
            line_t2 = f"  {'':>5s} {'T²':>6s}  "
            for frac in skeleton_fractions:
                ea = errs_adaptive[f'{frac:.2f}']
                et = errs_t2[f'{frac:.2f}']
                line_adapt += f"{ea:6.3f} "
                line_t2 += f"{et:6.3f} "
            # Show winner per fraction
            line_win = f"  {'':>5s} {'win':>6s}  "
            for frac in skeleton_fractions:
                ea = errs_adaptive[f'{frac:.2f}']
                et = errs_t2[f'{frac:.2f}']
                if et < ea - 0.001:
                    line_win += f"  T²   "
                elif ea < et - 0.001:
                    line_win += f"  adp  "
                else:
                    line_win += f"  tie  "

            print(line_adapt + f" ({elapsed:.1f}s)")
            if L == 0 or L == transition_layer or L == n_layers - 1:
                print(line_t2)
                print(line_win)

    # ── Summary ──────────────────────────────────────────────────────────
    # Find the optimal strategy per regime
    early_layers = [r for r in results['per_layer'] if r['regime'] == 'early']
    late_layers = [r for r in results['per_layer'] if r['regime'] == 'late']

    def mean_error_for(layers, strategy, frac_key):
        if not layers:
            return 1.0
        return float(np.mean([l['strategies'][strategy]['errors'][frac_key] for l in layers]))

    if verbose:
        print(f"\n{'='*72}")
        print(f"  REGIME SUMMARY")
        print(f"{'='*72}")

        # For each skeleton fraction, show which strategy wins in each regime
        print(f"\n  Mean relative L2 error by strategy and regime:")
        print(f"  {'Skeleton':>10s}", end='')
        for strat in strategies:
            print(f"  {strat:>10s}", end='')
        print()

        for frac in skeleton_fractions:
            frac_key = f'{frac:.2f}'
            k = int(frac * analysis_dim)
            compress_ratio = analysis_dim / k

            print(f"\n  --- K={k} ({frac:.0%} of {analysis_dim}, "
                  f"{compress_ratio:.1f}× compression) ---")

            for regime_name, regime_layers in [('Early', early_layers), ('Late', late_layers)]:
                print(f"  {regime_name:>10s}", end='')
                best_err = 1.0
                best_strat = ''
                for strat in strategies:
                    err = mean_error_for(regime_layers, strat, frac_key)
                    print(f"  {err:10.4f}", end='')
                    if err < best_err:
                        best_err = err
                        best_strat = strat
                print(f"  ◀ {best_strat}")

    # Compute recommended strategy
    recommendations = {}
    for frac in skeleton_fractions:
        frac_key = f'{frac:.2f}'
        k = int(frac * analysis_dim)

        early_best = min(strategies, key=lambda s: mean_error_for(early_layers, s, frac_key))
        late_best = min(strategies, key=lambda s: mean_error_for(late_layers, s, frac_key))

        early_err = mean_error_for(early_layers, early_best, frac_key)
        late_err = mean_error_for(late_layers, late_best, frac_key)

        # Also compute the two-regime combined error (best-of per layer)
        two_regime_err = 0
        for lr in results['per_layer']:
            if lr['regime'] == 'early':
                two_regime_err += lr['strategies'][early_best]['errors'][frac_key]
            else:
                two_regime_err += lr['strategies'][late_best]['errors'][frac_key]
        two_regime_err /= n_layers

        # Compare against uniform best
        uniform_best = min(strategies,
                          key=lambda s: mean_error_for(results['per_layer'], s, frac_key))
        uniform_err = mean_error_for(results['per_layer'], uniform_best, frac_key)

        improvement = (uniform_err - two_regime_err) / uniform_err if uniform_err > 0 else 0

        recommendations[frac_key] = {
            'skeleton_k': k,
            'compression_ratio': round(analysis_dim / k, 1),
            'early_strategy': early_best,
            'early_error': round(early_err, 6),
            'late_strategy': late_best,
            'late_error': round(late_err, 6),
            'two_regime_error': round(two_regime_err, 6),
            'uniform_strategy': uniform_best,
            'uniform_error': round(uniform_err, 6),
            'improvement_pct': round(improvement * 100, 2),
        }

    results['recommendations'] = recommendations

    if verbose:
        print(f"\n{'─'*72}")
        print(f"  RECOMMENDATIONS")
        print(f"{'─'*72}")
        for frac_key, rec in sorted(recommendations.items()):
            print(f"\n  K={rec['skeleton_k']} ({rec['compression_ratio']}× compression):")
            print(f"    Early (L0-L{transition_layer-1}): {rec['early_strategy']:>10s}  "
                  f"error={rec['early_error']:.4f}")
            print(f"    Late  (L{transition_layer}-L{n_layers-1}): {rec['late_strategy']:>10s}  "
                  f"error={rec['late_error']:.4f}")
            print(f"    Two-regime combined:           error={rec['two_regime_error']:.4f}")
            print(f"    vs uniform {rec['uniform_strategy']:>10s}:     "
                  f"error={rec['uniform_error']:.4f}")
            imp = rec['improvement_pct']
            if imp > 1:
                print(f"    → Two-regime wins by {imp:.1f}%")
            elif imp < -1:
                print(f"    → Uniform wins by {-imp:.1f}% (two-regime not beneficial here)")
            else:
                print(f"    → Negligible difference ({imp:+.1f}%)")

        print(f"\n{'='*72}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# K vector loading (same as diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def load_k_vectors(path: str):
    """Load K vectors from .npz or .pt file."""
    if path.endswith('.npz'):
        data = np.load(path)
        if 'k_vectors' in data:
            k = data['k_vectors']
            return k, k.shape[0], k.shape[1], k.shape[3]

        # Per-head format: k_layer{L}_head{H}
        keys = sorted(data.files)
        layers = set()
        heads = set()
        for key in keys:
            if key.startswith('k_layer'):
                parts = key.replace('k_layer', '').replace('_head', ' ').split()
                if len(parts) == 2:
                    layers.add(int(parts[0]))
                    heads.add(int(parts[1]))

        n_layers = max(layers) + 1
        n_heads = max(heads) + 1
        sample = data[keys[0]]
        n_pos, head_dim = sample.shape

        k_vectors = np.zeros((n_layers, n_heads, n_pos, head_dim), dtype=np.float32)
        for key in keys:
            if key.startswith('k_layer'):
                parts = key.replace('k_layer', '').replace('_head', ' ').split()
                L, H = int(parts[0]), int(parts[1])
                k_vectors[L, H] = data[key]

        return k_vectors, n_layers, n_heads, head_dim

    elif path.endswith(('.pt', '.pth')):
        import torch
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict) and 'k_vectors' in data:
            k = data['k_vectors'].float().numpy()
            return k, k.shape[0], k.shape[1], k.shape[3]
        k = data.float().numpy()
        return k, k.shape[0], k.shape[1], k.shape[3]

    raise ValueError(f"Unsupported format: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Shannon-Prime Phase 11 — Two-Regime Reconstruction Analysis')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to K vectors (.npz or .pt)')
    parser.add_argument('--output', '-o', type=str,
                        help='Path to save JSON results')
    parser.add_argument('--sqfree', action='store_true',
                        help='Pad to sqfree dimensions for multi-prime analysis')
    parser.add_argument('--sample-positions', type=int, default=64,
                        help='Number of positions to sample per head (default: 64)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    args = parser.parse_args()

    print(f"Loading K vectors from {args.input}...")
    k_vectors, nl, nh, hd = load_k_vectors(args.input)
    print(f"  Shape: {k_vectors.shape}")

    results = analyze_regime(
        k_vectors,
        use_sqfree=args.sqfree,
        n_sample_positions=args.sample_positions,
        verbose=not args.quiet,
    )

    if args.output:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
