# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
sp_chord_diagnostic.py — Phase 11 Prime Chord Diagnostic

Analyzes extracted K vectors to test the Prime Chord hypothesis:
  - Do attention heads form "prime chords" (subsets of active primes)?
  - Does chord diversity expand in middle layers and collapse in late layers?
  - Are there "Ghost Heads" off the arithmetic manifold?
  - Are ghosts genuinely stochastic, or high-prime specialists (p=13,17,19)?

Input:
  K vectors as .npz file with arrays keyed by "k_layer{L}_head{H}" or
  a single "k_vectors" array of shape (n_layers, n_heads, n_positions, head_dim).
  Also accepts .pt / .pth files with the same structure as a dict of tensors.

Output:
  - Per-head prime histograms (console + JSON)
  - Chord entropy curve across layers (plot-ready)
  - Jaccard adjacency matrices per layer
  - Cross-layer persistence scores
  - Ghost Head classification with extended prime check (p=2..19)
  - Summary verdict: harmonic ratio, manifold dimensionality estimate

Usage:
    python sp_chord_diagnostic.py --input kv_vectors.npz
    python sp_chord_diagnostic.py --input kv_vectors.npz --output results.json
    python sp_chord_diagnostic.py --input kv_vectors.npz --plot
    python sp_chord_diagnostic.py --synthetic --n-layers 32 --n-heads 32 --head-dim 128

The --synthetic flag generates RoPE-structured test K vectors for development.
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Core 5-torus primes
PRIMES_5 = [2, 3, 5, 7, 11]

# Extended prime set for Ghost Head analysis
PRIMES_EXT = [2, 3, 5, 7, 11, 13, 17, 19, 23]

# Entropy thresholds for chord classification
ENTROPY_PURE_TONE = 1.5    # H < 1.5 bits → single dominant prime
ENTROPY_CHORD_MAX = 3.0    # 1.5 < H < 3.0 → chord (subset of primes)
ENTROPY_GHOST = 4.0        # H > 4.0 → ghost head (near-uniform)

# Ghost Head reclassification: if a "ghost" has >40% energy on extended primes
# (13, 17, 19, 23), reclassify as high-prime specialist
HIGH_PRIME_RECLAIM_THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# Math utilities
# ─────────────────────────────────────────────────────────────────────────────

def factorize(n: int) -> Dict[int, int]:
    """Prime factorization of n. Returns {prime: exponent}."""
    if n < 2:
        return {}
    factors = {}
    for p in PRIMES_EXT:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    # Remaining factor (if any prime > 23)
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def prime_signature(n: int) -> frozenset:
    """Set of distinct prime factors of n (ignoring multiplicity)."""
    return frozenset(factorize(n).keys())


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits from a count vector."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def jaccard_similarity(set_a: frozenset, set_b: frozenset) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


# ─────────────────────────────────────────────────────────────────────────────
# VHT2 (Vilenkin-Hartley Transform) — minimal numpy implementation
# ─────────────────────────────────────────────────────────────────────────────

def _hartley_kernel(p: int) -> np.ndarray:
    """p×p Hartley matrix H[i,j] = cas(2πij/p) / √p."""
    idx = np.arange(p)
    angles = (2.0 * np.pi / p) * np.outer(idx, idx)
    H = (np.cos(angles) + np.sin(angles)) / np.sqrt(p)
    return H


def _factor_small_primes(n: int) -> List[int]:
    """Factor n into primes {2,3,5,7,11}. Raises if unfactorable."""
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
    """Forward VHT2: in-place-style, returns spectral coefficients.

    Works on vectors whose length factors into {2,3,5,7,11}.
    For power-of-2, this is identical to the Walsh-Hadamard butterfly / √N.
    """
    n = len(x)
    primes = _factor_small_primes(n)
    out = x.copy().astype(np.float64)

    # Apply Kronecker stages in ascending prime order (matches GPU vht2_forward)
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


def vht2_forward_pow2(x: np.ndarray) -> np.ndarray:
    """Fast WHT butterfly for power-of-2 dimensions."""
    n = len(x)
    out = x.copy().astype(np.float64)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = out[j]
                b = out[j + h]
                out[j] = a + b
                out[j + h] = a - b
        h *= 2
    out /= np.sqrt(n)
    return out.astype(np.float32)


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def vht2(x: np.ndarray) -> np.ndarray:
    """VHT2 forward transform — auto-selects butterfly or general path."""
    if is_power_of_2(len(x)):
        return vht2_forward_pow2(x)
    return vht2_forward(x)


# ─────────────────────────────────────────────────────────────────────────────
# Sqfree padding — enables multi-prime analysis on power-of-2 head dims
# ─────────────────────────────────────────────────────────────────────────────

# Known sqfree pad dimensions (matches core/shannon_prime.h)
SQFREE_PAD = {64: 66, 96: 110, 128: 154, 256: 330}
# 66  = 2 × 3 × 11       (primes: 2,3,11)
# 110 = 2 × 5 × 11       (primes: 2,5,11)
# 154 = 2 × 7 × 11       (primes: 2,7,11)
# 330 = 2 × 3 × 5 × 11   (primes: 2,3,5,11)


def _is_sqfree_rich(n: int, min_distinct: int = 3) -> bool:
    """Check if n is squarefree and has >= min_distinct prime factors from {2,3,5,7,11}.

    Squarefree means no prime appears more than once in the factorization.
    This ensures the VHT2 Vilenkin decomposition has multiple distinct prime
    dimensions to work with.
    """
    distinct = 0
    d = n
    for p in [2, 3, 5, 7, 11]:
        if d % p == 0:
            distinct += 1
            d //= p
            if d % p == 0:
                return False  # Not squarefree — p appears more than once
    return d == 1 and distinct >= min_distinct


def sqfree_pad_dim(head_dim: int) -> int:
    """Find next squarefree-rich dimension >= head_dim.

    Returns the smallest n >= head_dim such that n factors entirely into
    {2,3,5,7,11} with each prime appearing at most once, and at least
    3 distinct primes are used.
    """
    if head_dim in SQFREE_PAD:
        return SQFREE_PAD[head_dim]
    n = head_dim
    while n < head_dim * 2:  # Safety bound: don't pad more than 2×
        if _is_sqfree_rich(n):
            return n
        n += 1
    # Fallback: return head_dim unchanged (no viable pad found)
    return head_dim


def sqfree_pad_vector(x: np.ndarray, pad_dim: int) -> np.ndarray:
    """Pad vector from head_dim to pad_dim with mean-fill."""
    hd = len(x)
    if hd >= pad_dim:
        return x[:pad_dim].copy()
    out = np.empty(pad_dim, dtype=x.dtype)
    out[:hd] = x
    out[hd:] = np.mean(x)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton extraction (variance-ranked mask)
# ─────────────────────────────────────────────────────────────────────────────

def build_skeleton_mask(spectra: np.ndarray, skeleton_k: int = 96) -> np.ndarray:
    """Build variance-ranked skeleton mask from a batch of spectral vectors.

    Args:
        spectra: (n_positions, head_dim) array of VHT2 coefficients
        skeleton_k: number of skeleton indices to keep

    Returns:
        skeleton_indices: (skeleton_k,) array of spectral indices, ranked by variance
    """
    # Variance per spectral index across positions
    var = np.var(spectra, axis=0)
    ranked = np.argsort(var)[::-1]  # descending variance
    return ranked[:skeleton_k]


# ─────────────────────────────────────────────────────────────────────────────
# Prime Chord Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_skeleton_primes(skeleton_indices: np.ndarray,
                            extended: bool = True,
                            spectral_energy: Optional[np.ndarray] = None) -> Dict:
    """Analyze the prime factorization of skeleton indices.

    Args:
        skeleton_indices: array of spectral index positions in the skeleton
        extended: use extended prime set (up to 23) for ghost detection
        spectral_energy: optional (dim,) array of per-index energy (variance).
            When provided, prime counts are ENERGY-WEIGHTED rather than
            index-counted. This captures "how much energy lives on the Z/pZ
            subspace" rather than just "how many indices are divisible by p."

    Returns:
        dict with:
          - prime_counts: {prime: count_or_energy} for each prime in PRIMES_EXT
          - prime_fractions: {prime: fraction} normalized
          - chord_entropy: Shannon entropy of prime distribution (bits)
          - dominant_chord: frozenset of primes with >15% representation
          - signatures: list of (index, prime_signature) pairs
          - ghost_score: 0.0 (pure tone) to 1.0 (uniform/ghost)
    """
    primes_to_check = PRIMES_EXT if extended else PRIMES_5
    use_energy = spectral_energy is not None

    # Count how many skeleton indices are divisible by each prime,
    # optionally weighted by spectral energy at each index
    prime_counts = {p: 0.0 for p in primes_to_check}
    signatures = []

    for i, idx in enumerate(skeleton_indices):
        idx = int(idx)
        weight = 1.0
        if use_energy and idx < len(spectral_energy):
            weight = float(spectral_energy[idx])

        if idx == 0:
            # DC component: assign to all primes equally (it's universal)
            for p in primes_to_check:
                prime_counts[p] += weight
            signatures.append((idx, frozenset()))
            continue

        sig = prime_signature(idx)
        signatures.append((idx, sig))
        for p in primes_to_check:
            if idx % p == 0:
                prime_counts[p] += weight

    # Normalize
    total = sum(prime_counts.values())
    prime_fractions = {}
    if total > 0:
        prime_fractions = {p: c / total for p, c in prime_counts.items()}

    # Entropy of prime distribution
    counts_arr = np.array([prime_counts[p] for p in primes_to_check], dtype=np.float64)
    chord_ent = shannon_entropy(counts_arr)

    # Dominant chord: primes with >15% representation
    CHORD_THRESHOLD = 0.15
    dominant = frozenset(p for p, f in prime_fractions.items() if f > CHORD_THRESHOLD)

    # Chord confidence: minimum distance any core prime has from the threshold.
    # For primes IN the chord, margin = fraction - threshold (positive = safely above).
    # For primes NOT in the chord, margin = threshold - fraction (positive = safely below).
    # Overall confidence = min of all margins. Near 0 = borderline. Negative = misclassified.
    core_primes_for_conf = PRIMES_5  # Only measure confidence on the 5-torus primes
    margins = []
    for p in core_primes_for_conf:
        f = prime_fractions.get(p, 0.0)
        if p in dominant:
            margins.append(f - CHORD_THRESHOLD)
        else:
            margins.append(CHORD_THRESHOLD - f)
    chord_confidence = min(margins) if margins else 0.0

    # Ghost score: how close to uniform? (1.0 = perfectly uniform)
    max_entropy = np.log2(len(primes_to_check))
    ghost_score = chord_ent / max_entropy if max_entropy > 0 else 0.0

    # Check if "ghost" is actually a high-prime specialist
    core_energy = sum(prime_counts[p] for p in PRIMES_5)
    ext_energy = sum(prime_counts[p] for p in [13, 17, 19, 23] if p in prime_counts)
    high_prime_fraction = ext_energy / total if total > 0 else 0.0

    return {
        'prime_counts': prime_counts,
        'prime_fractions': prime_fractions,
        'chord_entropy': chord_ent,
        'dominant_chord': dominant,
        'chord_confidence': chord_confidence,
        'signatures': signatures,
        'ghost_score': ghost_score,
        'high_prime_fraction': high_prime_fraction,
    }


def classify_head(analysis: Dict) -> str:
    """Classify a head based on its prime chord analysis.

    Returns one of:
      'pure_tone'   — H < 1.5, dominated by 1-2 primes
      'chord'       — 1.5 <= H < 3.5, clear subset of active primes
      'ghost'       — H >= 3.5, near-uniform across all primes
      'high_prime'  — was ghost but >40% energy on extended primes (13,17,19,23)
    """
    ent = analysis['chord_entropy']
    hp_frac = analysis['high_prime_fraction']

    if ent < ENTROPY_PURE_TONE:
        return 'pure_tone'
    elif ent < 3.5:
        return 'chord'
    else:
        # Ghost candidate — check for high-prime specialization
        if hp_frac >= HIGH_PRIME_RECLAIM_THRESHOLD:
            return 'high_prime'
        return 'ghost'


# ─────────────────────────────────────────────────────────────────────────────
# Cross-layer analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_jaccard_matrix(chords: List[frozenset]) -> np.ndarray:
    """Compute pairwise Jaccard similarity between head chords in a layer."""
    n = len(chords)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = jaccard_similarity(chords[i], chords[j])
    return J


def compute_persistence(layer_chords_a: List[frozenset],
                        layer_chords_b: List[frozenset]) -> float:
    """Fraction of heads whose dominant chord is identical between two layers."""
    assert len(layer_chords_a) == len(layer_chords_b)
    n = len(layer_chords_a)
    if n == 0:
        return 0.0
    matches = sum(1 for a, b in zip(layer_chords_a, layer_chords_b) if a == b)
    return matches / n


def is_tridiagonal_like(J: np.ndarray, threshold: float = 0.5) -> float:
    """Measure how "tridiagonal" a Jaccard matrix is.

    Returns a score from 0 (no tridiagonal structure) to 1 (perfectly tridiagonal).
    Compares mean similarity on the tri-diagonal band vs off-band.
    """
    n = J.shape[0]
    if n < 3:
        return 0.0

    band_vals = []
    off_vals = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(i - j) <= 1:
                band_vals.append(J[i, j])
            else:
                off_vals.append(J[i, j])

    if not band_vals or not off_vals:
        return 0.0

    band_mean = np.mean(band_vals)
    off_mean = np.mean(off_vals)

    if band_mean <= off_mean:
        return 0.0
    return float((band_mean - off_mean) / band_mean)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_k_vectors(path: str) -> Tuple[np.ndarray, int, int, int]:
    """Load K vectors from .npz or .pt file.

    Returns:
        k_vectors: (n_layers, n_heads, n_positions, head_dim)
        n_layers, n_heads, head_dim
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.pt', '.pth'):
        try:
            import torch
            data = torch.load(path, map_location='cpu', weights_only=False)
        except ImportError:
            print("ERROR: .pt files require PyTorch. Use .npz instead.")
            sys.exit(1)

        if isinstance(data, dict):
            # Try structured format: k_layer{L}_head{H}
            layers = set()
            heads = set()
            for key in data.keys():
                if key.startswith('k_layer'):
                    parts = key.replace('k_layer', '').split('_head')
                    if len(parts) == 2:
                        layers.add(int(parts[0]))
                        heads.add(int(parts[1]))

            if layers and heads:
                n_layers = max(layers) + 1
                n_heads = max(heads) + 1
                sample = data[f'k_layer0_head0']
                if hasattr(sample, 'numpy'):
                    sample = sample.numpy()
                n_pos, hd = sample.shape
                k_vectors = np.zeros((n_layers, n_heads, n_pos, hd), dtype=np.float32)
                for L in range(n_layers):
                    for H in range(n_heads):
                        key = f'k_layer{L}_head{H}'
                        if key in data:
                            t = data[key]
                            if hasattr(t, 'numpy'):
                                t = t.numpy()
                            k_vectors[L, H] = t
                return k_vectors, n_layers, n_heads, hd

            # Try single tensor
            if 'k_vectors' in data:
                t = data['k_vectors']
                if hasattr(t, 'numpy'):
                    t = t.numpy()
                return t, t.shape[0], t.shape[1], t.shape[3]

        # Single tensor
        if hasattr(data, 'numpy'):
            data = data.numpy()
        if data.ndim == 4:
            return data, data.shape[0], data.shape[1], data.shape[3]

    elif ext == '.npz':
        data = np.load(path, allow_pickle=True)

        # Try structured format
        layers = set()
        heads = set()
        for key in data.files:
            if key.startswith('k_layer'):
                parts = key.replace('k_layer', '').split('_head')
                if len(parts) == 2:
                    layers.add(int(parts[0]))
                    heads.add(int(parts[1]))

        if layers and heads:
            n_layers = max(layers) + 1
            n_heads = max(heads) + 1
            sample = data[f'k_layer0_head0']
            n_pos, hd = sample.shape
            k_vectors = np.zeros((n_layers, n_heads, n_pos, hd), dtype=np.float32)
            for L in range(n_layers):
                for H in range(n_heads):
                    key = f'k_layer{L}_head{H}'
                    if key in data.files:
                        k_vectors[L, H] = data[key]
            return k_vectors, n_layers, n_heads, hd

        # Single array
        if 'k_vectors' in data.files:
            arr = data['k_vectors']
            return arr, arr.shape[0], arr.shape[1], arr.shape[3]

        # First array
        arr = data[data.files[0]]
        if arr.ndim == 4:
            return arr, arr.shape[0], arr.shape[1], arr.shape[3]

    raise ValueError(f"Cannot parse K vectors from {path}")


def generate_synthetic_k_vectors(n_layers: int = 32, n_heads: int = 32,
                                 head_dim: int = 128, n_pos: int = 256,
                                 seed: int = 42) -> np.ndarray:
    """Generate synthetic RoPE-structured K vectors for testing.

    Simulates head specialization: each head gets a different RoPE frequency
    profile, creating the expected prime chord structure.
    """
    rng = np.random.RandomState(seed)
    k_vectors = np.zeros((n_layers, n_heads, n_pos, head_dim), dtype=np.float32)

    # RoPE base frequencies for each dimension pair
    base = 10000.0
    dim_pairs = head_dim // 2
    thetas = base ** (-2.0 * np.arange(dim_pairs) / head_dim)

    for L in range(n_layers):
        # Layer-dependent specialization: deeper layers → sharper frequency focus
        layer_sharpness = 1.0 + 2.0 * (L / n_layers)

        for H in range(n_heads):
            # Each head has a "center frequency" that shifts across heads
            # This creates the sliding-window chord structure
            center = H / n_heads  # 0.0 to ~1.0
            freq_weights = np.exp(-layer_sharpness *
                                  (np.linspace(0, 1, dim_pairs) - center) ** 2)

            for pos in range(n_pos):
                # RoPE rotation at this position
                angles = pos * thetas
                cos_a = np.cos(angles) * freq_weights
                sin_a = np.sin(angles) * freq_weights

                # K vector: mix of RoPE-rotated random projections
                raw = rng.randn(dim_pairs).astype(np.float32)
                k = np.zeros(head_dim, dtype=np.float32)
                k[0::2] = raw * cos_a - rng.randn(dim_pairs) * sin_a
                k[1::2] = raw * sin_a + rng.randn(dim_pairs) * cos_a

                # Add some non-harmonic noise (simulates Ghost Head potential)
                if H in [7, 23]:  # Designated "ghost" heads
                    k += rng.randn(head_dim).astype(np.float32) * 0.5

                k_vectors[L, H, pos] = k

    return k_vectors


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic(k_vectors: np.ndarray, skeleton_k: int = 96,
                   verbose: bool = True, use_sqfree: bool = False) -> Dict:
    """Run the full Prime Chord diagnostic.

    Args:
        k_vectors: (n_layers, n_heads, n_positions, head_dim) array
        skeleton_k: number of skeleton indices per head
        verbose: print progress and results
        use_sqfree: pad power-of-2 head dims to sqfree dimensions
            (128→154=2·7·11) to enable multi-prime analysis

    Returns:
        Full results dict with all metrics
    """
    n_layers, n_heads, n_pos, head_dim = k_vectors.shape

    # Determine analysis dimension
    analysis_dim = head_dim
    if use_sqfree:
        analysis_dim = sqfree_pad_dim(head_dim)

    if verbose:
        print(f"\n{'='*72}")
        print(f"  Shannon-Prime Phase 11 — Prime Chord Diagnostic")
        print(f"{'='*72}")
        print(f"  Layers: {n_layers}  Heads: {n_heads}  "
              f"Positions: {n_pos}  Head dim: {head_dim}")
        if use_sqfree and analysis_dim != head_dim:
            primes_of_dim = _factor_small_primes(analysis_dim)
            print(f"  Sqfree pad: {head_dim} → {analysis_dim} "
                  f"({' × '.join(str(p) for p in sorted(set(primes_of_dim)))})")
        print(f"  Skeleton K: {skeleton_k}")
        print(f"  Extended primes: {PRIMES_EXT}")
        print(f"{'='*72}\n")

    results = {
        'config': {
            'n_layers': n_layers, 'n_heads': n_heads,
            'n_positions': n_pos, 'head_dim': head_dim,
            'analysis_dim': analysis_dim,
            'sqfree_padded': use_sqfree and analysis_dim != head_dim,
            'skeleton_k': skeleton_k,
        },
        'per_layer': [],
        'cross_layer': {},
        'summary': {},
    }

    # ── Per-layer analysis ────────────────────────────────────────────────
    all_layer_chords = []  # List[List[frozenset]]  — per layer, per head

    for L in range(n_layers):
        t0 = time.time()
        layer_result = {
            'layer': L,
            'heads': [],
            'chord_entropy_mean': 0.0,
            'n_pure_tone': 0,
            'n_chord': 0,
            'n_ghost': 0,
            'n_high_prime': 0,
            'unique_chords': 0,
            'tridiagonal_score': 0.0,
        }

        head_chords = []
        head_analyses = []

        for H in range(n_heads):
            # VHT2 transform all positions for this head
            kv = k_vectors[L, H]  # (n_pos, head_dim)

            # Optionally pad to sqfree dimension for multi-prime analysis
            if use_sqfree and analysis_dim != head_dim:
                spectra = np.zeros((n_pos, analysis_dim), dtype=np.float32)
                for p in range(n_pos):
                    padded = sqfree_pad_vector(kv[p], analysis_dim)
                    spectra[p] = vht2(padded)
            else:
                spectra = np.zeros_like(kv)
                for p in range(n_pos):
                    spectra[p] = vht2(kv[p])

            # Build variance-ranked skeleton
            skel_idx = build_skeleton_mask(spectra, skeleton_k)

            # Compute per-index energy (variance across positions) for weighting
            spectral_var = np.var(spectra, axis=0)

            # Analyze prime structure of skeleton indices (energy-weighted)
            analysis = analyze_skeleton_primes(skel_idx, extended=True,
                                              spectral_energy=spectral_var)
            classification = classify_head(analysis)

            head_result = {
                'head': H,
                'classification': classification,
                'chord_entropy': analysis['chord_entropy'],
                'chord_confidence': analysis['chord_confidence'],
                'ghost_score': analysis['ghost_score'],
                'high_prime_fraction': analysis['high_prime_fraction'],
                'dominant_chord': sorted(analysis['dominant_chord']),
                'prime_fractions': {str(k): round(v, 4)
                                    for k, v in analysis['prime_fractions'].items()},
            }
            layer_result['heads'].append(head_result)
            head_chords.append(analysis['dominant_chord'])
            head_analyses.append(analysis)

            # Count classifications
            if classification == 'pure_tone':
                layer_result['n_pure_tone'] += 1
            elif classification == 'chord':
                layer_result['n_chord'] += 1
            elif classification == 'ghost':
                layer_result['n_ghost'] += 1
            elif classification == 'high_prime':
                layer_result['n_high_prime'] += 1

        # Layer-level metrics
        entropies = [a['chord_entropy'] for a in head_analyses]
        layer_result['chord_entropy_mean'] = float(np.mean(entropies))
        layer_result['chord_entropy_std'] = float(np.std(entropies))

        # Unique chords
        unique = set(frozenset(c) for c in head_chords)
        layer_result['unique_chords'] = len(unique)

        # Jaccard adjacency matrix
        J = compute_jaccard_matrix(head_chords)
        layer_result['tridiagonal_score'] = is_tridiagonal_like(J)
        layer_result['jaccard_mean'] = float(np.mean(J[np.triu_indices(n_heads, k=1)]))

        all_layer_chords.append(head_chords)
        results['per_layer'].append(layer_result)

        elapsed = time.time() - t0
        if verbose:
            print(f"  Layer {L:2d}: entropy={layer_result['chord_entropy_mean']:.3f} "
                  f"±{layer_result['chord_entropy_std']:.3f}  "
                  f"unique={layer_result['unique_chords']:2d}  "
                  f"tone={layer_result['n_pure_tone']} "
                  f"chord={layer_result['n_chord']} "
                  f"ghost={layer_result['n_ghost']} "
                  f"hi-p={layer_result['n_high_prime']}  "
                  f"tridiag={layer_result['tridiagonal_score']:.3f}  "
                  f"({elapsed:.1f}s)")

    # ── Cross-layer analysis ──────────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*72}")
        print(f"  Cross-Layer Persistence")
        print(f"{'─'*72}")

    def _safe_mean(arr):
        """np.mean that returns 0.0 for empty arrays without warnings."""
        if not arr:
            return 0.0
        return float(np.mean(arr))

    persistence_scores = []
    for L in range(1, n_layers):
        p = compute_persistence(all_layer_chords[L - 1], all_layer_chords[L])
        persistence_scores.append(p)
        if verbose:
            bar = '█' * int(p * 40) + '░' * (40 - int(p * 40))
            print(f"  L{L-1:2d}→L{L:2d}: {p:.3f} {bar}")

    results['cross_layer'] = {
        'persistence_scores': persistence_scores,
        'persistence_mean': float(np.mean(persistence_scores)) if persistence_scores else 0.0,
        'persistence_early': _safe_mean(persistence_scores[:max(1,n_layers//4)]),
        'persistence_middle': _safe_mean(persistence_scores[max(1,n_layers//4):max(2,3*n_layers//4)]) if len(persistence_scores) > 2 else _safe_mean(persistence_scores),
        'persistence_late': _safe_mean(persistence_scores[max(1,3*n_layers//4):]) if len(persistence_scores) > 2 else _safe_mean(persistence_scores),
    }

    # ── Chord Atlas ──────────────────────────────────────────────────────
    # For each layer, group heads by their chord class (dominant_chord as
    # a frozenset of primes), and identify transition layers where chord
    # reorganization occurs (persistence cliff < 0.15).

    chord_atlas = []
    for L in range(n_layers):
        lr = results['per_layer'][L]
        # Group heads by their dominant chord
        chord_groups = defaultdict(list)
        for h_result in lr['heads']:
            chord_key = tuple(sorted(h_result['dominant_chord']))
            chord_groups[chord_key].append(h_result['head'])

        # Sort chord classes by size (most heads first)
        sorted_classes = sorted(chord_groups.items(),
                                key=lambda x: (-len(x[1]), x[0]))

        # Build a compact label for each chord class
        layer_atlas = {
            'layer': L,
            'n_classes': len(sorted_classes),
            'classes': [],
        }
        for chord_primes, head_ids in sorted_classes:
            label = '×'.join(str(p) for p in chord_primes) if chord_primes else '∅'
            # Compute mean prime fractions and confidence for heads in this class
            class_fracs = defaultdict(list)
            class_confs = []
            borderline_heads = []
            for h_result in lr['heads']:
                if h_result['head'] in head_ids:
                    for p_str, frac in h_result['prime_fractions'].items():
                        class_fracs[int(p_str)].append(frac)
                    conf = h_result.get('chord_confidence', 0.0)
                    class_confs.append(conf)
                    if abs(conf) < 0.02:  # within 2% of boundary
                        borderline_heads.append((h_result['head'], conf))
            mean_fracs = {p: float(np.mean(v)) for p, v in class_fracs.items()}
            # Top-3 primes by mean fraction
            top3 = sorted(mean_fracs.items(), key=lambda x: -x[1])[:3]
            mean_conf = float(np.mean(class_confs)) if class_confs else 0.0
            min_conf = float(np.min(class_confs)) if class_confs else 0.0

            layer_atlas['classes'].append({
                'chord': list(chord_primes),
                'label': label,
                'heads': head_ids,
                'n_heads': len(head_ids),
                'mean_prime_fractions': {str(k): round(v, 4) for k, v in mean_fracs.items()},
                'top3_primes': [[p, round(f, 3)] for p, f in top3],
                'confidence_mean': round(mean_conf, 4),
                'confidence_min': round(min_conf, 4),
                'borderline_heads': [[h, round(c, 4)] for h, c in borderline_heads],
            })
        chord_atlas.append(layer_atlas)

    results['chord_atlas'] = chord_atlas

    # Identify transition layers (persistence cliff)
    transition_layers = []
    for i, p in enumerate(persistence_scores):
        if p < 0.15:
            transition_layers.append(i + 1)  # L(i+1) is where the shift lands

    # Also always include the layer with peak diversity
    diversity_curve_pre = [la['n_classes'] for la in chord_atlas]
    peak_div_layer = int(np.argmax(diversity_curve_pre))

    # Verbose chord atlas output
    if verbose:
        print(f"\n{'─'*72}")
        print(f"  Chord Atlas — Per-Head Prime Chord Signatures")
        print(f"{'─'*72}")

        # Show interesting layers: first, last, transitions, peak diversity
        interesting = sorted(set(
            [0, n_layers - 1, peak_div_layer] + transition_layers +
            # Also show layers adjacent to transitions
            [max(0, t - 1) for t in transition_layers]
        ))

        for L in interesting:
            la = chord_atlas[L]
            is_transition = L in transition_layers
            marker = " ◀ TRANSITION" if is_transition else ""
            if L == peak_div_layer:
                marker += " ◀ PEAK DIVERSITY"

            print(f"\n  Layer {L:2d} — {la['n_classes']} chord class(es){marker}")
            print(f"  {'·'*60}")

            for cls in la['classes']:
                heads_str = ','.join(f'H{h}' for h in cls['heads'])
                top3_str = '  '.join(f"p{p}={f:.0%}" for p, f in cls['top3_primes'])
                conf_mean = cls['confidence_mean']
                conf_min = cls['confidence_min']
                # Confidence gauge
                if conf_min < 0:
                    conf_tag = "MISCLASSIFIED"
                elif conf_min < 0.005:
                    conf_tag = "RAZOR-EDGE"
                elif conf_min < 0.01:
                    conf_tag = "BORDERLINE"
                elif conf_min < 0.02:
                    conf_tag = "SOFT"
                elif conf_min < 0.05:
                    conf_tag = "MODERATE"
                else:
                    conf_tag = "FIRM"
                print(f"    {cls['label']:>12s}  │ {cls['n_heads']:2d} heads: [{heads_str}]")
                print(f"    {'':>12s}  │ energy: {top3_str}")
                print(f"    {'':>12s}  │ confidence: mean={conf_mean:+.3f}  "
                      f"min={conf_min:+.3f}  [{conf_tag}]")
                if cls['borderline_heads']:
                    bl_str = ', '.join(f"H{h}({c:+.3f})" for h, c in cls['borderline_heads'])
                    print(f"    {'':>12s}  │ ⚠ borderline: {bl_str}")

        # Cross-transition chord migration report
        if transition_layers:
            print(f"\n  {'─'*60}")
            print(f"  Chord Migration at Transition(s)")
            print(f"  {'─'*60}")
            for T in transition_layers:
                pre = chord_atlas[T - 1]
                post = chord_atlas[T]
                pre_set = set(tuple(c['chord']) for c in pre['classes'])
                post_set = set(tuple(c['chord']) for c in post['classes'])
                born = post_set - pre_set
                died = pre_set - post_set
                survived = pre_set & post_set

                print(f"\n    L{T-1:2d} → L{T:2d}  "
                      f"(persistence={persistence_scores[T-1]:.3f})")
                if survived:
                    labels = ['×'.join(str(p) for p in c) if c else '∅'
                              for c in sorted(survived)]
                    print(f"      Survived:  {', '.join(labels)}")
                if born:
                    labels = ['×'.join(str(p) for p in c) if c else '∅'
                              for c in sorted(born)]
                    print(f"      Born:      {', '.join(labels)}")
                if died:
                    labels = ['×'.join(str(p) for p in c) if c else '∅'
                              for c in sorted(died)]
                    print(f"      Died:      {', '.join(labels)}")

                # Head-level migration: where did each head go?
                pre_map = {}
                for cls in pre['classes']:
                    for h in cls['heads']:
                        pre_map[h] = cls['label']
                post_map = {}
                for cls in post['classes']:
                    for h in cls['heads']:
                        post_map[h] = cls['label']

                # Show heads that changed class
                changed = []
                for h in sorted(set(pre_map.keys()) | set(post_map.keys())):
                    old = pre_map.get(h, '?')
                    new = post_map.get(h, '?')
                    if old != new:
                        changed.append(f"H{h}:{old}→{new}")
                if changed:
                    # Wrap at ~70 chars
                    line = "      Migrated:  "
                    lines = [line]
                    for ch in changed:
                        if len(lines[-1]) + len(ch) + 2 > 72:
                            lines.append("                 ")
                        lines[-1] += ch + "  "
                    print('\n'.join(lines))

    # ── Summary ──────────────────────────────────────────────────────────
    total_heads = n_layers * n_heads
    total_ghost = sum(lr['n_ghost'] for lr in results['per_layer'])
    total_high_prime = sum(lr['n_high_prime'] for lr in results['per_layer'])
    total_harmonic = total_heads - total_ghost

    # Chord diversity curve
    diversity_curve = [lr['unique_chords'] for lr in results['per_layer']]
    peak_diversity_layer = int(np.argmax(diversity_curve))

    # Entropy curve
    entropy_curve = [lr['chord_entropy_mean'] for lr in results['per_layer']]

    # Ghost head layer distribution
    ghost_curve = [lr['n_ghost'] for lr in results['per_layer']]
    peak_ghost_layer = int(np.argmax(ghost_curve)) if max(ghost_curve) > 0 else -1

    results['summary'] = {
        'total_heads': total_heads,
        'harmonic_count': total_harmonic,
        'ghost_count': total_ghost,
        'high_prime_count': total_high_prime,
        'harmonic_ratio': total_harmonic / total_heads if total_heads > 0 else 0.0,
        'ghost_ratio': total_ghost / total_heads if total_heads > 0 else 0.0,
        'peak_diversity_layer': peak_diversity_layer,
        'peak_diversity_value': diversity_curve[peak_diversity_layer],
        'peak_ghost_layer': peak_ghost_layer,
        'entropy_curve': entropy_curve,
        'diversity_curve': diversity_curve,
        'ghost_curve': ghost_curve,
        'persistence': results['cross_layer'],
    }

    # Manifold dimensionality estimate:
    # Count distinct primes that appear as dominant in >10% of harmonic heads
    all_dominant_primes = defaultdict(int)
    for L in range(n_layers):
        for h_result in results['per_layer'][L]['heads']:
            if h_result['classification'] in ('pure_tone', 'chord', 'high_prime'):
                for p in h_result['dominant_chord']:
                    all_dominant_primes[p] += 1
    active_primes = [p for p, c in all_dominant_primes.items()
                     if c > 0.10 * total_harmonic]
    results['summary']['active_primes'] = sorted(active_primes)
    results['summary']['manifold_dim_estimate'] = 2 * len(active_primes)  # T^k = 2k real dims

    # ── Verdict ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*72}")
        print(f"  VERDICT")
        print(f"{'='*72}")
        print(f"  Harmonic heads:  {total_harmonic}/{total_heads} "
              f"({100*results['summary']['harmonic_ratio']:.1f}%)")
        print(f"  Ghost heads:     {total_ghost}/{total_heads} "
              f"({100*results['summary']['ghost_ratio']:.1f}%)")
        print(f"  High-prime spec: {total_high_prime}/{total_heads} "
              f"({100*total_high_prime/total_heads:.1f}%)")
        print()
        print(f"  Active primes:   {results['summary']['active_primes']}")
        print(f"  Manifold dim:    ~{results['summary']['manifold_dim_estimate']}D "
              f"(T^{len(active_primes)})")
        print()
        print(f"  Peak diversity:  Layer {peak_diversity_layer} "
              f"({diversity_curve[peak_diversity_layer]} unique chords)")
        if peak_ghost_layer >= 0:
            print(f"  Peak ghosts:     Layer {peak_ghost_layer} "
                  f"({ghost_curve[peak_ghost_layer]} ghost heads)")
        print()
        print(f"  Persistence (early  L0-L{n_layers//4}):  "
              f"{results['cross_layer']['persistence_early']:.3f}")
        print(f"  Persistence (middle L{n_layers//4}-L{3*n_layers//4}): "
              f"{results['cross_layer']['persistence_middle']:.3f}")
        print(f"  Persistence (late   L{3*n_layers//4}-L{n_layers}):  "
              f"{results['cross_layer']['persistence_late']:.3f}")

        # Sliding window verdict
        tridiag_scores = [lr['tridiagonal_score'] for lr in results['per_layer']]
        mean_tridiag = np.mean(tridiag_scores)
        print(f"\n  Tridiagonal score (sliding window): {mean_tridiag:.3f}")
        if mean_tridiag > 0.3:
            print(f"  → CONFIRMED: Heads form a sliding window on the torus")
        elif mean_tridiag > 0.15:
            print(f"  → PARTIAL: Some sliding-window structure detected")
        else:
            print(f"  → NOT DETECTED: Heads do not show sliding-window adjacency")

        # Global chord confidence summary
        all_confs = []
        all_borderline = []
        for L in range(n_layers):
            for h_result in results['per_layer'][L]['heads']:
                c = h_result.get('chord_confidence', 0.0)
                all_confs.append(c)
                if abs(c) < 0.02:
                    all_borderline.append((L, h_result['head'], c))
        global_conf_mean = float(np.mean(all_confs)) if all_confs else 0.0
        global_conf_min = float(np.min(all_confs)) if all_confs else 0.0
        n_borderline = len(all_borderline)
        pct_borderline = 100 * n_borderline / total_heads if total_heads > 0 else 0.0

        results['summary']['chord_confidence_mean'] = round(global_conf_mean, 4)
        results['summary']['chord_confidence_min'] = round(global_conf_min, 4)
        results['summary']['n_borderline_heads'] = n_borderline
        results['summary']['pct_borderline'] = round(pct_borderline, 1)

        print(f"\n  Chord confidence:")
        print(f"    Global mean:  {global_conf_mean:+.4f}")
        print(f"    Global min:   {global_conf_min:+.4f}")
        print(f"    Borderline (<2%): {n_borderline}/{total_heads} "
              f"({pct_borderline:.1f}%)")
        if all_borderline:
            # Show worst 10 borderline heads
            worst = sorted(all_borderline, key=lambda x: abs(x[2]))[:10]
            for lyr, h, c in worst:
                chord = results['per_layer'][lyr]['heads'][h]['dominant_chord']
                chord_str = '×'.join(str(p) for p in chord) if chord else '∅'
                print(f"      L{lyr:2d} H{h:2d}: conf={c:+.4f}  chord={chord_str}")

        # Phase 11 feasibility
        hr = results['summary']['harmonic_ratio']
        print(f"\n  Phase 11 feasibility:")
        if hr > 0.90:
            print(f"  → STRONG: {hr:.0%} harmonic — algebraic reconstruction viable")
        elif hr > 0.75:
            print(f"  → MODERATE: {hr:.0%} harmonic — viable with ghost-head fallback")
        else:
            print(f"  → WEAK: {hr:.0%} harmonic — manifold may need expansion "
                  f"(T^{len(active_primes)} → T^{len(active_primes)+2})")

        # Confidence-adjusted feasibility
        if hr > 0.90 and pct_borderline > 30:
            print(f"  ⚠ CAUTION: {pct_borderline:.0f}% borderline heads — "
                  f"chord boundaries may be unstable under different prompts")

        print(f"\n{'='*72}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ASCII visualization (works everywhere)
# ─────────────────────────────────────────────────────────────────────────────

def print_ascii_plots(results: Dict):
    """Print ASCII visualizations of key metrics."""
    n_layers = results['config']['n_layers']
    entropy_curve = results['summary']['entropy_curve']
    diversity_curve = results['summary']['diversity_curve']
    ghost_curve = results['summary']['ghost_curve']

    # Chord Entropy Curve
    print(f"\n  Chord Entropy by Layer (higher = more diverse prime usage)")
    print(f"  {'─'*60}")
    max_ent = max(entropy_curve) if entropy_curve else 1.0
    for L in range(n_layers):
        bar_len = int(40 * entropy_curve[L] / max_ent) if max_ent > 0 else 0
        bar = '▓' * bar_len + '░' * (40 - bar_len)
        label = f"L{L:2d}"
        print(f"  {label} {bar} {entropy_curve[L]:.2f}")

    # Ghost Head Distribution
    print(f"\n  Ghost Heads by Layer")
    print(f"  {'─'*60}")
    max_g = max(ghost_curve) if ghost_curve and max(ghost_curve) > 0 else 1
    for L in range(n_layers):
        bar_len = int(40 * ghost_curve[L] / max_g) if max_g > 0 else 0
        bar = '█' * bar_len
        label = f"L{L:2d}"
        g = ghost_curve[L]
        print(f"  {label} {bar:<40s} {g}")

    # Unique Chords (Diversity)
    print(f"\n  Chord Diversity by Layer (unique chord signatures)")
    print(f"  {'─'*60}")
    max_d = max(diversity_curve) if diversity_curve else 1
    for L in range(n_layers):
        bar_len = int(40 * diversity_curve[L] / max_d) if max_d > 0 else 0
        bar = '▒' * bar_len
        label = f"L{L:2d}"
        print(f"  {label} {bar:<40s} {diversity_curve[L]}")

    # Chord Class Heatmap — one symbol per head, consistent across layers
    if 'chord_atlas' in results:
        atlas = results['chord_atlas']
        n_heads = results['config']['n_heads']

        # Build a global chord-to-symbol mapping (stable across layers)
        all_chords = set()
        for la in atlas:
            for cls in la['classes']:
                all_chords.add(tuple(cls['chord']))
        # Sort: biggest chords (most primes) first, then by prime values
        sorted_chords = sorted(all_chords, key=lambda c: (-len(c), c))
        symbols = '●◆▲■★◇▽□○◈⬟⬡⊕⊗⊞⊠'
        chord_sym = {}
        for i, c in enumerate(sorted_chords):
            chord_sym[c] = symbols[i % len(symbols)]

        # Legend
        print(f"\n  Chord Class Heatmap (each column = one head)")
        print(f"  {'─'*60}")
        print(f"  Legend:")
        for c, s in chord_sym.items():
            label = '×'.join(str(p) for p in c) if c else '∅'
            print(f"    {s} = {label}")
        print()

        # Head index header (compact)
        if n_heads <= 40:
            hdr = ''.join(f"{h%10}" for h in range(n_heads))
            print(f"       {hdr}")
            print(f"       {'─'*n_heads}")

        for L in range(n_layers):
            la = atlas[L]
            # Build head→symbol map for this layer
            head_sym = ['·'] * n_heads
            for cls in la['classes']:
                sym = chord_sym.get(tuple(cls['chord']), '?')
                for h in cls['heads']:
                    if h < n_heads:
                        head_sym[h] = sym
            row = ''.join(head_sym)
            # Mark transition layers
            marker = ''
            if L > 0 and L - 1 < len(results['cross_layer']['persistence_scores']):
                if results['cross_layer']['persistence_scores'][L - 1] < 0.15:
                    marker = ' ◀'
            print(f"  L{L:2d}  {row}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# Optional matplotlib visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: Dict, output_path: Optional[str] = None):
    """Generate matplotlib plots of diagnostic results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots")
        return

    n_layers = results['config']['n_layers']
    layers = list(range(n_layers))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Shannon-Prime Phase 11 — Prime Chord Diagnostic', fontsize=14)

    # 1. Chord Entropy Curve
    ax = axes[0, 0]
    ax.plot(layers, results['summary']['entropy_curve'], 'b-o', markersize=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Chord Entropy (bits)')
    ax.set_title('Chord Entropy by Layer')
    ax.axhline(y=ENTROPY_GHOST, color='r', linestyle='--', alpha=0.5, label='Ghost threshold')
    ax.axhline(y=ENTROPY_PURE_TONE, color='g', linestyle='--', alpha=0.5, label='Pure tone threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Chord Diversity Curve
    ax = axes[0, 1]
    ax.plot(layers, results['summary']['diversity_curve'], 'g-o', markersize=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Unique Chords')
    ax.set_title('Chord Diversity by Layer')
    ax.grid(True, alpha=0.3)

    # 3. Ghost Head Distribution
    ax = axes[0, 2]
    ax.bar(layers, results['summary']['ghost_curve'], color='red', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Ghost Head Count')
    ax.set_title('Ghost Heads by Layer')
    ax.grid(True, alpha=0.3)

    # 4. Cross-Layer Persistence
    ax = axes[1, 0]
    if results['cross_layer']['persistence_scores']:
        ax.plot(range(1, n_layers), results['cross_layer']['persistence_scores'],
                'm-o', markersize=3)
    ax.set_xlabel('Layer Transition')
    ax.set_ylabel('Persistence')
    ax.set_title('Cross-Layer Chord Persistence')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 5. Head Classification Stacked Bar
    ax = axes[1, 1]
    tones = [lr['n_pure_tone'] for lr in results['per_layer']]
    chords = [lr['n_chord'] for lr in results['per_layer']]
    ghosts = [lr['n_ghost'] for lr in results['per_layer']]
    hi_ps = [lr['n_high_prime'] for lr in results['per_layer']]
    ax.bar(layers, tones, label='Pure Tone', color='#2196F3')
    ax.bar(layers, chords, bottom=tones, label='Chord', color='#4CAF50')
    ax.bar(layers, hi_ps, bottom=[t+c for t,c in zip(tones, chords)],
           label='High-Prime', color='#FF9800')
    ax.bar(layers, ghosts, bottom=[t+c+h for t,c,h in zip(tones, chords, hi_ps)],
           label='Ghost', color='#F44336')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Head Count')
    ax.set_title('Head Classification by Layer')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Tridiagonal Score
    ax = axes[1, 2]
    tridiag = [lr['tridiagonal_score'] for lr in results['per_layer']]
    ax.plot(layers, tridiag, 'c-o', markersize=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Tridiagonal Score')
    ax.set_title('Sliding Window Structure')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Confirmed threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to {output_path}")
    else:
        plt.savefig('chord_diagnostic.png', dpi=150, bbox_inches='tight')
        print(f"  Plot saved to chord_diagnostic.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Shannon-Prime Phase 11 — Prime Chord Diagnostic')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to K vectors (.npz or .pt)')
    parser.add_argument('--output', '-o', type=str,
                        help='Path to save JSON results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate matplotlib plots')
    parser.add_argument('--plot-output', type=str, default=None,
                        help='Path for plot image (default: chord_diagnostic.png)')
    parser.add_argument('--skeleton-k', type=int, default=96,
                        help='Skeleton size (default: 96)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic RoPE-structured K vectors')
    parser.add_argument('--n-layers', type=int, default=32,
                        help='Number of layers (synthetic mode)')
    parser.add_argument('--n-heads', type=int, default=32,
                        help='Number of heads (synthetic mode)')
    parser.add_argument('--head-dim', type=int, default=128,
                        help='Head dimension (synthetic mode)')
    parser.add_argument('--n-positions', type=int, default=256,
                        help='Number of positions (synthetic mode)')
    parser.add_argument('--sqfree', action='store_true',
                        help='Pad power-of-2 head dims to sqfree dimensions '
                             '(128→154=2·7·11) for multi-prime analysis')
    parser.add_argument('--ascii', action='store_true', default=True,
                        help='Print ASCII visualizations (default: on)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    args = parser.parse_args()

    # Load or generate K vectors
    if args.synthetic:
        print("Generating synthetic RoPE-structured K vectors...")
        k_vectors = generate_synthetic_k_vectors(
            n_layers=args.n_layers, n_heads=args.n_heads,
            head_dim=args.head_dim, n_pos=args.n_positions)
        print(f"  Shape: {k_vectors.shape}")
    elif args.input:
        print(f"Loading K vectors from {args.input}...")
        k_vectors, nl, nh, hd = load_k_vectors(args.input)
        print(f"  Shape: {k_vectors.shape}")
    else:
        parser.print_help()
        print("\nERROR: Provide --input or --synthetic")
        sys.exit(1)

    # Run diagnostic
    results = run_diagnostic(k_vectors, skeleton_k=args.skeleton_k,
                             verbose=not args.quiet, use_sqfree=args.sqfree)

    # ASCII plots
    if args.ascii and not args.quiet:
        print_ascii_plots(results)

    # Save JSON
    if args.output:
        # Convert frozensets and other non-serializable types
        def make_serializable(obj):
            if isinstance(obj, frozenset):
                return sorted(obj)
            if isinstance(obj, set):
                return sorted(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                result = make_serializable(obj)
                if result is not obj:
                    return result
                return super().default(obj)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=CustomEncoder)
        print(f"\n  Results saved to {args.output}")

    # Generate plots
    if args.plot:
        plot_results(results, args.plot_output)


if __name__ == '__main__':
    main()
