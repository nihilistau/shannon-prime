#!/usr/bin/env python3
"""
Shannon-Prime: Hierarchical Vilenkin Skeleton — can a small skeleton recreate the cache?

The Vilenkin basis on sqfree-padded dims has Kronecker structure:
    hd=128 → pad=154 = 2·7·11 → Z/2Z × Z/7Z × Z/11Z

This script tests whether the low-prime sub-projections (Z/2Z×Z/7Z = 14 coeffs)
can linearly predict the high-prime refinement (the other 140 coeffs), using
calibration data from real KV vectors.

If the linear predictor is good, we can compress to:
    - Store: 14 skeleton coefficients (banded quantized, ~5 bits each = 70 bits)
    - Predict: 140 coefficients via 14×140 matrix (calibrated, ~31KB per layer×head)
    - Residual: tiny correction on the predicted 140

That's ~9% skeleton instead of 50%, with the predictor matrix amortized across
all sequence positions.

Usage:
    python sp_hierarchical_skeleton.py model.gguf [--n-tokens 256]

Copyright (C) 2026 Ray Daniels. All Rights Reserved.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# ── Vilenkin transform (matching C implementation) ──────────────────────

SMALL_PRIMES = [2, 3, 5, 7, 11]

def sqfree_pad_dim(head_dim: int) -> int:
    """Find next sqfree-factorable dim >= head_dim."""
    for n in range(max(head_dim, 2), head_dim * 4):
        d = n
        for p in SMALL_PRIMES:
            while d % p == 0:
                d //= p
        if d == 1:
            # Check squarefree: each prime divides at most once
            d2 = n
            ok = True
            for p in SMALL_PRIMES:
                c = 0
                while d2 % p == 0:
                    d2 //= p
                    c += 1
                    if c > 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok and d2 == 1:
                return n
    return head_dim

def factorize(n: int) -> list:
    """Factorize n into small primes."""
    factors = []
    for p in SMALL_PRIMES:
        while n % p == 0:
            factors.append(p)
            n //= p
    return factors if n == 1 else []

def vilenkin_forward_f32(data: np.ndarray) -> np.ndarray:
    """In-place staged Hartley transform, 1/√p per stage. Self-inverse."""
    n = len(data)
    out = data.astype(np.float64).copy()
    factors = factorize(n)
    if not factors:
        raise ValueError(f"Cannot factorize {n}")

    stride = 1
    for p in factors:
        inv_sqrt_p = 1.0 / np.sqrt(p)
        block_size = stride * p
        n_blocks = n // block_size

        for blk in range(n_blocks):
            for s in range(stride):
                base = blk * block_size + s
                # Gather p elements
                vals = np.array([out[base + k * stride] for k in range(p)])
                # Apply p×p Hartley: H[k] = Σ_n cas(2πkn/p) · x[n]
                tmp = np.zeros(p)
                for k in range(p):
                    for ni in range(p):
                        angle = 2.0 * np.pi * k * ni / p
                        cas = np.cos(angle) + np.sin(angle)
                        tmp[k] += cas * vals[ni]
                    tmp[k] *= inv_sqrt_p
                # Scatter back
                for k in range(p):
                    out[base + k * stride] = tmp[k]
        stride *= p

    return out.astype(np.float32)


# ── Kronecker sub-projection ───────────────────────────────────────────

def kronecker_indices(pad_dim: int, keep_primes: list) -> np.ndarray:
    """Get coefficient indices belonging to the sub-projection over keep_primes.

    In the Kronecker product Z/p1 × Z/p2 × ... × Z/pk, each index i
    decomposes as i = i1 + p1*(i2 + p2*(...)).
    The sub-projection keeps only indices where the coordinates for
    non-kept primes are zero.

    Returns sorted array of indices.
    """
    factors = factorize(pad_dim)
    unique_primes = []
    seen = set()
    for f in factors:
        if f not in seen:
            unique_primes.append(f)
            seen.add(f)

    # Build multi-index ranges
    # Each index i in [0, pad_dim) maps to (i_p1, i_p2, ..., i_pk)
    # where i = i_p1 + p1 * (i_p2 + p2 * (...))
    indices = []
    for idx in range(pad_dim):
        remaining = idx
        coords = {}
        stride_acc = 1
        for p in factors:
            coord = (remaining // stride_acc) % p
            if p not in coords:
                coords[p] = coord
            # For repeated primes this gets tricky, but sqfree dims have
            # each prime exactly once
            stride_acc *= p

        # Actually, for the staged transform, the decomposition is simpler.
        # Let's just use the mixed-radix representation.
        remaining = idx
        mr = []
        for p in factors:
            mr.append(remaining % p)
            remaining //= p

        # Check: for primes NOT in keep_primes, the coordinate must be 0
        keep = True
        fi = 0
        remaining2 = idx
        for p in factors:
            digit = remaining2 % p
            remaining2 //= p
            if p not in keep_primes and digit != 0:
                keep = False
                break

        if keep:
            indices.append(idx)

    return np.array(sorted(set(indices)))


def measure_linear_predictability(vectors: np.ndarray, pad_dim: int,
                                  skeleton_indices: np.ndarray,
                                  target_indices: np.ndarray) -> dict:
    """Measure how well skeleton coefficients linearly predict target coefficients.

    Uses ridge regression on calibration data, reports R² and reconstruction error.
    """
    n_vecs = len(vectors)
    if n_vecs < 2:
        return {'r2': 0.0, 'rmse': float('inf'), 'n_skeleton': len(skeleton_indices)}

    # Split into train/test
    n_train = max(1, int(n_vecs * 0.8))
    X_train = vectors[:n_train][:, skeleton_indices]
    Y_train = vectors[:n_train][:, target_indices]
    X_test = vectors[n_train:][:, skeleton_indices]
    Y_test = vectors[n_train:][:, target_indices]

    if len(X_test) == 0:
        X_test = X_train
        Y_test = Y_train

    # Ridge regression: W = (X^T X + λI)^{-1} X^T Y
    n_skel = len(skeleton_indices)
    lam = 1e-4 * n_train  # Scale regularization with data size
    XtX = X_train.T @ X_train + lam * np.eye(n_skel)
    XtY = X_train.T @ Y_train

    try:
        W = np.linalg.solve(XtX, XtY)  # n_skel × n_target
    except np.linalg.LinAlgError:
        return {'r2': 0.0, 'rmse': float('inf'), 'n_skeleton': n_skel,
                'n_target': len(target_indices), 'predictor_bytes': 0}

    # Predict on test set
    Y_pred = X_test @ W
    residual = Y_test - Y_pred

    # R² per target coefficient, then mean
    ss_res = np.sum(residual ** 2, axis=0)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2, axis=0)
    r2_per_coeff = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)
    mean_r2 = float(np.mean(r2_per_coeff))

    # RMSE
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    # Correlation of full reconstructed vector vs original
    all_indices = np.concatenate([skeleton_indices, target_indices])
    sort_order = np.argsort(all_indices)

    full_pred = np.zeros_like(vectors[n_train:])
    full_pred[:, skeleton_indices] = X_test
    full_pred[:, target_indices] = Y_pred

    corrs = []
    for i in range(len(X_test)):
        orig = vectors[n_train + i]
        pred = full_pred[i]
        c = np.corrcoef(orig, pred)[0, 1]
        corrs.append(c if not np.isnan(c) else 0.0)
    mean_corr = float(np.mean(corrs))

    # Predictor storage cost: W matrix in fp16
    predictor_bytes = n_skel * len(target_indices) * 2

    return {
        'r2': round(mean_r2, 6),
        'rmse': round(rmse, 6),
        'correlation': round(mean_corr, 6),
        'n_skeleton': n_skel,
        'n_target': len(target_indices),
        'skeleton_pct': round(100.0 * n_skel / pad_dim, 1),
        'predictor_bytes': predictor_bytes,
        'predictor_kb': round(predictor_bytes / 1024, 1),
        'r2_p25': round(float(np.percentile(r2_per_coeff, 25)), 4),
        'r2_p50': round(float(np.percentile(r2_per_coeff, 50)), 4),
        'r2_p75': round(float(np.percentile(r2_per_coeff, 75)), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test hierarchical Vilenkin skeleton predictability')
    parser.add_argument('model', help='GGUF model path')
    parser.add_argument('--n-tokens', type=int, default=256,
                        help='Number of tokens to process for calibration')
    parser.add_argument('--layers', type=str, default='0,1,2,-1',
                        help='Comma-separated layer indices to analyze')
    parser.add_argument('--prompt', type=str,
                        default='The meaning of life is',
                        help='Prompt text for KV extraction')
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python required. pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    llm = Llama(model_path=args.model, n_ctx=max(512, args.n_tokens * 2),
                n_gpu_layers=0, verbose=False)

    # Get model dims
    n_embd = llm.n_embd()
    n_head = llm.metadata.get('llama.attention.head_count', None)
    n_head_kv = llm.metadata.get('llama.attention.head_count_kv', None)
    n_layer = int(llm.metadata.get('llama.block_count', 32))

    if n_head is None or n_head_kv is None:
        # Try alternate keys
        for k, v in llm.metadata.items():
            if 'head_count_kv' in k:
                n_head_kv = int(v)
            elif 'head_count' in k and 'kv' not in k:
                n_head = int(v)

    n_head = int(n_head)
    n_head_kv = int(n_head_kv)
    head_dim = n_embd // n_head
    pad_dim = sqfree_pad_dim(head_dim)
    factors = factorize(pad_dim)
    unique_primes = list(dict.fromkeys(factors))

    print(f"\nModel: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}")
    print(f"head_dim={head_dim}, pad_dim={pad_dim} = {'·'.join(str(p) for p in unique_primes)}")
    print(f"Factorization: {' × '.join(f'Z/{p}Z' for p in unique_primes)}")

    # Build sub-projection hierarchies
    hierarchies = []
    for i in range(1, len(unique_primes) + 1):
        primes = unique_primes[:i]
        idx = kronecker_indices(pad_dim, primes)
        sub_dim = 1
        for p in primes:
            sub_dim *= p
        label = ' × '.join(f'Z/{p}Z' for p in primes)
        hierarchies.append({
            'label': label,
            'primes': primes,
            'indices': idx,
            'expected_size': sub_dim,
        })
        print(f"  Level {i}: {label} → {len(idx)} coeffs ({100*len(idx)/pad_dim:.1f}%)")

    # Also test variance-ranked at various sizes for comparison
    variance_sizes = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    # Tokenize and extract KV vectors
    tokens = llm.tokenize(args.prompt.encode('utf-8'))
    if len(tokens) < args.n_tokens:
        # Generate more tokens
        output = llm(args.prompt, max_tokens=args.n_tokens - len(tokens),
                     temperature=0.0)
        full_text = args.prompt + output['choices'][0]['text']
        tokens = llm.tokenize(full_text.encode('utf-8'))[:args.n_tokens]

    print(f"\nProcessing {len(tokens)} tokens...")

    # We need to extract KV cache. Use eval + internal state.
    llm.reset()
    llm.eval(tokens)

    # Parse layer indices
    layer_list = []
    for l in args.layers.split(','):
        li = int(l.strip())
        if li < 0:
            li = n_layer + li
        if 0 <= li < n_layer:
            layer_list.append(li)

    print(f"Analyzing layers: {layer_list}")
    print()

    # Extract K vectors from internal state
    # llama-cpp-python doesn't expose raw KV easily, so we'll generate
    # synthetic calibration data using the model's embedding + projection
    # patterns. For a proper test, use sp-engine's KV capture.
    #
    # ALTERNATIVE: use the SHANNON_PRIME_DUMP_K env var to dump real KV
    # vectors, then load them here.

    dump_path = os.environ.get('SHANNON_PRIME_DUMP_K', None)
    if dump_path and os.path.exists(dump_path):
        print(f"Loading KV dump from {dump_path}")
        raw = np.fromfile(dump_path, dtype=np.float32)
        n_vecs = len(raw) // head_dim
        raw_vecs = raw[:n_vecs * head_dim].reshape(n_vecs, head_dim)
        print(f"Loaded {n_vecs} vectors of dim {head_dim}")
    else:
        # Generate synthetic KV-like vectors from embeddings
        # Not ideal but demonstrates the pipeline
        print("No SHANNON_PRIME_DUMP_K set — generating synthetic calibration vectors")
        print("(For real results, run sp-engine with SHANNON_PRIME_DUMP_K=kv_dump.bin)")
        print()
        np.random.seed(42)
        n_vecs = len(tokens) * n_head_kv
        # Simulate: structured signal + noise (mimics attention head patterns)
        base_signal = np.random.randn(n_vecs, head_dim).astype(np.float32) * 0.5
        # Add low-frequency structure
        for i in range(n_vecs):
            for f in range(1, 5):
                amp = 1.0 / f
                phase = np.random.uniform(0, 2 * np.pi)
                base_signal[i] += amp * np.cos(
                    2 * np.pi * f * np.arange(head_dim) / head_dim + phase)
        raw_vecs = base_signal

    # Transform to Vilenkin domain
    print(f"Transforming {len(raw_vecs)} vectors to Vilenkin domain (pad_dim={pad_dim})...")
    vilenkin_vecs = np.zeros((len(raw_vecs), pad_dim), dtype=np.float32)
    for i in range(len(raw_vecs)):
        padded = np.zeros(pad_dim, dtype=np.float32)
        padded[:head_dim] = raw_vecs[i]
        if pad_dim > head_dim:
            padded[head_dim:] = np.mean(raw_vecs[i])
        vilenkin_vecs[i] = vilenkin_forward_f32(padded)

    # Compute per-coefficient variance for baseline comparisons
    var = np.var(vilenkin_vecs, axis=0)
    var_ranked = np.argsort(var)[::-1]

    print(f"\n{'='*75}")
    print(f"HIERARCHICAL VILENKIN SKELETON ANALYSIS")
    print(f"{'='*75}")
    print(f"pad_dim={pad_dim}, n_vectors={len(vilenkin_vecs)}, head_dim={head_dim}")
    print()

    # ── Test 1: Kronecker sub-projection hierarchies ────────────────
    print(f"{'─'*75}")
    print(f"TEST 1: Kronecker sub-projection as skeleton → predict remainder")
    print(f"{'─'*75}")
    print()

    for h in hierarchies:
        skel_idx = h['indices']
        target_idx = np.array([i for i in range(pad_dim) if i not in set(skel_idx)])
        if len(target_idx) == 0:
            print(f"  {h['label']}: FULL (no targets to predict)")
            continue

        result = measure_linear_predictability(
            vilenkin_vecs, pad_dim, skel_idx, target_idx)

        print(f"  {h['label']} ({len(skel_idx)} coeffs = {result['skeleton_pct']}%)")
        print(f"    Linear R²:    {result['r2']:.4f}  "
              f"(p25={result['r2_p25']}, p50={result['r2_p50']}, p75={result['r2_p75']})")
        print(f"    Correlation:  {result['correlation']:.4f}")
        print(f"    RMSE:         {result['rmse']:.4f}")
        print(f"    Predictor:    {result['predictor_kb']:.1f} KB "
              f"({result['n_skeleton']}×{result['n_target']} fp16 matrix)")
        print()

    # ── Test 2: Variance-ranked at various sizes ────────────────────
    print(f"{'─'*75}")
    print(f"TEST 2: Variance-ranked skeleton at various sizes → predict remainder")
    print(f"{'─'*75}")
    print()

    for frac in variance_sizes:
        k = max(1, int(frac * pad_dim))
        skel_idx = var_ranked[:k]
        target_idx = var_ranked[k:]
        if len(target_idx) == 0:
            continue

        result = measure_linear_predictability(
            vilenkin_vecs, pad_dim, skel_idx, target_idx)

        print(f"  Top {frac*100:.0f}% by variance ({k} coeffs)")
        print(f"    Linear R²:    {result['r2']:.4f}  "
              f"(p25={result['r2_p25']}, p50={result['r2_p50']}, p75={result['r2_p75']})")
        print(f"    Correlation:  {result['correlation']:.4f}")
        print(f"    RMSE:         {result['rmse']:.4f}")
        print(f"    Predictor:    {result['predictor_kb']:.1f} KB")
        print()

    # ── Test 3: Hybrid — Kronecker skeleton + variance-ranked extras ─
    print(f"{'─'*75}")
    print(f"TEST 3: Kronecker skeleton + variance-ranked extras")
    print(f"{'─'*75}")
    print()

    if len(hierarchies) >= 2:
        base_h = hierarchies[-2]  # Second-to-last (e.g., Z/2Z×Z/7Z for 154)
        base_idx_set = set(base_h['indices'].tolist())

        for extra_pct in [0.05, 0.10, 0.15, 0.20]:
            n_extra = max(1, int(extra_pct * pad_dim))
            # Add top-variance indices not already in the Kronecker sub-projection
            extras = [i for i in var_ranked if i not in base_idx_set][:n_extra]
            combined_idx = np.array(sorted(set(base_h['indices'].tolist() + extras)))
            target_idx = np.array([i for i in range(pad_dim) if i not in set(combined_idx)])

            if len(target_idx) == 0:
                continue

            result = measure_linear_predictability(
                vilenkin_vecs, pad_dim, combined_idx, target_idx)

            total_pct = 100.0 * len(combined_idx) / pad_dim
            print(f"  {base_h['label']} + top {extra_pct*100:.0f}% variance "
                  f"({len(combined_idx)} coeffs = {total_pct:.1f}%)")
            print(f"    Linear R²:    {result['r2']:.4f}  "
                  f"(p25={result['r2_p25']}, p50={result['r2_p50']}, p75={result['r2_p75']})")
            print(f"    Correlation:  {result['correlation']:.4f}")
            print(f"    RMSE:         {result['rmse']:.4f}")
            print(f"    Predictor:    {result['predictor_kb']:.1f} KB")
            print()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"{'='*75}")
    print(f"SUMMARY")
    print(f"{'='*75}")
    print()
    print("If linear R² > 0.90 for a small skeleton, that skeleton can recreate")
    print("the cache: store the skeleton + predictor matrix, reconstruct on read.")
    print()
    print("The predictor matrix is per-(layer, head) but amortized across ALL")
    print("sequence positions. For a 32-layer, 8-head model with 14→140 predictor:")
    print(f"  Total predictor storage: 32 × 8 × 14 × 140 × 2 = "
          f"{32 * 8 * 14 * 140 * 2 / 1024:.0f} KB")
    print(f"  Per-position K storage:  14 coeffs × 5 bits = 70 bits = 8.75 bytes")
    print(f"  Current L/2 K storage:   77 coeffs × 4.25 bits avg ≈ 41 bytes")
    print(f"  Compression improvement: ~4.7× more compact per position")
    print()
    print("Run with SHANNON_PRIME_DUMP_K=kv_dump.bin for real KV data.")


if __name__ == '__main__':
    main()
