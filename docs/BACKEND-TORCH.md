# Shannon-Prime: PyTorch Backend

## Overview

The PyTorch backend is the reference implementation and the integration path for Python-based inference engines. It runs on any device PyTorch supports — CPU, CUDA, MPS — with no compiled extensions required. It's used by the ComfyUI integration and is the easiest backend to test against.

## Components

All in `backends/torch/shannon_prime_torch.py`:

| Class / Function | Purpose |
|-----------------|---------|
| `vht2(x)` | **The** transform — staged Hartley on the last dim (any dim factoring into {2,3,5,7,11}), orthonormal, self-inverse (`vht2(vht2(x)) ≈ x` with no 1/N). At p=2 this is the WHT butterfly scaled by 1/√2. |
| `wht_inplace(x)` / `iwht(x)` | Deprecated aliases that now route to `vht2`. Old callers that did `wht_inplace(x); ...; wht_inplace(x); x.div_(n)` for a round-trip should drop the division. |
| `sqfree_pad_dim(head_dim)` | Next squarefree multiple of {2,3,5,7,11} ≥ head_dim (64 → 66, 128 → 154, 256 → 330). |
| `MobiusMask(n)` | Squarefree-first coefficient reordering. |
| `BandedQuantizer(n, bits)` | Per-band abs-max quantization to int8. |
| `ShadowCache(...)` | Complete ship-path KV cache with write/read pipeline. |
| `correlation(a, b)` | Pearson correlation for validation. |

The sqfree + spinor aggressive variant lives alongside in
`backends/torch/shannon_prime_sqfree.py` (`SqfreeShadowCache`), also using
`vht2` as the underlying transform.

## Quick Start

```python
from shannon_prime_torch import ShadowCache

# Initialize for a specific model
cache = ShadowCache(
    head_dim=128,
    n_layers=32,
    n_heads_kv=8,
    max_seq_len=4096,
    k_band_bits=[5, 5, 4, 3],  # Ship default
    v_band_bits=[3],            # Flat for V
    use_mobius=True,
)

# Write a K vector (after RoPE)
cache.write_k(layer=0, head=0, pos=0, k_vec=k_tensor)

# Write a V vector
cache.write_v(layer=0, head=0, pos=0, v_vec=v_tensor)

# Read back
k_reconstructed = cache.read_k(layer=0, head=0, pos=0)
v_reconstructed = cache.read_v(layer=0, head=0, pos=0)

# Check quality
from shannon_prime_torch import correlation
print(f"K correlation: {correlation(k_tensor, k_reconstructed):.4f}")

# Memory estimation
mem = cache.memory_bytes(seq_len=32768)
print(f"Compressed: {mem['total_bytes']/1024/1024:.0f} MB")
print(f"Baseline:   {mem['baseline_bytes']/1024/1024:.0f} MB")
print(f"Ratio:      {mem['ratio']:.1f}×")
```

## Using Individual Components

```python
import torch
from shannon_prime_torch import wht_inplace, MobiusMask, BandedQuantizer

# WHT on a batch of vectors
vectors = torch.randn(16, 128)  # 16 vectors of dim 128
wht_inplace(vectors)            # In-place, operates on last dim

# Möbius reorder
mask = MobiusMask(128)
reordered = mask.reorder(vectors)
restored = mask.unreorder(reordered)

# Banded quantization
bq = BandedQuantizer(128, [5, 5, 4, 3])
scales, quants = bq.quantize(vectors)
reconstructed = bq.dequantize(scales, quants)

print(f"Bytes per vector: {bq.compressed_bytes_per_vec()}")  # 76
print(f"Compression: {128*2 / bq.compressed_bytes_per_vec():.1f}×")
```

## Vilenkin Basis (Research)

```python
from shannon_prime_torch import VilenkinBasis

# 4-prime basis: N = 2×3×5×7 = 210
vb = VilenkinBasis(n_primes=4)

# Forward/inverse (self-inverse: V·V = I)
x = torch.randn(64)
coeffs = vb.forward(x.unsqueeze(0), head_dim=64)
recon = vb.inverse(coeffs, head_dim=64).squeeze(0)

print(f"Round-trip error: {(x - recon).abs().max():.2e}")  # ~1e-6
```

## Device Support

The PyTorch backend works on any device. Tensors are processed wherever they live:

```python
# CPU
cache = ShadowCache(head_dim=128, n_layers=1, n_heads_kv=1, device='cpu')

# CUDA (if available)
k = torch.randn(128, device='cuda')
cache.write_k(0, 0, 0, k)
k_recon = cache.read_k(0, 0, 0)  # Returns on same device

# MPS (Apple Silicon)
k = torch.randn(128, device='mps')
```

Note: the Möbius permutation index tensors are moved to the input tensor's device on first use.

## Testing

```bash
cd shannon-prime
python3 tests/test_torch.py  # 28 tests
```

Validates: WHT roundtrip (hd=32–256), Möbius function values, Möbius roundtrip, all 5 banded quant configs, K/V spectral asymmetry, Vilenkin roundtrip (2/3/4-prime), full pipeline, Möbius quality improvement, compression ratios, and memory estimation.
