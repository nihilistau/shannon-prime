# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
Pure PyTorch implementation of Shannon-Prime VHT2 KV cache compression.

This backend runs on any device PyTorch supports (CPU, CUDA, MPS).
It's the reference for ComfyUI integration and the easiest to test against.

Usage:
    from shannon_prime_torch import ShadowCache

    cache = ShadowCache(head_dim=128, n_layers=32, n_heads_kv=8)
    cache.write_k(layer=0, head=0, pos=0, k_vec=tensor)
    k_reconstructed = cache.read_k(layer=0, head=0, pos=0)
"""

import torch
import math
from typing import Optional, Tuple, List


# =============================================================================
# WHT — Walsh-Hadamard Transform
# =============================================================================

def wht_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    In-place Walsh-Hadamard Transform (iterative butterfly).
    
    x: (..., n) where n is power of 2.
    Returns x transformed in-place.
    
    Self-inverse: wht(wht(x)) = n * x
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"

    h = 1
    while h < n:
        # Reshape to (..., n//(2*h), 2, h) for butterfly
        x_view = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x_view[..., 0, :].clone()
        b = x_view[..., 1, :].clone()
        x_view[..., 0, :] = a + b
        x_view[..., 1, :] = a - b
        h <<= 1

    return x


def iwht(x: torch.Tensor) -> torch.Tensor:
    """Inverse WHT = forward WHT / n."""
    n = x.shape[-1]
    wht_inplace(x)
    x.div_(n)
    return x


# =============================================================================
# Möbius Mask
# =============================================================================

class MobiusMask:
    """
    Squarefree-first coefficient ordering for WHT coefficients.
    
    The Möbius function μ(n) is non-zero iff n is squarefree.
    Prioritizing squarefree indices during banded quantization
    improves quality by +0.14 PPL at identical coefficient budget.
    
    Cross-platform invariant: K correlation 0.997 on both hd=128 and hd=64.
    """

    def __init__(self, n: int):
        self.n = n
        self.mu = self._compute_mobius(n)
        
        # Build permutation: squarefree first, then non-squarefree
        squarefree = [i for i in range(n) if self.mu[i] != 0]
        non_squarefree = [i for i in range(n) if self.mu[i] == 0]
        self.order = squarefree + non_squarefree
        self.n_squarefree = len(squarefree)
        
        # Build inverse permutation
        self.inv_order = [0] * n
        for i, idx in enumerate(self.order):
            self.inv_order[idx] = i

        # Pre-compute as tensors (will be moved to device on first use)
        self._order_tensor = torch.tensor(self.order, dtype=torch.long)
        self._inv_order_tensor = torch.tensor(self.inv_order, dtype=torch.long)

    @staticmethod
    def _compute_mobius(n: int) -> List[int]:
        mu = [0] * n
        if n > 1:
            mu[1] = 1
        
        for i in range(2, n):
            # Factor i and check for squared primes
            val = i
            n_factors = 0
            has_square = False
            p = 2
            while p * p <= val:
                if val % p == 0:
                    n_factors += 1
                    val //= p
                    if val % p == 0:
                        has_square = True
                        break
                p += 1
            if not has_square and val > 1:
                n_factors += 1
            
            if has_square:
                mu[i] = 0
            else:
                mu[i] = 1 if n_factors % 2 == 0 else -1
        
        return mu

    def reorder(self, x: torch.Tensor) -> torch.Tensor:
        """Squarefree-first reordering. x: (..., n)."""
        idx = self._order_tensor.to(x.device)
        return x[..., idx]

    def unreorder(self, x: torch.Tensor) -> torch.Tensor:
        """Restore original index order. x: (..., n)."""
        idx = self._inv_order_tensor.to(x.device)
        return x[..., idx]


# =============================================================================
# Banded Quantization
# =============================================================================

class BandedQuantizer:
    """
    VHT2 banded quantization of WHT coefficients.
    
    Splits n coefficients into k equal bands. Each band gets:
      - 1 fp16 scale (max(abs(band)) / (2^(bits-1) - 1))
      - Packed signed integers at the band's bit depth
    
    Key results from paper:
      - 5/5/4/3 BEATS lossless fp16 by 0.04% (spectral regularization)
      - 4/4/4/4 is off the Pareto frontier
      - 3-bit floor: 2-bit on any band is catastrophic
      - Flat beats banded for V vectors (no exceptions)
    """

    def __init__(self, n: int, band_bits: List[int]):
        self.n = n
        self.n_bands = len(band_bits)
        self.band_bits = band_bits
        self.band_size = n // self.n_bands
        assert n % self.n_bands == 0, \
            f"head_dim {n} must be divisible by n_bands {self.n_bands}"

    def quantize(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Quantize WHT coefficients into banded format.
        
        x: (..., n) float tensor
        Returns: (scales, quant_vals) per band
          scales[b]: (...,) fp16 scale
          quant_vals[b]: (..., band_size) int8/int16 quantized values
        """
        scales = []
        quant_vals = []

        for b in range(self.n_bands):
            start = b * self.band_size
            end   = start + self.band_size
            band  = x[..., start:end]
            bits  = self.band_bits[b]
            max_val = (1 << (bits - 1)) - 1

            # Scale per vector
            amax = band.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
            scale = amax / max_val

            # Quantize
            q = (band / scale).round().clamp(-max_val, max_val).to(torch.int8)

            scales.append(scale.squeeze(-1).half())
            quant_vals.append(q)

        return scales, quant_vals

    def dequantize(self, scales: List[torch.Tensor],
                   quant_vals: List[torch.Tensor]) -> torch.Tensor:
        """
        Dequantize banded format back to float WHT coefficients.
        
        Returns: (..., n) float tensor
        """
        bands = []
        for b in range(self.n_bands):
            scale = scales[b].float().unsqueeze(-1)
            q     = quant_vals[b].float()
            bands.append(q * scale)

        return torch.cat(bands, dim=-1)

    def compressed_bytes_per_vec(self) -> int:
        """Bytes per compressed vector."""
        total = 0
        for bits in self.band_bits:
            total += 2  # fp16 scale
            total += (self.band_size * bits + 7) // 8  # packed data
        return total


# =============================================================================
# Vilenkin-Hartley Transform (research path)
# =============================================================================

class VilenkinBasis:
    """
    Multi-prime Vilenkin-Hartley basis.
    
    Generalizes WHT from Z/2Z to Z/p1Z × Z/p2Z × ... × Z/pkZ.
    Hartley kernel: cas(x) = cos(x) + sin(x)
    Self-inverse: V·V = I (when normalized by 1/sqrt(p) per factor)
    Round-trip error: 0.0000
    
    Progressive prime expansion monotonically increases correlation:
      Walsh (Z/2Z):              0.9490
      Z/2Z × Z/3Z:              0.9493
      Z/2Z × Z/3Z × Z/5Z:      0.9500
      Z/2Z × Z/3Z × Z/5Z × Z/7Z: 0.9513
    """

    PRIMES = [2, 3, 5, 7, 11, 13]

    def __init__(self, n_primes: int, device: str = 'cpu'):
        assert 1 <= n_primes <= len(self.PRIMES)
        self.n_primes = n_primes
        self.primes = self.PRIMES[:n_primes]
        self.n = 1
        for p in self.primes:
            self.n *= p

        # Build basis via Kronecker product
        basis = torch.ones(1, 1, device=device)
        for p in self.primes:
            h = torch.zeros(p, p, device=device)
            for i in range(p):
                for j in range(p):
                    angle = 2 * math.pi * i * j / p
                    h[i, j] = (math.cos(angle) + math.sin(angle)) / math.sqrt(p)
            basis = torch.kron(basis, h)

        self.basis = basis  # (n, n) orthonormal matrix

    def forward(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Project head_dim vector into Vilenkin space (zero-pads if needed)."""
        if head_dim < self.n:
            pad = torch.zeros(*x.shape[:-1], self.n - head_dim,
                            device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        return x @ self.basis.T.to(x.device)

    def inverse(self, coeffs: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Reconstruct from Vilenkin coefficients (truncates to head_dim)."""
        # V is orthonormal (V·V = I), so inverse = V
        full = coeffs @ self.basis.T.to(coeffs.device)
        return full[..., :head_dim]


# =============================================================================
# Shadow Cache — the main integration point
# =============================================================================

class ShadowCache:
    """
    VHT2 compressed KV cache for transformer models.
    
    Write path: raw KV → WHT → Möbius reorder → band quantize → store
    Read path:  load → band dequantize → Möbius unreorder → iWHT → KV
    
    Ship-safe configuration:
      K: 4 bands at 5/5/4/3 bits (with Möbius reorder)
      V: 1 band at 3 bits (flat, no reorder)
    
    Achieves 3.4–3.8× total compression at <1.25% PPL cost.
    Spectral regularization: 5/5/4/3 BEATS fp16 by 0.04%.
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_layers: int = 32,
        n_heads_kv: int = 8,
        max_seq_len: int = 4096,
        k_band_bits: List[int] = [5, 5, 4, 3],
        v_band_bits: List[int] = [3],
        use_mobius: bool = True,
        device: str = 'cpu',
    ):
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv
        self.max_seq_len = max_seq_len
        self.device = device

        self.k_quantizer = BandedQuantizer(head_dim, k_band_bits)
        self.v_quantizer = BandedQuantizer(head_dim, v_band_bits)
        self.mobius = MobiusMask(head_dim) if use_mobius else None

        # Storage: lists of (scales, quant_vals) per position
        # Indexed as cache[layer][head][pos]
        n_slots = n_layers * n_heads_kv
        self.k_scales = [[None] * max_seq_len for _ in range(n_slots)]
        self.k_quants = [[None] * max_seq_len for _ in range(n_slots)]
        self.v_scales = [[None] * max_seq_len for _ in range(n_slots)]
        self.v_quants = [[None] * max_seq_len for _ in range(n_slots)]

    def _slot(self, layer: int, head: int) -> int:
        return layer * self.n_heads_kv + head

    def write_k(self, layer: int, head: int, pos: int,
                k_vec: torch.Tensor):
        """
        Compress and store a K vector.
        k_vec: (head_dim,) float tensor (already RoPE'd)
        """
        x = k_vec.clone()

        # WHT forward
        wht_inplace(x.unsqueeze(0))
        x = x.squeeze(0) if x.dim() > 1 else x

        # Möbius reorder (squarefree first)
        if self.mobius is not None:
            x = self.mobius.reorder(x)

        # Band quantize
        scales, quants = self.k_quantizer.quantize(x.unsqueeze(0))
        scales = [s.squeeze(0) for s in scales]
        quants = [q.squeeze(0) for q in quants]

        slot = self._slot(layer, head)
        self.k_scales[slot][pos] = scales
        self.k_quants[slot][pos] = quants

    def write_v(self, layer: int, head: int, pos: int,
                v_vec: torch.Tensor):
        """Compress and store a V vector."""
        x = v_vec.clone()
        wht_inplace(x.unsqueeze(0))
        x = x.squeeze(0) if x.dim() > 1 else x

        # No Möbius for V (uniform spectrum — no benefit)
        scales, quants = self.v_quantizer.quantize(x.unsqueeze(0))
        scales = [s.squeeze(0) for s in scales]
        quants = [q.squeeze(0) for q in quants]

        slot = self._slot(layer, head)
        self.v_scales[slot][pos] = scales
        self.v_quants[slot][pos] = quants

    def read_k(self, layer: int, head: int, pos: int) -> torch.Tensor:
        """Reconstruct a K vector from compressed storage."""
        slot = self._slot(layer, head)
        scales = [s.unsqueeze(0) for s in self.k_scales[slot][pos]]
        quants = [q.unsqueeze(0) for q in self.k_quants[slot][pos]]

        x = self.k_quantizer.dequantize(scales, quants).squeeze(0)

        if self.mobius is not None:
            x = self.mobius.unreorder(x)

        wht_inplace(x.unsqueeze(0))
        x = x.squeeze(0) if x.dim() > 1 else x
        x.div_(self.head_dim)

        # NaN guard
        x = torch.clamp(x, -65504.0, 65504.0)
        x = torch.nan_to_num(x, nan=0.0)

        return x

    def read_v(self, layer: int, head: int, pos: int) -> torch.Tensor:
        """Reconstruct a V vector from compressed storage."""
        slot = self._slot(layer, head)
        scales = [s.unsqueeze(0) for s in self.v_scales[slot][pos]]
        quants = [q.unsqueeze(0) for q in self.v_quants[slot][pos]]

        x = self.v_quantizer.dequantize(scales, quants).squeeze(0)

        wht_inplace(x.unsqueeze(0))
        x = x.squeeze(0) if x.dim() > 1 else x
        x.div_(self.head_dim)

        x = torch.clamp(x, -65504.0, 65504.0)
        x = torch.nan_to_num(x, nan=0.0)

        return x

    def compression_ratio(self) -> float:
        """Total K+V compression ratio vs fp16 baseline."""
        baseline = self.head_dim * 2  # fp16 bytes per element
        k_bytes = self.k_quantizer.compressed_bytes_per_vec()
        v_bytes = self.v_quantizer.compressed_bytes_per_vec()
        return (2 * baseline) / (k_bytes + v_bytes)

    def memory_bytes(self, seq_len: int) -> dict:
        """Estimate memory usage for given sequence length."""
        n_slots = self.n_layers * self.n_heads_kv
        k_bytes = n_slots * seq_len * self.k_quantizer.compressed_bytes_per_vec()
        v_bytes = n_slots * seq_len * self.v_quantizer.compressed_bytes_per_vec()
        baseline = n_slots * seq_len * self.head_dim * 2 * 2  # K+V fp16
        return {
            'k_bytes': k_bytes,
            'v_bytes': v_bytes,
            'total_bytes': k_bytes + v_bytes,
            'baseline_bytes': baseline,
            'ratio': baseline / (k_bytes + v_bytes),
        }


# =============================================================================
# Convenience: compute correlation
# =============================================================================

def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    num = (a_centered * b_centered).sum()
    den = (a_centered.norm() * b_centered.norm()).clamp(min=1e-12)
    return (num / den).item()
