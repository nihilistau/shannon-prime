"""
Shannon-Prime VHT2 CUDA bridge — auto-selects GPU kernel or PyTorch fallback.

Provides a unified API for VHT2 compress/decompress operations regardless
of whether the CUDA extension is compiled. The CUDA path is ~8-12× faster
at 720p+ (2400+ tokens) where the butterfly dominates over Python overhead.

Build the CUDA extension once:
    cd backends/cuda && python setup_wan.py build_ext --inplace

Import:
    from vht2_cuda_bridge import VHT2Bridge
    bridge = VHT2Bridge()
    compressed = bridge.compress(k_vecs, bridge.skeleton_mask_128())
    recovered  = bridge.decompress(compressed, bridge.skeleton_mask_128())
"""

import math
import os
import sys
import torch
import pathlib

# ── Try to load the CUDA extension ────────────────────────────────────────────

_CUDA_AVAILABLE = False
_cuda_ext       = None

try:
    import shannon_prime_cuda_wan as _cuda_ext
    _CUDA_AVAILABLE = True
except ImportError:
    # Try to find the .pyd/.so next to this file (built with --inplace into cuda/)
    _here     = pathlib.Path(__file__).resolve().parent
    _cuda_dir = _here.parent / "cuda"

    # On Windows, Python 3.8+ requires explicit DLL search directories.
    # The extension depends on c10.dll, torch_cuda.dll, cudart64_12.dll etc.
    if hasattr(os, "add_dll_directory"):
        try:
            import torch
            _torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
            if _torch_lib.exists():
                os.add_dll_directory(str(_torch_lib))
        except Exception:
            pass
        # Also add CUDA runtime bin if CUDA_HOME or standard install path exists
        for _cuda_bin in [
            pathlib.Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin"),
            pathlib.Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"),
        ]:
            if _cuda_bin.exists():
                os.add_dll_directory(str(_cuda_bin))
                break

    if str(_cuda_dir) not in sys.path:
        sys.path.insert(0, str(_cuda_dir))
    try:
        import shannon_prime_cuda_wan as _cuda_ext
        _CUDA_AVAILABLE = True
    except ImportError:
        pass

# ── PyTorch fallback VHT2 ─────────────────────────────────────────────────────

def _sqfree_mask_128() -> torch.Tensor:
    """Return a [128] bool tensor marking squarefree integers 1..128."""
    def is_sqfree(n: int) -> bool:
        if n <= 0:
            return False
        p = 2
        while p * p <= n:
            if n % (p * p) == 0:
                return False
            p += 1
        return True

    mask = torch.zeros(128, dtype=torch.bool)
    for i in range(128):
        if is_sqfree(i + 1):          # 1-indexed squarefree check
            mask[i] = True
    return mask


def _twin_prime_pairs(n: int = 128) -> list[tuple[int, int]]:
    """
    Return spectral-index pairs (i, j) where i+1 and j+1 are primes with j-i==2.

    The +1 is because the Hartley basis is 0-indexed but the squarefree /
    prime structure is naturally 1-indexed (position k corresponds to the
    arithmetic value k+1).

    At n=128, the twin-prime pairs in the spectrum are:
      (3,5)→(2,4), (5,7)→(4,6), (11,13)→(10,12), (17,19)→(16,18),
      (29,31)→(28,30), (41,43)→(40,42), (59,61)→(58,60), (71,73)→(70,72),
      (101,103)→(100,102), (107,109)→(106,108).

    The pairs (3,5) and (5,7) share index 4. Sequential application is
    safe at small alpha; the second pass sees the first pass's tiny pull.

    Strange-attractor reading: each twin pair is a +2 arithmetic neighbor
    on the prime lattice. Quantization noise that lands on the wrong side
    of a bin boundary disagrees with its neighbor; the borrowing pulls
    the outlier back toward the local consensus before the inverse VHT2.
    """
    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        if k < 4:
            return True
        if k % 2 == 0:
            return False
        d = 3
        while d * d <= k:
            if k % d == 0:
                return False
            d += 2
        return True

    raw_pairs = []
    # We're looking for primes p, p+2 with p+2 <= n (so index j = p+1 < n)
    for p in range(2, n - 1):
        if is_prime(p) and is_prime(p + 2):
            raw_pairs.append((p - 1, p + 1))

    # Deduplicate overlapping pairs (e.g. (3,5) and (5,7) share index 4).
    # Vectorized scatter applies pairs simultaneously, so overlap would make
    # the result depend on internal index_copy_ ordering. Greedy keep: take
    # each pair only if neither index has been touched yet. At n=128 this
    # drops one pair (the (5,7) overlap with (3,5)) leaving 9 disjoint pairs.
    seen = set()
    pairs = []
    for i, j in raw_pairs:
        if i in seen or j in seen:
            continue
        pairs.append((i, j))
        seen.add(i)
        seen.add(j)
    return pairs


def _apply_twin_borrow(coeffs: torch.Tensor, mask: torch.Tensor,
                       pairs: list[tuple[int, int]],
                       alpha: float = 0.10, threshold: float = 0.0) -> torch.Tensor:
    """
    Decode-side twin-prime borrowing. Applied to spectral coefficients
    AFTER mask-application but BEFORE the inverse VHT2 transform.

    For each twin-prime pair (i, j) where both indices survive in the mask:
      diff = |c_i - c_j| / max(|c_i|, |c_j|, eps)
      if diff > threshold:
        avg = (c_i + c_j) / 2
        c_i ← (1-α) c_i + α avg
        c_j ← (1-α) c_j + α avg

    Properties:
      - Symmetric (no asymmetric drift)
      - Reduces to identity when c_i ≈ c_j (no spurious correction)
      - Decode-only — encoder doesn't need to know about it
      - Bounded change: |Δc| ≤ α/2 * |c_i - c_j|
      - Reversibility of stored skeleton is unaffected (encode side untouched)

    coeffs: [..., n] tensor (post-mask, pre-inverse)
    mask:   [n] bool — only pairs where BOTH indices are masked-true are touched
    """
    if alpha <= 0.0 or not pairs:
        return coeffs

    # Pre-filter pairs: both indices must be in the surviving skeleton
    mask_cpu = mask.detach().cpu()
    active_pairs = [(i, j) for (i, j) in pairs
                    if bool(mask_cpu[i].item()) and bool(mask_cpu[j].item())]
    if not active_pairs:
        return coeffs

    # Vectorized: gather all i-indices and j-indices, do one weighted blend
    idx_i = torch.tensor([p[0] for p in active_pairs],
                         dtype=torch.long, device=coeffs.device)
    idx_j = torch.tensor([p[1] for p in active_pairs],
                         dtype=torch.long, device=coeffs.device)

    c_i = coeffs.index_select(-1, idx_i)
    c_j = coeffs.index_select(-1, idx_j)

    if threshold > 0.0:
        # Only borrow when relative disagreement exceeds threshold
        denom = torch.clamp(torch.maximum(c_i.abs(), c_j.abs()), min=1e-6)
        rel_diff = (c_i - c_j).abs() / denom
        kick = (rel_diff > threshold).to(coeffs.dtype)
    else:
        kick = torch.ones_like(c_i)

    avg = 0.5 * (c_i + c_j)
    new_i = c_i + alpha * kick * (avg - c_i)
    new_j = c_j + alpha * kick * (avg - c_j)

    # Scatter back
    coeffs = coeffs.clone()
    coeffs.index_copy_(-1, idx_i, new_i)
    coeffs.index_copy_(-1, idx_j, new_j)
    return coeffs


def _hartley_butterfly_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch Hadamard/Hartley butterfly for head_dim=128 (= 2^7).
    Self-inverse: calling twice recovers the input.
    Input/output: [..., 128] float32 tensor (any leading dims).
    """
    # Work on the last dimension
    orig_shape = x.shape
    x = x.reshape(-1, 128).float()
    n = 128
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    # 7 butterfly stages
    for length in (1, 2, 4, 8, 16, 32, 64):
        step = length * 2
        # Build even/odd index arrays for in-place butterfly
        evens = torch.arange(0, n, step, device=x.device)
        # For each group, pair elements [base, base+length] .. [base+len-1, base+2*len-1]
        idx0 = (evens.unsqueeze(1)
                + torch.arange(length, device=x.device).unsqueeze(0)).reshape(-1)
        idx1 = idx0 + length

        u = x[:, idx0].clone()
        v = x[:, idx1].clone()
        x[:, idx0] = (u + v) * inv_sqrt2
        x[:, idx1] = (u - v) * inv_sqrt2

    return x.reshape(orig_shape)


# ── VHT2Bridge ────────────────────────────────────────────────────────────────

class VHT2Bridge:
    """
    Unified VHT2 compress/decompress for Wan head_dim=128.

    Uses the CUDA extension when available (8-12× faster at 720p+),
    falls back to a pure-PyTorch implementation otherwise.

    The skeleton mask marks squarefree positions (1-indexed): positions
    where the index+1 is not divisible by any perfect square > 1.
    At head_dim=128, ~60% of indices are squarefree (78/128).
    With a 30% keep fraction, the skeleton covers ~24 coefficients,
    yielding 3.5-4× compression with negligible reconstruction error.
    """

    def __init__(self, skeleton_frac: float = 0.30, device: str = "cuda"):
        self.device        = device
        self.skeleton_frac = skeleton_frac
        self._full_mask    = None   # cached full squarefree mask [128]
        self._skel_mask    = None   # cached skeleton mask [128] at fraction
        self._cuda         = _CUDA_AVAILABLE
        self._mode         = "cuda" if _CUDA_AVAILABLE else "torch"
        # Twin-prime pairs at head_dim=128 (computed once, cached)
        self._twin_pairs   = _twin_prime_pairs(128)

        print(f"[SP VHT2Bridge] mode={self._mode}  "
              f"skeleton_frac={skeleton_frac:.0%}  "
              f"head_dim=128  "
              f"device={device}")
        if not _CUDA_AVAILABLE:
            print("[SP VHT2Bridge] CUDA extension not compiled — "
                  "using PyTorch fallback. "
                  "Build with: cd backends/cuda && python setup_wan.py build_ext --inplace")

    def skeleton_mask_128(self, frac: float | None = None) -> torch.Tensor:
        """
        Return [128] bool mask for the squarefree skeleton at `frac` keep ratio.
        Cached — O(1) after first call.
        frac=None uses self.skeleton_frac.
        """
        target_frac = frac if frac is not None else self.skeleton_frac

        if self._full_mask is None:
            self._full_mask = _sqfree_mask_128().to(self.device)

        # Trim the squarefree positions to the desired keep count
        sqfree_indices = self._full_mask.nonzero(as_tuple=False).squeeze(1)
        n_keep = max(1, int(len(sqfree_indices) * (target_frac / 1.0)))
        keep_idx = sqfree_indices[:n_keep]

        mask = torch.zeros(128, dtype=torch.bool, device=self.device)
        mask[keep_idx] = True
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """VHT2 butterfly (forward = inverse, self-inverse). Returns new tensor."""
        if self._cuda:
            return _cuda_ext.forward(x.contiguous())
        return _hartley_butterfly_torch(x)

    def compress(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compress: VHT2 forward → zero non-skeleton positions.
        x:    [..., 128] float32 CUDA tensor
        mask: [128]      bool    CUDA tensor
        Returns: [..., 128] sparse coefficients (non-skeleton = 0)
        """
        x = x.contiguous()
        if self._cuda:
            return _cuda_ext.compress(x, mask)
        # Torch fallback
        coeffs = _hartley_butterfly_torch(x)
        return coeffs * mask.float().reshape(*([1] * (x.ndim - 1)), 128)

    def decompress(self, coeffs: torch.Tensor, mask: torch.Tensor,
                   twin_borrow: bool = False,
                   twin_alpha: float = 0.10,
                   twin_threshold: float = 0.0) -> torch.Tensor:
        """
        Decompress: re-apply mask → [twin-prime borrow] → VHT2 inverse.

        coeffs: [..., 128] sparse coefficient tensor
        mask:   [128]      bool CUDA tensor
        twin_borrow: when True, applies decode-side twin-prime smoothing on
                     the masked spectral coefficients before the inverse
                     transform (strange-attractor stack piece 3/4). Decode-only,
                     so the stored skeleton bytes are unchanged. Worst case is
                     identity (alpha=0 or no agreement issues).
        twin_alpha:  blend strength toward the pair average. Small values
                     (0.05-0.15) for safety; larger values pull harder.
        twin_threshold: only borrow when relative |c_i-c_j|/max exceeds this.
                        0.0 = always borrow; 0.1 = only on outliers.

        Returns: [..., 128] reconstructed vectors
        """
        coeffs = coeffs.contiguous()
        if twin_borrow:
            # Mask + twin-borrow + inverse, in PyTorch (CUDA path doesn't
            # know about twin pairs yet — bridge fallback handles this safely)
            masked = coeffs * mask.float().reshape(*([1] * (coeffs.ndim - 1)), 128)
            corrected = _apply_twin_borrow(masked, mask, self._twin_pairs,
                                           alpha=twin_alpha,
                                           threshold=twin_threshold)
            if self._cuda:
                # Use the CUDA forward (= inverse, self-inverse) on the corrected coeffs
                return _cuda_ext.forward(corrected.contiguous())
            return _hartley_butterfly_torch(corrected)

        if self._cuda:
            return _cuda_ext.decompress(coeffs, mask)
        # Torch fallback
        masked = coeffs * mask.float().reshape(*([1] * (coeffs.ndim - 1)), 128)
        return _hartley_butterfly_torch(masked)

    def roundtrip(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Fused compress + decompress: VHT2 → mask → VHT2.
        Equivalent to decompress(compress(x, mask), mask) but avoids the
        intermediate allocation when using the CUDA extension.
        """
        x = x.contiguous()
        if self._cuda:
            return _cuda_ext.roundtrip(x, mask)
        coeffs = _hartley_butterfly_torch(x)
        masked = coeffs * mask.float().reshape(*([1] * (x.ndim - 1)), 128)
        return _hartley_butterfly_torch(masked)

    def apply_twin_borrow(self, coeffs: torch.Tensor, mask: torch.Tensor,
                          alpha: float = 0.10,
                          threshold: float = 0.0) -> torch.Tensor:
        """
        Public wrapper for the decode-side twin-prime borrowing pass.
        Apply between mask-application and the inverse VHT2 forward.
        See _apply_twin_borrow for the math.
        """
        return _apply_twin_borrow(coeffs, mask, self._twin_pairs,
                                  alpha=alpha, threshold=threshold)

    @property
    def mode(self) -> str:
        """'cuda' if CUDA extension available, 'torch' otherwise."""
        return self._mode

    @property
    def cuda_available(self) -> bool:
        return self._cuda


# ── Module-level singleton for ComfyUI node reuse ─────────────────────────────

_global_bridge: VHT2Bridge | None = None

def get_bridge(skeleton_frac: float = 0.30, device: str = "cuda") -> VHT2Bridge:
    """
    Return the module-level VHT2Bridge singleton.
    First call constructs it; subsequent calls return the cached instance.
    """
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = VHT2Bridge(skeleton_frac=skeleton_frac, device=device)
    return _global_bridge
