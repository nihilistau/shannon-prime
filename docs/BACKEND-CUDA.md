# Shannon-Prime: CUDA Backend

## Overview

The CUDA backend runs the entire VHT2 pipeline on NVIDIA GPUs: VHT2 at p=2 butterfly (self-inverse, 1/√2 per stage), Möbius permutation, banded quantization, and shadow cache storage all operate in GPU memory. The compressed KV cache never leaves the GPU — no PCIe transfers needed.

## Architecture

```
GPU Memory Layout
═══════════════════════════════

  ┌──────────────────────────────────┐
  │  Model Weights (read-only)       │
  ├──────────────────────────────────┤
  │  Activation Buffers              │
  ├──────────────────────────────────┤
  │  Shannon-Prime Shadow Cache      │
  │  ┌────────────────────────────┐  │
  │  │ K cache: [layers×heads×    │  │
  │  │          seq×k_bytes]      │  │  ← 3.4× smaller than fp16
  │  │ V cache: [layers×heads×    │  │
  │  │          seq×v_bytes]      │  │
  │  ├────────────────────────────┤  │
  │  │ Möbius tables (constant)   │  │
  │  │ Scratch buffers            │  │
  │  └────────────────────────────┘  │
  └──────────────────────────────────┘
```

## Kernels

**`kernel_vht2_p2`:** One thread block per vector, shared-memory VHT2 at p=2 butterfly. For hd=128: 7 passes with `__syncthreads()` between each, each stage scaled by 1/√2 so the kernel is self-inverse (no separate inverse kernel, no 1/N on reconstruction). The host entry point is `sp_cuda_vht2_forward`; there is no `sp_cuda_iwht_inplace` anymore. Bandwidth-bound, not compute-bound — appropriate for single-token decode where latency matters more than throughput.

**`kernel_mobius_reorder` / `kernel_mobius_unreorder`:** Gather/scatter permutation. One thread per coefficient, one block per vector. The Möbius permutation tables are uploaded once at init and remain constant.

**`kernel_band_quantize_simple`:** One thread per vector processes all bands sequentially. Suitable for decode (1 vector at a time). For prefill, launch with more threads per vector. Honours `sp_band_config_t.head_dim` (not `band_size * n_bands`) so the last band absorbs any head_dim-% n_bands remainder — matches the CPU `sp_band_span` helper for the v1.03 10-band configs on hd=128 / pad_dim=154.

**`kernel_band_dequantize_simple`:** Inverse of quantize. Unpacks bit-packed integers, scales by fp16 scale factor. If a band's fp16 scale round-trips to a non-finite value (the real origin of the cascading NaN previously handled by the blanket output guard), the kernel zeros that band — the only surviving guard, and it lives at the root cause. Same `head_dim`-anchored iteration as the quantize kernel.

**Sqfree batched read kernels (`shannon_prime_sqfree.cu`).** `sp_cuda_sqfree_read_k_batch` / `_v_batch` process n_pos positions per (layer, head) in a single-per-step kernel dispatch series: `scatter_batch`, `reconstruct_residual_batch`, `dequantize_residual_batch`, `unpack_levels_batch`, `gather_mag_batch`. About 9 launches instead of 9 × n_pos. One caveat: the sqfree cache slot layout is `[total_bytes skeleton][4 mag][res_bytes]` (stride `bytes_per_pos > total_bytes`), while `sp_cuda_band_dequantize` assumes contiguous `total_bytes` strides across n_vecs. A `cudaMemcpy2DAsync` repacks the skeleton bytes into a contiguous buffer (reusing `d_pad_scratch`) before the band dequantise — without it, K_corr collapses from 0.94 to 0.04 on synthetic data.

## Building

Requires NVIDIA CUDA Toolkit (nvcc):

```bash
nvcc -O2 -arch=sm_80 \
    -o build/test_cuda \
    backends/cuda/shannon_prime_cuda.cu \
    core/shannon_prime.c \
    -lm
```

Supported architectures: sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere), sm_86 (GA10x), sm_89 (Ada), sm_90 (Hopper).

## API

```c
#include "backends/cuda/shannon_prime_cuda.h"

// Ship-path cache (4-band K, flat-3 V, power-of-2 hd):
sp_cuda_cache_t cc;
sp_cuda_cache_init(&cc, &cfg, max_seq_len, cuda_stream);

// Single-vector (decode hot path)
sp_cuda_write_k(&cc, layer, head, pos, d_k_vec);
sp_cuda_read_k(&cc, layer, head, pos, d_k_out);

// Batched (prefill; also the decode read path, with n_pos = kv_len)
sp_cuda_write_k_batch(&cc, layer, head, start_pos, n_pos, d_k_matrix);
sp_cuda_read_k_batch(&cc, layer, head, start_pos, n_pos, d_k_out_matrix);

// Sqfree cache (Knight skeleton + residual + Möbius predictor):
sp_cuda_sqfree_cache_t sc;
sp_cuda_sqfree_cache_init(&sc, &cfg, max_seq_len, /*residual_bits=*/3,
                          /*use_spinor=*/0, cuda_stream);

// Per-position write (decode)
sp_cuda_sqfree_write_k(&sc, layer, head, pos, d_k_vec);

// Batched read — processes all n_pos positions per (layer, head)
// in ~9 kernel launches instead of 9 × n_pos.
sp_cuda_sqfree_read_k_batch(&sc, layer, head, start_pos, n_pos, d_k_out);

// All pointers are device pointers — no host↔device transfers.
```

## Memory Overhead

The shadow cache, Möbius tables, and scratch buffers add minimal overhead compared to the fp16 baseline they replace:

| Component | Size (hd=128) |
|-----------|---------------|
| Möbius order table | 512 bytes (constant) |
| Möbius inverse table | 512 bytes (constant) |
| Scratch buffer | 512 bytes (per-stream) |
| K compressed per vec | 76 bytes (vs 256 fp16) |
| V compressed per vec | 50 bytes (vs 256 fp16) |
