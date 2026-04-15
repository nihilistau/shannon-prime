# Shannon-Prime: CUDA Backend

## Overview

The CUDA backend runs the entire VHT2 pipeline on NVIDIA GPUs: WHT butterfly, Möbius permutation, banded quantization, and shadow cache storage all operate in GPU memory. The compressed KV cache never leaves the GPU — no PCIe transfers needed.

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

**`kernel_wht_inplace`:** One thread block per vector, shared memory butterfly. For hd=128: 7 passes with `__syncthreads()` between each. Bandwidth-bound, not compute-bound — appropriate for single-token decode where latency matters more than throughput.

**`kernel_mobius_reorder` / `kernel_mobius_unreorder`:** Gather/scatter permutation. One thread per coefficient, one block per vector. The Möbius permutation tables are uploaded once at init and remain constant.

**`kernel_band_quantize_simple`:** One thread per vector processes all bands sequentially. Suitable for decode (1 vector at a time). For prefill, launch with more threads per vector.

**`kernel_band_dequantize_simple`:** Inverse of quantize. Unpacks bit-packed integers, scales by fp16 scale factor.

**`kernel_nan_guard`:** Clamps values and replaces NaN with 0. Defense-in-depth — the ship configuration (5/5/4/3) doesn't produce NaN, but aggressive configurations can accumulate errors in softmax over long context.

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

// Init cache on GPU
sp_cuda_cache_t cc;
sp_cuda_cache_init(&cc, &cfg, max_seq_len, cuda_stream);

// Single-vector operations (decode)
sp_cuda_write_k(&cc, layer, head, pos, d_k_vec);
sp_cuda_read_k(&cc, layer, head, pos, d_k_out);

// Batch operations (prefill)
sp_cuda_write_k_batch(&cc, layer, head, start_pos, n_pos, d_k_matrix);
sp_cuda_read_k_batch(&cc, layer, head, start_pos, n_pos, d_k_out_matrix);

// All pointers are device pointers — no host↔device transfers
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
