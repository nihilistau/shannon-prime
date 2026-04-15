# Shannon-Prime VHT2

**Exact Spectral KV Cache Compression via the Multiplicative Lattice**

Copyright (C) 2026 Ray Daniels. Dual-licensed AGPLv3 / Commercial.

---

Shannon-Prime compresses the transformer KV cache by exploiting the spectral structure that RoPE imprints on key vectors. Ship configuration: **3.4–3.8× total KV compression at <1.25% perplexity cost.** The 5/5/4/3 bit allocation beats lossless fp16 by 0.04%.

```bash
make test-all   # 109 tests across 6 suites — all must pass
```

## How It Works

```
Write: raw KV → WHT → Möbius reorder → Band quantize → Store
Read:  Load → Band dequantize → Möbius unreorder → Inverse WHT → KV
```

K vectors (position, via RoPE) concentrate 80%+ energy in the first WHT bands. V vectors (content) have uniform energy. They get different compression: K gets 4-band Möbius-ordered quantization (5/5/4/3), V gets flat 3-bit.

## Project Structure

```
shannon-prime/
├── core/                      C reference: WHT, Möbius, banded quant, Vilenkin
│   ├── shannon_prime.h        Full API
│   └── shannon_prime.c        Implementation (31/31 tests)
├── backends/
│   ├── cuda/                  NVIDIA GPU kernels
│   ├── vulkan/                Cross-platform GPU (+ GLSL shaders)
│   ├── torch/                 Pure PyTorch (28/28 tests)
│   └── adreno/                Qualcomm Snapdragon: NEON, Hexagon, big.LITTLE
├── tools/
│   ├── shannon_prime_llama.*   llama.cpp integration (7/7 tests)
│   └── shannon_prime_comfyui.py ComfyUI + Wan 2.1/2.2 (25/25 tests)
├── tests/                     109 tests across 6 suites
└── docs/                      Full documentation
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/Shannon-Prime.md](docs/Shannon-Prime.md) | Theory, pipeline, license rationale, key results |
| [docs/INTEGRATION-LLAMA.md](docs/INTEGRATION-LLAMA.md) | llama.cpp: hook points, API, env vars, GQA support |
| [docs/INTEGRATION-COMFYUI.md](docs/INTEGRATION-COMFYUI.md) | ComfyUI: Wan 2.1/2.2 MoE, expert switching, linear wrapper |
| [docs/BACKEND-CUDA.md](docs/BACKEND-CUDA.md) | CUDA kernels, memory layout, building |
| [docs/BACKEND-VULKAN.md](docs/BACKEND-VULKAN.md) | Vulkan compute shaders, standalone vs shared device |
| [docs/BACKEND-ADRENO.md](docs/BACKEND-ADRENO.md) | Snapdragon: NEON tiers, Hexagon HVX, big.LITTLE, fp16 |
| [docs/BACKEND-TORCH.md](docs/BACKEND-TORCH.md) | PyTorch API, Vilenkin research path |
| [docs/TESTING.md](docs/TESTING.md) | How to run, what to look for, interpreting failures |

## Quick Start

### llama.cpp

```bash
export SHANNON_PRIME_ENABLED=1
export SHANNON_PRIME_K_BITS=5,5,4,3
export SHANNON_PRIME_V_BITS=3
./llama-server -m model.gguf -c 32768
```

### ComfyUI (Wan 2.2 MoE)

```python
from shannon_prime_comfyui import WanVHT2Wrapper

wrapper = WanVHT2Wrapper(head_dim=128, model_type='wan22_moe', task_type='t2v')
for step, sigma in enumerate(sigmas):
    wrapper.set_expert_from_sigma(sigma)
    for block_idx in range(40):
        k, v = wrapper.get_or_compute(f"block_{block_idx}", compute_fn)
```

## Key Results

| Config | K Corr | Compression | PPL Impact |
|--------|--------|-------------|------------|
| 5/5/4/3 + Möbius (ship) | 0.992+ | 3.4× | −0.04% (better) |
| 4/4/4/3 | 0.990 | 3.8× | +0.39% |
| 3/3/3/3 (floor) | 0.976 | 4.6× | +3.90% |

## Test Summary

| Suite | Tests | What It Validates |
|-------|-------|-------------------|
| Core math | 31 | WHT, Möbius, banded quant, Vilenkin, full pipeline |
| Adreno/ARM | 14 | NEON tiers, fp16, absmax, affinity, counters |
| Vulkan | 4 | CPU fallback, batch ops, memory reporting |
| llama.cpp | 7 | Multi-layer, multi-head, batch, memory |
| PyTorch | 28 | Same invariants as C core, in PyTorch |
| ComfyUI | 25 | Wan 2.1/2.2 MoE, expert switching, 50×40 simulation |
| **Total** | **109** | |

## License

**AGPLv3** for open-source, academic, and non-proprietary use. Everyone can use it and benefit. Derivative works must share alike.

**Commercial license** available for proprietary integration. The commercial aspect is secondary — the primary goal is that the work belongs to the commons and is protected from closure.

Contact: Ray Daniels — raydaniels@gmail.com
