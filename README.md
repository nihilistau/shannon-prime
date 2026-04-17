# Shannon-Prime VHT2

**Exact Spectral KV Cache Compression via the Multiplicative Lattice**

---

Shannon-Prime compresses the transformer KV cache by exploiting the spectral
structure that RoPE imprints on key vectors. The single transform is **VHT2**
— the Vilenkin-Hartley Transform, a staged orthonormal generalization of the
Walsh-Hadamard Transform. At power-of-2 head_dim it reduces to the WHT
butterfly scaled by 1/√2 per stage (self-inverse, no 1/N needed); at
sqfree-padded dimensions it factors across small primes {2, 3, 5, 7, 11} and
unlocks the Möbius predictor + SU(2) spinor sheet bit for aggressive 3.3×
compression on Q8+ backbones.

Ship configuration: **3.4–3.8× total KV compression at <1.25% perplexity cost.**
The 5/5/4/3 bit allocation beats lossless fp16 by 0.04%. Aggressive config
(sqfree + spinor): **3.3× at MOBIUS-default quality** on Qwen3-8B Q8 hd=128.

```bash
make test-all   # 187/188 tests across 8 suites (one synthetic-K flake, see CLAUDE.md)
```

## How It Works

### Ship path (default)
```
Write: raw KV → VHT2 forward → Möbius reorder (K only, self-attn)
       → Band quantize (5/5/4/3 K, flat 3 V) → Store
Read:  Load → Band dequantize → Möbius unreorder → VHT2 forward (self-inverse)
       → NaN guard → KV
```
K (post-RoPE) concentrates 80%+ energy in the first VHT2 bands; V (content)
spreads uniformly. K gets 4-band Möbius-ordered quantization, V gets flat 3-bit.

### Sqfree+spinor aggressive path (opt-in, Q8+ backbones)
```
Write: raw KV → sqfree_pad → VHT2 → Knight skeleton extract
       → Band quantize → Möbius CSR predict residual
       → Quantize residual (3-bit) → Spinor sheet bit → Store
Read:  Load → Band dequant → Möbius predict → Spinor correct
       → Dequant residual → Scatter → VHT2 → sqfree_unpad → KV
```
Pads hd → next sqfree multiple (64 → 66, 128 → 154, 256 → 330) so the
Möbius predictor gets r = 0.40–0.58 (vs ≈ 0 on pure power-of-2). Gated on
`SHANNON_PRIME_SQFREE=1`; auto-enabled with `SHANNON_PRIME_SPINOR=1`.

## Project Structure

```
shannon-prime/
├── core/                         C reference: VHT2, Möbius, banded quant,
│   ├── shannon_prime.h           sqfree pad, Knight mask CSR, residual,
│   ├── shannon_prime.c           spinor, shadow cache. 31 tests.
│   └── shannon_prime_sqfree.c    Sqfree + spinor C implementation
├── backends/
│   ├── cuda/                     NVIDIA GPU kernels
│   ├── vulkan/                   Cross-platform GPU + GLSL shaders
│   ├── torch/                    Pure PyTorch (31 + 69 sqfree tests)
│   └── adreno/                   Qualcomm: NEON tiers, Hexagon HVX, big.LITTLE
├── tools/
│   ├── shannon_prime_llama.*            llama.cpp ship-path (7 tests)
│   ├── shannon_prime_llama_sqfree.c     llama.cpp sqfree hook
│   ├── shannon_prime_comfyui.py         ComfyUI + Wan 2.1/2.2 (25 tests)
│   ├── shannon_prime_comfyui_sqfree.py  ComfyUI sqfree variant
│   ├── sp_scaling_law.py                K-corr → PPL design rule
│   ├── sp_inject_freqs.py               GGUF frequency injection
│   ├── sp_compress_model.py             Weight spectral analysis
│   └── sp_benchmark.py                  Compression benchmark
├── tests/                        8 test suites, 187/188 passing
└── docs/                         Full documentation
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/Shannon-Prime.md](docs/Shannon-Prime.md) | Theory, pipeline, license rationale, key results |
| [docs/TOOLS.md](docs/TOOLS.md) | Command-line tools: benchmark, freq injection, weight compression |
| [docs/INTEGRATION-LLAMA.md](docs/INTEGRATION-LLAMA.md) | llama.cpp: hook points, API, env vars, GQA support |
| [docs/INTEGRATION-COMFYUI.md](docs/INTEGRATION-COMFYUI.md) | ComfyUI: Wan 2.1/2.2 MoE, expert switching, linear wrapper |
| [docs/BACKEND-CUDA.md](docs/BACKEND-CUDA.md) | CUDA kernels, memory layout, building |
| [docs/BACKEND-VULKAN.md](docs/BACKEND-VULKAN.md) | Vulkan compute shaders, standalone vs shared device |
| [docs/BACKEND-ADRENO.md](docs/BACKEND-ADRENO.md) | Snapdragon: NEON tiers, Hexagon HVX, big.LITTLE, fp16 |
| [docs/BACKEND-TORCH.md](docs/BACKEND-TORCH.md) | PyTorch API, VHT2 reference, sqfree+spinor path |
| [docs/TESTING.md](docs/TESTING.md) | How to run, what to look for, interpreting failures |

## Quick Start

### llama.cpp — ship path
```bash
export SHANNON_PRIME_ENABLED=1
export SHANNON_PRIME_K_BITS=5,5,4,3
export SHANNON_PRIME_V_BITS=3
./llama-server -m model.gguf -c 32768
```

### llama.cpp — sqfree+spinor aggressive path (Q8+ only)
```bash
export SHANNON_PRIME_ENABLED=1
export SHANNON_PRIME_SQFREE=1
export SHANNON_PRIME_SPINOR=1
export SHANNON_PRIME_RESIDUAL_BITS=3
export SHANNON_PRIME_K_BITS=3,3,3,3,3
./llama-server -m Qwen3-8B-Q8_0.gguf -c 32768
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

### Ship path
| Config | K Corr | Compression | PPL Impact |
|--------|--------|-------------|------------|
| 5/5/4/3 + Möbius (ship) | 0.992+ | 3.4× | −0.04% (better) |
| 4/4/4/3 | 0.990 | 3.8× | +0.39% |
| 3/3/3/3 (floor) | 0.976 | 4.6× | +3.90% |

### Sqfree+spinor (Qwen3-8B Q8 hd=128)
| Config | PPL | K Corr | Compression |
|--------|-----|--------|-------------|
| MOBIUS default 5/5/5/5/5 | 7.31 | 0.999 | 2.6× |
| K+μ+3bit+spinor 5/4/4/4/5 | 7.30 | 0.988 | 2.8× |
| K+μ+3bit+spinor 3/3/3/3/3 | 7.32 | 0.972 | 3.3× |

### Scaling law
```
log(PPL / PPL_base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
```
Use as a pre-bench filter — skip configs with predicted ΔPPL > 5%.

## Test Summary

| Suite | Tests | What It Validates |
|-------|-------|-------------------|
| Core math | 31 | VHT2 round-trip, Möbius, banded quant, sqfree, full pipeline |
| Adreno/ARM | 14 | NEON tiers, fp16, absmax, affinity, counters |
| Vulkan | 4 | CPU fallback, batch ops, memory reporting |
| CUDA | 7 | Multi-layer, multi-head, batch K+V roundtrip |
| llama.cpp | 7 | Integration layer, env-var driven config |
| PyTorch | 31 | VHT2 self-inverse at p-of-2 and sqfree, full pipeline |
| ComfyUI | 25 | Wan 2.1/2.2 MoE, expert switching, 50×40 simulation |
| Sqfree+spinor | 69 | Knight CSR, residual quant, spinor no-regress, scaling law |
| **Total** | **187/188** | *(1 known synthetic-K flake — see CLAUDE.md)* |

## Environment Variables

### Ship path
| Variable | Default | Description |
|----------|---------|-------------|
| `SHANNON_PRIME_ENABLED` | 0 | Enable the VHT2 shadow cache |
| `SHANNON_PRIME_K_BITS` | 5,5,4,3 | K band bit allocation |
| `SHANNON_PRIME_V_BITS` | 3 | V flat bit allocation |
| `SHANNON_PRIME_MOBIUS` | 1 | Möbius squarefree-first reorder (K only) |
| `SHANNON_PRIME_VERBOSE` | 0 | Print config + init |

### Sqfree+spinor (opt-in)
| Variable | Default | Description |
|----------|---------|-------------|
| `SHANNON_PRIME_SQFREE` | 0 | Enable sqfree prime-Hartley basis (pads hd) |
| `SHANNON_PRIME_SPINOR` | 0 | Enable SU(2) sheet bit (auto-enables SQFREE) |
| `SHANNON_PRIME_RESIDUAL_BITS` | 3 | Residual depth (1–4; 3 is the Pareto point) |
| `SHANNON_PRIME_SK_FRAC` | 0.75 | Skeleton fraction of pad_dim |

## License

**AGPLv3** for open-source, academic, and non-proprietary use. Everyone can use
it and benefit. Derivative works must share alike.

**Commercial license** available for proprietary integration. The commercial
aspect is secondary — the primary goal is that the work belongs to the commons
and is protected from closure.
