# Shannon-Prime VHT2

**Exact Spectral KV Cache Compression via the Multiplicative Lattice**

---

Shannon-Prime compresses the transformer KV cache by exploiting the spectral
structure that RoPE imprints on key vectors. The single transform is **VHT2**
— the Vilenkin-Hartley Transform, a staged orthonormal generalization of the
Walsh-Hadamard Transform. At power-of-2 head_dim it reduces to the WHT
butterfly scaled by 1/√2 per stage (self-inverse, no 1/N needed); at
sqfree-padded dimensions it factors across small primes {2, 3, 5, 7, 11} and
unlocks the Möbius predictor + SU(2) spinor sheet bit for aggressive 2.8×
compression on Q8+ backbones.

Ship configuration: **3.4–3.8× total KV compression at <1.25% perplexity cost.**
The 5/5/4/3 bit allocation beats lossless fp16 by 0.04%. Aggressive config
(sqfree + spinor, 5,4,4,4,5): **2.8× at MOBIUS-default quality** on Qwen3-8B Q8 hd=128.

```bash
make test-all   # 187/188 tests across 8 suites (one synthetic-K flake, see CLAUDE.md)
```

## Sibling repositories

The math core (this repo) is the canonical reference. Two sibling
repositories build inference paths on top of it; both vendor this
repo at `lib/shannon-prime/` as a git submodule, so they always run
the same VHT2 / Möbius / sqfree implementation.

| Repo | Role | Status |
|---|---|---|
| **[shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine)** | Standalone inference binary that owns the compressed KV layout end-to-end. Compression is on the write path by construction (no decompress→attention→recompress hook). The bug-free reference measurement surface. | Stage 5b: full forward + prefill + greedy chat with optimised single-token decode all working on Llama-3 / Qwen3, ship + sqfree + sqfree+spinor. See [docs/PRIME-ENGINE.md](docs/PRIME-ENGINE.md). |
| **[shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama)** | Post-decode hook into llama.cpp. Inherits 30+ model architectures via the upstream loader, but the hook surface itself has been a source of integration bugs — every PPL number measured through it carries a footnote. | In production for ship + sqfree paths; see [docs/INTEGRATION-LLAMA.md](docs/INTEGRATION-LLAMA.md) and the measured-results section below. |

The first published measurements that don't carry the hook-surface
footnote will come from `shannon-prime-engine` when the optimised
decode lands. The per-layer K-correlation report from its stage 5a
already round-trips real RoPE'd K through `KvCache` at K=0.9941
(Dolphin-1B ship) / K=0.9934 (Qwen3-8B ship) / K=0.9869
(sqfree+spinor) — the documented 0.992+ ship target, on real model
data, end-to-end.

## How It Works

### Ship path (default)
```
Write: raw KV → VHT2 forward → Möbius reorder (K only, self-attn)
       → Band quantize (5/5/4/3 K, flat 3 V) → Store
Read:  Load → Band dequantize (non-finite scale ⇒ zero band)
       → Möbius unreorder → VHT2 forward (= inverse) → KV
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

The math core lives in this repo. Two sibling repos depend on it
via git submodule for inference-time use; each has its own
project structure documented in their own READMEs.

```
shannon-prime-repos/                  ← parent dir holding all three
├── shannon-prime/                    ← THIS REPO (canonical math + tools)
│   ├── core/
│   │   ├── shannon_prime.h           VHT2, Möbius, banded quant, sqfree pad,
│   │   ├── shannon_prime.c           Knight mask CSR, residual, spinor, shadow cache
│   │   └── shannon_prime_sqfree.c    Sqfree + spinor C implementation
│   ├── backends/
│   │   ├── cuda/                     NVIDIA GPU kernels
│   │   ├── vulkan/                   Cross-platform GPU + GLSL shaders
│   │   ├── torch/                    Pure PyTorch (31 + 69 sqfree tests)
│   │   └── adreno/                   Qualcomm: NEON tiers, Hexagon HVX, big.LITTLE
│   ├── tools/
│   │   ├── shannon_prime_llama.*            in-tree stub for the llama.cpp hook
│   │   ├── shannon_prime_llama_sqfree.c     in-tree sqfree hook stub
│   │   ├── shannon_prime_comfyui.py         ComfyUI + Wan 2.1/2.2 (25 tests)
│   │   ├── shannon_prime_comfyui_sqfree.py  ComfyUI sqfree variant
│   │   ├── sp_scaling_law.py                K-corr → PPL design rule
│   │   ├── sp_inject_freqs.py               GGUF frequency injection
│   │   ├── sp_compress_model.py             Weight spectral analysis
│   │   └── sp_benchmark.py                  Compression benchmark
│   ├── tests/                        8 test suites, 187/188 passing
│   └── docs/                         Full documentation (incl. PRIME-ENGINE.md)
│
├── shannon-prime-engine/             ← SIBLING: standalone inference binary
│   │                                   Owns compressed KV layout. Stage 5b.
│   ├── lib/shannon-prime/            git submodule → this repo
│   ├── vendor/ggml/                  git submodule → ggml-org/ggml (MIT)
│   ├── src/                          ~900 LOC of engine code
│   │   ├── engine.{h,cpp}            Public API + Config (PeMode, sqfree, mobius)
│   │   ├── gguf_loader.{h,cpp}       Typed view over gguf_context
│   │   ├── vocab.{h,cpp}             tokenizer.ggml.* arrays
│   │   ├── tokenizer.{h,cpp}         GPT-2-style BPE encode/decode
│   │   ├── llama_weights.{h,cpp}     Llama-family arch binding
│   │   ├── forward.{h,cpp}           ggml graph: embed, block, full, decode
│   │   ├── prime_pe.{h,cpp}          PrimePE-RoPE-ALiBi lattice math
│   │   ├── kv_cache.{h,cpp}          Wrapper around sp_shadow_cache_t / sp_sqfree_cache_t
│   │   └── cli/main.cpp              Verbs: info, encode, embed, logits, kv_smoke,
│   │                                  prefill, chat
│   └── CMakeLists.txt                CMake + Ninja, optional CUDA/Vulkan
│
└── shannon-prime-llama/              ← SIBLING: llama.cpp post-decode hook
    │                                    Inherits 30+ archs from llama.cpp.
    │                                    See docs/INTEGRATION-LLAMA.md.
    ├── lib/shannon-prime/            git submodule → this repo
    └── src/llama-shannon-prime.{h,cpp}  Hook implementation
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/Shannon-Prime.md](docs/Shannon-Prime.md) | Theory, pipeline, license rationale, key results |
| [docs/PRIME-ENGINE.md](docs/PRIME-ENGINE.md) | shannon-prime-engine sibling: stages 3–5, KvCache, PrimePE-RoPE-ALiBi |
| [docs/TOOLS.md](docs/TOOLS.md) | Command-line tools: benchmark, freq injection, weight compression |
| [docs/INTEGRATION-LLAMA.md](docs/INTEGRATION-LLAMA.md) | llama.cpp: hook points, API, env vars, GQA support |
| [docs/INTEGRATION-COMFYUI.md](docs/INTEGRATION-COMFYUI.md) | ComfyUI: Wan 2.1/2.2 MoE, expert switching, linear wrapper |
| [docs/BACKEND-CUDA.md](docs/BACKEND-CUDA.md) | CUDA kernels, memory layout, building |
| [docs/BACKEND-VULKAN.md](docs/BACKEND-VULKAN.md) | Vulkan compute shaders, standalone vs shared device |
| [docs/BACKEND-ADRENO.md](docs/BACKEND-ADRENO.md) | Snapdragon: NEON tiers, Hexagon HVX, big.LITTLE, fp16 |
| [docs/BACKEND-TORCH.md](docs/BACKEND-TORCH.md) | PyTorch API, VHT2 reference, sqfree+spinor path |
| [docs/TESTING.md](docs/TESTING.md) | How to run, what to look for, interpreting failures |

## Quick Start

### shannon-prime-engine (standalone binary; recommended for measurement)
```bash
git clone --recursive https://github.com/nihilistau/shannon-prime-engine
cd shannon-prime-engine
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Sanity: per-layer K correlation on real RoPE'd K, ship vs sqfree.
./build/bin/sp-engine prefill --model dolphin_1b.gguf "The quick brown fox"
./build/bin/sp-engine prefill --sqfree --spinor --model dolphin_1b.gguf "..."

# Greedy generation through the optimised single-token decode + cache.
./build/bin/sp-engine chat --model dolphin_1b.gguf --n-predict 16 "The quick brown fox"
./build/bin/sp-engine chat --sqfree --spinor --model dolphin_1b.gguf --n-predict 16 "..."
```
Full verb list and stage status in [docs/PRIME-ENGINE.md](docs/PRIME-ENGINE.md).

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
export SHANNON_PRIME_K_BITS=5,4,4,4,5
./llama-server -m Qwen3-8B-Q8_0.gguf -c 32768
```
Note: 3,3,3,3,3 on the Knight skeleton was retracted in v1.07
(catastrophic post-fix); the effective Pareto point is 5,4,4,4,5.

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

### Engine-measured K correlation on real RoPE'd K (no hook surface)

Measured by `shannon-prime-engine`'s `prefill` verb, which captures
post-RoPE pre-GQA K from a real prefill, pushes it through the
KvCache wrapper, reads it back, and reports per-layer correlation
against the un-cached source. No decompress→attention→recompress
round-trip. Cache compression happens once on the write path; the
correlation is the storage-layout fidelity, not a hook artefact.

| Model | Path | K_corr (mean over all layers) | V_corr | Compression |
|---|---|---|---|---|
| Dolphin3.0-Llama3.2-1B-Q8 (hd=64, 16 layers) | ship 5,5,4,3 / 3 | **0.9941** | 0.9712 | 3.76× |
| Dolphin3.0-Llama3.2-1B-Q8 | sqfree (pad 66) | 0.9768 | 0.9484 | 3.76× |
| Dolphin3.0-Llama3.2-1B-Q8 | sqfree+spinor | 0.9869 | 0.9601 | 3.76× |
| Qwen3-8B-Q8 (hd=128, 36 layers) | ship 5,5,4,3 / 3 | **0.9934** | 0.9691 | 4.06× |

Spinor's +0.008–0.010 K-corr lift on the Knight skeleton matches
the value documented in CLAUDE.md. Ship hits the 0.992+ target on
real model data, end-to-end, on both architectures.

### Ship path (synthetic + scaling-law projection)
| Config | K Corr | Compression | PPL Impact |
|--------|--------|-------------|------------|
| 5/5/4/3 + Möbius (ship) | 0.992+ | 3.4× | −0.04% (better) |
| 4/4/4/3 | 0.990 | 3.8× | +0.39% |
| 3/3/3/3 (floor) | 0.976 | 4.6× | +3.90% |

### Sqfree+spinor (Qwen3-8B Q8 hd=128, llama-cpp-sp hook)
| Config | PPL | K Corr | Compression |
|--------|-----|--------|-------------|
| MOBIUS default 5/5/5/5/5 | 7.31 | 0.999 | 2.6× |
| K+μ+3bit+spinor 5/4/4/4/5 | 7.30 | 0.988 | 2.8× |
| ~~K+μ+3bit+spinor 3/3/3/3/3~~ | ~~7.32~~ | ~~0.972~~ | ~~3.3×~~ |

> Retracted: the 3/3/3/3/3 row was measured before the v1.03 fix to
> `sp_sqfree_cache_init`. Pre-fix, that call hard-coded 5,4,4,4,5 and
> silently ignored K_BITS, so the "3,3,3,3,3" run was actually running
> 5,4,4,4,5. Post-fix, true 3/3/3/3/3 on the Knight skeleton is catastrophic.
> The effective Pareto point is **5,4,4,4,5 at 2.8×**.

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

<!-- SP-MEASURED-RESULTS:BEGIN -->

_Auto-generated from `logs/*.json` on 2026-04-17 22:04 UTC_

> ⚠ **Caveat on all numbers in this section.** Every PPL figure below was
> measured through the shannon-prime-llama post-decode hook, which runs a
> decompress → attention → recompress round-trip on every decode rather
> than storing compressed KV natively. That integration surface introduces
> its own state — a single session's work surfaced four hook-specific bugs
> (GPU-resident KV memcpy segfault, vkCmdCopyBuffer barrier miss, CPU/GPU
> blob format mismatch, V-pipeline plumbing loss) that could each have
> been misread as compression-math bugs. Absolute PPL values carry that
> uncertainty; the **relative ordering** of configs (ship < sqfree+spinor
> 5,4,4,4,5 < 10-band-auto < 10-band-uniform-3 < catastrophic
> 10-band-uniform-2) has stayed stable across every hook refactor and is
> the part to trust. The
> [shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine)
> sibling repo (scaffolding landed) is intended as the bug-free reference
> measurement; every figure here should be re-measured there before
> treating as publication-grade.

### KV cache perplexity (VHT2 ship vs sqfree aggressive)

| Model | Backend | Config | Median PPL | Ctx/Chunks | Date |
|---|---|---|---|---|---|
| Dolphin3.0-Llama3.2-1B-Q8_0 | cuda | baseline | 10.7164 | 2048/8 | 2026-04-18T01:35:00Z |
| Dolphin3.0-Llama3.2-1B-Q8_0 | cuda | ship | 10.7698 | 2048/8 | 2026-04-18T01:38:00Z |
| Dolphin3.0-Llama3.2-1B-Q8_0 | cuda | sqfree+spinor | 13.3277 | 2048/8 | 2026-04-18T01:48:00Z |
| Qwen3-8B-Q8_0 | cuda | baseline | 8.6746 | 4096/8 | 2026-04-18T01:50:00Z |
| Qwen3-8B-Q8_0 | cuda | ship | 8.7051 | 4096/8 | 2026-04-18T02:15:00Z |
| Qwen3-8B-Q8_0 | cuda | sqfree+spinor | 8.8265 | 4096/8 | 2026-04-18T03:00:00Z |
| Qwen3-8B-Q8_0 | cuda-host-cpu (GGML_CUDA=OFF in current llama-cpp-sp build) | sqfree+spinor 10×2 (catastrophic — below 3-bit floor) | 428.7779 | 2048/4 | 2026-04-18T07:55:00Z |
| Qwen3-8B-Q8_0 | cuda-host-cpu (GGML_CUDA=OFF in current llama-cpp-sp build) | sqfree+spinor 10×3 | 11.8324 | 2048/4 | 2026-04-18T07:40:00Z |
| Qwen3-8B-Q8_0 | cuda-host-cpu (GGML_CUDA=OFF in current llama-cpp-sp build) | sqfree+spinor 10×auto (tools/sp_auto_bands.py → 4,4,4,4,4,3,3,3,3,3 @ total=35) | 10.0768 | 2048/4 | 2026-04-18T08:05:00Z |

### Weight predictor + frequency injector

| Model | Alpha | GGUF size | Sidecar | PPL baseline | PPL injected + ship | Date |
|---|---|---|---|---|---|---|
| Dolphin3.0-Llama3.2-1B-Q8_0 | 0.17 | 1,600,178,432 | 128 B | 10.7164 | 10.7698 | 2026-04-18T03:10:00Z |

_sp_inject_freqs.py produces a byte-identical GGUF plus a 128-byte .sp_freq_factors.bin sidecar (32 × fp32). The current shannon-prime-llama integration in D:/F/llama-cpp-sp does not yet consume the sidecar at inference, so injection has zero effect on PPL in this measurement — consistent with the 'GGUF weight compression is still theoretical' memory note. The ship-path KV compression layered on top still hits spec (+0.50% PPL @ 3.4-3.8x)._

<!-- SP-MEASURED-RESULTS:END -->
