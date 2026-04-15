# Shannon-Prime: Spectral KV Cache Compression via the Multiplicative Lattice

**Version 1.0 — 2026**
**Author: Ray Daniels**
**License: AGPLv3 / Commercial Dual License**

---

## What Shannon-Prime Is

Shannon-Prime is a KV cache compression system for transformer models that exploits a structural property no other system uses: the multiplicative lattice of the integers encoded by rotary position embeddings.

Every modern transformer uses RoPE (Rotary Position Embedding) to give tokens position awareness. RoPE applies angular rotations to key vectors at specific frequencies. Shannon-Prime's insight is that these angular rotations create predictable spectral structure — when you apply the Walsh-Hadamard Transform to a K vector, the energy concentrates in the first spectral bands. V vectors, which carry content rather than position, have uniform energy across all bands.

This asymmetry is the foundation. K and V are structurally different signals. They should be compressed differently.

The system achieves **3.4–3.8× total KV compression at less than 1.25% perplexity cost**. The ship configuration (5/5/4/3 bit banding) actually **beats lossless fp16 by 0.04%** — a spectral regularization effect where quantizing the low-energy noise tail improves quality.

## Where It Sits in the Transformer

```
                     ┌─────────────────────────────┐
                     │       Transformer Layer       │
                     │                               │
  Input ──► Q,K,V = W_q·x, W_k·x, W_v·x            │
              │         │         │                   │
              │    RoPE(K)        │                   │
              │         │         │                   │
              │    ┌────▼─────────▼────┐              │
              │    │  Shannon-Prime    │              │
              │    │  Shadow Cache     │              │
              │    │                   │              │
              │    │  Write path:      │              │
              │    │  K/V → WHT →      │              │
              │    │  Möbius reorder → │              │
              │    │  Band quantize → │              │
              │    │  Store (3.4×     │              │
              │    │   compressed)    │              │
              │    │                   │              │
              │    │  Read path:       │              │
              │    │  Load → Dequant → │              │
              │    │  Unreorder →      │              │
              │    │  Inverse WHT →    │              │
              │    │  K/V              │              │
              │    └────┬─────────┬────┘              │
              │         │         │                   │
              ▼         ▼         ▼                   │
           Attention = softmax(Q·K^T / √d) · V        │
              │                                       │
              ▼                                       │
           Output + Residual                          │
                     └─────────────────────────────┘
```

Shannon-Prime intercepts KV vectors **after** RoPE is applied to K and **before** they enter the KV cache. The shadow cache stores the compressed representation. When attention needs the cached K/V, Shannon-Prime reconstructs them. The inference engine never touches fp16 KV storage — it only sees the compressed shadow cache.

## The Compression Pipeline

Each vector goes through five stages:

```
Write: raw KV → WHT → Möbius reorder → Band quantize → Store
Read:  Load → Band dequantize → Möbius unreorder → Inverse WHT → KV
```

**WHT (Walsh-Hadamard Transform):** A self-inverse butterfly transform. Takes a vector and decomposes it into spectral components. K vectors with RoPE show strong concentration — 80%+ of energy in the first half of bands. V vectors are uniform. This is the Z/2Z case of the Vilenkin-Hartley basis described in the companion papers.

**Möbius reorder:** Rearranges WHT coefficients so squarefree indices come first. The Möbius function μ(n) identifies which indices carry independent information versus structural echoes. This reordering ensures the highest-information coefficients get the most quantization bits. Free quality improvement: +0.14 PPL at identical storage. Cross-platform invariant — K correlation 0.997 on both hd=128 and hd=64.

**Band quantize:** Splits the reordered coefficients into 4 equal bands. Each band gets its own fp16 scale factor and packed signed integers at a specified bit depth. The allocation follows WHT energy decay:

| Band | Energy | Bits | Role |
|------|--------|------|------|
| 0 | Highest | 5 | Core spectral structure |
| 1 | High | 5 | Primary harmonics |
| 2 | Medium | 4 | Detail |
| 3 | Lowest | 3 | Noise tail (beneficial compression) |

V vectors get flat 3-bit quantization with no banding — their uniform spectrum means all bands are equally important and equally tolerant of quantization.

## The Theoretical Foundation

The companion papers ("Position Is Arithmetic" and "The KV Cache Is a View") establish the mathematical basis:

1. Language follows Zipf's law with exponent s ≈ 1.
2. The generating function of Zipf is the Riemann zeta function.
3. The zeta zeros encode the prime harmonic structure via the explicit formula.
4. Therefore the multiplicative lattice generated by primes provides a natural spectral basis for positional encoding.

The Walsh-Hadamard Transform is the Z/2Z case of this spectral basis. The Vilenkin-Hartley Transform extends it to the full multi-prime decomposition (Z/2Z × Z/3Z × Z/5Z × ...). When the full Vilenkin basis is used at 95% energy threshold, Walsh catastrophically fails (PPL +477%) while Vilenkin achieves +9.9% — proving the multiplicative structure is real and measurable.

Shannon-Prime as shipped uses the WHT (Z/2Z) path because it works with all power-of-2 head dimensions. The Vilenkin path is included as a research extension for teams working on non-power-of-2 architectures or seeking maximum compression.

## Key Results

| Configuration | K Correlation | V Correlation | Compression | PPL Impact |
|---------------|---------------|---------------|-------------|------------|
| 5/5/4/3 + Möbius (ship) | 0.992+ | 0.957+ | 3.4× | −0.04% (better than fp16) |
| 5/4/4 (mobile safe) | 0.997 | 0.996 | 3.2× | <1.25% |
| 4/4/4/3 | 0.990 | 0.950+ | 3.8× | +0.39% |
| 3/3/3/3 (floor) | 0.976 | 0.940+ | 4.6× | +3.90% |

Critical rules discovered through experimentation:

1. **Skeleton size must equal head_dim.** Using sk=32 on hd=64 gives PPL +47%.
2. **3-bit floor.** 2-bit on any band is catastrophic — the quantization error exceeds the signal.
3. **5/5/4/3 mirrors WHT energy decay.** Each band's optimal bit depth tracks its energy.
4. **4 bands beats 5 or 8.** The 2-byte fp16 scale overhead per band erases gains from finer granularity.
5. **Flat beats banded for V.** No exceptions across the entire sweep.

## Platform Coverage

Shannon-Prime ships with four backend implementations:

| Backend | Language | Target | Status |
|---------|----------|--------|--------|
| **Core** | C11 | Universal reference | 31/31 tests |
| **CUDA** | CUDA C++ | NVIDIA GPUs | Kernels written, needs nvcc |
| **Vulkan** | C + GLSL | Cross-platform GPU | Shaders + CPU fallback, 4/4 tests |
| **Torch** | Python | PyTorch ecosystem, ComfyUI | 28/28 tests |
| **Adreno** | C + NEON | Qualcomm Snapdragon mobile | 14/14 tests |

Integration layers for llama.cpp and ComfyUI are included. See the dedicated integration guides.

## License: Why AGPLv3 + Commercial

Shannon-Prime uses a dual license:

**AGPLv3 (open source path):** Anyone can use, modify, and distribute Shannon-Prime for open-source, academic, and non-proprietary purposes. The AGPLv3 requires that derivative works — including modifications deployed as network services (SaaS) — must also be released under AGPLv3 with full source code disclosure.

**Commercial License (proprietary path):** For closed-source deployment, proprietary integration, or commercial products/services without AGPLv3 obligations, a separate Commercial License Agreement is available.

**Why this structure:**

The primary goal is that everyone benefits from this work. AGPLv3 ensures that:

- Researchers and open-source developers can use it freely.
- Anyone who improves it contributes those improvements back to the community.
- Companies that build on it in the open benefit from the same protections.
- No one can take the work, close it, and deny others access.

The commercial license exists for companies whose business model genuinely requires proprietary distribution — they can contact for terms. But the default is open. The code, the math, the results — they belong to the commons. The license structure protects that.

Code released prior to April 12, 2026 (tagged v1.0-public-domain) was released under a Public Domain waiver. That historical release remains under its original terms. All new development from April 12, 2026 onward is governed by the dual-license structure.

## Contact

Ray Daniels — raydaniels@gmail.com

For commercial licensing, research collaboration, or questions about integration.
