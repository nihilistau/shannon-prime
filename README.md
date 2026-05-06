# Shannon-Prime

**Exact Spectral KV Cache Compression via the Multiplicative Lattice**

Shannon-Prime compresses the transformer KV cache by exploiting the spectral structure that rotary position embeddings (RoPE) imprint on key vectors. The single transform is VHT2 — the Vilenkin-Hartley Transform — a staged orthonormal generalization of the Walsh-Hadamard Transform. At power-of-2 head dimensions it reduces to the p=2 Hartley butterfly scaled by 1/√2 per stage (self-inverse, no 1/N needed); at squarefree-padded dimensions it factors across small primes {2, 3, 5, 7, 11} and unlocks the Möbius predictor, Knight skeleton, and SU(2) spinor sheet bit for aggressive compression.

The transform is self-inverse: `VHT2(VHT2(x)) = x`. The same function compresses and decompresses. No separate encoder/decoder. No training. No fine-tuning.

---

## Headline Numbers

| Configuration | Compression | Quality | Notes |
|---|---|---|---|
| **Ship** (5/5/4/3 banded) | 3.4–3.8× | < 1.25% PPL cost | Beats lossless fp16 by 0.04% via spectral regularization |
| **Sqfree + Spinor** (5/4/4/4/5) | 2.8× | MOBIUS-default quality | On Qwen3-8B Q8 hd=128 |
| **Hierarchical Vilenkin** (14-skeleton) | 7.0× | Calibrated predictor | 9% skeleton → linear map → 2-bit residual |
| **Aggressive Hierarchical** (14×4 + 1-bit) | 12.6× | Research path | 24.5 bytes per position (from 308 fp16) |

**Speculative Decoding (Phone):** Qwen2.5-Coder-3B IQ2 + 0.5B Q8 draft + spec-decode --draft 8 + SP-Hexagon FUSED_KQ = **43.72 t/s eval (3.58× vs vanilla 12.20)**. Phone CPU only, no GPU/NPU offload.

**LM Studio:** Qwen3.6-35B-A3B MoE at **26.92 tok/sec** with Shannon-Prime KV cache active. LM Studio v2.14.0, custom runtime DLLs.

**ComfyUI Video:** Wan 2.2 5B at **4.6× step speed** (32 → 7 s/step) via block-skip + cross-attention cache.

```
make test-all   # 187/188 tests across 8 suites
```

---

## What It Does

Shannon-Prime intercepts KV vectors on the write path, transforms them into the spectral domain via VHT2, quantizes the spectral coefficients into bands (high-energy bands get more bits, low-energy tail gets fewer), and stores the compressed representation. On read, it dequantizes, applies the inverse transform (which is the same transform — VHT2 is self-inverse), and returns the reconstructed vector.

The write path: `raw KV → VHT2 → Möbius reorder → banded quantize → store`
The read path: `load → band dequantize → Möbius unreorder → VHT2 → reconstructed KV`

This works because RoPE creates predictable spectral structure in K vectors. The VHT2 coefficients concentrate energy in the early bands and decay toward the tail. The 5/5/4/3 bit allocation exploits this decay: 5 bits for the high-energy bands, 3 bits for the noise tail. The Möbius reorder pushes squarefree indices (61.4% of N=210) to the front, improving quality by 0.14 PPL at identical bit budget.

---

## The Compression Stack

Shannon-Prime provides three compression tiers, each building on the previous:

### Tier 1: Ship Path (3.4×)

The production-safe default. VHT2 → Möbius reorder → 4-band quantization at 5/5/4/3 bits. Works on every model, every architecture, every head dimension that is a power of 2. Zero calibration needed.

### Tier 2: Sqfree + Spinor (2.8×)

Pads the head dimension to a squarefree-factorable number (hd=128 → 154 = 2·7·11), applies the full Vilenkin-Hartley basis, extracts a Knight skeleton of the L/2 highest-variance squarefree indices, and uses the Möbius CSR predictor to reconstruct residuals. The spinor sheet bit adds a 1-bit SU(2) double-cover correction that shifts the Pareto frontier — not just moves along it.

### Tier 3: Hierarchical Vilenkin (7–12×)

Exploits the Kronecker product structure of the Vilenkin basis. For hd=128 (pad=154 = 2·7·11), the Z/2Z × Z/7Z sub-projection gives a 14-coefficient core skeleton (9.1% of the dimension). A calibrated linear predictor W maps the skeleton to all remaining coefficients. Per-position storage drops to 43.75 bytes (from 308 fp16).

---

## Key Features

**Self-Inverse Transform.** VHT2 is its own inverse. One function, one codepath, one set of SIMD kernels. No asymmetric encoder/decoder complexity.

**Stateless Compression.** Each KV vector is compressed independently. No sequential dependencies, no state to maintain across tokens. Enables arbitrary eviction, reordering, and parallel processing.

**Progressive Band Reads.** The v3 band-major disk format allows reading only the first N bands from storage. Band 0 alone (32 of 128 coefficients) gives 0.30 correlation; bands 0+1 give 0.86. Attention can answer most queries from partial data.

**System 1/2 Dual-Cache Switching.** Two cache modes: System 1 (fast, partial-band, good enough for 99% of tokens) and System 2 (full reconstruction, activated when attention entropy demands it). Inspired by Kahneman — the common case is cheap, the hard case pays full cost.

**Disk-Tier Architecture.** Compressed bands can live on different storage tiers: band 0 in RAM, band 1 on NVMe, bands 2–3 on Optane or network. Progressive loading makes 100K+ context feasible on memory-constrained devices.

**Cauchy Reset System.** The Ricci sentinel monitors p=3 spectral band energy drift; the Mertens oracle (driven by zeta-zero oscillations) predicts optimal reset windows. When compression errors become "timelike" (aligned with information flow), the cache re-anchors.

**PrimePE.** Lattice-aligned RoPE frequency injection. Blends multiplicative-lattice frequencies into standard geometric RoPE at zero runtime cost. Proven −0.6% to −0.8% PPL improvement across architectures, zero retraining.

**Scaling Law.** `log(PPL/base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)`. Fits 9 configurations across 4 orders of magnitude. Use as a pre-bench filter: if predicted PPL ratio exceeds your budget, skip the config.

**Model-Pack Registry.** Per-architecture compression defaults. phi3 calibrated, 7 architectures provisional. `--model-preset auto` resolves from GGUF metadata.

**Zero-Copy Pointers.** Optane and mmap-based reservoirs provide direct pointers into model weights. No copy, no allocation, no intermediate buffer. The shredder reads directly from the mapped file.

---

## Backends

Shannon-Prime ships 9 backend implementations:

| Backend | Target | Status | Key Feature |
|---|---|---|---|
| **CPU (C reference)** | Any x86/ARM | Production | The reference implementation. All features. |
| **CUDA** | NVIDIA GPU | Production | GPU-resident compressed cache. No host round-trip. |
| **Vulkan** | Any Vulkan 1.1+ GPU | Production | Cross-platform. Compute shaders. NVIDIA/AMD/Intel/Adreno. |
| **Adreno** | Qualcomm mobile GPU | Production | OpenCL 2.0 + Vulkan compute. Snapdragon 8 Gen 1+. |
| **Hexagon** | Qualcomm DSP | Production | HVX/HTP V69. FastRPC. 577 calls/sec ceiling. |
| **QNN AI Hub** | Qualcomm NPU/HTP | Production | Pre-compiled .bin execution via QNN runtime. |
| **Halide** | Cross-platform (research) | Scaffold | Halide scheduling language for auto-tuned kernels. |
| **Torch** | Python/PyTorch | Production | Reference for Python integrations. ComfyUI, notebooks. |
| **Beast Canyon** | Heterogeneous desktop | New | Optane reservoir + AVX-512 shredder + dual-GPU + phone sidecar. |

---

## Repository Structure

```
shannon-prime/                  ← You are here (core math library)
├── core/
│   ├── shannon_prime.h         # Complete public API (1133 lines)
│   ├── shannon_prime.c         # Ship path implementation
│   ├── shannon_prime_sqfree.c  # Sqfree + spinor + hierarchical
│   └── shannon_prime_modelpack.h
├── backends/
│   ├── cuda/                   # GPU-resident VHT2 + banded quant
│   ├── vulkan/                 # Compute shader pipeline
│   ├── adreno/                 # Mobile GPU (OpenCL + Vulkan)
│   ├── hexagon/                # Qualcomm DSP (HVX/HTP V69)
│   ├── qnn_aihub/              # QNN pre-compiled .bin driver
│   ├── halide/                 # Halide scheduling (research)
│   ├── torch/                  # PyTorch reference backend
│   ├── beast_canyon/           # Heterogeneous desktop engine
│   └── freethedsp/             # DSP unlock shim
├── docs/                       # Architecture, features, guides
├── tests/                      # 8 test suites, 187/188 passing
├── tools/                      # CLI utilities
└── scripts/                    # Build, benchmark, calibration
```

## Sibling Repositories

| Repository | Purpose |
|---|---|
| [shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine) | Standalone inference engine. Owns the compressed KV data path end-to-end. GGUF loader, native forward, GPU cache, multi-GPU, QNN HTP dispatch. **The reference implementation.** |
| [shannon-prime-comfyui](https://github.com/nihilistau/shannon-prime-comfyui) | 16 ComfyUI custom nodes. Video (Wan 2.x), image (Flux), audio (Stable Audio), TTS (Voxtral 4B). Block-skip caching + VHT2 KV compression. |
| [shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama) | llama.cpp integration for LM Studio. Patch-based bridge for the existing ecosystem. |

**Voxtral TTS forks** with integrated VHT2 KV compression:
[Python](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS),
[Rust](https://github.com/nihilistau/voxtral-mini-realtime-rs),
[C](https://github.com/nihilistau/voxtral-tts.c).

---

## Universal Across Modalities

The VHT2 spectral decomposition is not specific to language. Every transformer architecture that uses rotary position embeddings creates the same spectral concentration in its key vectors — the mathematics is identical whether the model generates text tokens, video frames, audio waveforms, or image patches.

| Modality | Model | Result |
|---|---|---|
| Text (LLM) | Qwen3-8B, Llama 3.2, Dolphin | 3.4× KV compression, < 1.25% PPL |
| Text (MoE) | Qwen3.6-35B-A3B | 26.92 tok/sec in LM Studio |
| Video | Wan 2.2 5B / A14B MoE | 4.6× / 3.5× step speed |
| Image | Flux v1/v2 | Block-level skip, dual-stream cache |
| Audio | Stable Audio | Cross-attention K/V cache |
| TTS | Voxtral 4B | VHT2 KV compression (Python, Rust, C) |

---

## Quick Start

```bash
# Build
make                    # Builds core + CPU backend
make test-all           # Run all 8 test suites

# With CUDA
make SP_WITH_CUDA=1

# With Vulkan
make SP_WITH_VULKAN=1

# Beast Canyon standalone test
cd backends/beast_canyon
cmake -B build -DSP_BEAST_STANDALONE=ON
cmake --build build
./build/sp_beast_test /path/to/model.gguf
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for full build instructions across all platforms.

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Complete system architecture — every component, every data path |
| [Compression Features](docs/COMPRESSION-FEATURES.md) | Deep dive on every compression feature with technical detail |
| [Backends](docs/BACKENDS.md) | All 9 backend implementations — capabilities, build, performance |
| [Mathematical Foundations](docs/MATHEMATICAL-FOUNDATIONS.md) | The theory: multiplicative lattice, VHT2, Möbius, scaling law |
| [Models & Hardware](docs/MODELS-AND-HARDWARE.md) | Supported models, hardware guide, future implications |
| [Quick Start](docs/QUICKSTART.md) | Build, configure, first run on every platform |
| [Testing](docs/TESTING.md) | Test suites, validation methodology, benchmarking |

---

## The Papers

- **Position Is Arithmetic** (v8) — The multiplicative lattice as the natural basis for positional encoding. PrimePE, the K-corr scaling law, the spinor sheet bit.
- **The KV Cache Is a View** (v2) — Spectral compression, the scaling law, basis-dependent Möbius prediction, the spinor sheet bit shifting the Pareto frontier.
- **The Multiplicative Lattice** — Combined mathematical framework. Vilenkin-Hartley transform, squarefree factorization, Knight-ranked masks, CSR predictor.

---

## License

Copyright (C) 2026 Ray Daniels. All Rights Reserved.

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).
Commercial license available — contact raydaniels@gmail.com.
