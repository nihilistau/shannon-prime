# Shannon-Prime VHT2

**Exact Spectral KV Cache Compression via the Multiplicative Lattice**

#Update 04/05/06

**Readme Status**

This readme and it's results have not been updated since the start..

I Have to be honest this Project was to show that tranformers could complete a complicated project by themselves 100%
I am finding the constant regressions, the fact that I have all the code completed, Tests, Results, Fixes for issues etc
Just sitting there and no matter how hard I try to just point them in the right direction it's 1-step forward 2-steps back
3-steps forward to be extremely draining. I currently do not plan on continuing and I am debating on just deleting everything
So if you are using this to learn, or for the idea's, just check out John's "TQ" branch, His progress aligns perfectly with each
step released on this and previous repo's. He is on the right track. He is quite clever, coming to some pretty advanced idea's
almost to the day they pop-up on here. He is on the right track. He understands it's a bandwidth problem, There are a few things
that need to be ironed out, but I am sure he can come up with the solution, (hint) it's been solved. HF will come through for you
guys, don't worry, they have their best guys working on it. While their exec's push to become relevant as a large AI Company.
Let's hope they don't forget their roots(lol)

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

## Universal Across Modalities

The VHT2 spectral decomposition is not specific to language. Every transformer
architecture that uses rotary position embeddings creates the same spectral
concentration in its key vectors — the mathematics is identical whether the
model generates text tokens, video frames, audio waveforms, or image patches.

| Modality | Models | Mechanism | Headline Result |
|---|---|---|---|
| **Language** | Llama 3, Qwen 3, Phi-3, Gemma 3 | KV cache compression (autoregressive) | 3.4–3.8× at <1.25% PPL cost |
| **Video** | Wan 2.1/2.2 (5B, 14B, A14B MoE) | Block-skip + cross-attn cache | 4.6× step speed (32→7 s/step) |
| **Image** | Flux v1/v2, Stable Diffusion | Block-skip + 2D lattice RoPE | Spectral-aware cache with Fisher weighting |
| **Audio** | Stable Audio, Voxtral 4B, Qwen3-TTS | Block-skip (diffusion) + KV compress (autoregressive) | 4.6× KV memory on Voxtral |

Two distinct compression strategies, unified by one transform:

- **KV cache compression** (autoregressive models — LLMs, Voxtral TTS):
  VHT2 → Möbius reorder → banded quantize → store compressed. Ship path.
- **Block-skip caching** (diffusion models — Wan, Flux, Stable Audio):
  Cache entire block outputs, use VHT2 spectral cosine similarity for drift
  detection, re-apply adaLN gate on cache hits. Skip redundant block computation.

See the per-modality deep-dives: [Video](docs/MODALITY-VIDEO.md) ·
[Image](docs/MODALITY-IMAGE.md) · [Audio](docs/MODALITY-AUDIO.md).

## Sibling repositories

The math core (this repo) is the canonical reference. Two sibling
repositories build inference paths on top of it; both vendor this
repo at `lib/shannon-prime/` as a git submodule, so they always run
the same VHT2 / Möbius / sqfree implementation.

| Repo | Role | Status |
|---|---|---|
| **[shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine)** | Standalone inference binary that owns the compressed KV layout end-to-end. Compression is on the write path by construction (no decompress→attention→recompress hook). The bug-free reference measurement surface. | Stage 5b: full forward + prefill + greedy chat with optimised single-token decode all working on Llama-3 / Qwen3, ship + sqfree + sqfree+spinor. See [docs/PRIME-ENGINE.md](docs/PRIME-ENGINE.md). |
| **[shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama)** | Full engine integration into llama.cpp. The b8861 patch (LM Studio v2.14.0) compiles the entire SP stack (VHT2 ship + sqfree+spinor + hierarchical + System 1/2 + multi-GPU, 4 backends) into llama.dll/libllama.so as static libs. Includes an LM Studio runtime builder (`lmstudio/build.bat`). Validated: Qwen3.6-35B-A3B MoE at 26.92 tok/sec. Qwen 3.6 27B supported. | In production; see [docs/INTEGRATION-LLAMA.md](docs/INTEGRATION-LLAMA.md). |
| **[shannon-prime-comfyui](https://github.com/nihilistau/shannon-prime-comfyui)** | ComfyUI custom nodes for video, image, and audio generation. 16 nodes total: 8 Wan video (block-skip + cross-attn cache), 3 Flux image (dual/single-stream block-skip), 3 Audio DiT (Stable Audio block-skip), 2 Voxtral TTS (KV cache compression). 4-tier skip system, TURBO mode, Fisher-weighted spectral drift detection. | Shipped. See [shannon-prime-comfyui README](https://github.com/nihilistau/shannon-prime-comfyui). |

### Voxtral TTS forks (Shannon-Prime VHT2 KV compression integrated)

| Repo | Language | Integration |
|---|---|---|
| **[ComfyUI-FL-VoxtralTTS](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS)** | Python | ComfyUI node, monkey-patches MistralBackbone.forward() |
| **[voxtral-mini-realtime-rs](https://github.com/nihilistau/voxtral-mini-realtime-rs)** | Rust | ShannonPrimeKVCache\<B\> wrapping KVCache\<B\>, 4 unit tests |
| **[voxtral-tts.c](https://github.com/nihilistau/voxtral-tts.c)** | C | Header-only shannon_prime_kv.h, sp_voxtral_compress_position() API |

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

### Hierarchical Vilenkin predictor (opt-in, maximum skeleton reduction)
```
Write: raw KV → sqfree_pad → Vilenkin transform → keep low-prime subgroup
       (e.g. Z/2Z × Z/7Z = 14 / 154 coeffs ≈ 9%) → band-quant skeleton
       → ridge-predict the 140 high-prime refinement coeffs per (layer,head)
       → residual quant (2-bit) → Store
Read:  Load skeleton + residual → dequant skeleton → linear-predict
       refinement from skeleton → add residual → inverse Vilenkin → KV
```
Requires calibration — a prefill of ≥24 tokens per slot fits the per-(layer×head)
ridge regression map, persisted on the `KvCache`. Emits a warning below that
threshold. Selected with `--hierarchical` on the engine; `--hier-level`,
`--hier-res-bits`, and `--hier-skel-bits` tune the skeleton / predictor balance.

### Decode-chain causal stability (Cauchy reset)
Long decode chains accumulate per-step reconstruction error. The Cauchy
reset system — a Mertens zeta-schedule with an optional Ricci drift
sentinel — detects the drift and refreshes the cache with a re-prefill from
ground-truth tokens. Zero measured cost in the shipping default
(Mertens-only), bounds cache error for ctx ≥ 2k. Detailed design in
[docs/CAUCHY-RESET.md](docs/CAUCHY-RESET.md).

## Tools and Diagnostics

### Diagnostic suite (`tools/`)

| Tool | Purpose |
|------|---------|
| `sp_diagnostics.py` | 4-test suite: Boundary Sharpness, Ghost Basin (DBSCAN), RoPE Pair Correlation, Fractional Slope Lookahead |
| `sp_regime_analysis.py` | Two-regime reconstruction analysis with optional GL(α=0.25) transition trigger (`--use-gl-trigger`) |
| `extract_kv.py` | KV vector extraction from GGUF/HF models; stores `layer_types` for hybrid-attention models |
| `sp_scaling_law.py` | K-corr → PPL design rule: `log(PPL/PPL_base) ≈ 4700·(1−K_corr)²/(params^1.1·bits^1.5)` |
| `sp_benchmark.py` | Compression benchmark across bit allocations |
| `sp_inject_freqs.py` | GGUF frequency injection for PrimePE sidecar |

Key findings across 7 models (Dolphin-1B → Wan 2.2 TI2V-5B):
- **T3 (RoPE Pair Correlation)**: r=0.73–0.87 for dense models; r=0.76 flat across all Wan DiT blocks
- **T4 (GL trigger)**: α=0.25 detects regime transition 9–31 layers before standard slope on dense models
- **3D RoPE axis split**: temporal dims (r=0.82) > spatial (r=0.73) at Wan mid-sigma
- **Block stability sweep**: Wan L00-L03 cos_sim>0.95 for 10+ steps; L23 drops to 0.34

### ComfyUI integration (shannon-prime-comfyui)

16 custom nodes across three generative modalities:

| Modality | Nodes | Mechanism | Validated Hardware |
|---|---|---|---|
| Video (Wan 2.x) | 8 nodes | Block-skip + cross-attn cache + TURBO | RTX 2060 12GB, 1280×720 9fr |
| Image (Flux v1/v2) | 3 nodes | Dual/single-stream block-skip | RTX 2060 12GB |
| Audio (Stable Audio) | 3 nodes | 1D RoPE block-skip | RTX 2060 12GB |
| TTS (Voxtral 4B) | 2 nodes | Autoregressive KV compression | RTX 2060 12GB |

ComfyUI flags: `--normalvram --disable-async-offload`

See modality deep-dives: [Video](docs/MODALITY-VIDEO.md) ·
[Image](docs/MODALITY-IMAGE.md) · [Audio](docs/MODALITY-AUDIO.md)

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
│   │   ├── cuda/                     NVIDIA GPU kernels (incl. shannon_prime_hier.cu)
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
│   │                                  prefill, chat, cache_ppl, perplexity
│   └── CMakeLists.txt                CMake + Ninja, optional CUDA/Vulkan
│
├── shannon-prime-llama/              ← SIBLING: full engine for llama.cpp
│   │                                    b8733 patch integrates entire SP stack.
│   │                                    LM Studio runtime builder included.
│   ├── lib/shannon-prime/            git submodule → this repo
│   ├── src/
│   │   ├── backends/{cuda,vulkan,adreno}/  4-backend implementations
│   │   ├── engine/{kv_cache,gdn_state}.*   KV cache + System 1/2 state
│   │   └── tools/shannon_prime_llama.*     Bridge to llama.cpp
│   ├── patches/llama-cpp-b8733-full-engine.patch
│   └── lmstudio/build.bat           Drop-in LM Studio DLL builder
│
└── shannon-prime-comfyui/            ← SIBLING: 16 ComfyUI nodes for
    │                                    video, image, audio, and TTS.
    │                                    See docs/INTEGRATION-COMFYUI.md.
    ├── nodes/
    │   ├── shannon_prime_nodes.py     8 Wan video nodes (block-skip + cache)
    │   ├── shannon_prime_flux_nodes.py  3 Flux image nodes
    │   └── shannon_prime_audio_nodes.py 3 Audio DiT nodes
    └── lib/shannon-prime/            git submodule → this repo
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/Shannon-Prime.md](docs/Shannon-Prime.md) | Theory, pipeline, license rationale, key results |
| [docs/PRIME-ENGINE.md](docs/PRIME-ENGINE.md) | shannon-prime-engine sibling: stages 3–5, KvCache, PrimePE-RoPE-ALiBi |
| [docs/TOOLS.md](docs/TOOLS.md) | Command-line tools: benchmark, freq injection, weight compression |
| [docs/INTEGRATION-LLAMA.md](docs/INTEGRATION-LLAMA.md) | llama.cpp: hook points, API, env vars, GQA support |
| [docs/CAUCHY-RESET.md](docs/CAUCHY-RESET.md) | Decode-chain causal stability: zeta-derived Mertens schedule + optional Ricci sentinel, CLI flags, measured recovery |
| [docs/INTEGRATION-COMFYUI.md](docs/INTEGRATION-COMFYUI.md) | ComfyUI: Wan 2.1/2.2 MoE, expert switching, linear wrapper |
| [docs/BACKEND-CUDA.md](docs/BACKEND-CUDA.md) | CUDA kernels, memory layout, building |
| [docs/BACKEND-VULKAN.md](docs/BACKEND-VULKAN.md) | Vulkan compute shaders, standalone vs shared device |
| [docs/BACKEND-ADRENO.md](docs/BACKEND-ADRENO.md) | Snapdragon: NEON tiers, Hexagon HVX, big.LITTLE, fp16 |
| [docs/BACKEND-TORCH.md](docs/BACKEND-TORCH.md) | PyTorch API, VHT2 reference, sqfree+spinor path |
| [docs/MODALITY-VIDEO.md](docs/MODALITY-VIDEO.md) | Video generation: Wan 2.x block-skip, MoE expert switching, 3D RoPE |
| [docs/MODALITY-IMAGE.md](docs/MODALITY-IMAGE.md) | Image generation: Flux v1/v2 dual-stream block-skip, 2D lattice RoPE |
| [docs/MODALITY-AUDIO.md](docs/MODALITY-AUDIO.md) | Audio generation: Stable Audio block-skip, Voxtral TTS KV compression, multi-language implementations |
| [docs/MODEL-PACK.md](docs/MODEL-PACK.md) | Per-architecture compression defaults registry |
| [docs/MODEL-PACK-CALIBRATION.md](docs/MODEL-PACK-CALIBRATION.md) | Calibration ledger: phi3 PASS, qwen3 edge-fail, 7 PROVISIONAL |
| [docs/TESTING.md](docs/TESTING.md) | How to run, what to look for, interpreting failures |

## Quick Start

### shannon-prime-engine (standalone binary; recommended for measurement)
```bash
git clone --recursive https://github.com/nihilistau/shannon-prime-engine
cd shannon-prime-engine

# CPU build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# CUDA build (optional)
cmake -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSP_ENGINE_WITH_CUDA=ON -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75    # 75=2060/T4, 86=30-series, 89=40-series
cmake --build build-cuda

# Sanity: per-layer K correlation on real RoPE'd K, ship vs sqfree.
./build/bin/sp-engine prefill --model dolphin_1b.gguf "The quick brown fox"
./build/bin/sp-engine prefill --sqfree --spinor --model dolphin_1b.gguf "..."

# Greedy generation through the optimised single-token decode + cache.
./build/bin/sp-engine chat --model dolphin_1b.gguf --n-predict 16 "The quick brown fox"

# Drift measurement: compressed PPL − baseline PPL on wikitext.
# --model-preset auto resolves arch → preset from the model-pack registry.
SP_ENGINE_BACKEND=gpu ./build-cuda/bin/sp-engine perplexity --cache \
    --model qwen3-8b-q8.gguf --model-preset auto \
    --ctx 2048 --chunks 8 data/wiki.test.raw

# Long-context decode-chain stability via Cauchy reset (Mertens schedule):
SP_ENGINE_BACKEND=gpu ./build-cuda/bin/sp-engine perplexity --cache \
    --model qwen3-8b-q8.gguf --cauchy-mode 2 \
    --ctx 4096 --chunks 8 data/wiki.test.raw

# Save compressed cache to disk after generation
./build/bin/sp-engine chat --model qwen3-8b-q8.gguf --save-cache /tmp/session1 "Hello"

# Resume from saved cache
./build/bin/sp-engine chat --model qwen3-8b-q8.gguf --load-cache /tmp/session1 "Continue"
```
Full verb list and stage status in [docs/PRIME-ENGINE.md](docs/PRIME-ENGINE.md).
Model-pack presets for auto-resolution: [docs/MODEL-PACK.md](docs/MODEL-PACK.md).
Cauchy design + ablation: [docs/CAUCHY-RESET.md](docs/CAUCHY-RESET.md).

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
Note: 3,3,3,3,3 on the Knight skeleton was retracted in v1.03
(catastrophic post-fix — see the longer note below the aggressive-path
results table); the effective Pareto point is 5,4,4,4,5.

### ComfyUI (Video / Image / Audio / TTS)

All 16 nodes are installed via the `shannon-prime-comfyui` custom node package. In ComfyUI workflows:

- **Video**: Model Loader → ShannonPrimeWanBlockSkip → ShannonPrimeWanTurbo → KSampler → ShannonPrimeWanCacheFlush
- **Image**: Model Loader → ShannonPrimeFluxBlockSkip → KSampler → ShannonPrimeFluxCacheFlush
- **Audio**: Model Loader → ShannonPrimeAudioBlockSkip → Generate → ShannonPrimeAudioCacheFlush
- **TTS**: VoxtralModelLoader → ShannonPrimeVoxtralKVCache → VoxtralGenerate → ShannonPrimeVoxtralCacheFlush

See the [shannon-prime-comfyui README](https://github.com/nihilistau/shannon-prime-comfyui) for full node documentation and workflow examples.

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
| Vulkan | 4+1 | CPU fallback, batch ops, memory reporting, dual-GPU cross-device (RTX 2060 + Intel UHD, correlation 1.0000) |
| CUDA basic | 7 | Multi-layer, multi-head, batch K+V roundtrip |
| CUDA advanced | 24 | sqfree GPU, spinor, batch read, hierarchical, cold storage, stress |
| llama.cpp | 7 | Integration layer, env-var driven config |
| PyTorch | 31 | VHT2 self-inverse at p-of-2 and sqfree, full pipeline |
| ComfyUI | 25 | Wan 2.1/2.2 MoE, expert switching, 50×40 simulation |
| Sqfree+spinor | 69 | Knight CSR, residual quant, spinor no-regress, scaling law |
| **Total** | **187/188** | *(1 known synthetic-K flake — see CLAUDE.md)* |

## Settings reference

### llama.cpp hook — ship path (env vars)
| Variable | Default | Description |
|----------|---------|-------------|
| `SHANNON_PRIME_ENABLED` | 0 | Enable the VHT2 shadow cache |
| `SHANNON_PRIME_K_BITS` | 5,5,4,3 | K band bit allocation |
| `SHANNON_PRIME_V_BITS` | 3 | V flat bit allocation |
| `SHANNON_PRIME_MOBIUS` | 1 | Möbius squarefree-first reorder (K only) |
| `SHANNON_PRIME_VERBOSE` | 0 | Print config + init |

### llama.cpp hook — sqfree+spinor (opt-in env vars)
| Variable | Default | Description |
|----------|---------|-------------|
| `SHANNON_PRIME_SQFREE` | 0 | Enable sqfree prime-Hartley basis (pads hd) |
| `SHANNON_PRIME_SPINOR` | 0 | Enable SU(2) sheet bit (auto-enables SQFREE) |
| `SHANNON_PRIME_RESIDUAL_BITS` | 3 | Residual depth (1–4; 3 is the Pareto point) |
| `SHANNON_PRIME_SK_FRAC` | 0.75 | Skeleton fraction of pad_dim |

### shannon-prime-engine — runtime env vars
| Variable | Default | Effect |
|----------|---------|--------|
| `SP_ENGINE_BACKEND` | *(unset)* | `gpu` / `cuda` / `vulkan` — route forward + weights + cache through the named GPU backend. Unset = CPU (mmap weights). |
| `SHANNON_PRIME_GPU_CACHE` | 1 | When backend is GPU, route `KvCache` to the GPU-resident path (`create_gpu`). Set `0` for the host-cache A/B diagnostic. |
| `SHANNON_PRIME_SYNC_CALIB_TO_GPU` | 0 | Upload calibrated variance-ranked mask (ship `d_mobius_order` + sqfree Knight CSR) to the GPU cache after `calibrate_end`. Measured +0.21 PPL asymmetric regression on Qwen3-8B ship; keep off unless investigating. **Note (2026-04-22):** the fp16 scale fix in band quantize (both CUDA and CPU now recompute `inv_scale` from the stored fp16 value) eliminates the quantize/dequantize asymmetry that variance-ranking amplified. The +0.21 PPL regression needs re-measurement with the fix applied. |
| `SHANNON_PRIME_SQFREE_NO_BATCH` | 0 | Fall back to per-vec sqfree GPU read (diagnostic — batched path is correct on kv_smoke, real-model PPL still validating). |
| `SHANNON_PRIME_NO_CALIBRATE` | 0 | Skip auto-calibration — for A/B against a static mask on the same run. |
| `SP_CALIBRATE` | 0 | Force the `prefill` CLI verb to calibrate before writing. |
| `SP_ENGINE_SAVE_CACHE` | *(unset)* | Prefix path for disk cache save. Per-layer files `{prefix}.L{n}.bin`; hier path also saves `{prefix}.hier_w.bin`. |
| `SP_ENGINE_LOAD_CACHE` | *(unset)* | Prefix path for disk cache load. Resumes from a previously saved VHT2 v2 binary session. |
| `SP_ENGINE_COLD_MB` | *(unset)* | Cold-storage pinned-RAM budget in MB. Enables hot/cold tiered eviction (GPU VRAM → CPU pinned RAM → disk). |
| `SP_ENGINE_EVICT_KEEP` | *(unset)* | Keep N most-recent positions in GPU VRAM; older positions spill to cold storage. |
| `SP_DEBUG_DECODE` | 0 | Print per-layer K/X correlation vs `forward_full` at each decode step. |

### shannon-prime-engine — CLI flags (shared across kv_smoke / prefill / chat / perplexity / cache_ppl)
| Flag | Effect |
|------|--------|
| *(none)* | **Ship path** — VHT2 + Möbius + 4-band banded quant. |
| `--sqfree` | Sqfree prime-Hartley + Knight skeleton (L/2) + 3-bit residual. |
| `--sqfree --spinor` | Adds SU(2) sheet-bit correction (+0.008–0.010 K_corr). |
| `--hierarchical` | Hierarchical Vilenkin predictor, 9% skeleton + calibrated ridge map + 2-bit residual. |
| `--model-preset auto\|off\|<name>` | Apply model-pack overlay (`auto` resolves from arch; see docs/MODEL-PACK.md). |
| `--k-bits CSV` / `--v-bits CSV` | Per-band bit allocation override. |
| `--residual-bits N` | Sqfree residual depth (1–4, default 3). |
| `--hier-level N` / `--hier-res-bits N` / `--hier-skel-bits CSV` | Hierarchical tuning knobs. |
| `--no-mobius` | Disable Möbius reorder (ship path). |
| `--ctx N` / `--chunks N` | Per-chunk context + number of chunks for perplexity/cache_ppl. |
| `--cache` | (perplexity verb) Run decode-chain through the compressed cache — this is the drift measurement. |
| `--cauchy-mode N` | 0 = off, 1 = fixed-N, 2 = dynamic Mertens (shipping default when enabled). |
| `--cauchy-fixed-n N` / `--cauchy-cooldown N` / `--cauchy-warmup N` | Cauchy schedule knobs (default 512 / 64 / 64). |
| `--cauchy-use-ricci` / `--cauchy-ricci-only` / `--cauchy-mertens-only` | Opt-in Ricci sentinel / ablation gates. |
| `--save-cache <prefix>` / `--load-cache <prefix>` | Disk serialization. VHT2 v2 binary format with 64-byte header (magic `0x56485432`). Per-layer files `{prefix}.L{n}.bin`; hier path also writes `{prefix}.hier_w.bin`. |
| `--cold` / `--cold-mb N` / `--evict-keep N` | Hot/cold tiered storage. GPU VRAM → CPU pinned RAM → disk. `--cold` enables, `--cold-mb` caps pinned RAM budget, `--evict-keep` keeps N recent positions in VRAM. Ring-buffer wrap-around for capped CPU allocation. |
| `--n-gpu-layers N` | Partial GPU offload via `ggml_backend_sched_t` scheduler bridge. Offloads first N transformer blocks to GPU, rest stays CPU. Cross-backend copies handled transparently. 0 = all CPU. |
| `--pe-mode standard\|primepe\|primepe_alibi\|alibi` + `--pe-alpha F` + `--pe-tier 0\|1` | PrimePE-RoPE-ALiBi positional-encoding family. |

## License

**AGPLv3** for open-source, academic, and non-proprietary use.
Everyone can use it and benefit. Derivative works must share alike.

**Dual License** — the primary goal is that the work belongs to the
commons and is protected from closure. A commercial license is
available for proprietary integration.

## Contact

Email: raydaniels@gmail.com

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
