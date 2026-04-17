# CLAUDE.md — Project Instructions for AI Agents

## What This Project Is

Shannon-Prime is a KV cache compression system for transformers. It exploits
the spectral structure RoPE imprints on K vectors via the **Vilenkin-Hartley
Transform (VHT2)** — a staged, orthonormal generalization of the Walsh-Hadamard
Transform that extends cleanly to non-power-of-2 dimensions. VHT2 composes
with Möbius-ordered coefficient reordering and per-band quantization to deliver
3.4–3.8× KV compression at <1.25% PPL cost in the ship configuration, or up
to 3.3× at equivalent quality on Q8+ backbones via the sqfree+spinor aggressive
variant of the same pipeline.

**VHT2 with p = 2 is the Walsh-Hadamard Transform.** Everywhere in the code —
torch reference, C core, GPU kernels, integrations — the single transform is
VHT2. On power-of-2 head dimensions it reduces to the WHT butterfly scaled by
1/√2 per stage (self-inverse, so no 1/N on the inverse). On squarefree-padded
dimensions it factors across small primes {2,3,5,7,11}, giving the same
spectral structure that underpins the Möbius predictor.

**This is NOT theory. Everything here is implemented and tested.**

- 187/188 tests across 8 suites (the single synthetic-K-pipeline flake is a
  known random-data edge case — real K vectors with RoPE hit 0.997+)
- Core math validated: VHT2 self-inverse round-trip on power-of-2 and sqfree
  dims, Möbius function, banded quant, sqfree prime-Hartley, Knight CSR mask,
  N-bit residual, spinor sheet bit, scaling law
- Production-validated on desktop (Qwen3-8B), mobile (Dolphin 1B / S22 Ultra),
  and video gen (Wan 2.2 TI2V-5B / ComfyUI)
- Ship config: 3.4–3.8× KV compression at <1.25% PPL cost
- Aggressive config (sqfree+spinor on Q8+ @ hd=128): 3.3× at MOBIUS-default quality

## Two Configurations of the Same Pipeline

### Ship path (default, production)
```
Write: raw KV → VHT2 forward → Möbius reorder (K only, self-attn)
       → Band quantize (5/5/4/3 for K, flat 3 for V) → Store
Read:  Load → Band dequantize (non-finite scale ⇒ zero band)
       → Möbius unreorder → VHT2 forward (= inverse) → KV
```
K (post-RoPE) concentrates 80%+ energy in the first VHT2 bands; V (content)
spreads uniformly. K gets 4-band Möbius-ordered quantization, V gets flat 3-bit.
Validated, ship-safe, works on all power-of-2 head_dim.

### Sqfree+spinor aggressive path (opt-in, Q8+ backbones)
```
Write: raw KV → sqfree_pad (hd → pad_dim) → VHT2 forward
       → Knight skeleton extract → Band quantize skeleton
       → Möbius CSR predict residual → Quantize residual (3-bit)
       → Spinor sheet bit → Store
Read:  Load → Band dequant skeleton → Möbius predict → Spinor correct
       → Dequant residual → Scatter → VHT2 forward → sqfree_unpad → KV
```
Uses the same VHT2 transform on a sqfree-padded dimension so the Möbius
predictor gets r = 0.40–0.58 (vs ≈ 0 on pure power-of-2). The spinor bit is
the SU(2) double-cover correction for the causal-mask boundary. On Qwen3-8B Q8
hd=128: PPL 7.32 @ 3.3× (matches MOBIUS default 7.31 @ 2.6×, +27% compression).

**Q8+ feature.** On Q3 and below, scaling-law γ=1.5 on weight precision says
the spinor's 1-bit correction is washed out by W-matrix quantization noise;
use the VHT2 ship path for heavily quantized models.

## Build and Test

```bash
make test-all     # MUST pass before any commit. 187/188 across 8 suites.
make test-core    # C core math (31 tests)
make test-torch   # PyTorch backend (31 tests)
make test-sqfree  # Sqfree + spinor path (PyTorch, 69 tests)
```

## Rules

### DO:
- Run `make test-all` after every change. Tests must pass.
- Read the relevant test file before modifying any source file.
- Add new tests when adding new functionality.
- Keep the copyright header on all new files (see COPYRIGHT_HEADER.txt).
- Use the scaling law (`tools/sp_scaling_law.py`) as a pre-bench filter —
  skip configs with predicted ΔPPL > 5%.

### DO NOT:
- Rewrite core/shannon_prime.c without understanding why each function exists.
  The VHT2, Möbius, and banded quant implementations are validated against
  the papers.
- Change test thresholds without justification. They are set to catch real bugs
  while tolerating statistical variation on random test data. Real K vectors
  with RoPE structure achieve 0.997+ correlation; random data occasionally
  hits 0.984.
- Treat any part of this as theoretical or unimplemented. If you find yourself
  writing "this would need to be..." — check if it's already in the codebase.
- Re-introduce a blanket output NaN guard. The read path now produces finite
  values by construction — the non-finite scale check in `sp_band_dequantize`
  is the one surviving guard and lives at the root cause. If a new code path
  regresses into producing NaN, **fix the origin**, don't add another clamp.
- Apply the sqfree+spinor path to Q4 or lower models. Use the VHT2 ship path.

### CRITICAL INVARIANTS (never violate):
1. **VHT2 is self-inverse**: VHT2(VHT2(x)) = x. Error < 1e-5 at power-of-2,
   < 1e-4 at sqfree dims. There is **no 1/N** on the inverse — each p-stage
   is already normalised by 1/√p.
2. **Möbius reorder + unreorder = identity**. Error exactly 0.
3. **3-bit floor on any band**. 2-bit is catastrophic. Never go below 3.
4. **V vectors get flat quantization** (no banding) in self-attention. In
   cross-attention both K and V get matching banded allocation.
5. **K gets Möbius reorder, V does not (self-attention). Both get reorder
   (cross-attention).**
6. **Möbius mask is cross-platform invariant.** K corr 0.997 on both hd=128
   and hd=64.
7. **Sqfree pad dimensions are deterministic**: hd=64 → 66, hd=128 → 154,
   hd=256 → 330.
8. **Knight mask CSR** is built once per (pad_dim, sk_k). Skeleton and
   residual index sets must be disjoint; CSR offsets monotone non-decreasing.
9. **Residual quantization saturates at 3 bits** (~8 meaningful levels).
   4-bit is flat against 3-bit; 1-bit is catastrophic (PPL 700+).
10. **Spinor sheet bits**: ~50% set rate on random data (sanity). The correction
    is the SU(2) double cover — two valid signs for the predictor, one bit to
    pick the right sheet.

## Backend Notes (GPU kernels)

All backends now run the literal VHT2 with 1/√p per stage (self-inverse, no
1/N on reconstruction). Per-band amax normalisation in BandedQuantizer makes
the spectrum magnitudes invariant to the transform scale, so this cleanup is
behaviour-neutral on the output side and simplifies cross-backend debugging.

Vulkan compute is currently routed through the CPU staged VHT2 inside the
host wrappers (`vk_compress_one` / `vk_decompress_one`) because the GPU
dispatch hits `VK_ERROR_DEVICE_LOST` at the first `vkWaitForFences` on
RTX 2060; pipelines are still created so `test_vulkan` exercises the full
init path, only the submit is skipped. Debugging the shader hang is tracked
separately and needs RenderDoc on the target GPU.

## WHY These Specific Choices (Do Not "Optimize" Without Understanding)

- **5/5/4/3** is not arbitrary. It mirrors measured VHT2 energy decay. The
  correct optimization direction is ADAPTIVE allocation (measure energy,
  allocate bits proportionally), not trying different fixed configs.

- **5/4/4/4/5** for the sqfree path is also not arbitrary. It's the
  torus-aligned 5-band allocation where edge bands (DC + high-freq structural
  anchors) get more bits than the middle interpolation bands, matching the
  variance profile of the Knight-ranked skeleton.

- **3-bit residual** is the Shannon saturation point. The μ-inversion residual
  distribution has ~8 meaningful quantization levels. Measured:
  1→2 bit = 29× PPL drop, 2→3 bit = 2.1× drop, 3→4 bit = flat.

- The **spinor sheet bit** captures systematic sign errors in the Möbius
  predictor at the causal-mask boundary where Z/7Z and Z/11Z phase winds get
  truncated. It is NOT a generic improvement — it specifically corrects the
  half-Möbius "shredding" and only works when the basis preserves divisibility
  structure (sqfree VHT2, not pure power-of-2 VHT2).

- **No output NaN guard.** Previously a blanket `nan_to_num + clamp` ran on
  every read path as defense-in-depth for aggressive bit configs (4/4/3 and
  below on hd=64). It was removed in favour of a root-cause guard: if the
  fp16 per-band scale round-trips to a non-finite value (the real origin of
  the cascading NaN), `sp_band_dequantize` zeros the band. Ship config
  (5/4/4) produces finite outputs without this branch ever firing; aggressive
  configs now surface any remaining NaN instead of silently masking it.

- **Cross-attention in Wan** models has NO RoPE on K/V. Both K and V get
  Möbius reorder and matching bit allocation. This is DIFFERENT from
  self-attention where only K gets reorder and V gets flat quant.

- **Measurement noise** on Vulkan fp16 (RTX 2060) is ±5 PPL at 1-chunk
  wiki.test in the 12–20 range. Tight ladder comparisons (< 0.5 PPL) are
  noise. Use ≥ 4 chunks and report median of 3–5 runs for sub-1-PPL claims.

## The Scaling Law — USE IT

```
log(PPL / PPL_base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
```

Empirical, fit across 9 configurations, ±20%:
- **α=2**: quadratic in K error (bilinearity of attention)
- **β=1.1**: sub-linear in params (bigger models absorb K error)
- **γ=1.5**: super-linear in weight bits (Q4 amplifies error 2.8× vs Q8)

Safe K_corr floors (3% PPL budget):
- Dolphin 1B Q8: 0.988
- Qwen3-8B Q8: 0.962
- Qwen3-8B Q3: 0.974
- Llama-70B Q8: 0.927
- Wan 2.2 14B bf16: 0.914

**Before running any bench**, compute `predicted_ppl_ratio(k_corr, params_b, bits)`.
If predicted ΔPPL > 5%, don't bother. Cuts sweep time ~10×.

```python
from tools.sp_scaling_law import predicted_ppl_ratio, is_pareto_viable
ratio = predicted_ppl_ratio(k_corr=0.97, params_b=8.0, bits=8)
print(f"Predicted: +{(ratio-1)*100:.2f}% PPL")  # +1.0%
```

## Architecture

```
core/                      Backend-agnostic C: VHT2, Möbius, banded quant,
                           sqfree pad, Knight mask CSR, residual quant,
                           spinor, shadow cache
backends/
  cuda/                    NVIDIA GPU kernels (butterfly at p=2, band quant,
                           Möbius CSR predict, spinor extract/reconstruct)
  vulkan/                  Vulkan compute shaders + CPU fallback
                           (vilenkin.comp, knight_predict.comp,
                            mobius_reorder.comp, band_quantize.comp)
  torch/                   Pure PyTorch reference (single vht2 function,
                           plus sqfree+spinor reference path)
  adreno/                  ARM NEON (3 tiers), Hexagon HVX stubs,
                           big.LITTLE affinity
tools/
  shannon_prime_llama            llama.cpp integration (ship path)
  shannon_prime_llama_sqfree     llama.cpp sqfree+spinor hook
  shannon_prime_comfyui          ComfyUI integration (Wan 2.1/2.2)
  shannon_prime_comfyui_sqfree   ComfyUI sqfree+spinor variant
  sp_inject_freqs.py             GGUF frequency injection
  sp_compress_model.py           Weight spectral analysis
  sp_benchmark.py                Compression benchmark
  sp_scaling_law.py              K-corr → PPL design rule (pre-bench filter)
tests/                     8 test suites
docs/                      Full documentation
```

## Key Files

| File | What It Does | Test Coverage |
|------|-------------|---------------|
| core/shannon_prime.h | Full public API | test_core.c |
| core/shannon_prime.c | C reference (VHT2 only, no WHT aliases) | test_core.c |
| core/shannon_prime_sqfree.c | Sqfree + spinor C implementation | test_core.c (sqfree section) |
| backends/torch/shannon_prime_torch.py | PyTorch (vht2, ShadowCache) | test_torch.py |
| backends/torch/shannon_prime_sqfree.py | PyTorch sqfree+spinor path | test_sqfree.py |
| backends/cuda/shannon_prime_cuda.cu | CUDA VHT2 p=2 butterfly + sqfree dispatch | test_cuda.c |
| backends/cuda/shannon_prime_sqfree.cu | CUDA Vilenkin+spinor kernels | test_cuda.c (sqfree section) |
| backends/vulkan/shannon_prime_vulkan.c | Vulkan + CPU fallback | test_vulkan.c |
| backends/vulkan/shaders/vilenkin.comp | Vulkan VHT2 shader | test_vulkan.c |
| backends/vulkan/shaders/knight_predict.comp | Vulkan CSR+spinor shader | test_vulkan.c |
| backends/adreno/shannon_prime_adreno.c | Mobile backend (NEON) | test_adreno.c |
| tools/shannon_prime_llama.c | llama.cpp ship-path hook | test_integration.c |
| tools/shannon_prime_llama_sqfree.c | llama.cpp sqfree hook | test_integration.c |
| tools/shannon_prime_comfyui.py | ComfyUI Wan cache (ship) | test_comfyui.py |
| tools/shannon_prime_comfyui_sqfree.py | ComfyUI Wan cache (sqfree) | test_comfyui.py |
| tools/sp_scaling_law.py | K-corr → PPL design rule | test_sqfree.py |

## Environment Variables

### Ship path (default)
| Variable | Default | Description |
|----------|---------|-------------|
| `SHANNON_PRIME_ENABLED` | 0 | Enable the VHT2 shadow cache hook |
| `SHANNON_PRIME_K_BITS` | 5,5,4,3 | K band bit allocation |
| `SHANNON_PRIME_V_BITS` | 3 | V flat bit allocation |
| `SHANNON_PRIME_MOBIUS` | 1 | Möbius squarefree-first reorder (K) |
| `SHANNON_PRIME_VERBOSE` | 0 | Print config and init |

### Sqfree+spinor aggressive path (opt-in)
| Variable | Default | Description |
|----------|---------|-------------|
| `SHANNON_PRIME_SQFREE` | 0 | Enable sqfree prime-Hartley basis |
| `SHANNON_PRIME_SPINOR` | 0 | Enable spinor sheet bit (auto-enables SQFREE) |
| `SHANNON_PRIME_RESIDUAL_BITS` | 3 | Residual quantization depth (1–4) |
| `SHANNON_PRIME_SK_FRAC` | 0.75 | Skeleton fraction of pad_dim |

When `SQFREE=1`:
- K_BITS default changes to 5,4,4,4,5 (5-band torus-aligned)
- V_BITS gets same allocation as K (cross-attn style)
- head_dim pads to the next sqfree multiple: 64→66, 128→154, 256→330

## Validated Results

### Ship path
| Config | K Corr | Compression | PPL Impact |
|--------|--------|-------------|------------|
| 5/5/4/3 + Möbius (ship) | 0.992+ | 3.4× | −0.04% (better) |
| 4/4/4/3 | 0.990 | 3.8× | +0.39% |
| 3/3/3/3 (floor) | 0.976 | 4.6× | +3.90% |

### Sqfree+spinor path (Qwen3-8B Q8 hd=128)
| Config | PPL | K Corr | Compression |
|--------|-----|--------|-------------|
| MOBIUS default 5/5/5/5/5 | 7.31 | 0.999 | 2.6× |
| K+μ+3bit+spinor 5/4/4/4/5 | 7.30 | 0.988 | 2.8× |
| K+μ+3bit+spinor 3/3/3/3/3 | 7.32 | 0.972 | 3.3× |

### Scaling law fit
All 9 calibration points within ±20% of K = 4700.

## Papers

Reference documents in the repo root:
- Position_Is_Arithmetic_v8.md — PE theory, falsification suite,
  ZetaZeroPredictor, scaling law as independent confirmation
- KV_Cache_Is_A_View_v2.md — KV compression, VHT2, Vilenkin, Möbius
  basis-dependency, spinor, scaling law, precision sensitivity, Knight closeout
- multiplicative_lattice_combined.md — Unified synthesis: eight tests, one scaling law
- Shannon-Prime_Targeted_Analysis.md — Competitive landscape, recommendations
