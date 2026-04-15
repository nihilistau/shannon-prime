# CLAUDE.md — Project Instructions for AI Agents

## What This Project Is

Shannon-Prime VHT2 is a KV cache compression system for transformers. It exploits
the spectral structure RoPE imprints on K vectors via Walsh-Hadamard Transform,
Möbius-ordered coefficient reordering, and per-band quantization.

**This is NOT theory. Everything here is implemented and tested.**

- 109 tests across 6 suites, all passing
- Core math validated: WHT round-trip, Möbius function, banded quant, Vilenkin
- Production-validated on desktop (Qwen3-8B), mobile (Dolphin 1B / S22 Ultra),
  and video gen (Wan 2.2 14B / ComfyUI)
- Ship config: 3.4–3.8× KV compression at <1.25% PPL cost

## Build and Test

```bash
make test-all    # MUST pass before any commit. Currently 109/109.
make test-core   # C core math only (31 tests)
make test-torch  # PyTorch backend (28 tests)
```

## Rules

### DO:
- Run `make test-all` after every change. All tests must pass.
- Read the relevant test file before modifying any source file.
- Read docs/Shannon-Prime.md for the theory behind any math you're touching.
- Add new tests when adding new functionality.
- Keep the copyright header on all new files (see COPYRIGHT_HEADER.txt).

### DO NOT:
- Rewrite core/shannon_prime.c without understanding why each function exists.
  The WHT, Möbius, and banded quant implementations are validated against the papers.
- Change test thresholds without justification. They are set to catch real bugs
  while tolerating statistical variation on random test data. Real K vectors with
  RoPE structure achieve 0.997+ correlation; random data occasionally hits 0.984.
- Treat any part of this as theoretical or unimplemented. If you find yourself
  writing "this would need to be..." — check if it's already in the codebase.
- Copy code from the old repos (Shannon-Prime-Private, Shannon-Prime-ComfyUI, etc.).
  This is a clean room build from the papers.
- Remove or weaken the NaN guard. It's defense-in-depth for aggressive bit configs.

### CRITICAL INVARIANTS (never violate):
1. WHT is self-inverse: WHT(WHT(x)) = N·x. Error must be <1e-5.
2. Vilenkin basis is orthonormal: V·V = I (NOT N·I). Do not divide by N in inverse.
3. Möbius reorder + unreorder = identity. Error must be exactly 0.
4. 3-bit floor on any band. 2-bit is catastrophic. Never go below 3.
5. V vectors get flat quantization (no banding). Flat beats banded for V — no exceptions.
6. K gets Möbius reorder, V does not (self-attention). Both get reorder (cross-attention).
7. The Möbius mask is cross-platform invariant. K corr 0.997 on both hd=128 and hd=64.

## Architecture

```
core/                 Backend-agnostic C: WHT, Möbius, banded quant, Vilenkin, shadow cache
backends/
  cuda/               NVIDIA GPU kernels (WHT butterfly, band quant, shadow cache)
  vulkan/             Vulkan compute shaders + CPU fallback host code
  torch/              Pure PyTorch reference implementation
  adreno/             ARM NEON (3 tiers), Hexagon HVX stubs, big.LITTLE affinity
tools/
  shannon_prime_llama   llama.cpp integration layer (C)
  shannon_prime_comfyui ComfyUI integration (Python, Wan 2.1/2.2 aware)
tests/                 6 test suites
docs/                  8 documentation files
```

## Key Files

| File | What It Does | Test Coverage |
|------|-------------|---------------|
| core/shannon_prime.h | Full public API | test_core.c (31 tests) |
| core/shannon_prime.c | Reference implementation | test_core.c |
| backends/torch/shannon_prime_torch.py | PyTorch impl | test_torch.py (28 tests) |
| backends/adreno/shannon_prime_adreno.c | Mobile backend | test_adreno.c (14 tests) |
| backends/vulkan/shannon_prime_vulkan.c | Vulkan + fallback | test_vulkan.c (4 tests) |
| tools/shannon_prime_llama.c | llama.cpp hooks | test_integration.c (7 tests) |
| tools/shannon_prime_comfyui.py | ComfyUI Wan cache | test_comfyui.py (25 tests) |

## Compression Pipeline

```
Write: raw KV → WHT → Möbius reorder → Band quantize (5/5/4/3) → Store
Read:  Load → Band dequantize → Möbius unreorder → Inverse WHT → KV
```

## Papers

The theory is in the project files (read-only reference):
- Position_Is_Arithmetic_v7.md — PE theory, falsification suite, ZetaZeroPredictor
- KV_Cache_Is_A_View_v1.md — KV compression, VHT2, Vilenkin, Möbius, production results
- Shannon-Prime_Targeted_Analysis.md — Competitive landscape, walls, recommendations
