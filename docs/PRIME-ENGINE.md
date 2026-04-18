# shannon-prime-engine

**Standalone reference inference engine that owns the compressed KV
data path end-to-end.**

Sibling repository: [github.com/nihilistau/shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine)
(local checkout: `D:/F/shannon-prime-repos/shannon-prime-engine`).

## What it is

`shannon-prime-engine` is the "own the data path" sibling of
`shannon-prime-llama`. Both share 100% of the core math via the
`lib/shannon-prime/` git submodule. They serve different user
stories:

| Repo | Approach | Trade-off |
|---|---|---|
| `shannon-prime-llama` | Patch into llama.cpp's existing decode loop via a post-decode hook. | Inherits 30+ model architectures, but the hook surface itself has been a source of integration bugs (V-pipeline plumbing loss, GPU/CPU blob mismatch, vkCmdCopyBuffer barrier miss, etc.). Compression happens in a decompress→attention→recompress round-trip on every decode rather than as a property of the storage layout. |
| `shannon-prime-engine` | Build a complete inference binary where the KV layout is owned by us; compression is on the write path by construction. | Stack of ~900 lines of engine code we control, currently scoped to the llama family (Llama, Qwen2/3, Phi3, Mistral, Granite). No host-side hook surface to bug-hunt; readback fidelity is intrinsic to the cache. |

The engine's role in the wider project: it is the **bug-free
reference measurement surface** for any compression result the
shannon-prime math claims. Numbers measured through the llama.cpp
hook should be re-measured here before being treated as
publication-grade.

## Status as of stage 5b

Pre-alpha but live. The full forward pass (token-embedding →
RMSNorm → attention with PrimePE-RoPE + optional ALiBi → SwiGLU
FFN → output_norm → output projection → logits) runs end-to-end on
both Dolphin-1B (Llama-3 family, hd=64) and Qwen3-8B-Q8 (hd=128
GQA-4×). Greedy generation is wired and produces clean
continuations through the `chat --naive` path; the optimised decode
path with cache-fed past K/V is shipped but has a known
correctness regression (see Stage 5b status below).

| Stage | Verb | What it proves | Status |
|---|---|---|---|
| 3a | `embed` | ggml graph + backend init + token-embedding lookup. | ✓ |
| 3b | `block1` | One transformer block: norm + attn + FFN + residual. | ✓ |
| 3c | `logits` | All `n_layer` blocks + output head → real logits. | ✓ |
| 3d | `logits --pe-mode primepe` | PrimePE-RoPE-ALiBi (composite/prime-tiered lattice, alpha-blended). | ✓ |
| 4   | `kv_smoke` | KvCache wrapper around `sp_shadow_cache_t` / `sp_sqfree_cache_t`. | ✓ |
| 5a  | `prefill` | Real RoPE'd K from prefill compressed through KvCache; per-layer K/V correlation reported. | ✓ |
| 5b  | `chat` | Stateful prefill + decode loop (naive path correct, optimised decode path has a logit-magnitude regression). | partial |
| 6   | (planned) | Persistent backend KV tensors; debug + ship optimised decode; perplexity verb. | — |
| 7   | (planned) | CUDA / Vulkan backend selection; release packaging. | — |

## Why a separate engine — the four-bug case

A representative session in the llama-cpp-sp (hook) integration surfaced four host-side
integration bugs in a single afternoon, none of which were
compression-math bugs:

1. GPU-resident KV tensor `cudaMemcpy` segfault (host pointer dereferenced as device).
2. Vulkan `vkCmdCopyBuffer` synchronisation barrier miss between
   compute and transfer queues.
3. CPU/GPU blob format mismatch on the same compression config (CPU
   wrote interleaved heads, GPU expected planar).
4. V-pipeline plumbing loss: a refactor dropped the V tensor on the
   prefill path, masquerading as "V correlation regression" for two
   sessions before it was caught.

Each could plausibly have been read as a math bug in `shannon-prime`
itself — they touch the same code paths the math relies on. The
engine deliberately removes that surface area: if the read path
disagrees with the write path, the bug has to be in
`lib/shannon-prime/` or in our ~900 lines, not in the integration
glue. Stage 5a's per-layer K correlation report is the first
artefact that closes that loop end-to-end on real RoPE'd K.

## What it shares with the math repo

Vendored as a git submodule at `lib/shannon-prime/`. The engine
links the C core directly:

```cmake
add_library(shannon_prime_core STATIC
    ${SP_CORE_DIR}/core/shannon_prime.c
    ${SP_CORE_DIR}/core/shannon_prime_sqfree.c)
```

This means **every shannon-prime invariant the test suite enforces
is enforced inside the engine binary too**: VHT2 self-inverse,
Möbius identity round-trip, sqfree pad dimensions (64→66,
128→154, 256→330), the 3-bit floor on band quantisation, spinor's
~50% sheet-bit set rate. The engine never re-implements the math —
it just calls in.

## PrimePE-RoPE-ALiBi (stage 3d)

The engine's positional encoding is wired in from the start as a
configurable family rather than retrofitted. The `PeMode` enum
selects:

```
Standard          geometric RoPE, no ALiBi (default; identical to llama.cpp)
PrimePe           lattice-drawn freq_factors, alpha-blended with geometric
PrimePeAlibi      PrimePe + per-head ALiBi slopes
AlibiOnly         standard RoPE + ALiBi (ablation)
```

Lattice math is the canonical three-tier composite/prime allocation
from `prime_rope.h`:

* **local tier** (dims 0..n/3): small numbers (2..64) — high freq
* **mid tier**   (dims n/3..2n/3): medium (64..1024)
* **long tier**  (dims 2n/3..n): large (1024..8192) — low freq

Composites (default, `--pe-tier 0`) are reducible lattices —
primes emerge as the compression invariant. Primes
(`--pe-tier 1`) are the ablation arm. Both are log-linearly
mapped into the geometric envelope the pretrained model expects;
factors are perturbations around 1.0 rather than full
substitutions.

Identity property: `--pe-alpha 0` is byte-identical to
`--pe-mode standard`. Verified against Dolphin-1B "Hello world".

## Compressed KV cache (stage 4–5a)

`KvCache` wraps either `sp_shadow_cache_t` (ship) or
`sp_sqfree_cache_t` (aggressive) behind a single C++ class. The
shadow cache requires the caller to allocate the per-(layer, head)
storage by design; the wrapper does this, the sqfree variant
allocates internally. Layout matches what
`ggml_backend_tensor_get` returns for a
`[head_dim, n_head_kv, n_tokens]` tensor, so write/read can plug
straight into the prefill path without transposes.

Measured on real RoPE'd K (CLAUDE.md ship-target: K corr ≥ 0.992):

| Model | Path | K_corr (mean) | V_corr | Compression |
|---|---|---|---|---|
| Dolphin-1B (hd=64, n_layer=16) | ship | 0.9941 | 0.9712 | 3.76× |
| Dolphin-1B | sqfree (pad 66) | 0.9768 | 0.9484 | 3.76× |
| Dolphin-1B | sqfree+spinor | 0.9869 | 0.9601 | 3.76× |
| Qwen3-8B-Q8 (hd=128, n_layer=36) | ship | 0.9934 | 0.9691 | 4.06× |

Spinor's +0.008–0.010 K-corr lift on the Knight skeleton matches
the value documented in CLAUDE.md.

## Stage 5b status — chat works, optimised decode pending

The stateful API
(`bind_cache` / `prefill` / `decode` / `kv_pos`) is in. The
`chat` verb does greedy generation. Two paths:

* **`chat --naive`** — re-runs `forward_full` over the running
  prefix each step. Slow (O(n²) total) but provably correct.
  Smoke test:
  > `./sp-engine chat --model dolphin_1b.gguf --n-predict 16 --naive "The quick brown fox"`  
  > → `"The quick brown fox jumps over the lazy dog. Here's a breakdown of the sentence: - "`
* **`chat`** (default) — uses the optimised single-token `decode`
  graph that reads past K/V from the cache as input tensors,
  concats with the freshly-projected new K/V along the position
  axis, runs attention over `[past_n + 1]` keys. Currently
  produces output logits at ~half the magnitude of `forward_full`
  at the same position, even with a near-lossless 8-bit cache.
  Structural checks all pass (F32 dtypes, `[hd, n_kv, *]` shapes,
  `ggml_concat` dim 2, GQA index math matches Llama-3 head→kv_head
  mapping). Bug is in `build_block_decode`'s graph; localising it
  needs a per-layer K diff against `forward_full` at the same
  position. Tracked in source as a `FIXME(stage 5b)`.

## Building (Windows / MinGW)

```bash
git clone --recursive https://github.com/nihilistau/shannon-prime-engine
cd shannon-prime-engine
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/sp-engine.exe banner
```

If you see exit code 127, copy three MinGW runtime DLLs next to
the binary (one-time):

```bash
cp /c/ProgramData/mingw64/mingw64/bin/libgcc_s_seh-1.dll   build/bin/
cp /c/ProgramData/mingw64/mingw64/bin/libstdc++-6.dll      build/bin/
cp /c/ProgramData/mingw64/mingw64/bin/libwinpthread-1.dll  build/bin/
```

MSVC release builds (planned) will static-link the runtime.

## CLI verbs at a glance

```
sp-engine version
sp-engine banner
sp-engine info     --model <gguf>
sp-engine encode   --model <gguf> <text>
sp-engine decode   --model <gguf> <id1> [id2 ...]
sp-engine embed    --model <gguf> <text>
sp-engine block1   --model <gguf> <text>
sp-engine logits   --model <gguf> [--pe-mode N --pe-alpha F --pe-tier N] <text>
sp-engine kv_smoke [--sqfree --spinor --head-dim N --n-tokens N]
sp-engine prefill  --model <gguf> [--sqfree --spinor] <text>
sp-engine chat     --model <gguf> [--n-predict N --naive --sqfree] <prompt>
```

## License

Same dual-license as `shannon-prime`: AGPLv3 for open-source /
academic / non-proprietary use, commercial license available
(contact raydaniels@gmail.com). The vendored `vendor/ggml/`
submodule retains its original MIT license.
