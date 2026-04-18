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

## Status

Pre-alpha but live. Full forward pass runs end-to-end on
Llama-family GGUFs (Llama-3 hd=64 and Qwen3 hd=128 GQA-4× both
validated). Stateful prefill + single-token decode through a
compressed KvCache — ship, sqfree, sqfree+spinor, and hierarchical
Vilenkin predictor (9% skeleton) all functional.

## Headline result

**Qwen3-8B-Q8 ship cache-mode PPL = 18.14 vs baseline 18.05 →
+0.5% at 4.06× KV compression** (`perplexity --cache`, ctx=512
chunks=2, wiki.test.raw, decode-per-token through the compressed
cache). First hook-free publishable measurement on a model whose
KV cache is actually worth compressing.

### Scaling-law projection at ship K_corr=0.993, Q8 weights

| Model size | Ship PPL hit @ 4× compression | Regime |
|---|---|---|
| 1B  | +10% (Dolphin measured +10.5% ✓) | noise — KV isn't the bottleneck |
| **8B**  | **+0.5% (Qwen3 measured +0.5% ✓)** | **sweet spot, ship-ready** |
| 14B | +0.2% predicted | ship-ready |
| 70B | +0.01% predicted | compression essentially free |
| 405B | +0.002% predicted | the regime this tech was built for |

Both measured points sit on the scaling-law curve within fit
error. The `params^1.1` denominator roughly halves the
compression penalty per doubling of model size, so results from
1B benchmarks are noise for the sizes people actually deploy.

## Stage table

| Stage | Verb | Status |
|---|---|---|
| 3a–3d | `embed` / `block1` / `logits` (+ PrimePE-RoPE-ALiBi) | ✓ |
| 4     | `kv_smoke` | ✓ |
| 5a    | `prefill` — real RoPE'd K through KvCache | ✓ |
| 5b    | `chat` — stateful prefill + decode-per-token | ✓ |
| 6a    | `perplexity` (baseline + `--cache` decode-chain) | ✓ |
| 6b    | `cache_ppl` — baseline PPL + K/V correlation + scaling term | ✓ |
| 6c    | Sidecar auto-load (`<model>.sp_freq_factors.bin`) | ✓ |
| 7     | CUDA / Vulkan backend, release packaging | planned |

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

## Compressed KV cache

`KvCache` wraps `sp_shadow_cache_t` (ship) / `sp_sqfree_cache_t`
(aggressive) / `sp_hier_cache_t` (hierarchical, v1.15+) behind a
single C++ class. Layout matches what `ggml_backend_tensor_get`
returns for a `[head_dim, n_head_kv, n_tokens]` tensor, so
write/read plugs straight into the prefill path without
transposes.

Ship is default; every non-ship mode is an opt-in flag that
auto-calibrates on the first prefill call (calibration state
persists across decode steps).

### Cache mode decision table

| Flag | Path | Skeleton | When it wins |
|---|---|---|---|
| *(none)* | **Ship** VHT2 + Möbius + banded | full | default, validated, K_corr≥0.993, deploy-ready on all Q8+ backbones ≥8B |
| `--sqfree` | Knight skeleton + residual + Möbius CSR predictor | 50% | when sqfree-aligned spectral structure matters for the specific model |
| `--sqfree --spinor` | + SU(2) sheet bit correction | 50% | small K-corr lift over `--sqfree` |
| `--hierarchical` | Kronecker sub-projection + per-slot calibrated linear predictor | **9%** | when storage-per-vector is the binding constraint; needs ≥24-token calibration prompt |

### cache_ppl roundtrip (baseline PPL + K/V correlation + scaling term)

| Model | Mode | K_corr | V_corr | Skeleton | Scaling term |
|---|---|---|---|---|---|
| **Qwen3-8B-Q8** | **ship** | **0.9930** | 0.9652 | full | 0.229 |
| Qwen3-8B-Q8 | hierarchical | 0.9833 | 0.9368 | 9% (14/154) | 1.314 |
| Dolphin-1B-Q8 | ship | 0.9947 | 0.9708 | full | 0.133 |
| Dolphin-1B-Q8 | hierarchical | 0.9752 | 0.9318 | 9% (6/66) | 2.895 |

Hierarchical at 9% skeleton is architecturally competitive with
ship's K-corr at 5.5× smaller skeleton. Scaling-law projection
at 70B Q8 puts hierarchical within 0.05% PPL of ship for prefill-
style (single-pass forward) workloads.

### Cache-mode PPL note

`perplexity --cache` (prefill + decode-per-token) amplifies the
compression penalty well beyond what the prefill-roundtrip
scaling law predicts, because each decode reads its past K/V
back through the compression pipeline. Measured on Qwen3-8B-Q8,
ctx=512, wiki.test.raw:

| Mode | Chunk 1 PPL | Chunk 2 PPL | Final |
|---|---|---|---|
| baseline (no cache) | 11.83 | 18.05 | 18.05 |
| ship cache | 11.72 | 18.14 | **18.14 (+0.5%)** |
| hierarchical cache | 17.43 | *(terminated)* | *(pending)* |

Hierarchical's chunk-1 partial (+48% vs ship chunk 1) exceeded
the CPU bench window before chunk 2 completed; the final number
is pending a persistent-KV-tensor decode path and/or a GPU
backend. The measurement shows decode-chain amplification is
order-of-magnitude larger than prefill-only scaling predicts —
same pattern seen on Dolphin-1B, where sqfree+spinor landed at
+325% PPL in decode-chain vs scaling law's single-pass
prediction of +~100%. Ship mode is dramatically more robust to
decode-chain compounding because its K_corr error is ~10× smaller.

### Sidecar injection (`<model>.sp_freq_factors.bin` auto-load)

`sp_inject_freqs.py` writes both a modified GGUF (with
`rope_freqs.weight` embedded) and a debug sidecar .bin. The
engine auto-discovers the sidecar at model path and uses its
factors in `ggml_rope_ext`. Useful for A/B-ing different alphas
against the same base GGUF without regenerating it.

| Model | Source | PPL |
|---|---|---|
| Dolphin-1B-Q8 | GGUF `rope_freqs.weight` (no sidecar) | 12.65 |
| Dolphin-1B-Q8 | α=0.17 sidecar auto-loaded | **12.38 (−2.1%)** |

## Stage 5b — chat with greedy decode works

The stateful API
(`bind_cache` / `prefill` / `decode` / `kv_pos`) is in. The
`chat` verb does greedy generation through the optimised
single-token decode graph: it reads past K/V from the bound
KvCache as input tensors, concats with the freshly-projected new
K/V along the position axis, runs attention over `[past_n + 1]`
keys, writes the new K/V back to the cache, and emits logits.

Smoke tests on Dolphin-1B and Qwen3-8B-Q8 (greedy, n_predict=20):

> `./sp-engine chat --model dolphin_1b.gguf --n-predict 20 "The quick brown fox"`  
> → `"The quick brown fox jumps over the lazy dog. The sentence \"The quick brown fox jumps over the lazy dog.\" is"`

> `./sp-engine chat --sqfree --spinor --model dolphin_1b.gguf --n-predict 20 "..."`  
> → `"...What is the correct order of the sentence? To determine the correct order"`

> `./sp-engine chat --model Qwen3-8B-Q8_0.gguf --n-predict 20 "..."`  
> → `"...This sentence is a well-known pangram. It is used to test"`

`chat --naive` (re-runs `forward_full` over the running prefix
each step) remains as a slower-but-trivially-correct fallback;
it now produces identical output to the optimised path.

### The decode bug we hunted (post-mortem)

The first cut of `decode` produced logits at ~half the magnitude
of `forward_full` at the same position, even with a near-lossless
8-bit cache. Captured layer-0 K matched at correlation 1.0 against
`forward_full`'s K, and the K_full data after concat matched too —
which made the bug confusing. Captured V was smaller than expected,
and the magnitude of the freshly-projected V differed *between two
forward_full calls on overlapping prefixes*, which was the tell.

Root cause: V at the projection capture point is a `ggml_reshape_3d`
**view** over the `mul_mat` output and does not own a buffer.
`ggml_set_output` on a view does not pin the underlying source buffer
through subsequent ops, so the gallocr was repurposing it during the
GQA broadcast / concat — corrupting both the captured V and the V
data the attention op actually consumed. K was unaffected because it
comes out of `ggml_rope_ext` (a real op output, own buffer).

Fix: materialise V via `ggml_cont` once, before the capture point,
and reuse the materialised tensor for both the capture and the
downstream concat / GQA path. Applied symmetrically in both
`build_block` (forward_full) and `build_block_decode`. The same
fix doubles as a stability improvement for stage 5a's `prefill`
verb, where per-layer V capture was returning data that depended
on what other graphs had run since.

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
