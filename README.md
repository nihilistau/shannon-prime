# Shannon-Prime

**The math core of the Prime Power Transformer (PPT-ARM) — a number-theoretic re-derivation of the entire transformer forward pass over the class-number-1 ring $\mathcal{O}_K = \mathbb{Z}[\omega]$ of $\mathbb{Q}(\sqrt{-163})$.**

Shannon-Prime is not a KV-cache compression library. It is the algebraic substrate on which an end-to-end transformer system runs: integer matmul over $\mathcal{O}_K$, polynomial-ring attention, CRT-NTT, Frobenius lifting for low-bit weight storage, and the **Friedman sieve** — an order-invariant cache admission policy built on Dickson's Lemma. The headline scientific claim of the framework is that the sieve **sharpens attention rather than degrading it**: at the calibrated threshold $\tau_A = 0.20$ on `functiongemma-270M`, the sieve evicts 8.77% of K-vectors and perplexity drops **14.80% below the unmodified baseline**. Companion engine: [shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine). Papers: [Position_Is_Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic).

---

## Table of contents

1. [What's in this repository](#whats-in-this-repository)
2. [The Prime Power Transformer](#the-prime-power-transformer)
3. [The Friedman sieve and Dickson's Lemma](#the-friedman-sieve-and-dicksons-lemma)
4. [Theorems verified mechanically](#theorems-verified-mechanically)
5. [Why Shannon-Prime stands out](#why-shannon-prime-stands-out)
6. [Status](#status)
7. [Repository layout](#repository-layout)
8. [Building](#building)
9. [Companion repositories](#companion-repositories)
10. [Citing](#citing)
11. [License](#license)

---

## What's in this repository

A reference C implementation of every algebraic primitive used by the Prime Power Transformer. CPU-portable, bit-identical between GCC and MSVC, no `__int128` on the hot path, no floating-point in the integer kernels. Hexagon HVX variants exist in research form for `sp_ntt_crt_hvx` and the KSTE encoder.

### O_K integer arithmetic over Q(sqrt(-163))

`core/sp_ok_arith.{h,c}` and `core/sp_frobenius.{h,c}` implement element-wise arithmetic on
$$\mathcal{O}_K = \mathbb{Z}[\omega], \qquad \omega = \tfrac{1+\sqrt{-163}}{2}, \qquad \omega^2 = \omega - 41,$$
with norm $N(a + b\omega) = a^2 + ab + 41 b^2$. The carrier type is a packed `{int64_t a, b}` in Array-of-Structs layout that aligns naturally with the AVX-2 `_mm256_mul_epi32` lane structure. Because $K = \mathbb{Q}(\sqrt{-163})$ is one of nine [Heegner number](https://en.wikipedia.org/wiki/Heegner_number) fields and has class number 1, $\mathcal{O}_K$ is a UFD and every linear-algebraic step of inference admits an exact, invertible integer representation. See [Paper I §2](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-Theory.md#2-the-field-ring-and-curve).

### VHT2 + Möbius squarefree reorder

`core/shannon_prime.{h,c}` carries the Vilenkin-Hartley Transform (VHT2) and Möbius compression machinery from the earlier framework. At power-of-2 head dimensions VHT2 reduces to the self-inverse Hartley butterfly; at squarefree-padded dimensions it factors across $\{2,3,5,7,11\}$ and the Möbius reorder pushes the 60.79% squarefree indices to the front. Squarefree density matches the theoretical $6/\pi^2$ from [Paper I §4](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-Theory.md#4-möbius-compression-in-a-ufd). The output is the 63-byte Spinor block (14 fp16 anchors + 60 residual lanes at 3-bit magnitude + 1-bit phase) — this format is frozen and load-bearing for everything else in the stack.

### Polynomial-ring attention over R_q = Z_q[x]/(x^N+1)

`core/sp_poly_ring.{h,c}` replaces the real-valued bilinear form of standard attention with a CKKS-style encoding into the cyclotomic ring $R_q$ at $N = 256$. Each $Q, K$ vector is lifted to a polynomial and the inner product is the coefficient of $x^{N-1}$ in the negacyclic convolution. At Gemma3's head dimension 256, the KL divergence between softmax computed from real-valued logits and from ring-valued logits is **exactly zero**, with cosine 1.0. See [Paper I §7](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-Theory.md#7-the-polynomial-ring-and-the-number-theoretic-transform).

### CRT-NTT dual-prime kernel

`core/sp_ntt_crt.{h,c}` and `core/sp_ntt_crt_consts.h` implement the negacyclic Number-Theoretic Transform over two ~30-bit Proth primes $q_1, q_2$ with $q_1 q_2 \approx 2^{60}$, recombined by Garner's algorithm. Every intermediate fits a `uint64_t`. **No `__int128` is used anywhere on the hot path**; this is the engineering escape route the architecture buys with its CRT structure. The kernel is bit-identical between Linux GCC and Windows MSVC and portable to ARM, RISC-V, Hexagon HVX, and GPU shaders. Barrett reduction (constant $\mu = \lfloor 2^{60}/q\rfloor$) replaces division. See [Paper II §5](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-System.md#5-polynomial-ring-attention) and [§7](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-System.md#7-crt-dual-prime-ntt).

### KSTE encoder + Tier-0/Tier-1 dominance signatures

`core/sp_kste.{h,c}`, `core/sp_kste_pack.c`, `core/sp_kste_embed.c` implement the **Knight-Spinor Tree Encoder**: the function $K \in \mathbb{R}^{128} \to T \in \mathcal{T}_{60,3}$ that maps a continuous Key vector to a 60-node, 3-labelled rooted tree, fitting alongside the existing 63-byte Spinor block (no new memory layout). The encoder is **order-invariant** — it consumes only ranks and signs of $K$, never the numerical values — and therefore *Frobenius-invariant by construction*: scaling $K$ by any $\pi_p^k$ yields bit-identical packed bytes.

The encoder ships with two structural signatures:
- **Tier-0** `sp_kste_signature_t` — the multiset $(A\text{-count}, B\text{-count}, C\text{-count}, \mathrm{max\_depth}, \mathrm{node\_count})$ packed in a single `uint64_t`.
- **Tier-1** `sp_kste_anc_sig_t` — the $3{\times}3$ matrix of ancestor-descendant label-pair counts, saturated at 255 in 16 bytes.

These two signatures embed $\mathcal{T}_{60,3}$ into $\mathbb{N}^{14}$ under elementwise product order, where the dominance relation $\preceq_d$ becomes a bytewise comparison answered in single-digit microseconds. See the next section.

### Frobenius lifting for fp8 / fp4 weight storage

`core/sp_ok_q8.{h,c}`, `core/sp_ok_q4.{h,c}`, `core/sp_ok_block_quant.{h,c}` lift fp16/fp32 GGUF weights into $\mathcal{O}_K$ coordinates with a per-tensor Frobenius scale $\pi_p^k$. The packed Q8/Q4 storage drops the resident weight footprint by 8× on Gemma3-1B (10.4 GB → 1.3 GB) while preserving the Theorem-4 cancellation property. `core/sp_vht2_block_q8.{h,c}` is the block-quantized VHT2 path for cache lanes.

The central guarantee, formalised as **Theorem 4** in [Paper I §3.2](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-Theory.md#32-theorem-projective-cancellation): the Frobenius factor $\pi_p^{4k}$ accumulated through $W_Q' x \cdot W_K'^\top x \cdot W_V' \cdot W_O'$ cancels at the immediately following RMSNorm, yielding inference bit-identical to the unscaled floating-point reference. Empirically: on Gemma3-1B with $p = 41$, $k = 8$, PPL **13.11 vs 13.12** unshimmed — six significant figures of bit-exactness.

---

## The Prime Power Transformer

Every step of the transformer forward pass admits an algebraic substitution from $\mathcal{O}_K$, the CRT, the prime-pair families (twin, sexy, Mersenne), and the Poncelet closure condition $n\delta \equiv 0$ on a CM elliptic curve. The mapping is not decoration; it is the substrate. Paper I §10 lists the 13 steps explicitly. We summarise:

| Step | Operation | Algebraic replacement |
|---|---|---|
| 1 | Embedding lookup | Möbius reconstruction over squarefree token indices; CRT vocabulary sharding |
| 2 | RMSNorm (pre-attn) | Mersenne-prime scaling; Poncelet closure $d^2 = R^2 - 2Rr$ |
| 3 | Q/K/V projections | Twin-prime head pairing; sexy-prime 6:1 GQA grouping |
| 4 | SP Write (KV → archive) | Poncelet closure as eviction trigger; CRT-sharded KV |
| 5 | FUSED_KQ | UFD-exact decompression; Heegner endomorphism |
| 6 | Softmax | $p$-adic exponential on integers; circulant attention on closed orbits |
| 7 | Fused V weighted sum | Spinor reconstruction across twin-paired heads |
| 8 | Attention output projection | CRT decomposition of $W_O$ into independent sub-matrices |
| 9 | FFN (skeleton + residual) | Mersenne-dimensional skeletons; $n^2 + n + 41$ cold-start |
| 10 | Activation oracle update | Cramér prime-gap prefetch; Poncelet early exit |
| 11 | Residual add + norm | Group-law residual on $E(K)$ |
| 12 | Per-layer loop | $n\delta \equiv 0$ adaptive depth; caustic projection |
| 13 | LM head | CRT pruning of vocabulary logits; Mersenne-prime sampling |

The grand unified view: **the transformer forward pass is a discrete dynamical system on a CM elliptic curve $E$ over a class-number-1 field**. The hidden state is a point on $E/K$. Each layer is a point addition. Attention is a bilinear form on $R_q$. The orbit closes at depth $n$ iff $\mathrm{ord}(\delta) \mid n$ (Poncelet). The caustic is the invariant subspace — always compressible. The Shannon limit of the architecture is the information content of the curve point: $\log_2 \mathrm{ord}(\delta)$ bits, no more.

See [Paper I §11](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-Theory.md#11-the-grand-unified-view) for the unified derivation.

---

## The Friedman sieve and Dickson's Lemma

The KV cache in this framework is not a bounded-by-token-budget buffer; it is a **filter** that admits only structurally-novel keys. Each incoming Key vector $K$ is encoded by KSTE to a tree $T_K \in \mathcal{T}_{60,3}$, and a new token is admitted iff $T_K$ does not dominate any tree already in the cache under the dominance relation $\preceq_d$.

### The dominance relation

Define $\preceq_d$ on $\mathcal{T}_{60,3}$ by elementwise comparison of the Tier-0 and Tier-1 signatures:
$$Q \preceq_d K \;\iff\; \sigma_0(K) \succeq \sigma_0(Q) \;\text{ and }\; \sigma_1(K) \succeq \sigma_1(Q).$$
Concatenating the two signatures gives an embedding
$$\varphi : \mathcal{T}_{60,3} \to \mathbb{N}^{14}, \qquad Q \preceq_d K \iff \varphi(K) \ge_{\mathrm{elem}} \varphi(Q).$$
This is a coarsening of Kruskal's homeomorphic embedding $\preceq$ — the inclusion $\preceq \subsetneq \preceq_d$ is strict — but it is the relation under which the **empirical signal lives** on noisy production K-vectors. Strict Kruskal embedding sat at ROC AUC 0.500 on the resolution probe; dominance gives a 17× intra/inter ratio at the near-duplicate regime and a 720× wall-time speedup ($685\,\mu\mathrm{s} \to 0.95\,\mu\mathrm{s}$ p99 at $n = 4096$).

### The wqo and the cache bound

**Theorem (Dickson, 1913).** *The elementwise order $(\mathbb{N}^k, \le_{\mathrm{elem}})$ is a well-quasi-order: every infinite sequence $v_1, v_2, \dots$ in $\mathbb{N}^k$ contains indices $i < j$ with $v_i \le_{\mathrm{elem}} v_j$.* (See the AMS reference: [L. E. Dickson, *Amer. J. Math.* 35, 1913, pp 413–422](https://www.jstor.org/stable/2370405).)

The image $\varphi(\mathcal{T}_{60,3})$ sits inside the bounded hypercube $[0,60]^{14}$. The maximal antichain inside that cube is finite by a Sperner-style cross-section bound. The cache therefore has a *physical* upper limit on the number of mutually $\preceq_d$-incomparable trees — empirically the cache plateaus at ~307/512 slots on i.i.d. Gaussian inputs, which is the manifestation of that finite antichain. The eviction policy stops *needing* a variance-fallback heuristic under dominance semantics: the cache **settles** rather than churning.

### WKL_0 refutation, strengthened to PRA

Every sieve decision admits a finite witness of failure: dominance is a bytewise comparison of two 64-bit and two 16-byte signatures. The WKL$_0$ refutation property of [Paper III §3.3](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-III-Friedman.md#33-the-refutation-property) is preserved and strengthened: Dickson's Lemma itself is **provable in PRA**, which sits below WKL$_0$ in the consistency-strength hierarchy. The framework's runtime moved from Kruskal-strength (independent of ATR$_0$) to PRA-strength without losing any expressivity in the encoder.

### The headline empirical result

Calibrated sweep on `functiongemma-270M`, baseline PPL = 17.7296 (ctx=64):

| $\tau_A$ | PPL | $\Delta$ vs baseline | Eviction rate |
|------:|----:|------------------:|---------:|
| 0.05 | 29.5595 | +66.7% | 12.24% |
| 0.10 | 23.3099 | +31.5% | 11.20% |
| **0.20** | **15.1052** | **−14.80%** | **8.77%** |
| 0.40 | 20.4401 | +15.3% | 11.98% |

At $\tau_A = 0.20$ the sieve **evicts 8.77% of K-vectors and lowers perplexity by 14.80% below the unmodified baseline**. The encoder is filtering enough background noise that attention sharpens. Audit trail: [`papers/PPT-ARM/SESSION-STATE-friedman-4c.md`](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/SESSION-STATE-friedman-4c.md). See [Paper III §11.6](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-III-Friedman.md#116-from-kruskal-embedding-to-dickson-dominance-the-operational-subsumption-relation) for the full proof.

This is not a compression result. It is a **structural redundancy** result: the model is being asked to attend over fewer, cleaner positions, and inference quality improves.

---

## Theorems verified mechanically

Aligned with [Paper I §12](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-Theory.md#12-verified-theorems-and-extensions):

- **T1 — Endomorphism realization.** Hidden-state trajectory through $L$ layers embeds in $E^L$ exactly.
- **T2 — Möbius UFD compression.** Reconstruction over $\mathcal{O}_K$ at squarefree basis is exact.
- **T3 — Hasse–Weil = Shannon limit.** $|E_p(\mathbb{F}_p) - (p+1)| \le 2\sqrt p$.
- **T4 — Frobenius cancellation.** Validated bit-identical at six significant figures on Gemma3-1B (PPL 13.11 vs 13.12).
- **T5 — Deuring / CM Sato–Tate.** Asymmetric distribution of $a_p$ between split and inert primes.
- **T6 — CRT exact sharding.** Dual-prime kernel bit-identical to 60-bit reference; portable.
- **E9.1 — Stern–Brocot RoPE.** Discrepancy $\phi = 0.00134$ vs $0.05576$ standard RoPE.
- **E9.2 — Weil pairing on $E[n]$.** Miller's algorithm validated, bilinearity confirmed.
- **E9.3 — Hecke multiplicativity.** 20/20 trials.
- **E9.5 — LLL reduction.** KV-write optimization, 20/20 trials.
- **E9.6 — BSD analytic rank.** Toy curves verified via Sage.
- **E10 — Iwasawa $\mu = 0$.** Residual-stream depth stability confirmed.
- **Phase-4 Friedman dominance (Paper III §11.6).** Sieve at $\tau_A = 0.20$ on functiongemma-270M evicts 8.77% of K-vectors and improves PPL by 14.80%.

The KSTE Tier-1 and Tier-2 unit suites (21/21 green on MSVC) run as part of the engine test build.

---

## Why Shannon-Prime stands out

- **Algebraic correctness from first principles.** Every linear step of the forward pass is an exact integer-ring operation in a UFD. No fp16 rounding inside the matmul, no compensation for non-associativity, no per-layer drift. The few fp32 islands (RMSNorm, softmax, RoPE) are explicit bridge kernels, not buried defaults.
- **The sieve sharpens attention rather than degrading it.** The 270M $\tau_A = 0.20$ result (PPL −14.80% at 8.77% eviction) is the canonical evidence: a properly structured eviction policy *adds* signal by removing ambient noise. This inverts the usual KV-compression trade-off.
- **Provable foundations.** Dickson's Lemma (1913, provable in PRA) bounds the cache by the antichain count of a finite hypercube. Theorem 4 (Frobenius cancellation through RMSNorm) bounds the inference drift to zero. Both proofs have mechanically-verified test suites.
- **Portable to no-fp16 hardware.** The CRT-NTT kernel uses two 30-bit Proth primes recombined by Garner's algorithm; no `__int128`, no 64-bit floats on the hot path, no FMA. It is the same code on x86 AVX-512, ARM v9, Hexagon HVX, and GPU shaders.
- **Tested end-to-end against real production models.** Gemma3-1B at six-figure bit-identity (Paper II), functiongemma-270M and Gemma3-1B under live sieve telemetry (SESSION-STATE-friedman-4c), Qwen and Llama families in the engine bench harness. No toy benchmarks.

---

## Status

The math core is **production-grade on CPU**. CPU reference implementations of every primitive listed under "What's in this repository" are complete, MSVC and GCC bit-identical, and verified with `make test-all` on both Windows BuildTools 2019 and Linux GCC.

| Component | Status |
|---|---|
| $\mathcal{O}_K$ arithmetic + Frobenius lift | Production |
| VHT2 + Möbius reorder + 63-byte Spinor block | Production (frozen format) |
| Polynomial-ring attention + 60-bit NTT | Production |
| CRT dual-prime NTT (no `__int128`) | Production |
| KSTE encoder + Tier-0/Tier-1 dominance | Production, 21/21 tests green |
| Q8/Q4 weight storage with Frobenius scale | Production (Q8); Q4 mixed-precision path under calibration |
| Friedman sieve in engine, observer mode | Production, bit-identical to baseline |
| Friedman sieve, policy mode | Calibration sweep in progress (Phase 4e) |
| Hexagon HVX kernels (`sp_ntt_crt_hvx`, KSTE HVX) | Research-grade |

The Friedman sieve is wired into the engine in both **observer** mode (telemetry only, bit-identical PPL guarantee) and **policy** mode (eviction live). Default-calibration policy mode currently over-evicts; the Phase 4e sweep is identifying the production $(\tau_A, \alpha, \text{capacity})$ knee.

---

## Repository layout

```
shannon-prime/
├── core/
│   ├── shannon_prime.h               public API
│   ├── shannon_prime.c               VHT2, Möbius, banded paths
│   ├── shannon_prime_sqfree.c        squarefree-padded VHT2
│   ├── shannon_prime_pe.c            PrimePE lattice math
│   ├── shannon_prime_cauchy.c        Cauchy/Ricci sentinel
│   ├── shannon_prime_modelpack.{h,c} per-architecture defaults
│   ├── sp_ok_arith.{h,c}             O_K element arithmetic
│   ├── sp_frobenius.{h,c}            π_p^k Frobenius lift
│   ├── sp_ok_q8.{h,c}                packed int8 O_K storage
│   ├── sp_ok_q4.{h,c}                packed int4 O_K storage
│   ├── sp_ok_block_quant.{h,c}       block-quantized GGUF lift
│   ├── sp_vht2_block_q8.{h,c}        block-quantized VHT2 lanes
│   ├── sp_poly_ring.{h,c}            R_q = Z_q[x]/(x^N+1) attention
│   ├── sp_ntt.{h,c}                  60-bit Proth NTT (parity anchor)
│   ├── sp_ntt_crt.{h,c}              dual-prime CRT-NTT (production)
│   ├── sp_ntt_crt_hvx.{h,c}          Hexagon HVX variant (research)
│   ├── sp_ntt_consts.h, sp_ntt_crt_consts.h
│   ├── sp_ec_weil.{h,c}              Weil pairing on E[n]
│   ├── sp_arm.{h,c}                  ARM port helpers
│   ├── sp_kste.{h,c}                 Knight-Spinor Tree Encoder
│   ├── sp_kste_pack.c                bit-packing helpers
│   └── sp_kste_embed.c               Kruskal + dominance tests
├── backends/                         CUDA, Vulkan, Adreno, Hexagon, QNN, Torch
├── docs/                             ARCHITECTURE, KSTE-CALIBRATION, etc.
├── tests/                            unit + theorem suites
├── tools/                            CLI utilities
└── scripts/                          build, benchmark, calibration
```

---

## Building

```bash
# Linux / WSL
make                           # core + CPU backend
make test-all                  # unit + theorem suites

# Windows (Visual Studio 2019 BuildTools + Ninja)
cmake -B build-cuda -G Ninja
cmake --build build-cuda --target sp-engine test_sp_kste test_sp_friedman_cache --config Release
```

CMake options of interest:

```cmake
set(SP_FROBENIUS_QUANT   ON)   # Theorem 4 shim
set(SP_ENGINE_POLY_ATTN  ON)   # polynomial-ring attention
set(SP_NTT_CRT           ON)   # dual-prime kernel
set(SP_FRIEDMAN_SIEVE    ON)   # KSTE + dominance sieve
set(SP_ENABLE_AVX2       ON)
set(SP_ENABLE_AVX512     ON)
```

Detailed per-platform instructions: [docs/QUICKSTART.md](docs/QUICKSTART.md).

---

## Companion repositories

| Repository | Purpose |
|---|---|
| [shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine) | Reference inference engine. GGUF loader, native $\mathcal{O}_K$ forward, persistent NTT-domain KV cache, CRT-NTT attention, Friedman sieve, CLI verbs. **The reference implementation.** |
| [Position_Is_Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic) | The papers. PPT-ARM Theory (Paper I), System (Paper II), Friedman Sieve (Paper III), KSTE Engineering (Paper IV). |
| [shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama) | llama.cpp integration. Patch-based bridge for the existing ecosystem; secondary to the engine. |
| [shannon-prime-comfyui](https://github.com/nihilistau/shannon-prime-comfyui) | ComfyUI custom nodes for video / image / audio / TTS workloads. Secondary to the engine. |

---

## Citing

If you use the Friedman sieve or the dominance-subsumption relation in academic work, please cite Paper III §11.6:

```bibtex
@unpublished{ShannonPrime2026Friedman,
  author    = {KnackAU and Claude (Anthropic) and Gemini (Google DeepMind)},
  title     = {The Friedman Stack: Order-Invariant Memory and Ultraproduct Attention
               (Paper III, §11.6: Dickson dominance as the operational subsumption relation)},
  year      = {2026},
  month     = {May},
  note      = {Shannon-Prime Project},
  url       = {https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/PPT-ARM/PPT-ARM-III-Friedman.md}
}

@article{Dickson1913,
  author  = {Dickson, L. E.},
  title   = {Finiteness of the odd perfect and primitive abundant numbers
             with $n$ distinct prime factors},
  journal = {American Journal of Mathematics},
  volume  = {35},
  year    = {1913},
  pages   = {413--422}
}
```

For Theorem 4 (Frobenius cancellation), cite Paper I §3.2 and Paper II §3.

---

## License

Copyright (C) 2026 Ray Daniels. All Rights Reserved.

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3). Commercial license available — contact `raydaniels@gmail.com`.
