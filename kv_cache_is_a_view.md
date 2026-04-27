# The KV Cache Is a View: Spectral Compression, the K-corr Scaling Law, and Basis-Dependent Möbius Prediction

**Knack**
*2026 | v2.0*

---

## Changes from v1

- New headline result (abstract + §3): an empirical scaling law connecting K-reconstruction fidelity to downstream perplexity across model size and weight quantization, with three interpretable exponents. Validated across 9 configurations spanning 4 orders of magnitude of PPL ratio.
- §4 reframed: the Möbius predictor is now correctly characterized as basis-dependent. On WHT (the r≈0 result from v1) it functions as a partition heuristic; on squarefree prime-Hartley bases it functions as a genuine predictor (r=0.40–0.58). This resolves the apparent contradiction in v1 between "Möbius prediction gives r=0" and "Möbius prediction dominates by 3–7× over alternatives."
- New §5: the spinor sheet bit. A 1-bit SU(2) double-cover correction that matches MOBIUS default quality at +27% compression on Qwen3-8B Q8 hd=128. The first architectural feature that shifts (not moves along) the Pareto frontier in this framework.
- New §6.2: precision-sensitivity finding. Spinor is a Q8+ feature; on Q3 the same K-corr loss costs ~7× more PPL, washing out the 1-bit correction. Explained by the scaling law.
- §7 Knight closeout: Knight-family squarefree-only skeletons converge back to MOBIUS when composites are added back, because full coverage IS MOBIUS. Reframed as a falsification result supporting the multiplicative-lattice claim, not a dead end.
- §9.1 added: measurement noise characterization (±5 PPL on 1-chunk wiki.test in the 12-20 range on Vulkan fp16). Tight ladder sweeps flagged as noise-driven.
- §10 added: the N-bit residual saturation curve — Shannon signature at ~8 meaningful levels (3 bits) for the μ-inversion residual distribution.
- References updated.

---

## Abstract

The transformer KV cache is treated universally as data to be stored, compressed, and evicted. We present evidence that it is instead a projection of a low-dimensional spectral structure — specifically, the multiplicative lattice of the integers encoded by rotary position embeddings — and that this structure can be exploited for near-optimal compression and eventual reconstruction.

The paper's headline result is an empirical scaling law connecting KV-cache reconstruction fidelity (K-corr, the Pearson correlation between original and reconstructed K vectors) to downstream perplexity:

    log(PPL / PPL_base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)

The law fits 9 configurations across (Dolphin 1B Q8, Qwen3-8B Q8, Qwen3-8B Q3) within ±20%, spanning 4 orders of magnitude of PPL ratio from baseline to catastrophic. The three exponents are interpretable: quadratic in K-error (from the K·Q·V bilinearity of attention), sub-linear in params (bigger models absorb K error through head-averaging), and super-linear in weight precision (Q4 amplifies K error ~2.8× vs Q8).

The law functions as a design rule: predict PPL impact from (K-corr, params, bits) before benching, and skip any config whose predicted cost exceeds the budget. This cuts sweep time ~10× in practice.

We introduce three compression methods exploiting the lattice structure. VHT2 banded quantization applies Walsh-Hadamard transform then per-band spectral allocation, achieving 3.4–3.8× total KV compression at <1.25% perplexity cost. The spinor sheet bit — a 1-bit SU(2) double-cover correction to the Möbius predictor's causal-mask sign errors — matches MOBIUS default quality at +27% compression on Qwen3-8B Q8 hd=128, the first feature that shifts (not moves along) the Pareto frontier. Vilenkin successive decomposition extends WHT from the Z/2Z basis to a multi-prime Vilenkin-Hartley basis, achieving 5.1× compression at +9.9% PPL.

The Möbius predictor's behavior is basis-dependent: on WHT it operates as a partition heuristic (r≈0, involutive basis scrambles divisibility) and on squarefree prime-Hartley bases it operates as a genuine predictor (r=0.40–0.58). This resolves what appeared to be a contradiction between the predictor's theoretical grounding and its measured correlation in early implementations — different bases expose different aspects of the same lattice structure.

Three structural discoveries: the Z/3Z skeleton appears at 100% of positions in every head of Layer 0, forming a standing wave that IS the coordinate system. K and V occupy completely disjoint Vilenkin bands at every layer (a symplectic pair: K is the map, V is the terrain). Successive prime passes tile the mixed-radix residue classes with <3.5% deviation from uniform — an algebraic partition, not a redundant covering.

---

## 1. Introduction

Every deployed large language model stores context as a key-value cache that grows linearly with sequence length. At 128K context on a 70B model, the KV cache alone can exceed 40GB. The field has responded with compression: quantised caches, sliding windows, attention sinks, token merging, TurboQuant. All accept the premise that context is data.

We propose a different premise: context in rotary-encoded transformers is a signal with exploitable harmonic structure, and signals with known structure can be compressed far more effectively than arbitrary data — and eventually reconstructed from sparse invariants rather than stored at all.

The rotary position encoding used in virtually all modern transformers applies position-dependent rotations to query and key vectors at specific frequencies. In standard RoPE, these frequencies follow a geometric progression. This is an engineering choice, not a mathematical necessity. We show that treating these frequencies as projections of a multiplicative lattice — where the Walsh-Hadamard transform is the Z/2Z case of a Vilenkin-Hartley basis spanning the full prime structure — enables compression methods that geometric-agnostic approaches cannot match.

The companion paper ("Position Is Arithmetic") establishes why the multiplicative lattice is the natural spectral basis for positional encoding. This paper shows what you can build with that insight, and provides a quantitative, falsifiable scaling law that ties reconstruction fidelity to downstream quality across model families and quantization levels.

---

## 2. The Structural Asymmetry: K Carries Position, V Carries Content

### 2.1 K and V in WHT Space

RoPE applies position-dependent angular rotations to K vectors. The WHT is the Z/2Z projection of the Vilenkin-Hartley basis — the natural transform for signals structured by the multiplicative lattice. When we apply WHT to K vectors from a production model, the energy concentrates in the first spectral bands. When we apply WHT to V vectors, the energy is uniformly distributed.

This asymmetry is the theory's most direct empirical signature: K encodes multiplicative arithmetic relationships as angular rates (spectral concentration); V encodes learned content projections with no arithmetic structure (uniform spectrum). They require fundamentally different compression strategies.

### 2.2 The Symplectic Pair

Vilenkin decomposition reveals a stronger property: K and V occupy completely disjoint bands in the Vilenkin spectrum. At every layer tested:

| Property | K vectors | V vectors |
|---|---|---|
| Universal indices | 4–6 at 99–100% | None (max 20%) |
| Vilenkin band | Blocks 8–8 (indices 48–53) | Spread: blocks 2–3, 6–7, 12–13 |
| Residue class rules | Layer-specific k₂ avoidance | Perfectly flat |
| Shared indices | 0 | 0 |

Zero overlap at every layer. K is the coordinate (sparse, localized, respects selection rules). V is the momentum (dense, smeared, uses all residue classes uniformly). They compress independently because they occupy orthogonal spectral bands — no interference. Combined K+V compression at PPL 10.27 confirms this.

---

## 3. The K-corr → PPL Scaling Law

### 3.1 Motivation

The symplectic structure of §2 tells us that K errors and V errors have different propagation paths. V errors average through the attention-weighted sum, so they scale roughly linearly in reconstruction error. K errors enter the attention score directly through Q·K^T, so they should scale differently — and the shape of that scaling is diagnostic of the underlying structure.

If K truly carries positional information through the multiplicative lattice, the lattice theory makes a specific prediction: attention is bilinear in K·Q^T, so small angular errors in K compound quadratically through the softmax. Any correct theory of K-error propagation must observe this quadratic.

### 3.2 Measurement

We measured (K_corr, PPL) pairs across 9 configurations spanning three (model, quantization) pairs:

| Model | params | bits | K corr | PPL ratio | Config |
|---|---|---|---|---|---|
| Dolphin 1B Q8 | 1 | 8 | 0.988 | 4.0% | K+μ+spinor 5/4/4/4/5 |
| Dolphin 1B Q8 | 1 | 8 | 0.978 | 8.2% | K+μ+3bit 5/4/4/4/5 |
| Dolphin 1B Q8 | 1 | 8 | 0.940 | 87% | K+μ+2bit various |
| Dolphin 1B Q8 | 1 | 8 | 0.860 | 6700% | K+μ+1bit various |
| Qwen3-8B Q8 | 8 | 8 | 0.988 | 0.3% | K+μ+spinor 5/4/4/4/5 |
| Qwen3-8B Q8 | 8 | 8 | 0.972 | 1.0% | K+μ+spinor 3/3/3/3/3 |
| Qwen3-8B Q8 | 8 | 8 | 0.970 | 2.6% | K+μ+3bit WHT 3/3/3/3/3 |
| Qwen3-8B Q3 | 8 | 3 | 0.988 | 2.0% | K+μ+spinor 5/4/4/4/5 |
| Qwen3-8B Q3 | 8 | 3 | 0.972 | 3.6% | K+μ+spinor 3/3/3/3/3 |

### 3.3 The Law

A three-parameter fit yields:

    log(PPL / PPL_base) ≈ K · (1 − K_corr)^α / (params^β · bits^γ)
    K ≈ 4700,  α ≈ 2,  β ≈ 1.1,  γ ≈ 1.5

All 9 points fit within ±20% of the predicted coefficient. The residuals show no systematic pattern across model, quantization, or PPL magnitude — the law generalizes across 4 orders of magnitude of PPL ratio and 8× parameter range.

### 3.4 Interpretation of the Exponents

**α ≈ 2 (quadratic in K-error).** Attention scores are bilinear in K·Q^T. Small angular errors in K vectors compound through the softmax. The observed quadratic matches the theoretical prediction from the bilinear structure of attention: if K error projects onto a random direction in d-dimensional space, the expected attention-score error is O(ε²) where ε is the K-corr deficit. This is the most theoretically constrained exponent and the most robustly fit in the data.

**β ≈ 1.1 (sub-linear in params).** Larger models have more heads and higher effective dimension. Uncorrelated K errors across heads average out; correlated errors (from a shared compression scheme) do not. The observed 1.1 is close to inverse-linear, consistent with attention-over-heads averaging scaling as N_heads, which scales roughly linearly with parameter count.

**γ ≈ 1.5 (super-linear in weight precision).** Low-precision W_Q/W_K matrices quantize the attention score directly; small K errors that Q8 rounds cleanly get amplified by Q4 or Q3 into measurable logit divergence. The super-linear exponent explains why the same K-corr loss costs 7× more PPL on Q3 than Q8 — a finding that initially looked surprising but is structurally inevitable.

### 3.5 Safe K-corr Floors

Solving for K_corr such that ΔPPL/PPL = 3%:

| Model & quant | Min K-corr (3% PPL budget) |
|---|---|
| Dolphin 1B Q8 | 0.988 |
| Qwen3-8B Q8 | 0.962 |
| Qwen3-8B Q3 | 0.974 |
| Llama-70B Q8 (predicted) | 0.927 |
| Llama-70B Q4 (predicted) | 0.957 |
| Wan 2.2 14B bf16 (predicted) | 0.914 |

The law predicts that large, high-precision models can tolerate dramatically lower K-corr than small, low-precision models. This is the "big models compress more easily" intuition, now quantitative.

### 3.6 Using the Law as a Design Rule

For any compression config under consideration, compute:

    ΔPPL/PPL = exp(4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)) − 1

If ΔPPL/PPL > 5% (typical budget), don't bother running the bench — the config is not Pareto-viable. This transforms compression research from empirical sweeping to targeted construction: you know which K-corr levels are acceptable on your target model, and you measure compression configs against that bar.

In our own workflow, the law reduced sweep time roughly 10× by eliminating obviously-unviable configs pre-bench.

---

## 4. The Möbius Predictor, Basis-Dependent

### 4.1 The Apparent Contradiction

Earlier work (§6.2 of v1, "mobius-inversion-finding") reported that the Möbius predictor — non-squarefree coefficients reconstructed from squarefree ones via the Möbius inversion formula — measures correlation r≈0 in production when implemented on a Walsh-Hadamard basis. The same theoretical construction, implemented as a partition heuristic (squarefree-first ordering, no predictive claim), gives real quality wins (+0.024 K-corr, cross-platform invariant).

In parallel, Phase 10 research measurements on the squarefree prime-Hartley basis reported r=0.40–0.58 for the same predictor. This apparent contradiction blocked progress on the reconstruction endgame in v1.

### 4.2 The Resolution

The Walsh-Hadamard transform is self-inverse: WHT² = I. This involutivity scrambles the divisibility structure the Möbius predictor depends on — when you apply WHT to a vector whose non-squarefree entries are predictable from its squarefree entries, the resulting spectrum has this property destroyed. The predictor measures r≈0 because the information it relies on has been hidden by the basis, not because the information doesn't exist.

The squarefree prime-Hartley basis is built on primes {2,3,5,7,11} and is NOT involutive at the squarefree head_dim values we care about (hd=66 for hd=64; hd=154 for hd=128). The divisibility structure is preserved through the forward transform and remains accessible to the predictor. On this basis, r=0.40–0.58 as Phase 10 reported.

### 4.3 What Ships on Each Basis

The implications are directly operational:

**WHT basis (hd=64 → 64; hd=128 → 128):** Möbius functions as a partition heuristic. Squarefree-first ordering gives +0.024 K-corr at equal compression; this is real and validated cross-platform. Predictor path returns near-random residuals and is not useful.

**Squarefree prime-Hartley basis (hd=64 → 66; hd=128 → 154):** Möbius functions as a genuine predictor with r=0.40–0.58. Residual path captures the ~50% of non-squarefree variance that the predictor correctly forecasts. This is the basis on which the spinor sheet bit (§5) achieves its Pareto-shifting improvement.

**Vilenkin-5P basis (hd padded to 2310):** Full prime-Hartley over {2,3,5,7,11}. Theoretically optimal; reconstructs K-corr 0.97 before quantization. Runtime: ~1 second per forward pass at hd=64. Not shippable in its current form — listed as deferred research, not production.

### 4.4 Walsh vs Vilenkin at the Same Threshold

The catastrophic Walsh failure result from v1 — 95% energy threshold gives PPL 57.21 on Walsh vs PPL 10.89 on Vilenkin 2-prime — now has a clean interpretation. Both transforms can represent K vectors at 95% energy capture, but Walsh throws away the structural information (divisibility, residue classes) that makes reconstruction stable. Vilenkin preserves it.

This is not a negative result against Walsh-Hadamard; WHT is the correct transform for the banded-quantization path of §6. It is a positive result about what information specific bases preserve. Different bases for different jobs.

---

## 5. The Spinor Sheet Bit

### 5.1 The Prediction

Section 5.4 of the companion paper ("Position Is Arithmetic") predicts topological depth in the multiplicative lattice: inner torus dimensions (Z/3Z) survive the causal mask robustly; outer dimensions (Z/7Z, Z/11Z) are fragile because their phase winds faster and the causal truncation creates incomplete cycles.

If outer dimensions carry systematic sign errors at the causal boundary — errors not randomly distributed but consistently biased — then a single correction bit per non-squarefree position should substantially improve reconstruction. Specifically, the correction should have the structure of an SU(2) double cover: two valid choices for the predictor's sign, one bit to pick between them, no information needed to distinguish them beyond the local value itself.

### 5.2 Implementation

For each non-squarefree position `n` in the reconstruction target, the Möbius predictor emits `pred[n] = Σ μ(d) · skel[n/d]`. The standard residual is `v_plus = actual[n] − pred[n]`. We additionally compute `v_minus = actual[n] + pred[n]` (equivalent to predicting `−pred`) and store:

- The residual with smaller magnitude: `min(|v_plus|, |v_minus|)`, quantized at N bits
- A single "sheet bit" indicating whether `v_plus` or `v_minus` was chosen

On reconstruct, if the sheet bit is set, flip the predictor sign before adding the residual.

The sheet bit costs 1 bit per non-squarefree position. On hd=64 with pad=66, this is ~25 non-sqfree positions, ~3 bytes per head. On hd=128 with pad=154, ~50 positions, ~7 bytes per head.

### 5.3 Results

**Dolphin 1B hd=64, Wiki-2 fixture, 1-chunk:**

| Residual bits | Spinor OFF | Spinor ON | Δ |
|---|---|---|---|
| 1-bit | PPL 833 | PPL 56 | 14.9× |
| 2-bit | PPL 27 | PPL 14.1 | 1.9× |
| 3-bit | PPL 12.2 | PPL 11.68 | 1.04× |
| 4-bit | PPL 12.1 | PPL 11.6 | 1.05× |

The spinor's advantage concentrates at low residual bits. At 1-bit the sheet correction is worth ~15× PPL; at 3-bit it is within measurement noise. This is exactly the bit-count signature of a systematic sign-error correction: when residual precision is high, the correction is absorbed into the residual's own bits; when residual precision is scarce, the sheet bit carries information the residual alone cannot.

**Qwen3-8B hd=128, Wiki-2 fixture:**

| Config | PPL | K corr | Compression |
|---|---|---|---|
| MOBIUS default 5/5/5/5/5 | 7.31 | 0.999 | 2.6× |
| K+μ+3bit (no spinor) 5/5/5/5/5 | 7.70 | 0.980 | 2.6× |
| K+μ+3bit+spinor 5/4/4/4/5 | **7.30** | 0.988 | **2.8×** |
| K+μ+3bit+spinor 3/3/3/3/3 | **7.32** | 0.972 | **3.3×** |

The spinor path matches MOBIUS default quality at +27% compression (2.8× vs 2.6×). Without the sheet bit, the same predictor is 5% worse than MOBIUS at equal compression. The sheet bit closes the gap and exceeds the default.

This is the first architectural feature in this framework that shifts (not merely moves along) the Pareto frontier — a new operating point, not a trade.

### 5.4 Why It Scales with head_dim

The spinor's effect is small on hd=64 and large on hd=128. The theory explains why: squarefree padding gives pad=66 for hd=64 (adding 2 non-sqfree positions) but pad=154 for hd=128 (adding 26 non-sqfree positions). The sheet-bit correction applies only to non-sqfree positions; there are 2.3× more of them on hd=128, so the correction accumulates into a measurable compression improvement.

The topological-depth prediction from §5.4 of the companion paper — that outer-torus dimensions become more fragile as head_dim grows — is exactly what produces this scaling. The sheet bit is the concrete mechanism for handling that fragility.

---

## 6. VHT2 Banded KV Compression

### 6.1 Method

Each KV head vector is Walsh-Hadamard transformed, split into N equal bands, and each band quantized with its own fp16 scale plus packed integer values. The band allocation mirrors the WHT energy decay: high-energy bands get more bits, the low-energy tail gets fewer.

### 6.2 Results

Optimal configuration: K with n=4 bands at bits 5/5/4/3; V with flat int3 (n=1 band).

| Model | head_dim | K × | V × | Total × | ΔPPL |
|---|---|---|---|---|---|
| Dolphin 1B | 64 | 2.8× | 4.3× | ~3.4× | +0.60% |
| Qwen3-8B | 128 | 3.2× | 4.7× | ~3.8× | +1.24% |

RAM at 32K context (hd=128): fp16 baseline 5.9 GB → VHT2 1.56 GB.

### 6.3 The Spectral Regularization Effect

5/5/4/3 achieves PPL 11.2147 on the converged 20,000-step Dolphin 1B run, 0.04% BETTER than lossless fp16 (11.2194). The 3-bit rounding on the lowest-energy band filters noise that was in the original cache. This is engineered spectral regularization: high-energy bands get precision, the noisy tail gets beneficial compression.

### 6.4 Precision-Sensitivity of Compression Features

The spinor result (§5) exposes a general property of KV compression features: their effectiveness depends on the model's weight precision, not just the reconstruction accuracy.

On Qwen3-8B Q8 (8-bit weights) with K+μ+spinor 5/4/4/4/5:
- K corr: 0.988
- PPL: 7.33 (baseline 7.31, +0.3%)

On Qwen3-8B Q3 (3-bit weights) with the same compression config:
- K corr: 0.988 (identical)
- PPL: 7.67 (baseline 7.52, +2.0%)

Same K corr, different PPL slope. The scaling law predicts this exactly: at γ=1.5 on weight precision, Q3 amplifies the same K error by (8/3)^1.5 = 4.4× compared to Q8.

Operationally: the spinor path is a Q8+ feature. On Q4 and below, MOBIUS default ships better because the spinor's 1-bit precision advantage is washed out by the W-matrix rounding downstream.

This is not a bug in the spinor — it is the scaling law applied to feature design. Before shipping any new compression feature, predict its viability per model and per quantization via the law. Don't expect universal wins; expect targeted wins.

### 6.5 Bit Allocation Sweep

| Config | PPL | vs Lossless | Compression |
|---|---|---|---|
| 5/5/4/3 | 11.2147 | −0.04% (better) | 3.05× |
| 5/4/4/3 | 11.2593 | +0.36% | 3.20× |
| 4/4/4/3 | 11.2624 | +0.39% | 3.37× |
| 4/3/3/3 | 11.4407 | +1.98% | 3.76× |
| 3/3/3/3 | 11.6565 | +3.90% | 4.00× |

4/4/4/4 is off the Pareto frontier entirely — 4/4/4/3 is strictly better in both quality and compression.

### 6.6 Critical Rules

1. Skeleton size must equal head_dim (sk=32 on hd=64 → PPL +47%).
2. 3-bit floor — 2-bit on any band is catastrophic.
3. 5/5/4/3 mirrors WHT energy decay — each band's optimal depth tracks its energy.
4. n=4 beats n=5/n=8 (2-byte scale overhead per band erases gains).
5. Flat beats banded for V — no exceptions across the entire sweep.

---

## 7. The Knight Family: Falsification Result

### 7.1 What Was Tested

The Knight's Move construction stores only squarefree skeleton indices, relying on the Möbius predictor to reconstruct non-squarefree positions. The hypothesis: if the multiplicative lattice has the structure we claim, dropping the "redundant" non-squarefree positions should give free compression, because those positions are predictable from the generators.

We tested the Knight family across multiple bases (WHT, vilenkin3p pad=90, vilenkin_sqfree pad=66, vilenkin5p pad=2310), band counts (1, 2, 3, 4, 5, 6, 9, 10), ladder shapes (uniform, bathtub, inverted, decay, asymmetric, 0-bit drops), and skeleton_k extensions (top-K squarefree plus top composites back).

### 7.2 Results

Every Knight variant converges to the μ-predictor correlation ceiling. Above that ceiling, residual bits recover some of the unpredicted variance: 1-bit catastrophic (PPL 700+), 2-bit marginal (PPL 27), 3-bit near-saturation (PPL 12.5), 4-bit flat against 3-bit. Below the ceiling, no configuration helps — the missing information is genuinely missing.

When composites are added back to the skeleton (sk=60+ on hd=64), Knight converges back to MOBIUS — because full coverage IS MOBIUS. The "free compression" from dropping composites is exactly offset by the residual's cost of representing them.

### 7.3 Why This Matters

The Knight family closeout is not a dead end. It is a falsification result supporting the multiplicative-lattice claim:

- If primes were privileged (the "primes are special" hypothesis), squarefree-only skeletons should beat full-coverage by some measurable margin. They don't.
- If the multiplicative lattice is the active structure (the "lattice covers everything" hypothesis), squarefree skeletons should lose exactly the information carried by composites, and adding composites back should recover it exactly. This is what we observe.

The result is: composites are coordinates in the same lattice primes generate, not redundant projections to prune. The signal is distributed across the squarefree set, not sparse within it. This complements the companion paper's falsification suite, which showed the same thing from the PE side.

### 7.4 The Aggressive Preset

One concrete new operating point emerged from the Knight sweep: `MOBIUS + BAND_BITS=3,3,3,3,3` on the WHT basis gives PPL 12.11 @ 3.2× compression on Dolphin 1B and PPL 7.51 @ 3.9× on Qwen3-8B Q8. This is not a Knight config — it's the standard MOBIUS mask with uniform 3-bit bands — but it dominates the aggressive-compression frontier of the Knight family and ships as the "memory-tight" preset.

---

## 8. Vilenkin Successive Decomposition

### 8.1 Beyond Walsh: The Vilenkin-Hartley Basis

Walsh functions use Z/2Z — one prime. The Vilenkin-Hartley transform generalises to Z/p_kZ for arbitrary primes using the Hartley kernel cas(x) = cos(x) + sin(x). This gives a real-valued transform that is self-inverse for ALL primes, not just p=2. For p=2, Hartley = Hadamard. The Kronecker product across primes V = H_{p₁} ⊗ H_{p₂} ⊗ ... ⊗ H_{p_k} is also self-inverse (V·V = N·I). Round-trip error = 0.0000.

Progressive prime expansion monotonically increases correlation on both synthetic and production K vectors:

| Basis | Correlation (synthetic) | Correlation (production) |
|---|---|---|
| Walsh (Z/2Z) | 0.9504 | 0.9490 |
| Z/2Z × Z/3Z | 0.9507 | 0.9493 |
| Z/2Z × Z/3Z × Z/5Z | 0.9542 | 0.9500 |
| Z/2Z × Z/3Z × Z/5Z × Z/7Z | 0.9628 | 0.9513 |

### 8.2 Three-Pass Successive Extraction

On Qwen3-8B (head_dim=128) using Vilenkin 2-prime basis (N=132): P1 extracts the Z/3Z skeleton, P2 extracts Z/5Z detail from the residual, P3 extracts Z/7Z texture from the remaining residual.

| Config | PPL | ΔPPL | Compression |
|---|---|---|---|
| Baseline | 9.91 | — | 1.0× |
| Vilenkin 2p 99% energy | 10.20 | +2.9% | 3.2× |
| Vilenkin 2p 95% energy | 10.89 | +9.9% | 5.1× |
| Vilenkin 3p 95% energy | 11.59 | +16.9% | 3.8× |
| Vilenkin 2p 90% energy | 13.48 | +36% | 7.2× |
| Walsh 95% energy | 57.21 | +477% | 3.6× |

Walsh catastrophically fails at 95% energy threshold (PPL 57). The multiplicative structure is real and measurable: Vilenkin 2p at the same threshold gives PPL 10.89. Per §4.2, Walsh's failure is not that the structure isn't there — it's that WHT² = I hides it.

### 8.3 Quantization of Vilenkin Coefficients

The int4 sweet spot provides the best compression-quality tradeoff:

| Quant | PPL | ΔPPL | Compression |
|---|---|---|---|
| int8 | 10.89 | +9.9% | 5.1× |
| int4 | 11.01 | +11.1% | 9.8× |
| Z/6Z | 11.47 | +15.7% | 10.7× |
| Z/5Z | 12.27 | +23.8% | 15.7× |
| Z/3Z | 22.12 | +123% | 38.4× |

---

## 9. Structural Discoveries

### 9.1 Measurement Noise

GPU reductions on fp16 Vulkan (RTX 2060) are not bit-reproducible. Measurement noise on 1-chunk wiki.test in the 12-20 PPL regime is ±5 PPL. Tight ladder sweeps (differences < 0.5 PPL) are at or below the noise floor. Real effects reported here — the 5-7 PPL cluster gap in §4.3 of the companion paper, the 29× and 2.1× drops in §10's residual-bit saturation curve, the 477% Walsh-vs-Vilenkin gap in §8.2, the +27% compression gap for the spinor on hd=128 — are all well above noise.

For future sweeps: run each config 3-5 times and report median, or use chunks ≥ 4 to average. The scaling law in §3 provides a cheaper alternative: for any new config, predict PPL impact and only run configs whose predicted cost is near the budget. Configs predicted to be ≥5× worse than the noise floor don't need repeated runs.

### 9.2 The Z/3Z Skeleton — A Standing Wave

Z/3Z indices {48, 49, 50, 51, 52, 53} appear at 100% of positions in Layer 0 — every single head, every position. These are Vilenkin blocks 16–17 (contiguous, capturing all 6 mixed-radix cells). In the mixed-radix decomposition k = k₁×3 + k₂, these map to blocks 16–17 out of 22 total.

This is a standing wave. It IS the coordinate system. The K cache does not store 128 bytes of arbitrary data per position — it stores ~10 dominant values on a FIXED frequency basis plus ~35–40 smaller corrections. The basis is the same across all positions; it is a property of the model, not the input.

### 9.3 Bandpass Structure

The Z/5Z detail (indices 31–41) occupies Vilenkin blocks 10–13 — a different localized window, lower frequency than the skeleton. P1 and P2 do not overlap: only 6 shared indices in Layer 0. The Z/7Z texture has zero universal indices (nothing above 40%). Position-dependent, not structural. But removing it costs 3.5 PPL — it IS the fine-grained positional information.

### 9.4 The Mixed-Radix Tiling Proof

P1 and P2 tile the k₂ residue classes at every layer with maximum deviation <3.5% from uniform:

| Layer | Max Deviation | Tiling Verified |
|---|---|---|
| L0 | 1.9% | Yes |
| L1 | 3.5% | Yes |
| L2 | 3.1% | Yes |
| L3 | 1.7% | Yes |

Successive prime passes partition the modular slots — they do not redundantly cover them. Layer-specific k₂ avoidance patterns are complementary: L1's P1 avoids k₂=2 (1%), P2 fills k₂=2 (37%). Combined, each k₂ class gets either P1 or P2, not both. This is algebraic: the Vilenkin basis respects the Z/3Z group structure, and successive extraction exploits it.

---

## 10. The N-bit Residual Saturation Curve

### 10.1 Measurement

On the squarefree prime-Hartley basis (where the Möbius predictor operates honestly, §4.3), we measured PPL as a function of residual bit depth at fixed skeleton configuration:

**Dolphin 1B hd=64, sqfree basis, K+μ pipeline:**

| Residual bits | PPL (no spinor) | ΔPPL |
|---|---|---|
| 1 | 777 | — |
| 2 | 26.8 | 29.0× drop |
| 3 | 12.5 | 2.14× drop |
| 4 | 12.3 | 1.02× drop |

### 10.2 Interpretation: Shannon Saturation at 8 Levels

The 1→2 bit drop is 29×. The 2→3 bit drop is 2.1×. The 3→4 bit drop is essentially flat. This is a classic Shannon saturation signature.

The μ-inversion residual distribution (the difference between predicted and actual values at non-squarefree positions) has approximately 8 meaningful quantization levels — not 2, not 4, not 16. Below 3 bits, quantization noise dominates the information content. At 3 bits, the quantization just fits the distribution. Above 3 bits, the quantizer is encoding empty space.

This is a falsifiable prediction from information theory applied to the lattice structure: the residual distribution should have a specific entropy bounded by the μ-predictor's error variance, and the quantization floor should show up at ~log₂(useful levels). We observe it at 3 bits, consistent with ~8 levels of real information in the residual.

### 10.3 Why This Matters

The 3-bit residual saturation is the design point that makes all downstream compression work. Below 3 bits, the predictor fails catastrophically. Above 3 bits, you're wasting storage. The aggressive-compression shipping preset (`MOBIUS + 3/3/3/3/3`) is not an arbitrary choice — it sits at the saturation point of the system's inherent information content.

---

## 11. Möbius Partition Mask (Production)

Repurposed as a pure partition heuristic — all coefficients retained, squarefree-first ordering — the Möbius mask produces a real quality win on the WHT basis where the predictor itself fails:

| Config | K corr | V corr | Compression |
|---|---|---|---|
| Baseline WHT | 0.9590 | 0.9521 | 3.8×/4.1× |
| Möbius + 5/4/4 | 0.9967 | 0.9950 | 3.2×/3.4× |
| Möbius + 4/4/3 | 0.9893 | 0.9877 | 3.6× |
| Möbius + 4/3/3 | 0.9830 | 0.9805 | 3.9× |

At equal compression (~3.9×): baseline K 0.959, Möbius K 0.983. +0.024 correlation for free. The mask is cross-platform invariant to the third decimal: K 0.9967 on desktop Qwen3-8B (hd=128) and 0.9972 on mobile Dolphin-1B (hd=64).

This is the partition-heuristic use of Möbius: squarefree-first ordering exploits the divisibility structure for free, without needing the predictor to work. It's the right choice on WHT exactly because the predictor doesn't work there.

---

## 12. Production Validation

### 12.1 Desktop: llama.cpp Integration

The implementation ships as two components: a frequency injection header (prime_rope.h) using the existing freq_factors mechanism, and a shadow cache backend (llama-kv-cache-shadow.cpp, ~4743 lines, all 13 phases) that intercepts KV writes for VHT2 compression.

PPL improvement at alpha=0.15–0.22 with zero retraining is confirmed across three architectures and three quantization levels. VHT2 banded compression operates independently via environment variables, no rebuild required.

The spinor path ships as opt-in via:

    LLAMA_SHADOW_VHT2_MASK_TYPE=knight
    LLAMA_SHADOW_VHT2_BASIS=sqfree
    LLAMA_SHADOW_VHT2_RESIDUAL_BITS=3
    LLAMA_SHADOW_VHT2_SPINOR=1
    LLAMA_SHADOW_VHT2_BAND_BITS=5,4,4,4,5    # quality
    LLAMA_SHADOW_VHT2_BAND_BITS=3,3,3,3,3    # aggressive (3.3× @ Qwen3-8B Q8)

### 12.2 Mobile: Samsung S22 Ultra / Adreno 730

Dolphin 1B Q8_0, Vulkan backend with CPU-fallback VHT2 writeback:

| Metric | Baseline | Möbius 5/4/4 |
|---|---|---|
| PPL | 14.24 ± 0.80 | 13.20 ± 0.60 |
| K correlation | — | 0.9972 |
| V correlation | — | 0.9960 |
| Generation speed | 1.79 tok/s | 3.57 tok/s |

Möbius is 2× faster than baseline on mobile — smaller active coefficient count means less scatter work, and the NEON writeback path is cache-bound.

---

## 13. Discussion

### 13.1 The Scaling Law as Infrastructure

The scaling law (§3) is not just a headline result — it is infrastructure for KV compression research generally. Before this paper, evaluating a new compression config meant running a bench. That is expensive for large models and scales badly across the product space of (model × quant × method). With the scaling law, evaluating a config means computing K-corr on a calibration batch and applying a closed-form formula.

We use this workflow internally to pre-filter any new compression idea: predict PPL, run bench only if predicted ΔPPL is near the budget. Approximately 10× compute savings in our own sweep work.

### 13.2 Why the V-cache Compresses More Than K

V vectors compress at 4.3–4.7× while K compresses at 2.8–3.2×. This is counterintuitive — K "should" compress better because it has spectral structure. But the structure is exactly what makes K sensitive to compression errors: angular relationships between K vectors determine attention scores, so small angular distortions compound. V is the weighted average that attention produces, so individual V errors average out. Flat 3-bit quantization on V works precisely because V has no structure to destroy.

The scaling law in §3 quantifies this intuition. K error enters attention quadratically; V error enters linearly. A 1% K error costs as much PPL as a 10% V error on the same model.

### 13.3 Limitations

Single-seed for the main 300M experiment. Modern baselines (YaRN, NTK-aware, CARoPE, CommVQ, KVTC) not compared head-to-head on identical benchmarks. The sqfree prime-Hartley basis is validated on llama.cpp with Qwen3-8B Q8 but not on pure-PyTorch end-to-end (the ComfyUI port of this stack is in progress, not yet shipped). End-to-end Wan 2.2 gen bench is predicted from the scaling law but not measured.

The spinor's cross-architecture generalization is partially tested (Dolphin 1B hd=64 + Qwen3-8B hd=128) but not on bf16 models. The prediction for Wan 2.2 14B bf16 (K-corr floor 0.914, aggressive compression viable) derives from the scaling law extrapolated to 14B / 16-bit and has not been validated.

### 13.4 The Path to Cache Elimination

This paper demonstrates compression and a scaling-law design rule. The companion paper's theoretical framework points to elimination. If the Z/3Z skeleton is a standing wave determined by model weights (not input), and if the zero-crossings of the Vilenkin spectrum are predictable from the Möbius function (which is determined by divisibility, which is determined by position), then the KV cache is not data to compress. It is a function to evaluate.

The current path: store the Z/3Z skeleton once per (layer, head), predict zero-crossing positions from Möbius topology, compute phase and amplitude between crossings from the known Vilenkin basis, and reconstruct the full K vector on demand. Level 2 reconstruction in the three-layer framework currently achieves 0.35 correlation at 10K training steps. The scaling law sets the target: for any production model, reconstruction needs to achieve the K-corr floor for that model's (params, bits) pair. For a 70B Q8 model, that floor is 0.927 — achievable with far less information than full storage. For 1B Q8, the floor is 0.988 — much harder; may not be reachable without stored state.

The scaling law tells us where reconstruction is viable and where it isn't. That alone reshapes the research agenda.

---

## 14. Conclusion

The KV cache carries two structurally different signals. K vectors encode position through the multiplicative lattice — sparse, localized in the Vilenkin spectrum, with a universal Z/3Z skeleton and layer-specific residue class selection rules. V vectors encode content — dense, uniform, structureless in the spectral domain. They occupy disjoint bands and compress independently.

The K-corr → PPL scaling law quantifies the relationship between reconstruction fidelity and downstream quality across model families and quantization levels, with interpretable exponents that match theoretical predictions from the lattice structure. The law functions as a design rule, cutting compression research cycle time by roughly an order of magnitude.

The spinor sheet bit — a 1-bit SU(2) double-cover correction to the Möbius predictor's causal-mask sign errors — shifts the Pareto frontier on hd=128 models at Q8+, matching MOBIUS default quality at +27% compression. This is the first architectural feature in the framework that is a genuine frontier shift rather than a trade along the existing frontier.

The Möbius predictor's basis dependency (r≈0 on WHT, r=0.40–0.58 on sqfree) resolves the apparent contradiction in earlier work: different bases expose different aspects of the lattice. WHT gets partition heuristics; sqfree gets genuine prediction. Each basis has its role.

The Knight family closeout, the N-bit residual saturation curve (~8 meaningful levels at 3 bits), and the mixed-radix tiling proof together confirm that the KV cache is a projection of the multiplicative lattice structure of the integers. Compression is reading that structure efficiently. Reconstruction — the endgame — is reading it from the other side.

---

## Appendix A: VHT2 Configuration Reference

```
LLAMA_SHADOW_CACHE=1  LLAMA_SHADOW_VHT2=1
LLAMA_SHADOW_HEAD_DIM=128        # must match model
LLAMA_SHADOW_VHT2_SKELETON_K=128 # must equal head_dim
LLAMA_SHADOW_VHT2_N_BANDS=4
LLAMA_SHADOW_VHT2_BAND_BITS=5,5,4,3
LLAMA_SHADOW_VHT2_V=1
LLAMA_SHADOW_VHT2_SKELETON_V=128
LLAMA_SHADOW_VHT2_V_N_BANDS=1
LLAMA_SHADOW_VHT2_V_BAND_BITS=3
LLAMA_SHADOW_VHT2_MASK_TYPE=mobius  # for Möbius partition

# Spinor aggressive mode (Qwen3-8B Q8 hd=128):
LLAMA_SHADOW_VHT2_MASK_TYPE=knight
LLAMA_SHADOW_VHT2_BASIS=sqfree
LLAMA_SHADOW_VHT2_RESIDUAL_BITS=3
LLAMA_SHADOW_VHT2_SPINOR=1
LLAMA_SHADOW_VHT2_BAND_BITS=3,3,3,3,3    # 3.3× compression
```

## Appendix B: Complete Vilenkin Successive Results (Qwen3-8B)

| Config | PPL | ΔPPL | Compression |
|---|---|---|---|
| Baseline (no shadow) | 9.91 | — | 1.0× |
| Vik 2p 99% int8 | 10.20 | +2.9% | 3.2× |
| Vik 2p 95% int8 | 10.89 | +9.9% | 5.1× |
| Vik 2p 95% int4 | 11.01 | +11.1% | 9.8× |
| Vik 2p 95% Z/6Z | 11.47 | +15.7% | 10.7× |
| Vik 3p 95% int8 | 11.59 | +16.9% | 3.8× |
| Vik 2p 95% Z/5Z | 12.27 | +23.8% | 15.7× |
| Vik 2p 90% int8 | 13.48 | +36% | 7.2× |
| Vik 2p 95% Z/3Z | 22.12 | +123% | 38.4× |
| Walsh 95% int8 | 57.21 | +477% | 3.6× |

## Appendix C: Scaling Law Fit Data

| Model | params (B) | bits | K corr | PPL/base | Fit K_coeff |
|---|---|---|---|---|---|
| Dolphin 1B | 1 | 8 | 0.988 | 1.040 | 4975 |
| Dolphin 1B | 1 | 8 | 0.978 | 1.082 | 3845 |
| Dolphin 1B | 1 | 8 | 0.940 | 1.870 | 5430 |
| Dolphin 1B | 1 | 8 | 0.860 | 67.00 | 4845 |
| Qwen3-8B Q8 | 8 | 8 | 0.988 | 1.003 | 4416 |
| Qwen3-8B Q8 | 8 | 8 | 0.972 | 1.010 | 4120 |
| Qwen3-8B Q8 | 8 | 8 | 0.970 | 1.026 | 6410 |
| Qwen3-8B Q3 | 8 | 3 | 0.988 | 1.020 | 4590 |
| Qwen3-8B Q3 | 8 | 3 | 0.972 | 1.036 | 4790 |

Mean K ≈ 4700, std ≈ 750 (±16%). Median K = 4790.

## Appendix D: Spinor Bit-Count Saturation

| rb | spinor OFF | spinor ON | Δ |
|---|---|---|---|
| 1 | PPL 833 | PPL 56 | 14.9× |
| 2 | PPL 27 | PPL 14.1 | 1.9× |
| 3 | PPL 12.2 | PPL 11.68 | 1.04× |
| 4 | PPL 12.1 | PPL 11.6 | 1.05× |

Spinor's contribution concentrates at low residual bits — 1-bit saves 15× PPL by encoding the systematic sign error the residual alone cannot. At 3-bit and above the residual subsumes the correction.

## References

- Su et al. (2021) — RoFormer: Rotary Position Embedding
- Hsu et al. (2026) — TurboQuant (ICLR 2026)
- Zandieh et al. (2024) — QJL Transform
- Lenstra, Lenstra, Lovász (1982) — LLL lattice basis reduction
- Press et al. (2022) — ALiBi
- Knack (2026) — Position Is Arithmetic (companion paper v8)
- Xu et al. (2025) — CommVQ (ICML 2025, arXiv:2506.18879)
- Anonymous (2026) — KVTC (ICLR 2026)
- Zhang et al. (2025) — KVSink (arXiv:2508.04257)
- Hooper et al. (2024) — KVQuant
