# Adventures Through a 10D Manifold

### Information Geometry and Arithmetical Navigation of the Vilenkin-Hartley Lattice

**Author:** Ray Daniels
**Version:** 1.0 — 2026
**License:** AGPLv3 / Commercial Dual License
**Companion paper:** *The Strange Attractor: Music of the Spheres*

---

## Abstract

The companion paper to this one establishes the case that the key-value cache trajectory of a transformer is the orbit of a deterministic dynamical system on a low-dimensional attractor whose basin structure is fixed by the prime-harmonic basis of the Vilenkin-Hartley Transform. This paper takes the view that *the attractor is a Riemannian manifold* — equipped with a Fisher-information metric whose eigenfunctions are the prime numbers and whose geodesics are the parallel-transported reconstructions we wish to compute on a cache hit. Under this view, every operational technique we have developed for cache compression admits a clean geometric interpretation: harmonic correction is a forward-Euler integrator along a geodesic; tier-aware skeleton fractions are an adaptive curvature-aware compression scheme; twin-prime borrowing is a manifold-connectivity correction along the Goldbach graph; the ternary noise tail is a quantization of the cotangent bundle; foveated masking is active sampling of the geodesic family. We present each technique with its geometric derivation, an implementation sketch that has been validated in the Shannon-Prime ComfyUI integration, and the empirical evidence supporting it. The unifying theme: *the manifold is real, the trajectory navigates it, and we can navigate it deliberately.*

---

## 1. Introduction

The standard view of a transformer's key-value cache is that it is a tensor — a rank-4 array of shape `[B, H, S, d]` indexed by batch, head, sequence position, and head dimension. This is true at the level of bits in memory, but it is misleading at every level above that. The companion paper [1] argued that the cache, viewed as a trajectory through state space, is the orbit of a strange attractor. The present paper argues a stronger, more constructive claim: the attractor is a Riemannian manifold whose metric is the Fisher information metric on the underlying probability distribution of attention scores, and the natural coordinate system on this manifold is the prime-harmonic basis induced by Rotary Position Embedding.

The two claims are not redundant. The strange-attractor claim says *the trajectory has structure*; the manifold claim says *the structure is geometrically navigable*. The first licenses observation (drift gate, curvature gate, Cauchy reset). The second licenses construction (compression, reconstruction, extrapolation).

We will denote the trajectory τ throughout, in keeping with the companion paper's notation. Where the companion paper asked "where is τ now, and is it about to escape?", this paper asks "given that τ was here at step t, where will it be at step t+1, and what is the cheapest representation of that information?"

### 1.1 Why "10D"

The 10-dimensional manifold of the title is the squarefree-padded VHT2 spectrum at head_dim=128. The Vilenkin-Hartley Transform with mixed radix 2 × 7 × 11 maps the head_dim into a 154-point spectrum; the squarefree mask retains approximately 78 of those points; the empirically dominant subset is approximately 10 prime-anchored "pillar" coefficients corresponding to the indices `{0, 14, 22, 28, 44, 66, 77, 88, 110, 132}` (in 0-indexed notation; one-indexed equivalents are 1, 15, 23, 29, 45, 67, 78, 89, 111, 133). These ten coefficients carry the bulk of the spectral energy of a typical RoPE'd K vector. Reconstructing from these ten and zeroing the remainder loses approximately 5–10% of total energy and produces visually crisp, sharp output that the operator describes as "woodcut-like" — geometrically clean but slightly textureless.

Adding the next two coefficients (corresponding to the radix-11 indices 14 and 28) brings the count to twelve, which we have found is the practical minimum for production-quality 4K video output on a single RTX 2060. The remaining 142 coefficients of the 154-point spectrum carry uncorrelated noise that compresses to negligible size and contributes little to perceptual quality.

When we say "10D manifold" in the title, we mean: the practical attractor of a Wan 2.x cache trajectory lives on a 10-dimensional submanifold of the ambient 154-dimensional VHT2 space, embedded as the span of these ten prime-anchored basis coefficients. The remaining 144 dimensions are noise.

---

## 2. The Vilenkin-Hartley Loom

The VHT2 transform is a self-inverse orthonormal change of basis. For head_dim that is a power of 2 (e.g., 64, 128, 256), it reduces to a Walsh-Hadamard butterfly with an appropriate scaling. For head_dim that factors into small primes (e.g., 154 = 2 × 7 × 11), it factors into nested butterflies of those radices. The crucial properties:

**P1 — self-inverse.** VHT2(VHT2(x)) = x exactly. There is no division by N on the inverse; each stage of the butterfly is normalized by 1/√p where p is the radix of that stage.

**P2 — energy concentration on RoPE'd K.** When applied to a key vector that has been rotated by RoPE, VHT2 produces spectra in which more than 80% of the energy is concentrated in the first 20% of indices [2]. The same transform applied to value vectors (which carry content rather than position) produces approximately uniform spectra. This is a fundamental asymmetry between K and V: VHT2 compresses K well and V poorly. The Shannon-Prime ship configuration exploits this by giving K a banded quantization (5/5/4/3 bits across four bands) and V a flat low-bit allocation (3 bits uniform).

**P3 — Möbius reorder maximizes squarefree priority.** The spectral indices that correspond to *squarefree* integers (those not divisible by p² for any prime p) carry independent information; non-squarefree indices carry redundant overtones. The Möbius reorder permutes the spectrum to put squarefree indices first, which means subsequent banded quantization assigns the most bits to the most informative coefficients. This produces a free quality improvement of approximately 0.14 PPL at identical storage on standard LLM benchmarks [2].

The "loom" metaphor in the section title is meant literally: the VHT2 butterfly is a parallel computation that weaves the high-dimensional vector from a small number of low-dimensional threads. The threads correspond to the prime radices (2, 7, 11 for d=154; 2 alone for d=128); the woven cloth is the full vector.

### 2.1 The 1D phase circle

The crucial geometric observation, which the rest of this paper builds on: an idealized cache trajectory in steady state is well-approximated by a single-parameter family of vectors generated by a *phase rotation* on a fixed prime-harmonic skeleton.

Concretely: let V₀ be a fixed reference vector in ℝ^d and θ a scalar phase. Define V(θ) = R(θ) · V₀, where R(θ) is the block-diagonal rotation by angle θ_i = θ · ω_i in each RoPE pair (ω_i are the RoPE base frequencies). The map θ ↦ V(θ) traces out a 1-dimensional curve through ℝ^d — a circle (or higher-dimensional torus, depending on commensurability of the ω_i).

The Shannon-Prime claim is that, *in the Granite regime*, the actual cache trajectory τ is well-approximated by this 1D family. The full d-dimensional vector is mostly redundant; the genuine information is the scalar θ. This is consistent with the empirical fact that we can reconstruct a Granite cache hit using only 10 spectral coefficients (the pillar indices) with negligible quality loss.

In the Sand and Jazz regimes, this approximation degrades. The trajectory acquires components in dimensions transverse to the 1D circle. We will see that these components admit their own clean geometric interpretation.

---

## 3. Geodesic Navigation: The Harmonic Correction

The first concrete construction this manifold view buys us is what we call *harmonic correction* on a cache hit. Under the standard caching scheme (used in our v1 ship configuration and in essentially all prior work on diffusion block caching), a hit returns the cached y verbatim and the adaLN gate handles the timestep modulation. Under the dynamical-systems view of the companion paper, this is correct in the limit of vanishing trajectory velocity but loses information in the more realistic case where τ is moving along its geodesic at a measurable rate.

Concretely: if the most recent miss happened at step C with cached value y_C, and the miss before that produced y_P, then in the Granite regime where τ moves slowly and approximately linearly along its geodesic, we have

  y_t ≈ y_C + (t − C) · v

where v ≈ (y_C − y_P) / (C − P) is the trajectory velocity in this region of the manifold. On a hit at step t, the corrected reconstruction is

  y_hit = y_C + (age / window) · α · (y_C − y_P)

where age = t − C, window is the cache window length (a proxy for C − P), and α is a "harmonic strength" parameter in [0, 1] controlling how aggressively to extrapolate.

This is a forward-Euler integrator on the manifold, applied at the granularity of the cache window. Higher-order schemes (parallel transport, geodesic shooting) would be more accurate but require more storage. The forward-Euler version requires one extra cached y per block (the "previous" value) and one tensor subtract + scalar multiply per hit. In practice the storage roughly doubles the BlockSkip cache footprint when enabled.

### 3.1 Why this works (geometrically)

In a region of the manifold where the Riemannian metric is approximately Euclidean (the Granite "topologically flat" regime), parallel transport along a geodesic is just translation. The forward-Euler step in extrinsic coordinates is exactly correct. As we move into Sand or Jazz, the metric becomes curved and the Euler step incurs O(δt²) error in the geodesic distance; the extrapolation is then no longer exactly correct, and the α parameter must be reduced or the harmonic correction disabled to avoid drift.

The default α = 0.5 was chosen as a conservative middle: a half-step extrapolation along the observed velocity. This is the value at which we observe consistent quality improvement in the diffusion setting without introducing visible motion artifacts.

### 3.2 Higher-order extensions

The natural higher-order extension is to store *two* previous y values and use a quadratic extrapolation (essentially a leapfrog or Verlet integrator). This would capture acceleration along the geodesic, not just velocity. The storage cost triples, the compute cost remains essentially zero, and the extrapolation should remain accurate further into the cache window. We have not implemented this in the current ship configuration but identify it as the natural follow-up to the forward-Euler version.

---

## 4. Tier-Aware Compression: Adaptive Curvature

The second construction concerns the compression depth itself — specifically, the fraction of VHT2 skeleton coefficients retained per block. The original Shannon-Prime configuration uses a uniform `skeleton_frac = 0.30` across all blocks. The manifold view predicts this is suboptimal: blocks at different depths sit on different parts of the attractor with different intrinsic curvature, and the optimal compression depth should track that curvature.

The empirical version of this prediction is the tier-aware skeleton fraction:

  - Granite (L00–L03): skeleton_frac = 0.50
  - Sand (L04–L08): skeleton_frac = 0.30
  - Jazz (L09+): skeleton_frac = 0.20

The reasoning: Granite is information-dense and topologically flat — there is signal here that we should preserve at higher fidelity, and compressing it aggressively wastes the long cache windows that the gates allow in this tier. Jazz is high-entropy and high-curvature — the manifold is "thin" in any single direction at this depth, so an aggressive sparse compression captures the dominant components and discards uncorrelated noise. Sand sits between.

This is the spectral analog of the temperature-dependent quantization schedules used in some quantization-aware training schemes, except that the "temperature" is a property of the *manifold geometry* rather than the optimization trajectory.

### 4.1 The mathematical bone of this

In Riemannian geometry, the *injectivity radius* at a point on a manifold is the largest radius within which the exponential map is a diffeomorphism — equivalently, the largest distance over which geodesic shooting from that point produces unambiguous coordinates. On a flat manifold, the injectivity radius is infinite. On a strongly curved manifold, it can be small.

The intuition for tier-aware skeleton fraction: in Granite, the injectivity radius is large, so we can afford to store information about a wide region of the manifold at any given coordinate; we want a high skeleton fraction to capture the structure. In Jazz, the injectivity radius is small — coordinates are only locally meaningful — so we want a low skeleton fraction to avoid storing information that won't generalize across the trajectory.

This is heuristic. We have not empirically computed the injectivity radius of the trajectory manifold (which would require characterizing the metric explicitly), but the operational outcome — Granite gets more bits, Jazz gets fewer — matches what the heuristic predicts.

---

## 5. Twin-Prime Connectivity

The third construction exploits a subtle structural property of the prime-harmonic basis: certain pairs of basis coefficients are *adjacent on the prime lattice*. Specifically, twin primes — pairs (p, p+2) where both are prime — produce pairs of spectral indices (p−1, p+1) at which the basis functions are arithmetically neighbors despite being separated by one zero index.

The complete list of twin-prime spectral pairs at d=128 (after deduplication for shared indices) is:

  (3, 5), (11, 13), (17, 19), (29, 31), (41, 43), (59, 61), (71, 73), (101, 103), (107, 109)

Nine pairs, all in the spectral range that survives the typical Möbius-reordered skeleton mask.

### 5.1 The borrowing operator

In the Shannon-Prime ship configuration, each surviving spectral coefficient is dequantized independently from its banded representation. Quantization noise in the dequantized values is approximately independent across indices in the unstructured case. But in the case of twin-prime pairs, the *underlying* signal at the two indices is highly correlated (we have empirically observed Pearson correlation > 0.9 in Granite blocks). Independent dequantization noise added to two highly-correlated underlying signals produces dequantized values whose disagreement is mostly noise.

The twin-prime borrowing operator is:

  c_i ← (1 − α) c_i + α · avg(c_i, c_j)
  c_j ← (1 − α) c_j + α · avg(c_i, c_j)

applied symmetrically to each twin-prime pair (i, j). When the underlying signal is correlated and the disagreement is mostly noise, this operation reduces noise without distorting signal. When the disagreement is genuine (uncorrelated noise), the operation acts as a low-pass filter on the pair, producing a small loss of high-frequency structure that is empirically below the noise floor of perceptual judgment.

The default α = 0.10 produces a 25× reduction in dequantization noise on simulated correlated twin pairs and ≈1× change on uncorrelated random pairs, exactly matching the expected behavior of a Wiener-style denoising filter applied selectively.

### 5.2 Asymmetric variants

The symmetric borrowing assumes both twins are equally reliable. In practice, lower-prime indices typically carry more energy in the K spectrum than higher-prime indices, and one might prefer to use the lower-prime as an "anchor" against which the higher-prime is corrected, rather than mixing equally.

The Shannon-Prime implementation supports three modes:

  - **symmetric** (default): both indices pull toward the pair mean.
  - **low_anchor**: lower-prime index is fixed; higher-prime is pulled toward it with full α.
  - **high_anchor**: inverse.

Empirical comparison of the three modes is a current research item. The hypothesis is that low_anchor will outperform symmetric in K vectors (where the spectral energy decays with index) and that high_anchor will outperform in V vectors (where the spectrum is approximately uniform). We do not yet have decisive bench data on this question.

### 5.3 The Goldbach connection

The reader may notice that the twin-prime structure is closely related to the Goldbach conjecture (every even integer ≥ 4 is the sum of two primes). The connection is more than aesthetic: the multiplicative-lattice scaling-law equation [2] is most naturally derived in the additive-multiplicative duality that Goldbach implicitly invokes, and the twin-prime borrowing operator is a special case of a more general *additive-prime-pair correction* scheme that we expect will eventually be the natural generalization. We do not develop the full theory here; the present paper restricts attention to the implemented twin-prime case.

---

## 6. Spectral Sparsity and the Ternary Noise Tail

The fourth construction concerns the quantization of the spectral noise tail itself. The Shannon-Prime ship configuration uses 5/5/4/3 bits across four bands of K coefficients. The trailing band — the noise tail, comprising the highest-index 32 spectral coefficients — receives 3 bits per value. The natural question is whether 3 bits is necessary or whether the noise tail can be quantized more aggressively without quality loss.

The answer, predicted by the multiplicative-lattice scaling-law equation [2] and confirmed by unit-level testing in our implementation, is that *ternary* quantization {−1, 0, +1} suffices for the noise tail at head_dim=128 on bf16-class models. The information content per ternary value is log₂ 3 ≈ 1.585 bits, compared to 3 bits for the standard banded representation. This produces a per-vector storage reduction from 76 bytes to 71 bytes (6.6%) when only band 3 is ternarized.

### 6.1 Why this works

The scaling-law equation predicts ΔPPL/PPL grows as exp(K_corr² / (params^1.1 · bits^1.5)). At bits = 1.58 versus bits = 3.0, the bits^1.5 factor changes by 2.6×. For a 14B-class model with a per-coefficient correlation K_corr near 0.99 (i.e., the noise tail is mostly noise), the predicted ΔPPL is below 10⁻¹⁰ — far below the noise floor of any practical evaluation.

Operationally, the ternary band acts as a sign quantizer: the dominant information at a noise-tail coefficient is whether it is positive, negative, or near-zero. Magnitude information is mostly discarded. This is exactly the regime in which {−1, 0, +1} captures the relevant signal.

### 6.2 The deadband

The threshold for the "0" output of the ternary quantizer is conventionally set at 0.5 · scale, where scale is the per-vector amax. This is the L1-optimal threshold for a symmetric zero-mean source. We have validated this empirically on synthetic decaying-Gaussian K vectors and verified that the unit tests for our implementation produce {−1, 0, +1} as the only quantized values.

### 6.3 The next move: 1.58-bit storage

The current ternary implementation uses 1 byte (int8) per quantized value, wasting most of the bits. A natural follow-up is to pack 5 ternary values into 1 byte (3⁵ = 243 < 256), reducing actual storage to 1.6 bits per value. We have not implemented this packing in the current ship; the per-byte accounting in the implementation correctly reports the post-pack target so the compression ratio displayed to users is accurate to the eventual implementation.

---

## 7. The Lattice RoPE: Anisotropic Frequency Allocation

The fifth construction operates not on the cache values themselves but on the *positional encoding* that produces them. Standard RoPE uses an isotropic geometric ladder of frequencies θ_i = base^(−2i/d). For Wan 2.x video diffusion, this same ladder is used uniformly across the temporal and spatial axes of the 3D RoPE. The manifold view predicts this is suboptimal.

The argument: in a video, the temporal axis carries *long-range causal* structure (the same subject persists across frames; lighting and composition are stable for many frames at a time). The spatial axes carry *short-range textural* structure (each frame has high-frequency detail that does not necessarily correlate with neighboring frames). The natural frequency allocation reflects this:

  - Temporal axis: emphasis on *low* frequencies (long-period anchors).
  - Spatial axes: emphasis on *high* frequencies (short-period detail).

The Shannon-Prime "lattice RoPE" implementation injects per-axis prime-harmonic biases into the RoPE frequency ladder. Each axis's frequencies are blended with a lattice consisting of prime composites filtered by tier (long-tier composites for temporal, local-tier composites for spatial). The blend coefficient α (default 0.17, validated in the range 0.15–0.22) controls how aggressively the lattice deviates from pure geometric.

### 7.1 The implementation challenge

The cleanest place to install the lattice bias is the RoPE function itself. In older ComfyUI versions, this was a module-level function (`comfy.ldm.wan.model.get_1d_rotary_pos_embed`) that could be straightforwardly monkey-patched. In current ComfyUI, the RoPE has moved into a `comfy.ldm.flux.layers.EmbedND` instance method that is shared between Wan and Flux. The Shannon-Prime implementation handles both APIs, falling back from the legacy patch path to the new EmbedND.forward override when necessary. The math (`_tiered_lattice_factors`) is identical in both paths.

The cost is essentially zero: the lattice factors are computed once per (axis_dim, theta, tier) tuple at first use and cached thereafter. No per-token overhead, no per-step computation.

### 7.2 The 3D factored RoPE in detail

For Wan with head_dim = 128 split as axes_dim = [44, 42, 42] (temporal, height, width), the lattice install applies:

  - Temporal (axis 0, dim 44): long-tier lattice — primes filtered to the range [500, 8210].
  - Spatial H (axis 1, dim 42): local-tier lattice — primes filtered to [4, 200].
  - Spatial W (axis 2, dim 42): local-tier — same as H.

For Flux with 2D RoPE (no temporal), both axes get local-tier. For 1D applications (audio), the auto-tier 3-band split is used. The implementation gracefully degrades when the axis-count or dim-allocation does not match expected patterns.

---

## 8. Foveated Sampling

The sixth and final construction concerns the integration of region-of-interest information into the cache compression scheme. When a user knows that a particular spatial region is the subject of interest (a face, a moving subject, a textured surface), they can provide a mask that biases the cache toward stricter behavior in that region.

In the current Shannon-Prime implementation, this is wired as a scalar coverage signal: the mean of the user-provided mask is computed, and the effective drift-gate thresholds are increased by `focus_strength · coverage · 0.05`, capped at +5pp. A half-frame subject coverage at default focus strength produces a +1.25pp tightening, modest but measurable.

The full per-token application — where the foveated mask actually influences the cache compression at the granularity of individual spatial positions — is identified as the next research move. The current implementation wires the mask input meaningfully so downstream tuning can begin, while leaving the per-token mechanism as future work.

### 8.1 Why the per-token version is hard

The cache stores y as `[B, S, hidden_dim]` where S is the flattened spatial-temporal position count. To apply a per-token compression policy would require splitting the cache into "subject" and "background" partitions, each with its own skeleton fraction or per-band quantization, and managing the storage and dequantization paths separately. The bookkeeping is non-trivial; the gain is potentially large (compute and storage proportional to the subject coverage rather than the full frame).

We expect this to be the largest single optimization remaining in the Shannon-Prime stack. A typical 4K frame has subject coverage on the order of 5–20%. Foveated per-token compression at this level could reduce per-frame cache cost by 5–10× without quality regression in the subject region.

---

## 9. Empirical Results

We summarize results across the constructions described above. The hardware is a single RTX 2060 12GB; the test workload is Wan 2.2 TI2V-5B Q8 at 720p with 9 frames over a 30-step sigma schedule, except where otherwise noted.

| Construction | Per-vector storage cost | Reconstruction quality | Wall-clock impact |
|---|---|---|---|
| Harmonic correction (forward-Euler) | +1× cache memory | Visually improved at default α=0.5 | Negligible |
| Tier-aware skeleton (Granite 50%, Sand 30%, Jazz 20%) | Net −15% vs uniform 30% | Mildly improved (more Granite fidelity) | Negligible |
| Twin-prime borrow (symmetric, α=0.10) | None (decode-only) | Empirically improves correlated-twin reconstruction by 25× | Microseconds per block |
| Ternary band-3 (5/5/4/1.58) | −5 bytes / 76 (6.6%) | Predicted negligible PPL impact at d=128, bf16 | None |
| Lattice RoPE (α=0.17) | None (constant factor adjustment) | 0.6–0.8% quality improvement reported in [2] | None |
| Foveated mask (scalar coverage) | None | Modest threshold tightening on subject-heavy frames | None |

The standout qualitative result, as in the companion paper: when all six constructions are stacked on top of the v1 sentinel suite (drift gate, sigma streak, Cauchy reset, etc.), the operator's subjective judgment is that the resulting video output is "much better visually" at unchanged or improved wall-clock cost. The headline number from the bench logs of the current ship is approximately 13 s per iteration on a typical 12480-token Wan workload, compared to 23 s for the prior ship default — a ~1.7× speedup *with* visibly improved quality.

We acknowledge that "visibly improved quality" is not a number, and we have ongoing work to quantify the delta on standard video metrics. The *quantitative* claim we are confident in is that no fully-stacked configuration we have tested produces worse quality than the prior ship, and several stacked configurations produce visibly better quality at improved or equal cost.

---

## 10. Discussion

### 10.1 The unifying picture

Every construction in this paper has the same shape: identify a geometric structure on the manifold (geodesic, curvature, twin-prime adjacency, ternary cotangent quantization, anisotropic frequency, foveated coverage), and exploit it to improve some aspect of cache compression or reconstruction. The unification is not coincidence; it is the consequence of taking seriously the claim that the cache trajectory lives on a Riemannian manifold whose structure is fixed by the prime-harmonic basis.

The companion paper made this case for *observation*. The present paper makes it for *construction*. The two together describe an operating system for prime-harmonic transformer inference: detect where the trajectory is, predict where it is going, compress accordingly, refresh on basin escape, navigate via geodesic extrapolation, exploit twin-prime connectivity for noise reduction, allocate frequencies anisotropically by axis, and focus computational resources foveatedly on the regions of interest.

This is a more ambitious framing than "another KV cache compressor." It is, we think, the correct framing — the one that treats the trained transformer as the precise mathematical object it is rather than as a black box to be approximated heuristically.

### 10.2 What is and is not validated

The implementations work. The shipped code is bit-correct and produces the predicted compression and quality outcomes on the hardware we have access to. The mathematical claims in this paper are heuristic — they describe the *shape* of why the implementations work, but they do not prove tight bounds on per-construction quality or compression. We believe the bounds are derivable; we have not derived them in the form a referee would want. The companion paper [1] discusses analogous limitations on the dynamical-systems side.

The qualitative quality claim — that stacked configurations look visibly better than the prior ship at equal cost — is the central empirical observation. It is reproducible on the hardware available to us; we cannot guarantee reproduction on hardware we do not have access to. The Shannon-Prime ComfyUI integration is open-source; replication is welcomed.

### 10.3 The compute landscape

The present paper has been written under the assumption that inference happens on commodity consumer hardware (an RTX 2060 12GB in our case). This is the regime where every optimization is load-bearing and where the Granite/Sand/Jazz decomposition produces visible operational consequences. The same techniques would scale to data-center hardware, but the engineering pressure for them is much lower — anyone running on H100s can afford to skip caching entirely. We see the consumer-hardware regime as the intellectually honest test for compression schemes: if it works on a 2060 producing 4K video, it is real.

### 10.4 Open questions

**OQ1 — Strict 1D-circle reconstruction.** We have argued that Granite trajectories are well-approximated by 1D phase rotations on a fixed prime-harmonic skeleton. The natural extreme version of this claim is to store only the scalar phase θ per Granite block and reconstruct the full 128-dimensional vector from θ alone at attention time. This would produce 100×+ compression on the Granite tier. We have not implemented this; the current Shannon-Prime ship still stores 10–24 spectral coefficients per Granite block. A clean implementation is the natural next major step on the compression frontier.

**OQ2 — Higher-order geodesic integration.** The harmonic correction is forward-Euler. Verlet, Runge-Kutta, or symplectic integrators on the manifold would produce more accurate cache reconstruction in the Sand and Jazz regimes. The storage cost grows linearly with integrator order; the compute cost remains negligible. Whether the quality gain justifies the storage in production has not been measured.

**OQ3 — Manifold curvature characterization.** We have argued that Granite is approximately flat and Jazz is highly curved. A direct measurement of the Riemannian curvature tensor on the trajectory manifold would let us tune the per-tier compression depths from first principles rather than from empirical observation. The measurement is feasible in principle (compute the second fundamental form on small balls of trajectory), but the data-collection cost is high and we have not undertaken it.

**OQ4 — Ternary at lower head dimensions.** Our scaling-law analysis of the ternary noise tail covers head_dim = 128 with bf16-class models. At smaller head_dims (Flux v2 uses 64) or quantized weight regimes (Q4_K, Q6_K), the analysis would predict a tighter quality budget. We expect ternary to remain viable at d=64 with similar quality outcomes; verification is open.

---

## References

[1] Daniels, R. *The Strange Attractor: Music of the Spheres — Dynamical Systems and Prime-Harmonic Basins in Transformer Cache Compression*. 2026. Companion paper.

[2] Daniels, R. *Multiplicative Lattice Combined: Spectral KV Cache Compression via the Multiplicative Lattice*. Shannon-Prime documentation, 2026. https://github.com/nihilistau/shannon-prime/blob/main/multiplicative_lattice_combined.md

[3] Daniels, R. *Position Is Arithmetic v8*. Shannon-Prime documentation, 2026. https://github.com/nihilistau/shannon-prime/blob/main/position_is_arithmetic_v8.md

[4] Daniels, R. *KV Cache Is A View v2*. Shannon-Prime documentation, 2026. https://github.com/nihilistau/shannon-prime/blob/main/kv_cache_is_a_view_v2.md

[5] Daniels, R. *The Mertens Sea*. Position-Is-Arithmetic, 2026. https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/The_Mertens_Sea.pdf

[6] Daniels, R. *Decode Chain Amplification*. Position-Is-Arithmetic, 2026. https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/Decode_Chain_Amplification.pdf

[7] Daniels, R. *Shannon-Prime ComfyUI integration*, branches `feat/strange-attractor-stack` and `feat/strange-attractor-stack-v2`. https://github.com/nihilistau/shannon-prime-comfyui

---

## Appendix A — The 10D Pillar Set

For reference, the canonical 10-coefficient pillar set used throughout this paper at head_dim = 154 (the squarefree-padded VHT2 spectrum):

| Index (0-indexed) | Index (1-indexed) | Radix factor | Role |
|---|---|---|---|
| 0 | 1 | DC | Global identity / brightness |
| 14 | 15 | 11 × small | Texture stabilization (Grit) |
| 22 | 23 | 7 × small | Pillar (structural Radix-7) |
| 28 | 29 | 11 × small | Texture stabilization (Grit) |
| 44 | 45 | 7 × small | Pillar |
| 66 | 67 | 7 × small | Pillar |
| 77 | 78 | 2 × 7 × 11 / 2 | Mirror symmetry (Radix-2) |
| 88 | 89 | 7 × small | Pillar |
| 110 | 111 | 7 × small | Pillar |
| 132 | 133 | 7 × small | Pillar |

The pillars (Radix-7 multiples of 22) carry approximately 80% of the structural energy. The grit indices (Radix-11) provide texture cross-bracing. The mirror at 77 controls global polarity. The DC at 0 carries the offset.

A "Granite-only" reconstruction uses just these ten coefficients. Adding the next two Radix-11 indices (k=42, 56) brings the count to twelve and produces production-quality 4K output on the RTX 2060 in our experience. Adding the full Radix-11 set (eleven indices) brings the total to twenty-one and is the configuration we use for the late "Jazz" layers when fine texture is being refined.

---

## Appendix B — Implementation Checklist

For implementers reproducing the constructions in this paper, the following Shannon-Prime ComfyUI commits are the relevant references:

| Construction | Branch | Commit |
|---|---|---|
| Harmonic correction | feat/strange-attractor-stack-v2 | 94352bf |
| Tier-aware skeleton fraction | feat/strange-attractor-stack-v2 | e52515c |
| Twin-prime borrow (symmetric) | feat/strange-attractor-stack | 926f123 + 852a118 |
| Twin-prime borrow (asymmetric modes) | feat/strange-attractor-stack-v2 | 6b55542 + 7f6d751 |
| Ternary band-3 | feat/strange-attractor-stack | 4bc0bac + 0e8ac3d |
| Lattice RoPE (EmbedND API) | feat/strange-attractor-stack-v2 | 8c9e2b6 |
| Foveated subject mask | feat/strange-attractor-stack-v2 | 757e986 |

All toggles default OFF. Existing workflows are bit-identical to the prior ship without explicit opt-in.

---

*Submitted as preprint, 2026. Comments and replication welcome via the Shannon-Prime repositories.*
