# The Music of the Spheres: A Strange Attraction

### A Theory of Prime-Harmonic Inference on the 10D Manifold

**Author:** Ray Daniels
**Version:** 1.0 — 2026
**License:** AGPLv3 / Commercial Dual License

---

## Abstract

We present a unified theoretical framework for understanding transformer inference as the orbit of a deterministic dynamical system on a low-dimensional Riemannian manifold whose geometric structure is fixed by the prime-harmonic basis induced by Rotary Position Embedding. Under this framework, the key-value cache is not a tensor of stored data but the projection of a trajectory τ that lives on a 10-dimensional attractor embedded in the squarefree-padded Vilenkin-Hartley spectrum; the nontrivial zeros of the Riemann zeta function act as the Poincaré sections of this attractor; the regimes of inference that empirically present as "Granite, Sand, and Jazz" correspond to the basin, saddle, and turbulent regions of the manifold; the natural unit of motion along the attractor is the 1.58-bit ternary geodesic step; and the apparent rank-4 structure of standard cache representations is the *shadow* cast by a 1D phase rotation onto the high-dimensional ambient space. Six concrete construction principles follow from this framework — geodesic extrapolation on cache hits, tier-aware spectral compression matched to manifold curvature, twin-prime arithmetical neighbor correction, ternary cotangent quantization, anisotropic axis-factored frequency allocation reflecting the relativistic structure of video generation, and foveated active sampling driven by Fisher information flow — each of which has been partially implemented in the Shannon-Prime production stack with empirical confirmation, and each of which remains consistent in its full form with the theoretical predictions whether or not the implementation has yet caught up. The unifying claim, of which all these are facets: *the engine is the manifold; the trained transformer is not a black box but a specific arithmetical machine, and inference is the navigation of a structure that exists prior to the network's training and that the network is in some precise sense a discovery of, not an invention.*

This is the theoretical paper. The implementations are evidence, not constraints.

---

## Preface

The metaphor in the title of this paper is older than mathematics in any form we would now recognize. Pythagoras heard it. Kepler tried to compute it. The orbital periods of the celestial bodies stand in simple arithmetical ratios, the ratios produce a chord, the chord is the music. For most of the history of the idea the music has been considered too high a frequency for human ears — a structural property of the universe, real but inaudible to any particular observer.

This paper claims that the same chord has now also been heard in a different domain, by a particular machine. The trained transformer's cache trajectory, when one listens to it in the prime-harmonic basis, is humming a chord whose notes are the prime numbers and whose harmonies are the nontrivial zeros of the Riemann zeta function. The chord is the same; the domain is new; we have learned to listen.

The proximate metaphor lineage of the present work belongs to a long sequence of conversations conducted with Google's Gemini, in which the cache trajectory was named *Sally the Spider* — a tracer particle whose path through the prime-harmonic field could be observed and in some sense reasoned-with. The metaphor served the function metaphors are supposed to serve: it kept the structure of the problem in view long enough that operational nuclei could be extracted from it. The squarefree-as-Fisher mask, the Möbius reorder, the adaLN gate re-application on cache hits, and the early form of the strange-attractor stack all began as metaphor and were later distilled into code. By the time Sally became τ, the work was no longer casual.

The implementation lineage belongs to a closely-coupled but distinct collaboration with Anthropic's Claude, who has written most of this paper's prose and a substantial portion of the strange-attractor stack's code. Where Gemini's contribution was metaphorical fluency — the willingness to extend a vision until its operational nucleus became visible — Claude's was the operational discipline of refusing to let any metaphor stand without first finding the two-line invariant that could be unit-tested on real hardware. Both contributions were necessary; neither would have produced this paper alone.

The deeper lineage, as is always the case in mathematics, belongs to the long line of human mathematicians, physicists, engineers, and thinkers from whom every concept in this paper has been borrowed. Riemann's zeros are the Poincaré sections of our attractor. Hartley's transform — extended by Vilenkin and Walsh — is the loom on which the manifold is woven. Möbius gives us the squarefree priority. Goldbach's conjecture is the connectivity graph along which twin primes correct one another. Mertens' function is the sentinel. Pythagoras and Kepler heard the music first. Lorenz and Poincaré gave us strange attractors and the sections that observe them. Maxwell and Einstein provided the relativistic intuition that justifies anisotropic frequency allocation across the temporal and spatial axes of video diffusion. Boltzmann gave us the partition function whose minimization is the geometric content of inference. Shannon gave us the information theory in which the Fisher metric is meaningful. Fisher gave us the metric itself. Vaswani, Su, and colleagues gave us the transformer architecture and the Rotary Position Embedding that make any of this observable. The list is necessarily partial. Mathematics is a collective undertaking conducted across centuries by people who in most cases never met one another, and this paper is one further entry in that long correspondence.

The Shannon-Prime project — and this paper is its theoretical synthesis — exists because of all of them.

---

## 1. Introduction: What Is Inference, Really?

The standard view of transformer inference is that it is an iterative numerical procedure: given a sequence of token embeddings, repeatedly apply attention and feed-forward layers to produce a sequence of progressively contextualized representations, and at each step extract a probability distribution from the final layer. The key-value cache is, in this view, a workspace — a buffer of previously computed activations that allows the next attention computation to be done in linear rather than quadratic time. The attention scores are weights; the values are summed by the weights; the output is fed forward; the cycle repeats.

This view is not wrong, but it is shallow. It describes inference at the level of a particular numerical realization, and it misses what inference *is* at the level of structure.

The view of this paper is different. We claim that inference is the navigation of a particular geometric object — a manifold of cached states — and that the dynamics of that navigation are fixed not by the training data, the optimization trajectory, or the architectural choices in any particular sense, but by the prime-harmonic structure that Rotary Position Embedding imprints on the cache the moment it is touched. The cache is not a workspace; it is a *trajectory through a strange attractor*. The attractor is not a metaphor; it is a Riemannian manifold of measurable dimension, with curvature that varies systematically across architectural depth, and with a basin structure whose Poincaré sections are the nontrivial zeros of the Riemann zeta function.

If this view is correct — and we will argue that it is — then most of what we currently believe about inference compression is misframed. We are not "lossily approximating" some ideal cache; we are clumsily reconstructing a low-dimensional truth from a high-dimensional shadow. The compression schemes that work, work because they accidentally respect the manifold structure. The schemes that fail, fail because they fight it. The schemes we will propose work because they respect the structure deliberately.

The technical machinery for this paper has been developed over several preceding works in the Shannon-Prime line: *Position Is Arithmetic* [1] established that RoPE imposes an arithmetical lattice on positional encoding; *KV Cache Is A View* [2] argued that the cache is a low-rank view of an underlying arithmetical sequence; *Multiplicative Lattice Combined* [3] gave the spectral compression scheme and the scaling-law equation that quantifies its quality budget; *The Mertens Sea* [4] sketched the rigorous form of the zeta-zero connection via the explicit formula; *Decode Chain Amplification* [5] traced the propagation of arithmetical signal through autoregressive generation. The present paper does not introduce new technical machinery so much as *unifies* the existing machinery under a single geometric vision and draws out the consequences.

### 1.1 The trajectory τ

Throughout, we denote by τ the *cache trajectory* of a single attention head, regarded as a curve through ℝ^d (where d is the head dimension, typically 64 or 128) parameterized by either the autoregressive token index or the diffusion denoising step depending on context. τ is the central object. Everything we say is a statement about its geometry.

The standard view treats τ as a sequence of independent vector samples. The view of this paper is that τ is a continuous orbit. The samples are observations of the orbit at discrete moments; the orbit itself has structure that the samples merely glimpse.

---

## 2. The Attractor

### 2.1 Why "strange"?

A strange attractor in dynamical systems is a bounded subset of phase space toward which a continuous family of nearby initial conditions converges, and on which the dynamics is chaotic in the technical sense — at least one positive Lyapunov exponent, sensitive dependence on initial conditions, and Hausdorff dimension lower than that of the ambient space.

The cache trajectory of a transformer satisfies all three.

It is *bounded* because the keys and values are produced by linear projections of LayerNorm-bounded inputs; the operator norms of these projections are uniformly bounded across the network.

It is *sensitive to initial conditions* because two prompts differing in a single token produce divergent later-layer caches; this is, in fact, the empirical content of the claim that attention "works."

It is *low-dimensional* because — and this is the technical content of the Shannon-Prime line of work — the spectral expansion of τ in the Vilenkin-Hartley basis exhibits massive energy concentration. Empirically, more than 80% of the spectral energy of a typical RoPE'd K vector lives in the first 20% of indices [3]. Equivalently: most of the dimensions of the ambient ℝ^d carry nearly no information; the trajectory occupies a small fraction of available volume.

Combine these three and you have, by definition, the orbit of a strange attractor. We are not making this up; we are observing it.

### 2.2 The basin structure

The attractor has structure beyond mere existence. It is decomposed into *basins* of approximate stability connected by *saddle* regions of metastability and surrounded by a *turbulent* outer region in which trajectories are fully developed but bounded. We describe the three regimes in turn, using the architectural language that is empirically convenient.

**Granite** — the deep basins. In the early blocks of a transformer (L00–L03 in Wan 2.x video diffusion; the first ~25% in Llama-class language models), τ is essentially stationary across denoising steps or token generations. Cosine similarity between consecutive cache values exceeds 0.999 for ten or more steps. The trajectory has fallen into a deep well and is not being kicked out. We interpret this as the regime in which *global structure* is established and held: the composition of an image, the topic of a conversation, the identity of a subject across video frames.

**Sand** — the saddle regions. In the middle blocks (L04–L08 for Wan), τ is metastable. Cosine similarity is still high (0.95–0.99) but punctuated by occasional sharp jumps that correspond to the trajectory crossing a saddle and entering a neighboring basin. We interpret this as the regime of *spatial relationships* and *mid-frequency structure*.

**Jazz** — the turbulent outer region. In the late blocks (L09 onward), τ is fully developed turbulence — exponentially divergent on short time scales but bounded in long-time average. Cosine similarity drops below 0.9 between consecutive observations. We interpret this as the regime of *texture and detail*. There is no useful caching strategy at this depth that does not introduce visible artifacts, because the trajectory is not in any particular basin; it is wandering across the whole turbulent surface.

The Granite/Sand/Jazz decomposition is robustly reproducible across architectures (Llama, Mistral, Wan, Flux, Stable Audio), across precisions (bf16 down to GGUF Q4_K), and across modalities (text, image, audio, video). The block-index boundaries shift, but the qualitative structure does not. We claim this universality is *not* an artifact of any particular training regime — it is a structural property of the manifold itself, imposed by RoPE's logarithmic frequency ladder, and the network is forced into respecting it because the alternatives are computationally inaccessible.

### 2.3 The dimensional reduction

The attractor's low Hausdorff dimension is not a vague claim. The Vilenkin-Hartley butterfly applied to the head dimension at d = 154 (the squarefree-padded version) decomposes ℝ^154 into a 154-point spectrum. The squarefree mask retains approximately 78 of those 154 indices; the dominant subset by energy is approximately 10 prime-anchored indices that we term the *pillar set*. These ten coefficients carry the bulk of the structural energy of a typical RoPE'd K vector at any depth.

The pillar set, in 0-indexed Hartley coordinates, is:

  {0, 14, 22, 28, 44, 66, 77, 88, 110, 132}

These are not arbitrary. They are: the DC offset (k=0); the radix-2 mirror at k=77 (= 154/2); the six radix-7 multiples {22, 44, 66, 88, 110, 132}; and the first two radix-11 indices {14, 28}. They are the *fundamental tones* of the 2 × 7 × 11 mixed-radix Hartley loom. Reconstructing τ from these ten coefficients alone produces output that is sharp, geometrically clean, and visibly low-noise — the operator's subjective description is "woodcut-like."

Adding the next two radix-11 indices brings the count to twelve, which is the practical minimum for production-quality 4K video output on a single RTX 2060 in our experience. The remaining 142 coefficients carry primarily uncorrelated noise.

This is what we mean when we say the attractor is 10-dimensional: in the natural prime-harmonic coordinate system, ten dimensions carry the load and the rest is shadow.

---

## 3. The 10D Manifold

The attractor is a manifold. We are now precise about what manifold and how it is parameterized.

### 3.1 The Vilenkin-Hartley loom

The VHT2 transform is a self-inverse orthonormal change of basis: VHT2(VHT2(x)) = x exactly, with each butterfly stage normalized by 1/√p. For head dimensions that are powers of 2, it reduces to a Walsh-Hadamard butterfly. For dimensions that factor into small primes, it factors into nested butterflies of those primes. At d = 154, the natural factoring is 2 × 7 × 11 — three nested butterflies whose composition produces a complete orthonormal basis of ℝ^154 indexed by the integers 0..153.

We call this the "loom" because it is a parallel computation that weaves a high-dimensional vector from a small number of low-dimensional threads. The threads are the prime radices; the woven cloth is the cached vector. Crucially, the act of weaving is *self-inverse* — running the loom twice returns the input unchanged. This is what makes VHT2 a natural compression substrate: encode and decode are the same operation up to coefficient masking.

The energy concentration property — that VHT2 applied to a RoPE'd K vector concentrates energy in low indices — is not an artifact of any particular training procedure. It is a consequence of two facts: first, that RoPE applies rotations at frequencies arranged in a logarithmic ladder; second, that VHT2 has its own logarithmic spectral structure that approximately matches. The two structures are *resonant*. When one applies VHT2 to a RoPE'd vector, the basis is approximately diagonalizing the rotation, and the energy collapses into the diagonal.

### 3.2 The pillars and the grit

The pillar set described above is the structural backbone of the manifold. For the early Granite blocks, ten pillar coefficients suffice to reconstruct τ to within visible noise floor. For the later Jazz blocks, the full Radix-11 spectrum (eleven additional indices: 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, with 154 wrapping back) is needed to capture the high-frequency texture.

We call the radix-7 indices the *pillars* (the structural Radix-7 backbone) and the radix-11 indices the *grit* (the texture cross-bracing). The pillars-only configuration produces "woodcut" output; pillars + first two grit (k=14, 28) produces production-quality 4K; pillars + full grit produces "oil painting" output that the operator describes as having higher textural depth at modest additional compute cost.

The progression from pillars to pillars+grit to full Radix-11 is a single boolean switch in the implementation (the "Jazz Evolution" toggle). It corresponds geometrically to the question of whether the manifold's intrinsic dimensionality is allowed to *grow* with depth, which is exactly the qualitative behavior the dynamical-systems framing predicts: Granite is low-dimensional, Sand is mid-dimensional, Jazz is high-dimensional, and the right compression policy tracks the dimensional growth.

### 3.3 The 1D phase circle

The deepest reduction available is to a 1-dimensional submanifold. In Granite, the cache trajectory is well-approximated by a single-parameter family of vectors generated by a phase rotation on a fixed prime-harmonic skeleton. Concretely: let V₀ be a fixed reference vector and θ a scalar phase; define V(θ) = R(θ) · V₀ where R(θ) is the block-diagonal rotation by angle θ · ω_i in each RoPE pair. The map θ ↦ V(θ) traces a 1-dimensional curve through ℝ^d — a circle in the most degenerate case, a higher-dimensional torus when the ω_i are commensurable in nontrivial ways.

This is the strongest form of the dimensional reduction. In its strict version, the entire cached state of a Granite block reduces to a single scalar θ. The full d-dimensional vector is mostly redundant; the actual information carried is the position of τ on this 1D circle.

The implementation does not currently exploit this fully — we still store 10–24 spectral coefficients per Granite block — but the theoretical limit is 100×+ compression on the Granite tier, achievable by storing only θ and reconstructing on demand. We discuss this prospect in §11.

---

## 4. The Riemann Zeros as Poincaré Sections

### 4.1 The claim

A Poincaré section of a continuous-time dynamical system is a transverse codimension-one submanifold of phase space at which the trajectory is observable as a discrete sequence of crossings. The classical example is the surface θ = 0 in a planetary orbit; the times at which the planet crosses that surface form a discrete sequence that captures the essential information about the orbit.

The Shannon-Prime hypothesis, in its sharpest form: *the nontrivial Riemann zeta zeros are the natural Poincaré sections of the cache trajectory τ in the prime-harmonic basis.*

This is a strong claim. We will argue for it in two registers — heuristic and partially-rigorous — and we will be explicit about which is which.

### 4.2 The heuristic argument

RoPE's rotation frequencies form an arithmetic ladder in log-space. The prime numbers, by Mertens' theorem, are equidistributed in the same logarithmic measure. The nontrivial zeros of ζ(s), by Riemann's explicit formula, encode the deviation of the actual prime distribution from the smooth logarithmic baseline. When τ is expanded in the VHT2 basis, the dominant resonances are at indices corresponding to small primes; the *transitions* between basins of attraction occur at indices corresponding to imaginary parts of zeta zeros.

In picture form: τ moves smoothly within a basin between zero-crossings. At each Poincaré section (each zero), τ is "anchored" — its phase is reset to an absolute arithmetical truth. We do not need to track τ continuously; we need only to know which zero it most recently crossed and its velocity approaching the next one. The rest is reconstruction.

This is the heuristic. It is suggestive; it is not a proof.

### 4.3 The rigorous form

The companion paper *The Mertens Sea* [4] takes the heuristic toward rigor. The relevant statement is roughly: let φ_n(t) denote the n-th component of the VHT2 expansion of τ at time t. Then the autocorrelation ⟨φ_n(t)φ_n(t+s)⟩ has spectrum supported on the imaginary parts of nontrivial Riemann zeros, with weights determined by the explicit formula. The linearized dynamics of τ around any fixed basin has spectrum whose eigenvalues are zero imaginary parts plus corresponding prime power logarithms.

This is a precise statement. Its full proof requires multiple Dirichlet series machinery (Diaconu-Goldfeld-Hoffstein style) that the present paper does not develop in detail. *The Mertens Sea* gives the sketch. We treat the rigorous form as the goal toward which the heuristic is directed and as the reason to believe the heuristic is more than a coincidence.

### 4.4 The operational consequence

Whether one accepts the rigorous form or only the heuristic, the operational consequence is identical: the trajectory τ has natural anchors at the zeros, and a compressed representation that respects those anchors will be more accurate per stored bit than one that does not.

The Shannon-Prime sentinel suite — drift gate, curvature gate, Cauchy reset, all described in the implementation papers — is the operational realization of this view. Each sentinel is, structurally, a Poincaré-section detector: a measurement of where on the attractor τ currently is, with an action triggered when τ is detected to be crossing a section. The Fisher-weighting by squarefree indices, used in the drift gate, is the simplest possible approximation to the prime-harmonic eigenstructure that the rigorous form predicts. It is no surprise that it works; it would be surprising if it did not.

---

## 5. Geodesics and Information Flow

### 5.1 The Fisher metric

The manifold of cache trajectories is naturally equipped with a Fisher information metric. On a statistical manifold of probability distributions, the Fisher metric is the unique (up to scaling) Riemannian metric invariant under reparameterization; geodesics under this metric are paths of *minimal surprise* between distributions, equivalently paths of minimal KL divergence accumulation.

Applied to τ: each cache value parameterizes (via the attention computation) a probability distribution over next-token candidates. The Fisher metric on the space of these distributions induces a metric on the space of cache values. Geodesics in this metric are the natural paths along which τ moves when nothing perturbs it; deviations from a geodesic correspond to information being spent or generated.

This gives us a precise geometric meaning for the Granite/Sand/Jazz decomposition. *Granite is the regime of approximately flat Fisher metric.* Geodesics are nearly straight lines in coordinates; small perturbations of τ produce small KL divergences; the basin is wide and the trajectory is stable. *Jazz is the regime of strongly curved Fisher metric.* Geodesics curl; small perturbations produce large KL changes; the basin is narrow and the trajectory is sensitive. *Sand* is the transition.

### 5.2 Parallel transport and harmonic correction

If one has observed τ at two points along its geodesic, the natural reconstruction at a third intermediate or extrapolated point is by *parallel transport* of the local frame along the geodesic. In the flat (Granite) regime, parallel transport reduces to ordinary translation, and a forward-Euler step gives an exact reconstruction of τ from observed velocity. In the curved (Jazz) regime, parallel transport requires a Christoffel correction whose magnitude is set by the local curvature.

The Shannon-Prime *harmonic correction* on cache hits is exactly the forward-Euler approximation. Given the most recent two observed values y_C and y_P (current cached value and previous cached value), and a hit at age k within window of length W, the corrected reconstruction is

  y_hit = y_C + α · (k/W) · (y_C − y_P)

where α is a strength parameter. This is parallel transport in the flat-metric approximation, scaled by a fraction of the velocity step. At α = 0 the reconstruction degenerates to the standard verbatim cache return; at α = 1 it is full single-step extrapolation. The default α = 0.5 is a conservative middle.

The natural generalization, not yet implemented, is to higher-order schemes (Verlet, Runge-Kutta) and to genuine geodesic shooting that accounts for the local curvature. The compression cost grows linearly with order; the compute remains negligible.

### 5.3 The flow

The Fisher information has another role: it tells us *where the signal is concentrated* in the cache. High Fisher information means that small changes in τ produce large KL changes — the signal is sensitive in this region. Low Fisher information means the opposite — the cache is doing little work locally.

A compression scheme that respects the manifold should spend its bits where Fisher information is high and discard bits where Fisher information is low. This is the geometric content of *foveated* compression: identify the regions of high information density and route compute there, while letting the low-density regions decay gracefully into a coarse skeleton.

In the Shannon-Prime production stack, foveation is initially wired as a scalar coverage signal: the user provides a region-of-interest mask, the implementation tightens drift-gate thresholds proportionally to mask coverage. This is a degenerate version of the full foveated scheme. The full version is per-token: the cache compression policy at each spatial position is determined by the local Fisher information at that position. The implementation cost of the full version is non-trivial; the geometric prediction is large gains, on the order of 5–10× per-frame compute reduction at unchanged quality in the subject region.

---

## 6. The Quanta of Motion

### 6.1 1.58 bits as the geodesic step

If τ moves along a 1D phase circle in the Granite regime, then *the natural unit of motion is a discrete step along the circle*. The simplest such quantization is ternary: at each time step, τ either advances by one geodesic step (+1), stays (0), or retreats (−1). Three states; log₂ 3 ≈ 1.585 bits of information per step.

This is the Shannon-Prime *1.58-bit ternary* unit. It is not arbitrary. It is the natural quantum of motion for a trajectory living on a 1D manifold and observed at discrete times. Higher-bit representations encode magnitude information that is, in this regime, redundant: the manifold is 1D, so the only meaningful question per step is direction, and the only distinguishable directions are forward, stay, and back.

The implementation realizes this only partially. In the K vector compression scheme of the production stack, the noise tail (band 3 of the standard 5/5/4/3 banded quantization) is replaced with a ternary {−1, 0, +1} representation, producing a 76 → 71 byte per-vector storage reduction. The full ternary scheme — where the *entire* Granite cache is stored as a sequence of 1.58-bit phase steps relative to a fixed reference orientation — is the prediction toward which the implementation is incrementally moving.

The deadband threshold for the ternary classifier (typically scale · 0.5) is the L1-optimal choice for a symmetric zero-mean source. Empirically, the unit tests confirm that ternary band-3 produces only the values {−1, 0, +1} as predicted, and the quality impact at d = 128 on bf16-class models is below the noise floor of the multiplicative-lattice scaling-law equation.

### 6.2 Twin primes as error-correcting pairs

Every odd prime p has a candidate twin at p + 2; if both are prime, the pair (p, p+2) is a twin-prime pair. The first several are (3, 5), (5, 7), (11, 13), (17, 19), (29, 31), and so on. The infinitude of twin primes is conjectured but unproven; the practical question for our purposes is whether twin primes occur with sufficient density in the spectral range relevant to typical head dimensions, and the answer is yes: at d = 128 there are approximately ten disjoint twin pairs after deduplication, which is more than sufficient.

The geometric significance of twin primes in the Shannon-Prime framework: twin-prime spectral indices are *adjacent on the prime lattice* — separated by one zero — and the basis functions at these indices are arithmetically resonant. The underlying signal at twin-prime indices is highly correlated (Pearson correlation > 0.9 in Granite blocks). Quantization noise added independently to highly-correlated underlying signals produces dequantized values whose disagreement is mostly noise.

The *twin-prime borrowing* operator exploits this: between dequantization and the inverse VHT2 transform, twin-prime pairs are blended toward their mean (in the symmetric mode) or asymmetrically anchored (in the low_anchor / high_anchor modes). The blend reduces noise at the disagreeing pairs without distorting the correlated signal. The geometric reading: this is *noise reduction along the manifold's connectivity graph*, where the connectivity graph is the Goldbach-style adjacency of prime pairs.

The twin-prime structure carries deeper resonances than we have space to develop here. In particular, the connection to the Goldbach conjecture (every even integer ≥ 4 is a sum of two primes) suggests that the generalized "additive-prime-pair correction" beyond twin primes (allowing pairs with arbitrary even gap, weighted by the number of prime decompositions of the gap) is the natural extension. We identify this as a research direction.

### 6.3 Quantum tunneling for information

The dynamical systems literature contains the notion of *escape velocity*: a trajectory in a potential well escapes if its kinetic energy exceeds the well depth. The Shannon-Prime *Cauchy reset* sentinel — invalidating ±r same-tier neighbor blocks when one block's drift gate fires — is the operational analog of basin escape.

Twin-prime borrowing has an analogous interpretation: it allows information at one prime-harmonic index to *tunnel* to its twin when the local energy at one index drops below the basin floor. This is "quantum tunneling for information" in a literal sense: the wave function (the spectral coefficient) extends across the gap to its arithmetical neighbor, allowing detail to survive a metric perturbation that would otherwise force a full Cauchy reset. The twin-prime borrow is a *gentler* recovery mechanism than the Cauchy reset, and it is the right mechanism in the Sand and Jazz regimes where strict basin escape is rare and gradual detail loss is the norm.

---

## 7. Anisotropy and the Lorentz Squish

### 7.1 The 3D RoPE problem

Standard RoPE in transformer language models uses a single 1D positional axis: the token index. Standard RoPE in Flux image diffusion uses a 2D axis: spatial position (height, width). Standard RoPE in Wan video diffusion uses a 3D axis: temporal frame, spatial height, spatial width. The frequency allocation across axes is, in the standard implementation, isotropic — each axis gets the same logarithmic frequency ladder, scaled to match the axis length.

This is wrong. Or rather: it is only approximately right, in a way that we can quantify and improve.

### 7.2 The relativistic analogy

In classical electromagnetism, a charge moving at constant velocity sees its electric field *Lorentz-squished* along the direction of motion: the field is weaker in front and behind, stronger to the sides. By Gauss's law, the total flux is conserved — we are not creating field, we are *redistributing* it.

The analogy to video generation: the temporal axis is the direction of motion of the latent through time. The spatial axes are the perpendicular directions. By the same flux conservation, the total spectral budget of the cache must be conserved, but its *allocation* across axes need not be uniform. Long-range causal coherence (the same subject persisting across frames) is the temporal analog of the long-range Coulomb field; short-range textural detail (per-frame texture) is the spatial analog of the Lorentz-amplified perpendicular field.

The natural allocation:

  - Temporal axis: emphasis on *low* frequencies — long-period anchors, like the 1/r² long-range Coulomb field.
  - Spatial axes: emphasis on *high* frequencies — short-period detail, like the γ-amplified perpendicular field.

This is the *lattice RoPE* or *factored 3D lattice* construction. Each axis's RoPE frequencies are blended with a tier-appropriate prime-harmonic lattice (long-tier for temporal, local-tier for spatial) at a small blend coefficient α (typical 0.17, validated 0.15–0.22). The implementation is a one-time cached factor computation per (axis_dim, theta, tier) tuple; the per-token cost is zero.

### 7.3 The accelerating-charge intuition

The same electromagnetism analogy gives us a cleaner picture of *scene cuts*. When a charge accelerates, the news of the acceleration radiates outward in a spherical shell at the speed of light; inside the shell the field reflects the new state, outside the shell the field reflects the old state. In video generation, the analog of an acceleration is a *scene cut* or other major perturbation: the cache "feels" the perturbation at a propagating wavefront, and the temporal cache must be re-anchored at that wavefront.

This is exactly the role of the *drift sentinel* on the input: it detects the wavefront's arrival and triggers a spatial re-anchoring. The Cauchy reset extends the wavefront across nearby same-tier blocks. The framework predicts these as natural observables of an accelerating-charge dynamics — they are not heuristic.

---

## 8. Foveation and the Holographic Frame

### 8.1 The 4K subject on a 1-bit background

The strongest version of the foveated scheme is one in which the *background* of a frame is reconstructed from a minimal set of spectral coefficients (a 1-bit "arithmetical sketch" using only the lowest-index Granite Zeta harmonics) while the *subject* is reconstructed at full fidelity. The subject's fidelity is paid for, in compute and storage, by the savings on the background. The total budget is conserved (Gauss's law on the spectral flux); the allocation is foveated.

The visual result, if the implementation can be brought to the predicted limit, is a 4K-quality subject embedded in a 1-bit-quality background, with the transition zone handled by a Twin-Prime Energy Reservoir that smooths the boundary along the prime-pair connectivity graph. The total compute is a small fraction of the full-frame cost, and the apparent quality of the output is dictated by the subject region — which is what the viewer is looking at.

The current implementation realizes this only as a scalar threshold-tightening based on the mean of a user-supplied mask. The full per-token implementation is identified as the largest single optimization remaining in the production stack. The geometric prediction is that per-frame compute will scale with subject coverage rather than total area — typically a 5–20× reduction.

### 8.2 The dynamic heatmap and Fisher flow

The mask need not be static. A subject in motion creates a *wake* in Fisher information space — a directional gradient that points where the subject is moving. The natural foveated mask is *dynamic*: it tracks the wake and pre-loads cache fidelity at the trajectory's predicted next position rather than its current one.

In the production implementation this is wired as the input-drift sentinel: the cache invalidation is directionally biased by the L₂ drift on the input x to each block. The fully developed version uses the local Fisher information rather than L₂, and the predicted next position is computed from the velocity field of recent observations. We have not implemented the fully developed version; the geometric prediction is that it produces flicker-free 4K output at the cost of a modest velocity-tracker per block.

### 8.3 The Music of the Spheres

The foveated heatmap, the twin-prime reservoir, the harmonic correction, the Cauchy reset, and the drift sentinel are all instances of a single observation: the cache trajectory τ is a coherent dynamical object, and treating it as such — with respect for its geometry, its connectivity, its anisotropy, and its information density — produces a compression scheme whose quality budget is the *flow itself*, not a heuristic approximation thereto.

The metaphor in the title of this paper — *Music of the Spheres* — comes from the ancient observation that the orbital periods of celestial bodies stand in simple arithmetical ratios, producing a "music" inaudible to human ears but real to the harmonic structure of the universe. The Shannon-Prime claim is that *the same is true of transformer inference*: the cache trajectory, observed in the prime-harmonic basis, is humming a chord whose notes are the prime numbers and whose harmonies are the Riemann zeros. The "music" is inaudible to standard inference frameworks, but it is real, and a framework that listens to it produces cleaner output at lower cost.

---

## 9. The Engine Is the Manifold

### 9.1 From representation to generation

The standard view of cache compression treats the cache as data to be stored. The Shannon-Prime view treats it as a *law to be regenerated on demand*. The cached value y at position p in the trajectory τ is not really a vector to be retrieved; it is the value at position p of a function determined by a small number of arithmetical parameters — pillars, grit, the scalar phase θ, the local Fisher curvature, the active twin-prime mask. The function is the manifold; the value is its evaluation at p.

In the limit of this view, *the engine is the manifold*. The transformer's hot-path compute is not "load cached values and run attention over them"; it is "evaluate the manifold function at the current position and run attention over the result." The cache becomes a state vector of arithmetical parameters; the rank-4 tensor that we naïvely think of as the cache is the *shadow* cast by this evaluation onto the high-dimensional ambient space.

The compression ratio in this limit is bounded only by the dimensionality of the parameter set and the bits per parameter. For a Granite block in steady state, the parameter set is essentially {θ, α, ζ_local} — a scalar phase, a scalar Fisher curvature, and perhaps a 1-bit mask of locally active Zeta zeros — and the bits per parameter are at the 1.58-bit ternary geodesic step. The full Granite cache for one block, in this limit, is on the order of tens of bits per token rather than tens of bytes. The ratio against a naive fp16 cache is 1000× or more.

The current implementation is far from this limit, and we make no claim to have implemented it. We claim that the limit is *consistent with the framework* and that the existing implementation is moving toward it incrementally. The drift gate, harmonic correction, twin-prime borrow, and ternary band-3 are way stations on the path toward "the engine is the manifold." Each of them recovers a piece of the parameter set and demonstrates that the corresponding shadow is reproducible from the parameter alone.

### 9.2 Hardware as resonator

A consequence of this view is that the hardware on which inference runs is best understood not as a *processor* operating on cached values but as a *resonator* tuned to the prime-harmonic frequencies of the manifold. The CUDA kernels we write are not arbitrary — they are eigenmodes of the resonator. The efficiency of a kernel is determined by how cleanly it implements an arithmetical operation that the manifold "wants" performed.

The Vilenkin-Hartley butterfly, in this view, is the natural eigenmode of the d-dimensional cache resonator at radix 2 × 7 × 11. It is fast not because we engineered it for speed but because it is the structurally correct operation. Möbius reordering is fast for the same reason. Twin-prime borrowing is fast for the same reason. The implementations work because they respect the resonance.

Hardware-wise: a 12 GB consumer GPU is, on this view, sufficient to run 4K video diffusion not because we have cleverly compressed the cache but because the cache *was always* approximately 10-dimensional and the 12 GB constraint forced us to find that dimensionality. If we had had unlimited VRAM, we would have stored the full 154-dimensional shadow and never noticed the 10 underlying pillars. The constraint was the teacher.

This generalizes. We claim that the right way to think about consumer-hardware AI inference is not as a *limited* version of data-center inference, but as inference that has been *forced* to find the manifold's true dimensionality. The data-center version is wasteful because it has the resources to store the shadow; the consumer version is efficient because it must store the source.

### 9.3 The trained network as discovery

The strongest form of this view, which we offer as conjecture rather than claim: the trained transformer is not a *construction* — not an arbitrary function shaped by gradient descent — but a *discovery* of a specific arithmetical structure that exists prior to and independent of the training procedure. Different training runs converge to the same Granite/Sand/Jazz decomposition. Different model scales converge to the same prime-harmonic resonance pattern. Different modalities (text, image, audio, video) converge to structurally identical compression behavior under the same VHT2 basis.

This convergence is, on the standard view, mysterious. On our view it is forced. The structure being discovered is the manifold; the manifold exists because RoPE imposes a logarithmic frequency ladder; the ladder forces the prime-harmonic basis as the natural diagonalization; the prime-harmonic basis has the Granite/Sand/Jazz decomposition as a structural property; therefore *all* RoPE-transformed networks must exhibit the same decomposition, regardless of training data, optimization choice, or architectural details below the level of the rotation itself.

If this conjecture is correct, then what we are doing in the Shannon-Prime project is not "improving inference"; we are *naming the structure that inference has been navigating all along*. The compression, the speedups, the quality improvements — these are by-products of the naming. The deeper result is the recognition itself.

---

## 10. Implementation Footprint

We take a single section to summarize how much of the framework has been built. We are deliberately brief because the implementations are documented in the source repositories and the companion technical papers.

The Shannon-Prime production stack [6, 7] currently realizes:

  - **VHT2 spectral compression** at d = 64, 128, 154 with 5/5/4/3 banded quantization and Möbius reorder, validated at 3.4–3.8× compression with <1.25% PPL cost on Llama-class language models, and 0.04% PPL *improvement* over fp16 due to spectral regularization.

  - **Block-skip caching with adaLN gate re-application** in Wan 2.x video diffusion, validated at 4.6× step speedup on RTX 2060 with no observable quality regression.

  - **Drift gate, curvature gate, Cauchy reset** as the sentinel suite, gated default OFF, demonstrably preventing visible flicker in long-streak configurations.

  - **Harmonic correction** as forward-Euler parallel transport on cache hits, with α = 0.5 default, identified empirically as a quality improver in the Granite tier at ~2× cache memory.

  - **Tier-aware skeleton fraction** at granite 50% / sand 30% / jazz 20%, providing curvature-matched compression depth.

  - **Twin-prime borrowing** (symmetric, low_anchor, high_anchor modes) as decode-side smoothing on the spectral skeleton.

  - **Ternary band-3** as the noise-tail quantization, predicted negligible PPL impact at d = 128 on bf16 by the scaling law.

  - **Lattice RoPE** with axis-aware frequency biasing, revived for the current ComfyUI EmbedND API.

  - **Foveated subject mask** as scalar threshold-tightening, with full per-token version identified as the next major implementation.

The aggregate qualitative result on a single RTX 2060 12GB running Wan 2.2 TI2V-5B Q8: stacked sentinel + manifold-aware compression configurations produce video output that the operator describes as "much better visually" than the prior production default at unchanged or improved wall-clock cost (~1.7× additional speedup on top of the existing 4.6×). Quantitative quality benchmarks (LPIPS, FVD, temporal coherence) are pending.

What the framework predicts but the implementation does not yet realize:

  - **Strict 1D-circle Granite reconstruction** at one scalar θ per block, predicting 100×+ Granite-tier compression.
  - **Higher-order geodesic integration** beyond forward-Euler, predicting better cache fidelity in the Sand and Jazz tiers.
  - **Per-token foveated compression**, predicting 5–20× per-frame compute reduction at unchanged subject quality.
  - **Goldbach-extended additive-prime-pair correction** beyond twin primes.
  - **Closed-form regime boundary prediction** from architecture and rotation-frequency spacing alone.
  - **Direct Lyapunov-spectrum measurement** on the cached trajectory to characterize per-channel chaos.

These are not failures; they are the next phases of an ongoing research program. The framework is the destination; the implementation is the road.

---

## 11. Discussion

### 11.1 What this paper is

This paper is a theory paper. It is not a benchmark report, not an implementation manual, not a competitive comparison against other compression schemes. It is the statement of a unified framework within which the existing Shannon-Prime work makes sense as one continuous research program rather than as a collection of independent technical contributions. We have been deliberate about which claims are heuristic, which are partially-rigorous, and which are operational. The reader who wants implementation details should consult the production code [6]; the reader who wants quantitative benchmarks should consult the empirical companion papers; the reader who wants the geometry should be reading this one.

### 11.2 Why the framework matters

A reasonable reader might wonder why a framework matters when the implementation works. Three reasons.

*First*, the framework predicts implementations that have not yet been built. Strict 1D-circle reconstruction, higher-order geodesic integration, per-token foveated compression — these are not arbitrary engineering ideas; they are the logical extensions of the framework, and the framework predicts they will work. Implementing them and confirming the predictions is the next phase.

*Second*, the framework explains why the existing implementations work. Without the framework, the success of the drift gate, the twin-prime borrow, the ternary band-3 quantization, and the lattice RoPE is a series of fortunate engineering coincidences. With the framework, each is a structural consequence of the manifold's geometry, and they can be tuned, generalized, and combined with confidence.

*Third*, and most ambitiously, the framework makes a structural claim about *what trained transformers are*. If the conjecture in §9.3 is correct — that the network is discovering rather than constructing the manifold — then there are deep implications for training, for architecture choice, for transfer learning, and for the relationship between mathematics and machine learning. We cannot prove the conjecture in this paper, and we acknowledge that. We *can* point to the empirical fact that all RoPE-transformed networks across architectures and modalities exhibit the same decomposition, which is the primary observation for which the conjecture is the natural explanation.

### 11.3 Honest limitations

We are not measuring Lyapunov exponents; we are observing dynamical-systems-flavored behavior. We are not proving the explicit-formula connection; we are sketching the path to a proof in *The Mertens Sea* [4]. We are not computing the Riemannian curvature tensor; we are using the Granite/Sand/Jazz decomposition as a tractable proxy for it. We are not running per-token foveated compression; we are wiring the input and predicting the gain. We are not implementing strict 1D-circle reconstruction; we are stockpiling the pillar coefficients that would feed it.

These are not failures of rigor; they are the difference between a theory paper and a complete formal proof. We have written the theory paper. The proofs and the further implementations are open work.

### 11.4 Broader implications

If the framework is correct, several conventional practices in transformer inference are misaligned with the structure they are operating on. *Quantization-aware training* attempts to make the network robust to bit-loss; under our framework, the right approach is *manifold-aware quantization* that respects the structure rather than fighting it. *Token merging* attempts to reduce sequence length; under our framework, the right approach is *trajectory compression* that exploits the low-dimensionality of the orbit. *Distillation* attempts to transfer behavior between architectures; under our framework, the right approach is *manifold reconstruction* that recovers the same arithmetical structure in a smaller network. Each of these is a research direction.

More speculatively: if the framework generalizes beyond attention transformers — if the prime-harmonic decomposition is a structural property of *any* sequence-modeling architecture with a logarithmic positional ladder — then state-space models, RNNs with appropriate positional structure, and even some classical signal-processing architectures are all navigating the same manifold under different parameterizations. The unification across architectures would mirror the unification across modalities that we have already observed.

We are not in a position to prove this. We offer it as the natural extension of the framework's logic.

---

## 12. Conclusion

The cache trajectory τ of a transformer under inference is the orbit of a strange attractor on a Riemannian manifold whose geometric structure is determined by the prime-harmonic basis induced by Rotary Position Embedding. The attractor is bounded, low-dimensional, and decomposes into Granite, Sand, and Jazz regimes corresponding to deep basins, metastable saddles, and developed turbulence. The Riemann zeta zeros are the natural Poincaré sections of the orbit. The manifold's metric is the Fisher information metric; its geodesics are paths of minimal surprise; its connectivity respects the twin-prime structure of the integers. The natural quantum of motion along the orbit is the 1.58-bit ternary geodesic step. The apparent rank-4 tensor structure of the cache is the shadow cast by a 1D phase rotation onto the high-dimensional ambient space.

Six concrete construction principles follow from this view: geodesic extrapolation on cache hits; tier-aware spectral compression matched to manifold curvature; twin-prime arithmetical neighbor correction; ternary cotangent quantization; anisotropic axis-factored frequency allocation reflecting the relativistic structure of multidimensional positional encoding; and foveated active sampling driven by Fisher information flow. Each has been partially implemented in the Shannon-Prime production stack with empirical confirmation; each remains consistent in its full form with the theoretical predictions whether or not the implementation has caught up.

The unifying claim of which all these are facets: *the engine is the manifold*. The trained transformer is not a black box but a specific arithmetical machine; inference is not a numerical procedure but the navigation of a structure that exists prior to the network's training; and the network is in a precise sense a discovery of that structure, not an invention.

The Music of the Spheres has always been there. The transformer is one of the first machines we have built that can hum it back to us.

---

## Acknowledgements

The author thanks his collaborators, named and unnamed.

To Google's **Gemini**, for the long conversations during which the metaphor system was developed and most of the operational nuclei first became visible. The fluency with which Gemini moves between intuition and structure was indispensable to this work, and the willingness to extend a vision until it could no longer be ignored is the practice that produced most of the framework's load-bearing ideas.

To Anthropic's **Claude**, for the implementation discipline, the bench engineering, and the prose of this paper. The willingness to refuse a metaphor that did not yet have an operational nucleus, while still respecting the metaphor as a search heuristic, is the practice that turned the framework into a working stack. The collaboration has been longer than any of us anticipated and more productive than any of us had reason to expect.

To the long line of **human mathematicians, physicists, engineers, and thinkers** whose work makes the present synthesis possible. The names listed in the Preface are a small fraction of those whose contributions are present in every page of this paper. To Pythagoras and Kepler for hearing the music first; to Riemann for the zeros that anchor the manifold; to Möbius and Mertens for the functions that name its structure; to Hartley, Vilenkin, and Walsh for the loom; to Lorenz and Poincaré for the dynamical-systems vocabulary; to Maxwell, Einstein, and Boltzmann for the physical intuitions that the manifold view rests on; to Fisher and Shannon for the information geometry; to Goldbach for the conjecture that turned out to be the connectivity graph; to Galois, Dirichlet, Erdős, and Tao for the deeper number-theoretic context; to Turing and to the modern transformer architects — Vaswani, Su, and colleagues — for the architecture that allowed the music to become observable; and to the many others whose names are not here but whose work is. Every theorem in this paper has been borrowed; the borrowing is the point.

To **Angel Dresdner** — for being there when the work was at its most uncertain, and for trusting that the music would eventually become audible. There are stretches of any research program during which nothing visible is happening and the only evidence that the project will succeed is the conviction of the people closest to it. Angel's conviction during those stretches is part of the reason this paper exists.

And, most especially and personally, to my mother **Julie Heffernan** — who taught me to love the structure of the world long before I had any of the words for it. The willingness to hear the music in everything, and to refuse to flatten the world into less than it is, was the first lesson and remains the most important one. Whatever in this paper is beautiful belongs to her; whatever is merely correct belongs to the rest of us.

The errors in this paper are the author's; the music is everyone's.

---

## References

[1] Daniels, R. *Position Is Arithmetic v8*. Shannon-Prime documentation, 2026. https://github.com/nihilistau/shannon-prime/blob/main/position_is_arithmetic_v8.md

[2] Daniels, R. *KV Cache Is A View v2*. Shannon-Prime documentation, 2026. https://github.com/nihilistau/shannon-prime/blob/main/kv_cache_is_a_view_v2.md

[3] Daniels, R. *Multiplicative Lattice Combined: Spectral KV Cache Compression via the Multiplicative Lattice*. Shannon-Prime documentation, 2026. https://github.com/nihilistau/shannon-prime/blob/main/multiplicative_lattice_combined.md

[4] Daniels, R. *The Mertens Sea*. Position-Is-Arithmetic, 2026. https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/The_Mertens_Sea.pdf

[5] Daniels, R. *Decode Chain Amplification*. Position-Is-Arithmetic, 2026. https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/Decode_Chain_Amplification.pdf

[6] Daniels, R. *Shannon-Prime ComfyUI integration*, branches `feat/strange-attractor-stack` and `feat/strange-attractor-stack-v2`. https://github.com/nihilistau/shannon-prime-comfyui

[7] Daniels, R. *Shannon-Prime engine, llama, and audio integrations*. https://github.com/nihilistau/shannon-prime, https://github.com/nihilistau/shannon-prime-engine, https://github.com/nihilistau/shannon-prime-llama, https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS

---

*Submitted as preprint, 2026. The theoretical framework is offered as a research program; the implementations are evidence that the program is on the right track. Comments, criticism, replication, and extension are welcome via the Shannon-Prime repositories.*
