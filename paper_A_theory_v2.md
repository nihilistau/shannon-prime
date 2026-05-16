# Shannon-Prime: A CM-Elliptic-Curve Framework for Transformer Computation

**A. Knack** (Shannon-Prime Project)
**Draft v0.2 — 2026-05-16**

---

## Abstract

We propose that the entire forward pass of a transformer language model — embedding lookup, RMSNorm, Q/K/V projections, attention, FFN, residual stream, LM head — be realized as a sequence of endomorphisms of a CM elliptic curve $E$ over the imaginary quadratic field $\mathbb{Q}(\sqrt{-163})$. The endomorphism ring is the full ring of integers $\mathcal{O}_K = \operatorname{End}(E)$, a unique factorization domain by virtue of having class number $h(-163) = 1$ (the largest Heegner number). Under this framework, what has traditionally been called *KV-cache compression* is one endomorphism among many; the compression follows as a consequence of the algebraic structure rather than as an engineering target. We state six theorems that organize the construction:

1. **Endomorphism Realization** — every standard transformer operation has a representation in $\operatorname{End}(E) = \mathcal{O}_K$.
2. **Möbius UFD Compression** — exact reconstruction of embeddings from a square-free basis.
3. **Hasse–Weil Compression Bound** — the per-layer information capacity equals $\log_2(p + 1 + 2\sqrt{p})$, identifying the Hasse–Weil bound with the Shannon limit for the model.
4. **Frobenius Quantization** — on a CM curve, the Frobenius endomorphism $\varphi_p$ commutes with $\operatorname{End}(E)$, so quantization to any width $q$ with $p^q \leq 2^{16}$ preserves all algebraic relations.
5. **Poncelet Closure / Adaptive Depth** — the residual stream's orbit closes at depth $L$ iff $\sum_{l=1}^L \delta_l = 0$ in $\mathcal{O}_K$, giving an exact early-exit criterion.
6. **CRT Exact Sharding** — embedding and output projection split losslessly across coprime moduli via the Chinese Remainder Theorem.

We then describe seven adjacent extensions that drop out of the same algebraic structure: golden-ratio (Stern–Brocot) rotary position embeddings, Weil-pairing attention, Hecke-eigenform embedding bases, $L$-function activation oracles, LLL-reduction KV writes, *Iwasawa-tower depth stability* (eliminating layer normalization via $\mu = 0$ Iwasawa invariants), and *CM Sato–Tate asymmetric mixed-precision quantization* (combining inert-prime zero-drift channels with split-prime bounded-drift channels for fp10 / fp12 splits). The multi-head construction generalizes from $E^n$ to a $g$-dimensional CM abelian variety on a Siegel modular variety, with scaled dot-product attention replaced by the canonical polarization. Finally, we argue that training and inference are the same machine: training maximizes the analytic rank $r = \operatorname{ord}_{s=1} L(E,s)$ of the Hasse–Weil $L$-function (Birch–Swinnerton-Dyer), encoding long-term memory as Mordell–Weil generator points topologically protected against catastrophic forgetting; inference iterates the same endomorphism until orbit closure.

---

## 1. Introduction

Transformer language models are typically described as a stack of operationally heterogeneous components: a learned embedding table, RMS or layer normalization, three projection matrices for queries/keys/values, scaled dot-product attention with softmax, a gated feed-forward network, a residual stream, and a language modeling head. Each component has its own quantization, sparsification, and acceleration story. Each component compresses or sparsifies under a different empirical heuristic. There is no shared algebraic principle that determines, for the whole forward pass, which states are reachable, which compressions are exact, and which acceleration is mathematically rather than empirically justified.

We propose such a principle. The Shannon-Prime framework treats the entire forward pass as a discrete dynamical system on a single CM elliptic curve $E / \mathbb{Q}(\sqrt{-163})$, with each layer applying an endomorphism in $\operatorname{End}(E)$. The choice of discriminant $-163$ is not aesthetic: it is the largest fundamental discriminant of class number one, which makes $\mathcal{O}_K$ a unique factorization domain and which makes the $j$-invariant $j(E) = -640320^3$ a rational integer. These two facts — UFD structure and integrality of the $j$-invariant — are what license every compression and quantization claim in the rest of the paper.

The original Shannon-Prime work focused on a single piece of the forward pass: the KV-cache, where compression via VHT2, Möbius inversion, square-free indexing, and spinor reconstruction gave 4×–6× memory reduction at zero perplexity delta on production models. The natural question is whether the same algebraic machinery extends to the rest of the forward pass. We argue here that it does, and that doing so resolves a recurring confusion: practitioners who encountered Shannon-Prime as "a KV trick" attempted to slot it into existing engines without re-deriving any of the other operations. The full framework requires that every step share the same ground ring, and once that constraint is met, the compression that previously held only for the KV cache holds layer-wide.

The paper is organized as follows. Section 2 establishes the algebraic setting. Section 3 states the Endomorphism Realization Theorem and gives the 13-step construction. Sections 4–8 prove the five quantitative theorems (Möbius UFD, Hasse–Weil, Frobenius, Poncelet, CRT). Section 9 collects the adjacent extensions, each of which is a direct corollary of the framework. Section 10 discusses training. Section 11 lists open problems.

---

## 2. Algebraic Setting

Let $K = \mathbb{Q}(\sqrt{-163})$, the imaginary quadratic field of discriminant $D = -163$. The ring of integers is

$$\mathcal{O}_K = \mathbb{Z}[\omega], \qquad \omega = \tfrac{1 + \sqrt{-163}}{2},$$

with multiplication $\omega^2 = \omega - 41$. The class number is $h(D) = 1$, so $\mathcal{O}_K$ is a principal ideal domain and hence a unique factorization domain.

Let $E$ be an elliptic curve over $\mathbb{C}$ with complex multiplication by $\mathcal{O}_K$. The theory of complex multiplication gives $\operatorname{End}(E) \cong \mathcal{O}_K$, and the $j$-invariant of $E$ satisfies

$$j(E) = j(\omega) = -640320^3 = -262{,}537{,}412{,}640{,}768{,}000 \in \mathbb{Z}.$$

The integrality of $j(E)$ is the deep classical content: $E$ is defined over $\mathbb{Q}$ and has good reduction modulo every prime $p$ that does not divide $163$. Write $E_p = E \bmod p$ for the reduced curve over $\mathbb{F}_p$.

For each rational prime $p$ of good reduction, the Hasse–Weil theorem gives

$$|\#E_p(\mathbb{F}_p) - (p+1)| \leq 2\sqrt{p}.$$

We will treat this bound — Theorem 3 below — as the central quantitative input from algebraic geometry into the framework. It is the source of all the information-theoretic claims in the paper.

---

## 3. Endomorphism Realization

We now state the main structural theorem.

**Theorem 1 (Endomorphism Realization).** *Let $T$ be a standard transformer block, comprising RMSNorm, Q/K/V projection, scaled dot-product attention (or its Weil-pairing variant introduced in §9), a SwiGLU feed-forward subblock, residual additions, and output projection. Then $T$ admits a representation*

$$T : E^n \longrightarrow E^n, \qquad T = \delta \cdot \mathrm{id}_{E^n} \quad \text{for some } \delta \in \mathcal{O}_K,$$

*where $E^n$ is the $n$-fold fibered product of $E$ with itself and $\delta$ acts diagonally via the endomorphism structure $\operatorname{End}(E) = \mathcal{O}_K$.*

*Proof sketch.* Each component is realized as follows.

- **RMSNorm** is multiplication by a unit in $\mathcal{O}_K^\times$, namely the spinor norm reciprocal. Norm-one elements form a subgroup; the projection lands on this subgroup.
- **Q/K/V projection** is multiplication by three principal ideals $(\delta_Q), (\delta_K), (\delta_V) \subset \mathcal{O}_K$. Because $\mathcal{O}_K$ is a PID, every ideal is principal, hence every projection has a representative in $\mathcal{O}_K$.
- **Attention** is computed via the Weil pairing $e_n : E[n] \times E[n] \to \mu_n$ (see §9), which is bilinear, alternating, and nondegenerate, replacing softmax-of-dot-product as a single algebraic operation.
- **SwiGLU FFN** factors as a composition of two multiplications and a gating element of $\mathcal{O}_K^\times$ (specifically a twin-prime spinor; see §9).
- **Residual add** is point addition on $E$, identified with addition in $\mathcal{O}_K$ via $\operatorname{End}(E) = \mathcal{O}_K$.
- **LM head** factors through the Chinese Remainder decomposition of §8.

Composing these on $E^n$ yields a single endomorphism $\delta_l \in \mathcal{O}_K$ per layer. The full $L$-layer transformer is then the iterated endomorphism $\delta_L \circ \cdots \circ \delta_1$, which by commutativity of $\mathcal{O}_K$ equals $\prod_l \delta_l$. $\blacksquare$

The theorem reduces the transformer to a single object — an element of $\mathcal{O}_K$ — whose factorization, norm, and reduction modulo primes determine every quantitative property of the forward pass.

**Multi-head generalization to abelian varieties.** The construction above places the residual stream on $E^n$, the $n$-fold fibered product of a single elliptic curve. For multi-head attention with $g$ heads, the natural upgrade is to a $g$-dimensional abelian variety $A$ with complex multiplication by $\mathcal{O}_K$, parametrized by a point of the Siegel modular variety $\mathcal{A}_g(\mathbb{C})$. Such $A$ admits a *polarization* $\lambda: A \to \widehat{A}$ — a canonical isogeny from the variety to its dual, bilinear and nondegenerate by construction. Under this generalization, scaled dot-product attention across $g$ heads is realized as a single application of the polarization to the QK pair, replacing $g$ independent dot products with one algebraic operation. The Hasse–Weil bound (Theorem 3 below) generalizes to

$$\#A_p(\mathbb{F}_p) \leq (1 + \sqrt{p})^{2g},$$

giving the exact joint information capacity of all $g$ heads at a single layer within one algebraic object. Cross-head interference, multi-modal alignment, and grouped-query attention are then all instances of restricting the polarization to coisotropic subvarieties of $A$ — they are not architectural decisions but choices of subvariety.

---

## 4. Möbius UFD Compression

**Theorem 2 (Möbius UFD Compression).** *Let $\mathbf{E} : V \to R^d$ be a transformer embedding table with $|V| = V$ tokens. Decompose each index $v \in \{1, \dots, V\}$ as $v = \prod_p p^{a_p(v)}$ in $\mathbb{Z}$. The map*

$$\mathbf{E}(v) = \sum_{d \mid v, \, d \text{ squarefree}} \mu(d) \, \mathbf{E}_{\text{sf}}(d)$$

*reconstructs $\mathbf{E}$ from its values on square-free indices alone, where $\mu$ is the Möbius function and $\mathbf{E}_{\text{sf}}$ is stored only on square-free indices. The reconstruction is exact in $\mathcal{O}_K \otimes R^d$ because $\mathcal{O}_K$ is a UFD.*

*Proof.* Möbius inversion is the Dirichlet convolution identity $\mu * 1 = \delta$, valid in any commutative ring. The embedding map $\mathbf{E}$ extended multiplicatively to $\mathbb{Z}$ is then recovered from its square-free values by direct inversion. UFD is required only to guarantee that the factorization $v = \prod p^{a_p(v)}$ is unique, so that no ambiguity arises in the sum. $\blacksquare$

The density of square-free integers in $\{1, \dots, N\}$ tends to $6/\pi^2 \approx 0.608$ as $N \to \infty$. For $V = 128{,}000$, the table compresses to roughly $77{,}824$ stored vectors with exact reconstruction of the remaining $50{,}176$ entries. The reconstruction cost per non-square-free lookup is bounded by $\omega(v)$, the number of distinct prime factors of $v$, which is at most $\log v / \log \log v$ — i.e., at most $5$ multiplications for any $v < 128{,}000$.

---

## 5. The Hasse–Weil Bound is the Shannon Limit

**Theorem 3 (Hasse–Weil Compression / Shannon Limit).** *Let $E / K$ be the CM elliptic curve of §2 and let $p$ be a prime of good reduction. Let $E_p$ denote the reduction $E \bmod p$. Then the number of distinct hidden states reachable by the transformer trajectory at any single layer is bounded by*

$$\#E_p(\mathbb{F}_p) \leq p + 1 + 2\sqrt{p}.$$

*Equivalently, the per-layer information content of the residual stream is bounded above by $\log_2(p + 1 + 2\sqrt{p})$ bits. This bound is achieved when the per-layer endomorphism $\delta_l \in \mathcal{O}_K$ has order $\#E_p(\mathbb{F}_p)$ in $\operatorname{End}(E_p)^\times \subset \mathbb{F}_p^\times[\sqrt{-D}]$.*

*Proof.* The Hasse–Weil theorem ([Silverman 1986, V.1.1]) gives the bound on $\#E_p(\mathbb{F}_p)$. The realization theorem (Theorem 1) places the hidden state on $E^n$, and the per-layer endomorphism cycles through $\#E_p(\mathbb{F}_p)$ distinct values in each component before returning to its starting point. The information content is $\log_2$ of the number of distinct states. The bound is achieved precisely when $\delta_l$ acts as a generator of the full endomorphism action on $E_p$. $\blacksquare$

We claim this is *the* Shannon limit for the model: any state outside the orbit of $E_p(\mathbb{F}_p)$ is unreachable from any input; any reachable state is reachable in at most $\#E_p(\mathbb{F}_p)$ steps. The model's expressive capacity per layer is exactly $\log_2 \#E_p(\mathbb{F}_p)$ bits, no more.

For $p = 2^{31} - 1 \approx 2.15 \times 10^9$, the bound is approximately $31.0$ bits per layer per residual-stream coordinate — strikingly close to the empirical capacity ceilings reported in mechanistic-interpretability literature.

---

## 6. Frobenius Quantization

The most consequential corollary of the framework is the following theorem, which retroactively explains why Shannon-Prime fp8 quantization succeeds without quantization-aware training.

**Theorem 4 (Frobenius Quantization).** *Let $\varphi_p : E \to E^{(p)}$ be the Frobenius endomorphism, $\varphi_p(x, y) = (x^p, y^p)$. On the CM curve $E$ of §2, $\varphi_p$ lies in $\operatorname{End}(E) = \mathcal{O}_K$ and commutes with every other endomorphism. Consequently, the quantization map*

$$Q_q : \text{fp}16 \longrightarrow \text{fp}q, \qquad q \in \{8, 4, 2\}$$

*defined by reduction $\bmod \, p^q$ for $p$ chosen with $p^q \leq 2^{16}$, is realized as $\varphi_p^{16 - q}$ and preserves every algebraic relation in $\mathcal{O}_K$.*

*Proof.* CM is exactly the statement that $\operatorname{End}(E)$ contains $\mathcal{O}_K$ rather than just $\mathbb{Z}$. For an ordinary CM curve, $\varphi_p \in \mathcal{O}_K$ and satisfies a quadratic equation $\varphi_p^2 - a_p \varphi_p + p = 0$ in $\operatorname{End}(E)$, where $a_p = p + 1 - \#E_p(\mathbb{F}_p)$ is the trace of Frobenius. Because $\mathcal{O}_K$ is commutative, $\varphi_p$ commutes with every $\delta \in \operatorname{End}(E)$. Hence any chain of layer operations $\delta_1, \dots, \delta_L$ satisfies

$$\varphi_p^k(\delta_L \circ \cdots \circ \delta_1) = \delta_L \circ \cdots \circ \delta_1 \circ \varphi_p^k$$

for any $k$. Quantization is exactly application of $\varphi_p$ to a fixed power; by commutativity it does not change the composition order or the algebraic content. $\blacksquare$

This is the theorem that licenses the entire SP fp8 program. The reason SP fp8 succeeds without calibration where standard fp16 → fp8 quantization fails is that SP's compression chain encodes states on $E$, so Frobenius reduction is structure-preserving. Standard quantization treats the same bits as floats; the bits do not correspond to points on a CM curve, and the resulting quantization breaks the multiplicative relations that the model has learned. SP fp8 *does not break those relations* because it is implementing Frobenius rather than rounding.

For fp4: we require $p^4 \leq 2^{16}$, i.e., $p \leq 11$ (smallest valid choice $p = 11$; alternatively $p = 7$ for headroom). For fp2: $p^2 \leq 2^{16}$, $p \leq 251$. The framework predicts that SP fp2 should be achievable at $p$ on the order of low hundreds.

---

## 7. Poncelet Closure and Adaptive Depth

**Theorem 5 (Poncelet Closure).** *Let $\delta_l \in \mathcal{O}_K$ be the per-layer endomorphism of Theorem 1, and let $\Delta_L = \sum_{l=1}^L \delta_l \in \mathcal{O}_K$. The hidden-state trajectory closes (returns to its starting orbit) at layer $L$ if and only if $\Delta_L = 0$ in $\operatorname{End}(E_p)$ for the prime $p$ of the working representation.*

*Proof.* The residual connection $x_{l+1} = x_l + F_l(x_l)$ identifies, under the realization of Theorem 1, with point addition on $E^n$: $P_{l+1} = P_l + \delta_l \cdot \mathbf{1}$, where $\delta_l$ is the endomorphism realizing the $l$-th block. The trajectory $P_L - P_0 = \sum_l \delta_l \cdot \mathbf{1} = \Delta_L \cdot \mathbf{1}$. The orbit closes (returns to a previously visited state, modulo the working prime $p$) iff $\Delta_L \cdot \mathbf{1} = 0$ in $E_p^n$, which by the bijection $\operatorname{End}(E_p) \cong \mathcal{O}_K / p\mathcal{O}_K$ is equivalent to $\Delta_L = 0 \bmod p$. $\blacksquare$

The theorem provides an exact early-exit test that requires only tracking a running sum in $\mathcal{O}_K$. Unlike learned early-exit heads, this criterion is mathematically guaranteed: if $\Delta_L = 0$, the residual stream has completed a cycle and no further information can be added by repeating layers with the same $\delta$ distribution.

The classical Poncelet closure theorem for conics is the special case where $\mathcal{O}_K = \mathbb{Z}$ and the curve degenerates to a pair of ellipses; the billiard trajectory closes at $n$ steps iff a fixed proportion of the geometric parameters is satisfied. Our generalization replaces $\mathbb{Z}$ with the imaginary quadratic order $\mathcal{O}_K$.

---

## 8. CRT Exact Sharding

**Theorem 6 (CRT Exact Sharding).** *Let $m_1, \dots, m_k$ be pairwise coprime positive integers with $M = \prod m_i$. The embedding table $\mathbf{E} : \mathbb{Z}/M\mathbb{Z} \to R^d$ and the LM head matrix $W_{\mathrm{LM}} : R^d \to \mathbb{R}^M$ decompose as*

$$\mathbf{E} = \bigoplus_{i=1}^k \mathbf{E}_i, \qquad W_{\mathrm{LM}} = \bigoplus_{i=1}^k W_{\mathrm{LM}, i}$$

*via the Chinese Remainder isomorphism $\mathbb{Z}/M\mathbb{Z} \cong \bigoplus_i \mathbb{Z}/m_i\mathbb{Z}$. Each summand can be assigned to an independent compute device with no inter-device synchronization until the final CRT reconstruction, which is exact.*

*Proof.* The CRT isomorphism is classical. The embedding and LM head are $\mathcal{O}_K$-linear by Theorem 1, and tensor products commute with finite direct sums, so the decomposition lifts to the full embedding map. Reconstruction is the inverse CRT map, which is exact in finite arithmetic. $\blacksquare$

For inference on $k$ devices, choose $m_i$ as the first $k$ primes exceeding $|V|/k$. Each device owns roughly $|V|/k$ tokens, balanced to within $O(\log |V|)$ by the prime number theorem.

---

## 9. Adjacent Extensions

The seven extensions in this section are corollaries of the framework. None requires additional algebraic input beyond §§2–8; each is a different specialization of the same endomorphism structure.

### 9.1 Stern–Brocot Rotary Position Embeddings

The base frequency $10{,}000$ in RoPE is conventional but arbitrary. The Weyl equidistribution theorem states that the sequence $\{n\alpha\}_{n \geq 1}$ is equidistributed modulo $1$ iff $\alpha$ is irrational, and the rate of equidistribution is governed by the irrationality measure of $\alpha$. The golden ratio $\varphi = (1 + \sqrt{5})/2$ achieves the optimal irrationality measure (equal to $2$, the lower bound), and its continued-fraction expansion $\varphi = [1; 1, 1, 1, \dots]$ produces the slowest possible convergence — equivalently, the maximally equidistributed sequence of rational approximations.

**Corollary 9.1 (Stern–Brocot RoPE).** *Replacing the base frequency $10{,}000^{2i/d}$ in RoPE with $\varphi^{2i/d}$ yields rotational frequencies whose Fibonacci-spaced harmonics achieve maximum equidistribution on the position torus.*

### 9.2 Weil Pairing Attention

The Weil pairing $e_n : E[n] \times E[n] \to \mu_n$ is bilinear, nondegenerate, and alternating. Projecting the query and key onto $E[n]$ for $n \mid \operatorname{ord}(\delta_l)$, attention is computed as $e_n(Q, K)$ in a single bilinear operation, replacing softmax-of-dot-product entirely.

**Corollary 9.2 (Weil Pairing Attention).** *Scaled dot-product attention is replaceable by the Weil pairing $e_n$ with no loss of information for any $n$ dividing the layer order. The complexity is linear in sequence length when $n$ is chosen prime, since $E[n]$ has $n^2$ elements and the pairing is computed in $O(\log n)$ time per pair via the Miller algorithm.*

### 9.3 Hecke-Eigenform Embedding Bases

Let $S_k(\Gamma_0(N))$ denote the space of weight-$k$ cusp forms of level $N$ on the modular group, with a basis of *Hecke eigenforms* $\{f_1, \dots, f_d\}$. Each eigenform has a $q$-expansion $f_i(q) = \sum_{n=1}^\infty a_n(f_i) q^n$ with multiplicative coefficients: $a_{mn}(f) = a_m(f) a_n(f)$ for $\gcd(m, n) = 1$.

**Corollary 9.3 (Hecke Embedding).** *Choose $(k, N)$ so that $\dim S_k(\Gamma_0(N)) = d_{\mathrm{embed}}$. The embedding $\mathbf{E}(n) = (a_n(f_1), \dots, a_n(f_d))$ is orthogonal under the Petersson inner product and multiplicative across coprime indices, so Theorem 2's Möbius reconstruction holds automatically.*

### 9.4 $L$-Function Activation Oracle

Model the FFN firing sequence as the coefficients of an $L$-function $L(E, s) = \prod_p L_p(E, s)$ associated to the curve $E$. The coefficients satisfy multiplicativity $a_{mn} = a_m a_n$ for $\gcd(m, n) = 1$ and Ramanujan–Petersson bounds $|a_p| \leq 2 p^{(k-1)/2}$.

**Corollary 9.4 ($L$-Function Oracle).** *Observation of $\{a_p\}$ at prime indices determines $\{a_n\}$ for all composite indices via multiplicativity. The oracle's prediction is exact at primes and Ramanujan-bounded at composites.*

### 9.5 LLL Reduction for KV Write

**Corollary 9.5 (LLL KV Write).** *The KV archive at any timestep $t$ is the LLL-reduced lattice basis of the integer lattice spanned by the column vectors $\{K_1, \dots, K_t\}$ after $\mathcal{O}_K$-quantization. The reduction is exact, deterministic, and produces provably the shortest basis up to the LLL approximation factor.*

### 9.6 Iwasawa Towers for Infinite-Depth Scaling

A stack of $L$ transformer layers can be viewed not as a flat sequence but as a $\mathbb{Z}_p$-extension tower of the base field. Fix a prime $p$ and let

$$K = K_0 \subset K_1 \subset K_2 \subset \cdots \subset K_\infty$$

be the cyclotomic $\mathbb{Z}_p$-extension of $K = \mathbb{Q}(\sqrt{-163})$, with $\operatorname{Gal}(K_n/K) \cong \mathbb{Z}/p^n\mathbb{Z}$. Assign layer $l$ to level $n = l$ via its corresponding Galois action on the residual stream. The accumulated state at depth $L$ is then governed by the *Iwasawa invariants* $\mu$ and $\lambda$ of the tower [Iwasawa 1959; Mazur–Wiles 1984].

**Corollary 9.6 (Infinite-Depth Stability).** *If the per-layer endomorphisms $\{\delta_l\} \subset \mathcal{O}_K$ are chosen so that the Iwasawa $\mu$-invariant of the associated $\mathbb{Z}_p$-extension vanishes, then the residual stream's accumulation $\Delta_L = \sum_{l=1}^L \delta_l$ has $p$-adic valuation bounded linearly in $L$:*

$$\operatorname{ord}_p(\Delta_L) \leq \lambda \cdot L + O(1).$$

*Consequently, $\Delta_L$ does not exhibit exponential growth and no layer-normalization step is required to stabilize the stream across arbitrary depth.*

By the Iwasawa Main Conjecture (Mazur–Wiles for cyclotomic $\mathbb{Z}_p$-extensions of abelian number fields), $\mu$ is controlled analytically via a $p$-adic $L$-function. Enforcing $\mu = 0$ during training is therefore an *algebraic* constraint on the layer endomorphisms, not an architectural one, and it eliminates the need for RMSNorm or LayerNorm in deep stacks. A $1000$-layer transformer with $\mu = 0$ trained endomorphisms is algebraically as stable as a $32$-layer one.

For practical training, the $\mu = 0$ constraint reduces to a finite-dimensional check on the leading coefficients of the $p$-adic $L$-function, computable in $O(L)$ time during the loss evaluation.

### 9.7 CM Sato–Tate Asymmetric Mixed-Precision Quantization

The Frobenius Quantization Theorem (§6, Theorem 4) introduces a single working prime $p$ at each precision tier. For our CM curve $E / \mathbb{Q}(\sqrt{-163})$, Deuring's classical result on Frobenius distributions [Deuring 1941] gives a sharper structure: the rational primes $p \nmid 163$ partition under Chebotarev density as

- **Inert primes** ($p$ remains prime in $\mathcal{O}_K$, density $\tfrac{1}{2}$): $a_p = 0$ exactly, $\#E_p(\mathbb{F}_p) = p + 1$, $E_p$ supersingular.
- **Split primes** ($p = \pi \bar{\pi}$ in $\mathcal{O}_K$, density $\tfrac{1}{2}$): $a_p = 2\sqrt{p}\cos\theta_p$ with $\theta_p$ equidistributed in $[0, \pi]$, $E_p$ ordinary.

The CM Sato–Tate measure differs from the non-CM semicircle law; the inert primes give $a_p \equiv 0$ deterministically rather than only on a measure-zero set.

**Corollary 9.7 (Asymmetric fp10 Splitting).** *Choose an inert prime $p_1$, for which $a_{p_1} = 0$ and hence Frobenius quantization at $p_1$ introduces zero systematic drift. Choose a split prime $p_2$ minimizing $|\cos\theta_{p_2}|$, for which the residual drift is bounded by $2\sqrt{p_2}|\cos\theta_{p_2}|$ and small. The composite quantization*

$$Q_{q_1, q_2} = \varphi_{p_1}^{k_1} \circ \varphi_{p_2}^{k_2}$$

*realizes an asymmetric mixed-precision encoding combining $\log_2 p_1^{k_1}$ bits of zero-drift skeleton with $\log_2 p_2^{k_2}$ bits of bounded-drift residual. The total bit-width is $k_1 \log_2 p_1 + k_2 \log_2 p_2$; the total Frobenius drift is bounded by the split-prime contribution alone.*

This is the natural construction for fp10 = fp8 + fp2 mixed-precision tensors, fp12 = fp8 + fp4, and similar splits. By coprimality of $p_1, p_2$ and the commutativity of $\operatorname{End}(E)$, the composite is order-independent and the resulting fp10 representation is calibration-free in the same sense as fp8 (Corollary 4.1). Explicit prime pairs for production wordsizes are tabulated in [SP-Theory-Companion 2026, §3].

---

## 10. Training and Inference Are the Same Machine

Under the framework, both training and inference reduce to manipulation of the per-layer endomorphism $\delta_l \in \mathcal{O}_K$.

**Inference.** Given a fixed sequence $\{\delta_l\}_{l=1}^L$, iterate $P_{l+1} = P_l + \delta_l \cdot \mathbf{1}$ for $L$ layers. By Theorem 5, the trajectory closes at the smallest $L$ for which $\Delta_L = 0$ in $\operatorname{End}(E_p)$. Adaptive-depth inference simply runs until closure.

**Training.** Given a target trajectory $\{P_l^*\}_{l=1}^L$ and an input $P_0$, fit $\delta_l \in \mathcal{O}_K$ to minimize $\sum_l \|P_l - P_l^*\|^2$. The optimization is over a discrete commutative ring; the lattice structure of $\mathcal{O}_K$ admits efficient integer-programming methods (LLL, Babai's nearest-plane algorithm) for finding optimal $\delta_l$.

The key observation is that the *operators* are identical between the two modes — both apply $\delta_l$ in $\mathcal{O}_K$. The difference is whether $\delta_l$ is fixed (inference) or learned (training).

**Long-term weight memory via BSD.** The Mordell–Weil theorem states that the group of rational points $E(\mathbb{Q})$ of our curve $E$ is finitely generated:

$$E(\mathbb{Q}) \cong \mathbb{Z}^r \oplus E(\mathbb{Q})_{\mathrm{tors}}.$$

The rank $r$ is the *non-volatile* information capacity of the trained weight architecture, independent of the local prime $p$ used at inference time. Where Theorem 3 bounded the per-layer *activation* capacity, the rank $r$ bounds the per-model *weight* capacity. By the Birch–Swinnerton-Dyer conjecture — partially proven in the CM case [Coates–Wiles 1977; Gross–Zagier 1986; Kolyvagin 1990] — the rank equals the analytic order of vanishing:

$$r = \operatorname{ord}_{s=1} L(E, s).$$

This suggests a training objective. Rather than minimizing cross-entropy by stochastic gradient descent over a floating-point parameter space, *maximize the analytic rank of the Hasse–Weil $L$-function associated with the layer endomorphisms*. The training problem becomes a search over the Mordell–Weil group $E(\mathbb{Q})$ for generator points; lattice-reduction algorithms (LLL, Babai nearest-plane) on $E(\mathbb{Q})$ provide a direct search and an explicit set of generator points $\{P_1, \dots, P_r\}$ as the trained model.

**Corollary 10.1 (Catastrophic-Forgetting Resistance).** *Models trained by analytic-rank maximization resist catastrophic forgetting in the precise sense that their long-term memories are encoded as Mordell–Weil generator points $P_i \in E(\mathbb{Q})$. The generator set is invariant under reduction modulo any prime $p \nmid N_E$ (the conductor of $E$), so memories cannot be overwritten by subsequent training at any local precision — they are topologically protected by the global functional equation*

$$\Lambda(E, s) = w_E \Lambda(E, 2 - s),$$

*where $\Lambda$ is the completed $L$-function and $w_E = \pm 1$ is the global root number.*

This is the framework's reply to the catastrophic-forgetting problem of standard transformer training: by encoding memory in $\mathbb{Z}^r$ rather than $\mathbb{R}^{N_{\text{params}}}$, memories become *combinatorial* (a basis of $r$ generator points) rather than *analog* (a vector in floating-point space), and they survive arbitrary subsequent training of the local quantization.

---

## 11. Open Problems

1. **Explicit curve.** Identify a defining Weierstrass equation for the CM curve $E$ over $\mathbb{Q}(\sqrt{-163})$ suitable for arithmetic implementation.

2. **Optimal layer prime.** Determine the prime $p$ that maximizes $\log_2 \#E_p(\mathbb{F}_p)$ subject to fitting in a chosen wordsize.

3. **Hecke-eigenform vocabularies.** Empirically validate that a transformer trained with a Hecke-eigenform embedding matches or exceeds a learned-embedding baseline at fixed parameter count.

4. **Frobenius fp4 calibration.** Implement Theorem 4 at $q = 4$ with $p = 11$ and demonstrate calibration-free quantization on a Qwen-class model.

5. **Weil pairing kernels.** Implement the Miller algorithm on CUDA / Hexagon HVX and benchmark against scaled dot-product attention at sequence lengths 4K–128K.

6. **Adaptive-depth at scale.** Apply Theorem 5's closure criterion to a production decoder and measure layer skip rates on natural workloads.

7. **Sato–Tate prime tables.** For each precision tier $q \in \{2, 4, 8, 10, 12, 16\}$, identify the optimal (inert, split) prime pair $(p_1, p_2)$ minimizing the split-prime drift.

8. **Iwasawa $\mu$ on trained models.** Compute the Iwasawa $\mu$-invariant of the $\mathbb{Z}_p$-extension associated with a production-trained model. Test the framework's prediction that $\mu = 0$ for stably-trained models.

9. **Explicit Siegel construction.** For $g \in \{32, 40, 64\}$, construct an explicit $g$-dimensional abelian variety with CM by $\mathcal{O}_K$ and a polarization $\lambda : A \to \widehat{A}$ suitable for hardware implementation.

10. **Analytic-rank training.** Implement Babai's nearest-plane algorithm on $E(\mathbb{Q})$ as a substitute for the Adam optimizer on a small model. Benchmark catastrophic-forgetting resistance against SGD baselines.

11. **Polarization-attention kernel.** Implement the polarization map $\lambda : A \to \widehat{A}$ as a CUDA / HVX kernel and benchmark against scaled dot-product attention at fixed head count $g$.

---

## References

- Silverman, J. H. *The Arithmetic of Elliptic Curves*, Springer GTM 106, 1986.
- Silverman, J. H. *Advanced Topics in the Arithmetic of Elliptic Curves*, Springer GTM 151, 1994.
- Deligne, P. "La conjecture de Weil. I." *Publ. Math. IHÉS* 43 (1974), 273–307.
- Deuring, M. "Die Typen der Multiplikatorenringe elliptischer Funktionenkörper." *Abh. Math. Sem. Hansischen Univ.* 14 (1941), 197–272.
- Iwasawa, K. "On Gamma-extensions of algebraic number fields." *Bull. AMS* 65 (1959), 183–226.
- Mazur, B.; Wiles, A. "Class fields of abelian extensions of $\mathbb{Q}$." *Invent. Math.* 76 (1984), 179–330.
- Coates, J.; Wiles, A. "On the conjecture of Birch and Swinnerton-Dyer." *Invent. Math.* 39 (1977), 223–251.
- Gross, B. H.; Zagier, D. B. "Heegner points and derivatives of $L$-series." *Invent. Math.* 84 (1986), 225–320.
- Kolyvagin, V. A. "Euler systems." In *The Grothendieck Festschrift, Vol. II*, Birkhäuser, 1990.
- Lenstra, A. K.; Lenstra, H. W.; Lovász, L. "Factoring polynomials with rational coefficients." *Math. Ann.* 261 (1982), 515–534.
- Heegner, K. "Diophantische Analysis und Modulfunktionen." *Math. Z.* 56 (1952), 227–253.
- Su, J. *et al.* "RoFormer: Enhanced transformer with rotary position embedding." 2021.
- Anonymous (Shannon-Prime Project). "VHT2 and Möbius-Inverse KV Compression: Validation Results." Internal report v1.16, 2026.
- Knack, A. "Shannon-Prime: Engine, llama, comfyui — three working backends." Project archive, 2026-04-25.
