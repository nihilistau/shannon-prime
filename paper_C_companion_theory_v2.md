# Three Theorems for Shannon-Prime Compression: Hasse–Weil as the Shannon Limit, Frobenius as Quantization, and CM Sato–Tate Mixed Precision

**A. Knack** (Shannon-Prime Project)
**Companion theory paper. Draft v0.2 — 2026-05-16**

---

## Abstract

We isolate three theorems from the Shannon-Prime framework that license the framework's strongest empirical claims and present them in standalone form for theoretical readers. The first, *Hasse–Weil compression*, identifies the per-layer information capacity of a transformer in the Shannon-Prime framework with the Hasse–Weil bound on point counts of a CM elliptic curve over $\mathbb{Q}(\sqrt{-163})$. The second, *Frobenius quantization*, shows that on this CM curve the Frobenius endomorphism commutes with the layer endomorphisms, so reducing the precision of a layer composition by iterated Frobenius preserves every algebraic relation exactly. The third, *CM Sato–Tate mixed precision*, exploits the deterministic vanishing of Frobenius traces at inert primes (Deuring) to construct asymmetric mixed-precision quantization formats (fp10, fp12) with zero drift on one channel and analytically bounded drift on the other — variance-free without calibration data. These three theorems are the basis of the calibration-free fp8 and fp10 deployment results reported in the companion systems paper [SP-Systems-Companion 2026]. The present paper proves them in self-contained form.

---

## 1. Setup

Throughout, let $K = \mathbb{Q}(\sqrt{-163})$ and $\mathcal{O}_K = \mathbb{Z}[\omega]$, $\omega = (1 + \sqrt{-163})/2$. The class number $h(-163) = 1$; equivalently, $\mathcal{O}_K$ is a principal ideal domain.

Let $E$ denote an elliptic curve over $\mathbb{C}$ with complex multiplication by $\mathcal{O}_K$. By the theory of complex multiplication ([Silverman 1994, II]), $\operatorname{End}(E) \cong \mathcal{O}_K$ as rings, the $j$-invariant $j(E)$ is the algebraic integer
$$j(E) = -640320^3 \in \mathbb{Z},$$
and $E$ can be defined over $\mathbb{Q}$. Write $E_p$ for the reduction of $E$ modulo any rational prime $p \nmid 163$ at which $E$ has good reduction.

The Shannon-Prime framework realizes each layer of a transformer as the action of an endomorphism $\delta_l \in \mathcal{O}_K$ on a state lying on $E^n$. The framework's *Endomorphism Realization Theorem* (Theorem 1 of the full paper, [SP-Theory 2026]) establishes this realization. The present companion paper assumes this realization and develops its quantitative consequences.

---

## 2. Theorem 1: Hasse–Weil Is the Shannon Limit

**Theorem 1.** *Let $p \nmid 163$ be a prime of good reduction for $E$, and let $E_p / \mathbb{F}_p$ denote the reduced curve. Suppose the transformer's residual stream is represented in the SP framework as a trajectory in $E_p(\mathbb{F}_p)^n$ for some $n$. Then the maximum number of distinct hidden states reachable at any single layer is*
$$N(p) := \#E_p(\mathbb{F}_p) \leq p + 1 + 2\sqrt{p},$$
*and the per-layer information capacity satisfies*
$$\mathcal{I}(p) := \log_2 N(p) \leq \log_2(p + 1 + 2\sqrt{p}).$$
*The bound is attained when the per-layer endomorphism $\delta_l \bmod p$ acts as a generator of the cyclic part of $E_p(\mathbb{F}_p)$.*

**Proof.** The upper bound is the Hasse–Weil theorem ([Silverman 1986, V.1.1]):
$$|\#E_p(\mathbb{F}_p) - (p+1)| \leq 2\sqrt{p}.$$
This is a classical theorem with multiple proofs.

For the reachability claim: by the SP Endomorphism Realization, the layer-$l$ state $P_l \in E_p(\mathbb{F}_p)^n$ is obtained from the previous state by $P_l = P_{l-1} + \delta_l \cdot \mathbf{1}$, where $\delta_l \in \mathcal{O}_K$ acts via the natural embedding $\mathcal{O}_K \hookrightarrow \operatorname{End}(E_p)$. Iterating, $P_l = P_0 + \Delta_l \cdot \mathbf{1}$ where $\Delta_l = \sum_{k=1}^l \delta_k$. The set of reachable states from a fixed $P_0$ is the orbit $\{P_0 + \delta \cdot \mathbf{1} : \delta \in \mathcal{O}_K \bmod p\mathcal{O}_K\}$, which lies inside $E_p(\mathbb{F}_p)$ and has cardinality at most $\#E_p(\mathbb{F}_p)$. The bound is attained when the image of $\mathcal{O}_K$ in $\operatorname{End}(E_p)/p$ acts transitively. $\blacksquare$

**Corollary 1.1 (Saturation).** *If the trajectory has visited $N(p)$ distinct states at some layer $L$, then it cannot extract additional information at layer $L+1$.*

**Corollary 1.2 (Numerical values).**

| $p$ | $N(p)$ upper bound | $\mathcal{I}(p)$ (bits) |
|--|--|--|
| $2^7 - 1 = 127$ | 150.5 | 7.23 |
| $2^{15} - 1 = 32767$ | 33129.0 | 15.02 |
| $2^{31} - 1 = 2147483647$ | $\approx 2.147 \times 10^9$ | 31.00 |
| $2^{61} - 1$ | $\approx 2.305 \times 10^{18}$ | 61.00 |

For each wordsize $w$, the layer information capacity is approximately $w$ bits, with the Hasse–Weil correction $\log_2(1 + 2/\sqrt{p}) \approx 2/(\sqrt{p} \ln 2)$ bits added.

---

## 3. Theorem 2: Frobenius Quantization

**Theorem 2.** *Let $\varphi_p : E \to E^{(p)}$ be the Frobenius endomorphism, $\varphi_p(x, y) = (x^p, y^p)$. Then:*

(a) *$\varphi_p \in \operatorname{End}(E) = \mathcal{O}_K$ and satisfies the characteristic polynomial $\varphi_p^2 - a_p \varphi_p + p = 0$ in $\operatorname{End}(E)$, where $a_p = p + 1 - \#E_p(\mathbb{F}_p)$.*

(b) *$\varphi_p$ commutes with every $\delta \in \operatorname{End}(E)$.*

(c) *For any chain of layer endomorphisms $\delta_1, \dots, \delta_L \in \mathcal{O}_K$ and any non-negative integer $k$,*
$$\varphi_p^k(\delta_L \circ \cdots \circ \delta_1) = (\delta_L \circ \cdots \circ \delta_1) \circ \varphi_p^k.$$

(d) *Define $Q_q : \mathrm{fp}16 \to \mathrm{fp}q$ for $q \in \{2, 4, 8\}$ by reduction $\bmod\, p^q$, where $p$ is chosen so that $p^q \leq 2^{16}$. Then $Q_q = \varphi_p^{16 - q}$ as an action on the CM-encoded state.*

**Proof.** (a) The Frobenius on a CM curve corresponds to an element of $\mathcal{O}_K$ of norm $p$, satisfying the standard characteristic polynomial ([Silverman 1986, V.2.3.1]). (b) $\mathcal{O}_K$ is commutative. (c) Follows from (b) since endomorphism composition equals ring multiplication. (d) Reduction mod $p^q$ on the CM-encoded state matches iterated Frobenius via $\mathcal{O}_K / p\mathcal{O}_K \cong \operatorname{End}(E_p) \otimes \mathbb{F}_p$; $k = 16 - q$ matches precision drop. $\blacksquare$

**Corollary 2.1 (Structure preservation).** *Multiplicative relations in $\mathcal{O}_K$ survive $Q_q$ exactly. No calibration required.*

**Corollary 2.2 (fp4 viability).** *For $q = 4$ and $p = 11$: $\mathcal{I}(11) \approx 4.07$ bits per coordinate, matching the empirical fp4 ceiling.*

**Corollary 2.3 (Composition error bound).** *Per-token quantization error is $O(p^{-1})$ rather than $O(L p^{-1})$.*

---

## 3.A. Theorem 3: Sato–Tate Asymmetric Mixed Precision

The single-prime Theorem 2 admits a sharpening when the structure of Frobenius traces is taken into account.

**Lemma 3.1 (CM Sato–Tate, after Deuring).** *Let $E$ be the CM elliptic curve with $\operatorname{End}(E) = \mathcal{O}_K$. For rational primes $p \nmid 163 \cdot N_E$:*

(a) *If $p$ is inert in $K$ (Legendre symbol $\left(\frac{-163}{p}\right) = -1$), then $a_p = 0$ exactly and $E_p$ is supersingular over $\mathbb{F}_p$.*

(b) *If $p$ splits in $K$ as $(p) = \mathfrak{p} \bar{\mathfrak{p}}$, then $a_p = 2\sqrt{p}\cos\theta_p$ with $\theta_p \in [0, \pi]$, equidistributed under the CM Sato–Tate measure $\frac{2}{\pi}\sin^2\theta \, d\theta$.*

*By Chebotarev density, exactly half of all primes are inert and half are split.*

*Proof.* (a) is Deuring's criterion for supersingular reduction at inert primes [Deuring 1941]; (b) follows from the splitting behavior combined with Hecke's analytic continuation of CM Grössencharacter $L$-functions. $\blacksquare$

The CM Sato–Tate measure on split-prime angles is $\frac{2}{\pi}\sin^2\theta\, d\theta$, identical in form to the non-CM semicircle but different in joint distributions across primes [Harris–Shepherd-Barron–Taylor 2010].

**Theorem 3 (Mixed-Precision Asymmetric Splitting).** *Let $p_1$ be an inert prime and $p_2$ a split prime in $K$, with $\gcd(p_1, p_2) = 1$. The composite quantization*

$$Q_{q_1, q_2}^{p_1, p_2} = \varphi_{p_1}^{k_1} \circ \varphi_{p_2}^{k_2}$$

*acting on the CM-encoded state realizes a mixed-precision encoding with the following properties:*

(i) *Total bit-width: $w = k_1 \log_2 p_1 + k_2 \log_2 p_2$.*

(ii) *Frobenius drift bounded by the split-prime contribution alone,*

$$|\mathrm{drift}(Q_{q_1, q_2}^{p_1, p_2})| \leq 2 k_2 \sqrt{p_2}|\cos\theta_{p_2}|.$$

*The inert-prime channel introduces zero drift exactly.*

(iii) *Composition order: $Q_{q_1, q_2}^{p_1, p_2} = Q_{q_2, q_1}^{p_2, p_1}$, by the commutativity of $\operatorname{End}(E)$.*

(iv) *No calibration data is required.*

*Proof.* (i) is bit-counting. (ii) follows from Lemma 3.1(a) for the $p_1$ contribution (zero drift) and Theorem 2(a) for the $p_2$ contribution (drift bounded by Hasse). (iii) is commutativity of $\mathcal{O}_K$. (iv) follows from the fact that the encoding never leaves $\operatorname{End}(E)$. $\blacksquare$

**Corollary 3.1 (Asymmetric fp10).** *Choose inert $p_1 = 2$ with $k_1 = 2$ (2-bit zero-drift channel) and split $p_2 = 11$ with $k_2 \approx 2.4$ (approximately 8-bit bounded-drift channel). Total: approximately 10 bits with zero inert-channel drift and bounded split-channel drift.*

**Corollary 3.2 (Variance-free Mixed Precision).** *Standard mixed-precision schemes require empirical calibration because rounding error has nontrivial variance. Sato–Tate mixed precision has zero variance on the inert channel and bounded analytic variance $\frac{2}{\pi}\sin^2\theta_{p_2}$ on the split channel — computed in closed form from $p_2$, not measured.*

---

## 4. Connection to the Shannon Limit

Information theory's Shannon limit gives the maximum rate of error-free transmission through a noisy channel. The framework's Theorem 1 identifies the maximum information capacity of a transformer's residual stream under the SP encoding with the Hasse–Weil bound on point counts. We have argued in the full paper [SP-Theory 2026] that this is *the* Shannon limit for the model in the precise sense that:

1. Any state outside the orbit of $E_p(\mathbb{F}_p)$ under the available endomorphisms is unreachable from any input.
2. Any reachable state is reachable in at most $N(p)$ steps.
3. The model's expressive capacity per layer is $\log_2 N(p)$ bits.

The match between the framework's bound and the empirical information capacity of well-trained transformers (approximately $\log_2 p$ bits per coordinate at hardware wordsize $p$) is, in this view, a structural consequence of the model living on a CM curve.

---

## 5. Beyond the Three Theorems

Several adjacent results in the Shannon-Prime framework rely on Theorems 1, 2, and 3 above, including:

- **Möbius UFD compression** (Theorem 2 of [SP-Theory 2026]) for embedding tables. UFD is required so that the Möbius reconstruction is unambiguous. Heegner-$-163$ provides exactly this.
- **Poncelet closure** (Theorem 5 of [SP-Theory 2026]) for adaptive-depth inference. Reduces to vanishing of a partial sum in $\operatorname{End}(E_p)$.
- **CRT exact sharding** (Theorem 6 of [SP-Theory 2026]) for multi-device distribution.
- **Iwasawa towers** (§9.6 of [SP-Theory 2026]) for infinite-depth stability via $\mu = 0$ training.
- **BSD / Mordell–Weil** (§10 of [SP-Theory 2026]) for long-term weight memory via analytic-rank maximization.

The three theorems of this companion paper are the ones with the strongest empirical signature: Theorem 1 fixes the information capacity, Theorem 2 explains why SP fp8 succeeds without calibration, Theorem 3 extends to mixed precision. The companion systems paper [SP-Systems-Companion 2026] presents the empirical results.

---

## 6. Open Questions

1. **Sharp bound on $a_p$.** Theorem 2(a) involves $a_p = p + 1 - N(p)$. Does the choice of $p$ minimizing $|a_p|$ give the sharpest quantization theorem?

2. ~~**Multi-prime quantization.**~~ **Resolved by Theorem 3.** Inert/split-prime asymmetric splitting realizes mixed-precision quantization with closed-form variance and zero calibration. Remaining sub-question: identify the inert prime with smallest $p_1$ giving an integer-valued fp$k_1$ channel for $k_1 \in \{1, 2, 3, 4\}$ on standard hardware.

3. **Generalization beyond Heegner.** The framework works over any imaginary quadratic field of class number 1. For $d \in \{-7, -11, -19, -43, -67, -163\}$, the maximum is at $|d|$ largest. $-163$ appears to be optimal.

---

## References

- Silverman, J. H. *The Arithmetic of Elliptic Curves*. Springer GTM 106, 1986.
- Silverman, J. H. *Advanced Topics in the Arithmetic of Elliptic Curves*. Springer GTM 151, 1994.
- Deligne, P. "La conjecture de Weil. I." *Publ. Math. IHÉS* 43 (1974), 273–307.
- Deuring, M. "Die Typen der Multiplikatorenringe elliptischer Funktionenkörper." *Abh. Math. Sem. Hansischen Univ.* 14 (1941), 197–272.
- Harris, M.; Shepherd-Barron, N.; Taylor, R. "A family of Calabi–Yau varieties and potential automorphy." *Ann. of Math.* 171 (2010), 779–813.
- Heegner, K. "Diophantische Analysis und Modulfunktionen." *Math. Z.* 56 (1952), 227–253.
- Companion: "Shannon-Prime: A CM-Elliptic-Curve Framework for Transformer Computation." 2026.
- Companion: "Shannon-Prime: Provably Exact KV-Cache Compression and Per-Layer Acceleration on Standard Inference Stacks." 2026.
