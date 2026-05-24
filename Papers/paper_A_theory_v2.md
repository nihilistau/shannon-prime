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

We then describe seven adjacent extensions that drop out of the same algebraic structure: golden-ratio (Stern–Brocot) rotary position embeddings, Weil-pairing attention, Hecke-eigenform embedding bases, $L$-function activation oracles, LLL-reduction KV writes, Iwasawa-tower depth stability (eliminating layer normalization via $\mu = 0$ Iwasawa invariants), and CM Sato–Tate asymmetric mixed-precision quantization (combining inert-prime zero-drift channels with split-prime bounded-drift channels for fp10 / fp12 splits). The multi-head construction generalizes from $E^n$ to a $g$-dimensional CM abelian variety on a Siegel modular variety, with scaled dot-product attention replaced by the canonical polarization. Finally, we argue that training and inference are the same machine: training maximizes the analytic rank $r = \operatorname{ord}_{s=1} L(E,s)$ of the Hasse–Weil $L$-function (Birch–Swinnerton-Dyer), encoding long-term memory as Mordell–Weil generator points topologically protected against catastrophic forgetting; inference iterates the same endomorphism until orbit closure.

---

[...remainder omitted for brevity, will use full content in actual commit...]