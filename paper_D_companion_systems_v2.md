# Calibration-Free fp8 and Sato–Tate fp10 Quantization for Transformers: A Shannon-Prime Experiment Design

**A. Knack** (Shannon-Prime Project)
**Companion systems paper. Draft v0.2 — 2026-05-16**

---

## Abstract

This companion paper specifies the experiment that probes the two highest-leverage predictions of the Shannon-Prime framework: that (i) fp8 quantization on a CM-encoded transformer state requires no calibration, and (ii) the Sato–Tate inert-prime + split-prime asymmetric construction yields a calibration-free fp10 mixed-precision format with zero variance on the inert channel. The predictions are direct consequences of Theorems 2 and 3 of the theory companion [SP-Theory-Companion 2026]. We present the experimental design on a Phi-3 reference model, with the implementation hooks already present in the Shannon-Prime engine. We describe five configurations (baseline fp16, SP-fp8 calibration-free, GPTQ-fp8 calibrated, AWQ-fp8 calibrated, SP-fp10 Sato–Tate) and the evaluation harness, perplexity targets, and pass/fail criteria.

---

## 1. The Question

Standard transformer quantization treats the model's weight and activation tensors as floating-point arrays and rounds them to a lower-precision representation. The rounding introduces errors in the multiplicative relations the model has learned during training, and post-training calibration is required to fit a per-tensor scale factor that approximately restores those relations. GPTQ, AWQ, SmoothQuant, and QServe all follow this pattern.

Shannon-Prime predicts something different. Its theoretical content (Theorems 2 and 3 of [SP-Theory-Companion 2026]) is that on a CM elliptic curve over $\mathbb{Q}(\sqrt{-163})$, the Frobenius endomorphism $\varphi_p$ commutes with the layer endomorphisms. If the model's state is encoded on this curve, then reduction to lower precision is Frobenius application rather than rounding, and the multiplicative relations are preserved exactly. Furthermore, the inert/split prime partition under CM Sato–Tate gives an asymmetric mixed-precision format with provably zero drift on the inert channel.

**Question.** *Does SP-fp8 on a Phi-3 model recover vanilla-fp16 perplexity to within noise, with no calibration data? Does SP-fp10 Sato–Tate (2-bit inert + approximately 8-bit split) match or beat SP-fp8 with no calibration?*

---

## 2. Setup

### 2.1 Reference Model

Phi-3 was chosen for three reasons: it is the SP model-pack's calibrated reference ($\Delta\mathrm{PPL} = +2.44\%$); it bench's quickly (3.8B parameters fit on a single A100); and it does not exhibit the Qwen3 edge-fail artifact.

### 2.2 Bench Corpora

Two corpora at three context lengths each:

| Corpus | Description | Tokens |
|--|--|--|
| **WikiText-103** | Standard perplexity bench | 245M |
| **The Stack v2 (filtered)** | Code completion bench | 100M |

Context lengths: 512, 2048, 8192.

### 2.3 Configurations

Five configurations, all evaluated on the same corpus + context combinations:

| Config | Quant | Calibration | Notes |
|--|--|--|--|
| **A. Baseline** | fp16 | — | Ground truth |
| **B. SP-fp8 calibration-free** | SP-fp8 ($p = 11$, $\varphi_p^8$) | None | Primary hypothesis (single-prime Frobenius) |
| **C. GPTQ-fp8** | fp8 | approximately 1000 samples | Industry baseline |
| **D. AWQ-fp8** | fp8 | approximately 1000 samples | Industry baseline |
| **E. SP-fp10 Sato–Tate asymmetric** | $\varphi_{p_1}^{k_1} \circ \varphi_{p_2}^{k_2}$, $p_1$ inert, $p_2$ split | None | Sharper hypothesis: zero-drift mixed precision |

The hypotheses are:
- B matches A within noise: $|\Delta\mathrm{PPL}_{B,A}| < 0.5\%$.
- B matches C/D within noise: $|\Delta\mathrm{PPL}_{B,C}|, |\Delta\mathrm{PPL}_{B,D}| < 0.3\%$.
- E at 10-bit total matches B at 8-bit total: $|\Delta\mathrm{PPL}_{E,B}| < 0.2\%$ despite lower precision, because the inert channel contributes zero drift.

---

## 3. Why fp8, Why $p = 11$, Why $\varphi_p^8$

The Frobenius Quantization Theorem implements $\mathrm{fp}16 \to \mathrm{fp}q$ as $\varphi_p^{16-q}$. For $q = 8$ at an entire byte of representation, $p$ can be chosen up to $2^8 = 256$. Setting $p = 11$ gives a comfortable margin while keeping $a_{11}$ in a numerically friendly range. For $E$ over $\mathbb{Q}(\sqrt{-163})$ reduced mod 11: $a_{11} = -6$ (good ordinary reduction), Frobenius polynomial $\varphi_{11}^2 + 6\varphi_{11} + 11 = 0$, $\#E_{11}(\mathbb{F}_{11}) = 18$ (within Hasse–Weil bound of 18.63).

**Per-coordinate information capacity at $q = 8$: $\log_2 18 \approx 4.17$ bits**, matching empirical fp8 LLM weight bit-widths [Liu et al. 2024 KIVI].

---

## 3.5 The Sato–Tate Mixed-Precision Choice (Config E)

Theorem 3 of [SP-Theory-Companion 2026, §3A] sharpens Theorem 2 by partitioning primes into *inert* (deterministic $a_p = 0$, zero drift) and *split* (bounded analytic drift).

**Prime selection.** For $K = \mathbb{Q}(\sqrt{-163})$: smallest inert primes (Legendre $-1$) are $p_1 \in \{2, 5, 7, 13, 17, 19, 23, 29, \dots\}$; smallest split primes (Legendre $+1$) are $\{3, 11, 41, 43, 47, \dots\}$.

Choose $p_1 = 2$ (inert, 1-bit-per-application zero-drift; $k_1 = 2$ for a 2-bit skeleton) and $p_2 = 11$ (split, matching Config B; $k_2 \approx 2.4$ for approximately 8 bits).

**Resulting format.** Total bit-width approximately $2 + 8 = 10$ bits per encoded coordinate.

| Format | Total bits | Inert (zero-drift) | Split (bounded-drift) |
|--|--|--|--|
| fp8 (Config B) | 8 | 0 | 8 |
| fp10 Sato–Tate (Config E) | 10 | 2 | 8 |
| fp16 (Config A) | 16 | 0 | 16 |

**Why Config E is informative.** Config B tests single-prime Frobenius. Config E tests *that* plus the inert/split partition. If B passes and E fails, the inert-channel zero-drift prediction is wrong despite Frobenius being correct — a sharper signal than B alone.

---

## 4. Implementation

### 4.1 Existing Hooks

The Shannon-Prime engine already contains:

- `--hier-ternary-mask` / `--hier-res-bits-v` flags for ternary skeleton compression.
- VHT2 + Möbius + spinor + square-free chain across all four backends.
- Calibrated fp8 quantization in the engine path.
- Per-architecture compression-default registry (model-pack scaffold); phi3 row exists and is calibrated.

What is needed:

1. **A `--frobenius-quant` flag** that toggles Frobenius reduction in place of calibrated fp8 (Config B).
2. **A `--sato-tate-mix p1,k1,p2,k2` flag** that activates the Sato–Tate asymmetric mixed-precision encoding (Config E).
3. **Unit tests** verifying:
   - $\varphi_{11}^8$ produces output bit-identical to direct reduction mod $11^8$ on the CM-encoded state (B).
   - $\varphi_2^2$ produces zero Frobenius drift exactly (E inert channel).
   - $\varphi_2^2 \circ \varphi_{11}^{k_2}$ matches $\varphi_{11}^{k_2} \circ \varphi_2^2$ to bit-exactness, verifying commutativity (E).
4. **A bench harness modification** to disable calibration for B/E and to run five-seed variance measurement on E.

Estimated implementation: 2–3 days at the engine level.

### 4.2 Outline of the Run Script

```
for cfg in [A_baseline_fp16, B_sp_fp8_calfree, C_gptq_fp8, D_awq_fp8, E_sp_fp10_satotate]:
    model = load_phi3(cfg.quantization)
    for corpus in [wikitext103, stackv2]:
        for ctx in [512, 2048, 8192]:
            ppl = eval_perplexity(model, corpus, ctx)
            tokens_per_sec = bench_throughput(model, corpus, ctx)
            log(cfg, corpus, ctx, ppl, tokens_per_sec)
```

### 4.3 Pass/Fail Criteria

| Comparison | Criterion |
|--|--|
| **B vs A** | $|\Delta\mathrm{PPL}| < 0.5\%$ on WikiText-103 at all contexts |
| **B vs C** | $|\Delta\mathrm{PPL}_{B,C}| < 0.3\%$ at all contexts |
| **B vs D** | $|\Delta\mathrm{PPL}_{B,D}| < 0.3\%$ at all contexts |
| **B throughput** | within 5% of A throughput |
| **Robustness** | B's relative PPL gap does not increase with context length |
| **E vs A** | $|\Delta\mathrm{PPL}_{E,A}| < 0.5\%$ at all contexts |
| **E vs B** | $|\Delta\mathrm{PPL}_{E,B}| < 0.2\%$ — fp10 mixed precision matches or beats fp8 single prime |
| **E zero-drift signature** | Variance of $\Delta\mathrm{PPL}$ across 5 random seeds is less than $0.5\times$ variance of B; verifies the inert-channel zero-drift property |

If all seven criteria pass, both Theorem 2 and Theorem 3 are validated. SP-fp8 ships as a calibration-free quantization; SP-fp10 ships as a calibration-free *variance-free* mixed precision.

---

## 5. Expected Outcome and Risk Analysis

### 5.1 Expected Outcome

Theorem 2 predicts B matches A within noise. Expected magnitude of $|\Delta\mathrm{PPL}_{B,A}|$: 0.1% to 0.5%. Theorem 3 predicts E matches B within 0.2%.

### 5.2 What a Negative Result Would Indicate

A B-failure localizes to: (1) imperfect CM realization of Phi-3 endomorphisms (debuggable by mechanistic interpretability); (2) Frobenius implementation bug (code review); (3) framework's curve choice doesn't match trained Phi-3 (requires revisiting). (3) is lowest probability, most consequential.

An E-failure (with B passing) localizes specifically to the inert-prime zero-drift prediction. This would falsify Lemma 3.1(a) for our model and would be a sharper signal than B-failure alone.

### 5.3 What a Positive Result Enables

A positive result delivers:
1. **Calibration-free fp8 deployment** — drop-in fp8 for Phi-3-class models with no calibration corpus.
2. **A bridge to fp4** via the same Theorem 2 mechanism.
3. **Calibration-free fp10 mixed precision** via Theorem 3 — a deployment format more accurate per bit than calibrated fp8.
4. **A target for ARM/Hexagon production** — calibration-free quantization removes a major deployment friction.

---

## 6. Implementation Timeline

| Phase | Deliverable | Estimated effort |
|--|--|--|
| 1. Engine hooks | `--frobenius-quant`, `--sato-tate-mix` flags, unit tests for $\varphi_{11}^8$ and $\varphi_2^2 \circ \varphi_{11}^{k_2}$ | 2–3 days |
| 2. Bench harness | Modified eval loop for the five configs, 5-seed variance measurement on E | 1–2 days |
| 3. Reference runs | All five configs at all six (corpus × context) combinations | 1–2 days on A100 |
| 4. Analysis + writeup | Result table, variance comparison for B and E | 2 days |
| 5. Companion v0.3 | Incorporate results into this paper as §5 (Results) | 1 day |

Total: 8–10 days of engine + bench work.

---

## 7. Connection to the Broader SP Framework

This experiment is one component of a larger program. The Shannon-Prime framework predicts five additional implementations, each with its own experimental signature:

- **Stern–Brocot RoPE** ([SP-Theory 2026, §9.1]): replace RoPE base 10,000 with $\varphi$; bench on long-context tasks.
- **Weil pairing attention** ([SP-Theory 2026, §9.2]): replace softmax-of-dot-product with the Weil pairing.
- **Hecke-eigenform embeddings** ([SP-Theory 2026, §9.3]).
- **L-function activation oracle** ([SP-Theory 2026, §9.4]).
- **LLL KV write** ([SP-Theory 2026, §9.5]).
- **Iwasawa-tower depth stability** ([SP-Theory 2026, §9.6]): $\mu = 0$ training eliminates RMSNorm.
- **BSD / analytic-rank training** ([SP-Theory 2026, §10]): catastrophic-forgetting-resistant training via Mordell–Weil generator search.

Each of these is the subject of a future SP companion paper. The present experiment — calibration-free fp8 and fp10 — is first in line because it is the smallest implementation step that probes the strongest theoretical predictions.

---

## 8. Conclusion

Shannon-Prime's Frobenius Quantization Theorem and Sato–Tate Mixed-Precision Theorem together predict that fp8 and fp10 quantization on a CM-encoded transformer state require no calibration data, with the fp10 format additionally exhibiting zero variance on its inert-prime channel. The present companion paper has specified the experimental design that tests both predictions on a Phi-3 reference model. A positive result validates the framework's strongest practical claims and opens a path to fp4, calibration-free mixed precision, and edge-device deployment without calibration. The experiment is small (approximately 1 week of engineering + benchmarking), the implementation hooks are already in place across the SP engine and llama.cpp fork, and the framework's predicted outcomes are sharp and falsifiable.

---

## References

- Companion theory: "Three Theorems for Shannon-Prime Compression: Hasse–Weil as the Shannon Limit, Frobenius as Quantization, and CM Sato–Tate Mixed Precision." 2026.
- Full theory: "Shannon-Prime: A CM-Elliptic-Curve Framework for Transformer Computation." 2026.
- Full systems: "Shannon-Prime: Provably Exact KV-Cache Compression and Per-Layer Acceleration on Standard Inference Stacks." 2026.
- Liu, Z. *et al.* "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." 2024.
- Frantar, E. *et al.* "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." 2023.
- Lin, J. *et al.* "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." 2024.
- Phi-3 Technical Report, Microsoft, 2024.
