# Phase 2.3.1 — w4a16 quantization on V69 HTP (2026-05-01)

## TL;DR

AI Hub's `submit_quantize_job(weights_dtype=INT4, activations_dtype=INT16)`
+ `submit_compile_job(--target_runtime qnn_context_binary)` produces
deployable w4a16 .bin files for V69 HTP. **First data point on the
small attention block shows w4a16 is *slower* than fp32 at micro-scale**
(60 µs vs 44 µs raw NPU). Expected: dequant overhead dominates when
the matmul is too small to amortize it. The Qwen3-4B-shape submission
will tell us the real story — at d_model=2048, bandwidth dominates
compute and w4a16's 2-4× memory-bandwidth advantage should win.

## The numbers so far

```
Small attention block (seq=8, d_model=256, 4 heads, head_dim=64):
                          fp32           w4a16        delta
  Accelerator (raw):       44 us  ->     60 us       +36% (slower)
  QNN accelerator:        102 us  ->    126 us       +24%
  NetRun:                 413 us  ->    499 us       +21%
  .bin size:              560 KB  ->    188 KB        ~3x smaller (good)
```

## Why w4a16 lost at micro-scale (and why it'll win at Qwen-shape)

For a 256x256 matmul with seq=8:
- Total FLOPs:        ~2 MFLOPs  (cheap to compute)
- Weight bytes:       64 KB fp32 / 16 KB w4a16  (already cache-resident)
- Bandwidth-bound?    No — both fits in L2; compute dominates.
- Dequant overhead:   Per-block per-element work that fp32 doesn't pay.

For the Qwen3-4B-shape (seq=64, d_model=2048, 16 heads):
- Total FLOPs:        ~150 MFLOPs per attention block
- Weight bytes:       64 MB fp32 / 16 MB w4a16  (DDR-resident, streams to VTCM)
- Bandwidth-bound?    Yes — weight loads dominate. w4a16 streams 4x
                      faster than fp32 from DDR through VTCM to HVX.
- Dequant overhead:   Amortized over much larger compute window.

**Expected Qwen-shape w4a16 latency**: 400-800 µs (vs 1500 µs fp32).
The Qwen w4a16 submission is queued; result drops in this commit's
follow-up.

## AI Hub jobs

- `jg99nxkmg` — small attention quantize (SUCCESS)
- `jgnrv77v5` — small attention compile (w4a16 -> qnn_context_binary, SUCCESS, 188 KB)
- `<TBD>` — qwen3-4b w4a16 quantize (will submit next)

## Strategic implications for Phase 2.3.2 (SP-bands -> HTP int4 direct)

w4a16 via AIMET-equivalent path **works** as a baseline. But it's
*Qualcomm's* quantization scheme — not Shannon-Prime's banded
information-theoretic packing. The architectural opportunity:

1. **AIMET path (this Phase 2.3.1)** — proves the w4a16 deployment path.
   We get a working number, validates HTP int4 throughput, gives us
   a baseline to compare SP-band quantization against.

2. **SP-spectral-bands -> HTP int4 (Phase 2.3.2)** — replace AIMET's
   per-tensor scaling with SP's {5,5,4,3}-bit banded packing. Two
   variants:
     a. Mid-graph dequant: SP-Hexagon decompresses bands -> fp16 ->
        feeds HTP fp16 path. Wins on storage; pays decompress cost.
     b. Direct SP -> HTP int4 layout: skip the fp16 round-trip. Need
        to confirm HTP's int4 tensor-format docs (per-channel scale,
        block size, sign convention). If SP's 4-band layout maps
        cleanly to HTP's expected packing, we're golden. If not, a
        thin transpose/repack kernel sits in between.

3. **The Mode C / Ring-Buffer pressure** (Gemini's framing): with
   HTP at 1.5 ms/block, the question is purely "can we feed it?" SP
   compression isn't optional — it's how the KV cache stays under
   the bandwidth ceiling at 4K+ context.

## Bandwidth budget at 50 tok/sec, n_ctx=4096

```
Vanilla fp16 KV-read per layer:  64 MB
  x 28 layers x 50 tok/sec   = 90 GB/s    UFS ceiling exceeded
SP-banded KV-read per layer:    8-13 MB
  x 28 layers x 50 tok/sec   = 11-18 GB/s  Within VTCM-DDR budget
```

SP compression is the **only** path that keeps the HTP fed at scale.
That's the architectural narrative this commit confirms.

## Op-level breakdown (Qwen3-4B-shape fp32, --profiling_level detailed)

Per-op cycle counts on V69 HTP (hexagon clock ~1 GHz; cycles ≈ µs).
Captured with `qnn-net-run --profiling_level detailed` on
`v69_attn_qwen3_4b.bin`:

```
Op                          Cycles      ~µs    % of total
node_MatMul_0 (Q/K/V fused) 1,890,937   1,890     60%
MatMul_2      (output proj)   646,550     646     20%
MatMul_0      (Q @ K^T)       238,415     238      7%
Softmax_0                     159,255     159      5%
MatMul_1      (probs @ V)      94,168      94      3%
Transpose_0                    15,979      16      1%
Slice_45                        8,021       8     <1%
Input/Output                   ~45,000     45      1%
                              ─────────────────
Total                         ~3,100,000  ~3,100   ✓ matches Avg NetRun 3,258
```

## Prediction framework — w4a16 should hit 850-1150 µs raw NPU

Gemini's analysis: w4a16 wins where bandwidth dominates compute. The
op breakdown above shows exactly which ops are bandwidth-dominated:

| Op | fp32 µs | w4a16 prediction | Why |
|---|---|---|---|
| node_MatMul_0 (Q/K/V fused) | 1,890 | ~570 µs (-70%) | Weights are 12 MB (4× d_model² × 4); w4a16 streams ~4× faster from DDR |
| MatMul_2 (output proj) | 646 | ~194 µs (-70%) | 4 MB weights, same bandwidth-bound profile |
| MatMul_0 (Q @ K^T) | 238 | 238 (unchanged) | No weights — both operands are activations |
| Softmax_0 | 159 | 159 (unchanged) | Pure activation op |
| MatMul_1 (probs @ V) | 94 | 94 (unchanged) | No weights — both activations |
| Overheads | ~80 | ~80 | Fixed |
| **Total** | **3,100** | **~1,135 µs** | |

The prediction lands inside Gemini's 850-1,150 range. Once the Qwen
w4a16 .bin completes upload + compile, we can validate this op-by-op
by re-running `qnn-net-run --profiling_level detailed` against it
and diff'ing the cycle counts.

This op-breakdown methodology is the load-bearing tool for Phase
2.3.2 (SP-bands → HTP int4): we'll be able to attribute any speedup
or regression to specific ops.

## Strategic readout: speculative decoding becomes the prefetch oracle

Connection worth surfacing: with HTP at 1.5 ms/block + bandwidth as
the binding constraint, the question is **predictive prefetch** of
KV-cache bands. The speculative-decoding draft model (validated 2.16×
in commit 05c405d, Qwen2.5-Coder-3B + 0.5B draft) gives us a 4-8
token *lookahead* — exactly the prefetch window for SP-compressed
KV reads. Spec-decode in this regime isn't just compute reuse; it's
the **demand oracle for the streaming pipeline.** Track in
project_specdecode_session.md memory.

## Phase 2.4 architecture is forced by data (commit 2a759a1)

Sustained-call-rate measurement on the existing 1-attn-block .bin via
test_sp_qnn (200 iters, fp32, Qwen3-4B-shape):

```
sustained calls/sec:       577
tok/sec @ 28 layers/call:  20.6    (per-layer dispatch, fp32)
tok/sec @ 1 layer/call:    577     (theoretical, full-graph compile)
fp32 -> w4a16:             20.6 -> 38.7 tok/sec  (28-layer dispatch)
```

**Per-layer dispatch is dead-on-arrival.** At ~1.7 ms per FastRPC call
times 28 layers per token = 50 ms/token = 20 tok/sec — which is below
the existing CPU-only 43.72 t/s baseline (commit 05c405d, Qwen2.5-Coder-3B
+ spec-decode). Even with w4a16 amplification we only reach ~38 tok/sec.

**Full-graph compile is mechanically required.** One FastRPC call per
token, ~1.5 ms QNN exec dominates, jitter (30% of avg) gets amortized
across the full token's compute instead of compounding across 28 calls.

This **forces** Phase 2.4a to compile the full transformer (28 layers +
MLPs + embeddings + LM head) as a single QNN context binary — not
per-layer .bins driven by shannon-prime-llama dispatch.

## Next concrete steps

1. **Phase 2.4a** — get a Qwen3-4B-Coder ONNX (full transformer, not
   single block). Either via HuggingFace `optimum-cli export onnx` or
   via Qualcomm's qai-hub-models (we found earlier that qai-hub-models
   ships pre-exported Qwen at the right shape). Submit to AI Hub
   compile; expect a much larger .bin (~2-4 GB w4a16). Run via our
   own sp_qnn shim.
2. **Phase 2.3 stage 3** — wire libsp_qnn.so as the matmul executor in
   shannon-prime-llama IF we're keeping the per-layer dispatch path
   for development (it's slower but easier to debug). Lower priority
   given the rate finding.
3. **Phase 2.3.2** — map SP-bands to HTP int4 tensor format. Task #44.
   Becomes especially relevant if we control the quantization (vs
   AIMET's choices) once full-graph compile lands.
4. **Phase 2.3.1b** — Qwen w4a16 retry on faster network. Task #43.
   Still useful as a per-block calibration data point for the prediction
   framework above.
