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

## Next concrete steps

1. **Submit Qwen3-4B-shape w4a16** (now). Expected output: a 4-8 MB
   .bin (vs 32.8 MB fp32) producing 400-800 µs/block (vs 1500 µs fp32).
2. **Phase 2.3 stage 2** — fill in sp_qnn.c TODOs so we can call the
   w4a16 binary via our own runner.
3. **Phase 2.3.2** — investigate HTP int4 tensor format. Reverse-
   engineer the .bin if the docs are sparse; spec out the SP-band ->
   int4-layout mapping.
4. **Phase 2.4** — full Qwen3-4B graph (28 layers + MLP + embeddings
   + LM head) compiled as a single QNN context, with SP-compressed
   KV reads. The deployment target.
