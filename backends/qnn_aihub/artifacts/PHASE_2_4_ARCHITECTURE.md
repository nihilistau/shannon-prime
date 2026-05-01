# Phase 2.4 — Qwen3-4B deployment architecture (2026-05-01)

## TL;DR — Qualcomm has already split it for us

The official `qai_hub_models/models/qwen3_4b` package documents the
Qualcomm-blessed deployment shape for Qwen3-4B w4a16 on QNN. We don't
need to invent this; we follow it.

```
NUM_LAYERS              = 36       (not 28 — Qwen3-4B is deeper than I assumed)
HIDDEN_SIZE             = 2560     (not 2048)
NUM_SPLITS              = 4
NUM_LAYERS_PER_SPLIT    = 12       (split 4 takes the trailing 0)
DEFAULT_CONTEXT_LENGTHS = [512, 1024, 2048, 3072, 4096]
Min QNN SDK             = 2.45.0   (we have exactly this)
```

**The deployment is 8 context binaries, not 1:**
- 4 **Prompt Processor (PP)** splits — batch=128 tokens, prefill-shaped
- 4 **Token Generator (TG)** splits — batch=1 token, decode-shaped

Each split holds ~9-12 layers + their MLPs as one QNN context binary.
KV cache flows BETWEEN splits (each split's output K/V becomes the
next split's input). 4 FastRPC calls per generated token — comfortably
within the rate budget we measured (577 calls/sec) and orders of
magnitude better than the dead-on-arrival 28-call path.

## Realistic perf ceiling on V69 / S22 Ultra

Qualcomm's published numbers (perf.yaml, Genie runtime, w4a16,
n_ctx=4096):

```
Snapdragon 8 Elite Gen 5 QRD       29.2 t/s      (V81, latest)
Snapdragon X2 Elite CRD            30.3 t/s      (laptop)
Samsung Galaxy S25 (Snapdragon 8 Elite, V73)  24.4 t/s
Snapdragon X Elite                 17.5 t/s      (older)
Dragonwing IQ-9075 EVK             17.7 t/s      (V79)
```

**Our V69 (S22 Ultra, Snapdragon 8 Gen 1) is older than 8 Elite** —
extrapolating from the curve, vanilla Qwen3-4B w4a16 at n_ctx=4096
should hit ~12-17 t/s. With SP-banded KV compression layered on top
(reducing memory bandwidth from KV reads, the binding constraint at
this seq length), we project **20-25 t/s** as the realistic upper
bound for our Mode C deployment.

That's roughly half of our existing CPU-only Qwen2.5-Coder-3B + spec-
decode baseline (43.72 t/s, commit 05c405d). To beat the CPU baseline
with Qwen3-4B specifically, we need:
  - SP-banded KV compression on the bus (reduces bandwidth bottleneck)
  - Speculative decoding via 0.5B draft (also already validated)
  - Quantization-aware fine-tuning (Phase 2.3.2 — SP-bands → HTP int4
    direct rather than AIMET's per-tensor scaling)

## Path forward — Phase 2.4a stages

### Stage 2.4a-i: validate the qai-hub-models export pipeline runs on our setup

Without changing target device, run `qai-hub-models qwen3_4b.export`
and see what jobs land. The script auto-submits 8+ AI Hub compile
jobs targeting "Snapdragon 8 Elite QRD". Even though we can't run the
resulting V73-targeted .bins on our V69 device, this validates:
  - Our qai-hub-models install works
  - The export pipeline ends-to-end (PyTorch -> ONNX -> AI Hub -> .bin)
  - We see the 8 .bin output names + sizes
  - We learn what the calibration data flow looks like

### Stage 2.4a-ii: retarget to Samsung Galaxy S22 Ultra 5G

**Strategic clarification (2026-05-01)**: The `qai-hub-models` LLM export
declares `VALID_TARGET_RUNTIMES = Literal[TargetRuntime.GENIE]` — but
this is a Python type-hint only, NOT an API constraint. Inspection of
`models/common.py` confirms:

```
class TargetRuntime(Enum):
    TFLITE              = "tflite"
    QNN_DLC             = "qnn_dlc"
    QNN_CONTEXT_BINARY  = "qnn_context_binary"  ← what we want
    ...
    GENIE               = "genie"
```

`hub.submit_compile_job(target=TargetRuntime.QNN_CONTEXT_BINARY)` is
the unrestricted API path. The Genie wrapper is what produces the
Qualcomm-managed deployment artifacts (with their own runtime); the
underlying QNN context binaries are independent of that.

**The path: hijack the export pipeline at the final compile step.**

qai-hub-models gives us for free:
  - PyTorch checkpoint download (Qwen3-4B from HuggingFace)
  - Tokenizer + position processor
  - LLM-aware ONNX export (split into 4 PP + 4 TG ONNXes)
  - AIMET-based w4a16 quantization with calibration data
  - Per-split context-graph naming convention

We substitute at the final step:
  - Their default: `submit_compile_job(target=GENIE, device=8 Elite QRD)`
  - Ours:         `submit_compile_job(target=QNN_CONTEXT_BINARY, device=S22 Ultra 5G)`

Two implementation paths:
  (A) **Subclass / fork** `_shared/llm/export.py:export_main` —
      copy ~60 lines, change the target+device. Self-contained.
  (B) **Lower-level orchestration** — call the underlying ONNX-build
      and submit functions directly, skipping the wrapper entirely.
      More code but more control (e.g., we could use our own
      submit_w4a16.py / submit_qwen_shape.py harness).

Path (A) is faster to first .bin. Path (B) is cleaner for shannon-prime-
llama integration. Recommend (A) for stage 2.4a-ii, then refactor to
(B) as we land Phase 2.4c.

Risk: V69 may lack op coverage for Qwen3-4B shape. Mitigations:
  1. Try `Precision.w8a16` instead of w4a16 (broader op support)
  2. Try a smaller variant — `llama_v3_2_1b_instruct` (1B params,
     simpler arch) is on the qai-hub-models list and gives a faster
     iteration loop
  3. Compile against `qwen2_5_7b_instruct` (older arch, well-trodden
     V69 path)

### Stage 2.4a-iii: run the splits via sp_qnn shim

For each PP/TG split .bin we get:
  1. Push to /data/local/tmp/sp_qnn/
  2. Bench via test_sp_qnn (already burst-mode-tuned)
  3. Sum per-split latencies — that's the per-token latency floor

Per-token latency = sum(TG_split_i.exec_time) for i in 1..4. At ~1.5
ms/split (extrapolating from our small-block measurements scaled to
~9 layers), we'd see ~6 ms/token = 167 t/s ceiling — but real numbers
will be much lower due to KV-state passing overhead between splits.

### Stage 2.4a-iv: orchestrate via shannon-prime-llama

After per-split timing is known:
  - Add a "Qwen3-4B HTP path" mode to shannon-prime-llama
  - Sequence: tokenize → PP splits 1-4 (prefill batch=128) → TG splits
    1-4 per token → detokenize
  - SP-banded KV cache held between TG iterations (ours, not Qualcomm's)
  - Speculative decoding via 0.5B draft running on Adreno or Hexagon

This is the deployment target. After this lands, Phase 2.4d adds the
prefetch oracle.

## Risks specific to V69

- **Genie isn't a target** for our chip per perf.yaml (Snapdragon 8
  Elite onwards). But the underlying QNN context binaries shouldn't
  care — Genie is just a layer above QNN. We use sp_qnn shim instead.
- **w4a16 op support**: Qwen3 uses RMSNorm, RoPE, GQA. All standard
  in QNN HTP V69 per the SDK. Should be fine.
- **Memory**: 4B w4a16 weights = 2 GB; inflated for QNN context format
  ~ 3 GB. Plus KV cache. Plus runtime. Will need to mmap-load contexts
  rather than dlopen-into-RAM.

## File inventory

```
qai_hub_models/models/qwen3_4b/
  model.py              constants + PyTorch model (NUM_LAYERS=36, HIDDEN=2560)
  export.py             driver (uses _shared/llm/export.py:export_main)
  perf.yaml             Qualcomm-published reference numbers per chip
  info.yaml             Qwen3-4B metadata (genie_compatible=True, min QNN 2.45)
  quantize.py           AIMET-side quantization driver
  test.py               regression test (calibration data + accuracy check)
  demo.py               sample usage
```

The `_shared/llm/export.py:export_main` is the function we'd need to
study to understand the multi-split compile flow. Located at
`qai_hub_models/models/_shared/llm/export.py` in the install.

---

## Update 2026-05-02 — empirical findings invalidate the 4-split assumption

### What today validated
1. **libsp_qnn.so loads + executes** Phase 2.4a context binaries on V69 HTP. Single-split exec at 85 ms steady-state for the 12-layer middle splits, 142 ms for split 4 (12 layers + LM head). One-time graph finalize ~89 ms.
2. **The 4-split chain composes correctly** — residual stream and KV state pass cleanly between splits via `_model_model_layers_<L>_Add_1_output_0` naming. **Steady-state exec across all 4 splits = 391 tok/sec PP at 128-token chunking.** Matches the per-split projection.
3. **Backend+device shared across contexts** (refactored sp_qnn.c lines 44-67). Per-load time dropped from 1.5-2 sec to 600-1500 ms.

### What today proved is unworkable

**12-layer-per-split granularity is incompatible with V69 HTP working memory.**

Empirical residency table:

| State | Resident bytes (.bin sizes) | Result |
|---|---|---|
| Split 0 + Split 1 | 778 + 616 MB = 1394 MB | both load |
| Split 1 (released) → Split 2 | 778 + 616 MB | swap succeeds |
| Split 2 + Split 3 | 616 + 616 MB = 1232 MB | both held |
| Split 3 + Split 4 | 616 + 1007 MB = 1623 MB | **split 4 fails contextCreate** |

The ceiling is somewhere between 1.4 and 1.6 GB of resident context state on V69 HTP. Split 4 alone is 1007 MB (LM head + 12 layers) — leaves only ~600 MB headroom, insufficient for any pair-residency scheme involving it.

This kills both:
- Single-context unified .bin (would be ~3 GB total — wouldn't fit at all)
- 2-at-a-time pair residency (split 4 makes any pair containing it fail)

### The corrected architecture

The .bins as-shipped by Phase 2.4a are **wrong shape for HTP-resident execution on V69**. Per-token streaming requires:

#### 1. Re-export at finer granularity

Pass `NUM_LAYERS_PER_SPLIT=1` to `export_main` instead of 12. Result: **36 per-layer .bins** (~50-100 MB each), each fits comfortably in HTP. AI Hub re-export ~1 hour wall.

The LM head needs to be its own .bin too (since today's split 4 = 12 layers + LM head was the failure point — those need to separate so the LM head .bin alone fits).

#### 2. The Orchestrator (Mode C at split granularity)

```
[UFS]  ─── 36 per-layer .bins (50-100 MB) + 1 LM head + 1 embed ────┐
                                                                      │
       prefetch via DMA (existing rpcmem path proven on S22U)         │
       ┌────────────────────────────────────────────────────────┐     │
       │                                                        │     ▼
       ▼                                                        │  [HTP-resident]
  [DDR ring buffer]                                             │  layer N (active)
  4 layers' worth ──────────────────────────────► load N+1 ────►│  + layer N+1 (warming)
  prefetched ahead                                              │       │
                                                                │       │ exec layer N
                                                                │       ▼
                                                                │  [residual stream]
                                                                │       │
                                                                │       ▼ feed to N+1
                                                                │  exec layer N+1
                                                                │       │
                                                                │       ▼
                            [destroy(N), contextCreate(N+2)] ◄──┘  while DMA prefetches N+3
```

Per-layer cost projection:
- Single-layer .bin: ~50 MB, expected exec **~7-8 ms** (12 layers in one .bin = 85 ms / 12 ≈ 7 ms/layer)
- Per-token full pass: 36 × 7 = ~250 ms = **~510 tok/sec PP at 128-token chunking**
- Per-layer load: ~50-100 ms (proportional to .bin size — we measured 600 ms / 600 MB ≈ 100 ms / 100 MB)
- **Swap is FREE if exec time ≥ load time**, i.e., always with prefetch overlap

#### 3. Where Shannon-Prime layers in (Mode C front-end)

The KV cache and residual stream stay SP-compressed in DDR between layer calls:
- Output of layer N (residual + new KV bands) → SP encode (HVX or ISP) → DDR ring buffer
- Input to layer N+1 → fetch SP-encoded → SP decode → format for HTP fp16/int8 input
- Skeleton bands (B0+B1) live HTP-adjacent in shared memory; residual bands (B2+B3) stream from UFS via the DMA path

This is where "shit living in DDR vs disk vs ISP" gets controlled. The orchestrator is what decides what goes where:
- **HTP-resident**: current layer .bin + next layer .bin (warming)
- **DDR-resident**: skeleton bands for all layers (~50-100 MB total, hot-cached), residual bands for window of next 4 layers (prefetched), KV cache in SP-banded form (~10-20 MB working set)
- **UFS-resident**: residual bands for all 36 layers (~500 MB), per-layer .bins (~3 GB), full-precision skeleton archive (failsafe)

#### 4. Where Mode D layers in (Spectra ISP for spectral reconstruction)

The MFNR engine in Spectra 680 ISP does 18-bit fixed-point weighted accumulation across "frames" — exactly the operation needed to fuse skeleton + residual bands back into reconstructed weights. Run in parallel with HTP exec:

| Block | Role | Concurrency |
|---|---|---|
| **ISP** | Reconstruct layer N+1's weights from B0+B1+B2+B3 bands at 18-bit | parallel with HTP |
| **HTP** | Execute layer N (matmuls + attention + FFN) | the main compute |
| **HVX** | Format layer N+2's input + SP-decode N+2's KV history | hides behind ISP |
| **DMA** | Stream layer N+3's residual bands from UFS into DDR | hides behind HTP |

Three silicon blocks running concurrently, plus DMA. **This is the apex.**

### Concrete next-session work, in order

1. **Re-export Qwen3-4B at NUM_LAYERS_PER_SPLIT=1 + LM head split off.** Modify the qwen3_4b_v69_export.py wrapper to override these constants; submit; wait; download 36 per-layer .bins.
2. **Validate single-layer .bin execution** — test_sp_qnn_multi against one layer .bin, confirm 7-8 ms target.
3. **Validate 2-layer residency** — load layer 0 + 1, swap to 1+2, measure swap cost. With ~50 MB .bins, swap should be <50 ms = trivial vs 8 ms exec.
4. **Build the prefetch orchestrator** — async load of layer N+2 during layer N exec via threading or the QNN async load API.
5. **Wire SP encode/decode** between layer outputs/inputs (the Mode C front-end). Initial version on HVX/CPU, Mode D ISP path is follow-on.

### What today's wins enable

The .bins we have (12-layer granularity) don't get us to production for Qwen3-4B on V69. But they **proved the QNN context binary execution path works at 391 t/s exec rate**, which is the speed-of-light number we'd see in the per-layer pipeline if exec dominates load (which it will once we have 50 MB .bins instead of 600 MB).

The libsp_qnn.so shim, the chain test harness, the residency findings, the DCVS burst-mode setup — all reusable. The model export script is reusable (just different NUM_LAYERS_PER_SPLIT). The integration plan is now concrete, not hand-wavy.
