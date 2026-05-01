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
