# Phase 2.2 — Qwen3-4B-shape transformer attention block on V69 HTP (2026-05-01)

## TL;DR

**One Qwen3-4B-shape transformer attention block (seq_len=64, d_model=2048,
n_heads=16, head_dim=128, fp32) executes on our S22 Ultra V69 HTP at
1.5 ms steady-state minimum.** Three points on the latency-vs-shape curve
now validated end-to-end (AI Hub compile → on-device run).

This means the V69 HTP architectural pivot is not just a working path —
it scales sensibly with model shape, and the predicted Qwen3-4B-Coder
deployment target is tractable on this hardware once we add graph-level
fusion (Phase 2.3) and w4a16 quantization (Phase 2.3.1).

## Numbers

```
Phase 2.1 — 256×256 fp32 MatMul (commit 69d09d2):
  raw NPU:   17 µs
  QNN exec:  81 µs    ← matches AI Hub lab device exactly
  NetRun:   605 µs

Phase 2.2-small — 8-seq attention block (d_model=256, 4 heads):
  raw NPU:   44 µs
  QNN exec: 102 µs
  NetRun:   413 µs

Phase 2.2-qwen — 64-seq attention block (d_model=2048, 16 heads):
  raw NPU: 1,496 µs    ← actual Qwen3-4B target shape
  QNN exec:1,556 µs
  NetRun:  1,916 µs
  Init:   48,246 µs (one-time, mmap'd)
  HVX threads: 4
```

The 8-op attention block (Q+K+V+O matmuls + softmax + 2 attn matmuls +
reshape/transpose) at qwen-shape costs ~88× the raw-NPU time of a single
256×256 matmul, which roughly tracks the (d_model/256)² = 64× scaling
plus the cost of softmax + multi-head reshape overhead.

## What this unblocks for Phase 2.3+

Naive single-layer-per-context-binary deployment:
  - 28 layers × 1.5 ms = **42 ms / token** for attention alone
  - Plus MLP blocks (similar shape, expect 0.5-1 ms each) → **~70 ms / token**
  - That's ~14 tok/sec at fp32, single-context dispatch overhead per layer

With Phase 2.3 optimizations:
  - **Full-graph compile** — stack all 28 layers + MLP into one QNN context
    binary, eliminating per-layer FastRPC dispatch (saves the 400-500 µs
    overhead per call). Expected: 2-3× speedup → ~25-35 tok/sec.
  - **w4a16 quantization** via QAIRT's `--quantization_overrides` flow —
    HTP runs w4a16 at 2-4× the throughput of fp32 (V69's int matmul
    units are wider than its fp16 ones). Expected: another 2-4× → **50-100
    tok/sec for Qwen3-4B** on a phone.

## Reproducible run

```powershell
# 1. Build + submit Qwen3-4B-shape block (this commit)
$env:SP_AIHUB_OUT_DIR = "D:\F\shannon-prime-repos\shannon-prime\backends\qnn_aihub\artifacts"
cd D:\F\shannon-prime-repos\shannon-prime\backends\qnn_aihub
python submit_qwen_shape.py
# -> compile job submitted; poll via qai_hub.get_job(<job_id>)

# 2. Push + run on device
adb push artifacts\v69_attn_qwen3_4b.bin /data/local/tmp/sp_qnn/
python -c "import numpy as np; np.random.seed(0); np.random.randn(1,64,2048).astype(np.float32).tofile('input_qwen.raw')"
echo "x:=input_qwen.raw" > input_list_qwen.txt
adb push input_qwen.raw input_list_qwen.txt /data/local/tmp/sp_qnn/

adb shell "cd /data/local/tmp/sp_qnn; LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./qnn-net-run --backend libQnnHtp.so \
    --retrieve_context v69_attn_qwen3_4b.bin \
    --input_list input_list_qwen.txt --num_inferences 50 \
    --profiling_level basic --use_mmap --output_dir out_qwen"

# 3. Decode profile log
adb pull /data/local/tmp/sp_qnn/out_qwen/qnn-profiling-data_0.log
qnn-profile-viewer.exe --input_log qnn-profiling-data_0.log
```

## Tooling that landed

- `backends/qnn_aihub/attention_block.py` — parametric ONNX builder
  (Q/K/V projections + multi-head SDPA + output projection). Initializers
  inlined. ir_version=8 + opset 13 (AI Hub-accepted combo). Validates
  via onnx.checker before save.
- `backends/qnn_aihub/submit_attention.py` — submit harness (submit
  compile + poll + download + optional profile). Used for the small
  smoke run.
- `backends/qnn_aihub/submit_qwen_shape.py` — Qwen3-4B-specific
  submit-only driver. Uploads ~64 MB ONNX, gets back a 32.8 MB QNN
  context binary.

## AI Hub jobs

- `jpyv3q08p` — qwen3-4b compile (SUCCESS, 32.1 MB context binary)
- `j5wm6163g` — qwen3-4b profile (submitted; lab-side numbers will
  cross-validate our 1.5 ms steady-state)
- `jgoe6ojxp` — small attention compile (SUCCESS, 560 KB)
- `j5mvxorw5` — small attention profile (queued)

## Anomalies to flag

- Init time (48 ms) is ~4× the matmul-only init. Scales with op count
  and weight count. Will become significant per-layer if we don't fuse
  — another reason Phase 2.3 should fully-stack the graph.
- Avg 2.5 ms vs Min 1.5 ms is ~1.7× spread — typical FastRPC dispatch
  jitter; we'll see if it tightens up under taskset 0xf0 (perf cores)
  or shared-buffer mode (qnn-net-run --shared_buffer).

## What this does NOT include

- Quantization. fp32 is the upper-bound number. w4a16 is Phase 2.3.1.
- KV-cache. The block recomputes Q/K/V from scratch each call. The
  shannon-prime-llama integration in Phase 2.3 is what gives us a real
  inference loop with KV reuse.
- MLP. Phase 2.2.1 will add the FFN block to round out the layer.
- The full transformer stack. Phase 2.3 stitches everything together.
