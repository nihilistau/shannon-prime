# Phase 2.1 — V69 HTP latency reproduced locally on S22 Ultra (2026-05-01)

## TL;DR

**The QNN HTP execution path on V69 is reachable end-to-end on our actual
S22 Ultra (R5CT22445JA), no AI Hub round-trip required, and the steady-
state minimum latency of 81 µs/inference matches the AI Hub lab device's
measurement to the microsecond.**

This validates the architectural pivot to QNN-on-device as the lead path
for Phase 2 (replacing Halide DMA streaming, which Mode D Stage 1 proved
gated behind signed-PD on this device). The runtime is Qualcomm's own,
deploys with their official binaries, and works in unsigned PD by design.

## Numbers (basic profiling, 10 inferences over a 256×256 fp32 MatMul)

```
Execute Stats (Min — steady-state best case):
    Accelerator (execute excluding wait):  17 µs     ← raw NPU compute on V69 HTP
    Accelerator (execute):                 47 µs     ← NPU + dispatch wait
    QNN accelerator (execute):             81 µs     ← matches AI Hub job jp4jrjd2p exactly
    NetRun (end-to-end):                  605 µs     ← incl FastRPC + I/O

Execute Stats (Average):
    Accelerator (execute excluding wait):  21 µs
    QNN accelerator (execute):            180 µs
    NetRun:                               804 µs

Init/Load:
    QNN (load binary):                12,625 µs (one-time, mmap'd)
    Accelerator (load binary):         3,707 µs

HVX threads used: 4 (typical for V69 HTP)
```

The min-vs-avg gap (81 vs 180) is dominated by FastRPC dispatch jitter
and warmup; raw NPU compute is rock-steady at 17-21 µs.

## Run command (reproducible)

```
adb shell "cd /data/local/tmp/sp_qnn; \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context v69_sp_v69_baseline.bin \
    --input_list input_list.txt \
    --num_inferences 10 \
    --profiling_level basic \
    --use_mmap \
    --output_dir out"
```

## Bundle staged at /data/local/tmp/sp_qnn/

```
qnn-net-run                                    5.0 MB    on-device runner
libQnnHtp.so                                   2.9 MB    host backend
libQnnHtpV69Stub.so                          720.4 KB    host stub for V69
libQnnSystem.so                                3.6 MB    QNN system runtime
libQnnHtpPrepare.so                           87.7 MB    graph-prepare runtime
libQnnHtpV69Skel.so (from hexagon-v69/unsigned/) 10.7 MB    cDSP-side V69 skel (vendor-stripped for unsigned PD)
v69_sp_v69_baseline.bin                      176.1 KB    our QNN context binary (compile job j5wm6m34g)
input.raw                                      1.0 KB    1×256 fp32 input
input_list.txt                                 12 B      tensor binding
```

`hexagon-v69/unsigned/` is the key insight: Qualcomm pre-strips the
signature off the V69 skel for unsigned-PD use. Our process loads it
directly without any of the signature-bypass tricks Mode D Stage 1
needed.

## What this unblocks

- **Phase 2.2** — compile a transformer attention block (Qwen3-4B-shape:
  d_model=2048, n_heads=16, head_dim=128, seq_len=512 or so) via AI
  Hub. Use the existing `v69_workflow.py` as the harness; replace
  `build_tiny_matmul_onnx()` with `build_attention_block_onnx()`. Push
  the resulting .bin and re-run the same `qnn-net-run` invocation. See
  what fp32 attention runs at — this is the realistic deployment-perf
  number we've been wanting.
- **Phase 2.3** — integrate QNN execute() as the matmul kernel inside
  shannon-prime-llama. Pattern: SP-Hexagon FastRPC bridge already
  produces decompressed K rows; hand them to QNN's HTP-resident graph
  via a shared rpcmem buffer. Use `examples\QNN\SampleApp\` as the
  C++ integration template (already on disk at QAIRT install).
- **Phase 2.4** — the real prize: end-to-end Qwen3-4B-Coder running on
  V69 HTP with SP-compressed KV cache. This is the architectural goal
  of the V69 pivot project (project_v69_tensor_accelerator_pivot.md).

## What this does NOT touch

- Mode D streaming via Halide DMA — still gated by signed PD; that's a
  separate experiment that needs hardware (dev kit or testsig).
- Mode D ISP fusion — Stage 1 prereq, ditto.
- The shannon-prime-hexagon FastRPC + manual L2 + HVX path — keeps
  shipping; that's the SP-side compute that feeds QNN matmul.

## Reproduction prerequisites

- QAIRT 2.45.40.260406 installed at default location
- adb-connected S22 Ultra (or any V69 device)
- Our compile job artifact (`v69_sp_v69_baseline.bin`) — re-pull via
  `python v69_workflow.py --check j5wm6m34g` if SP_AIHUB_OUT_DIR set
  to a Windows path
- ~110 MB device-side disk for the lib bundle (libQnnHtpPrepare.so
  alone is 87 MB — Qualcomm includes the full graph-prepare runtime
  even for retrieve-context-only flows; can be omitted for inference-
  only deploys, will trim later)

## Anomalies / things-to-investigate-later

- Max NetRun of 1786 µs vs Min 605 µs — 3× spread on 10 inferences.
  Some FastRPC dispatch jitter, some Android task-scheduler effects.
  Will look cleaner on a larger sample (1000+ inferences) and once we
  pin to performance cores via `taskset 0xf0`.
- `libQnnHtpPrepare.so` (87 MB) feels oversized for retrieve-only.
  Likely contains the full ONNX→QNN graph compiler. For Phase 2.4
  deployment we should investigate whether it's needed at runtime or
  only for compile-time graph preparation.
