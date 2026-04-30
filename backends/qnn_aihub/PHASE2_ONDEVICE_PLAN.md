# Phase 2 — Run our AI-Hub-validated `.serialized.bin` directly on the S22U

## Why this is now the lead path

Mode D Stage 1 (commit `9e6ce90`) confirmed Halide DMA is unreachable in
unsigned PD on this production-locked S22U — `halide_hexagon_dma_allocate_engine`
hits a permissions wall that won't lift without a Qualcomm-issued testsig
or a dev kit. Meanwhile, the QNN HTP path is **already validated end-to-end
via Qualcomm AI Hub** at 81 µs/inference for a 256×256 fp32 MatMul on V69
(jobs `j5wm6m34g` compile + `jp4jrjd2p` profile, commit `e5b3963`).

What's been blocking Phase 2 isn't permissions — it's that we haven't yet
run a QNN context binary *locally on this exact device* (we've only
profiled through AI Hub's lab fleet). Time to do that.

## Toolchain inventory (everything is already on disk)

```
QAIRT 2.45.40.260406:    C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\
   bin\aarch64-android\qnn-net-run                # the on-device runner
   lib\aarch64-android\libQnnHtp.so               # host-side HTP backend
   lib\aarch64-android\libQnnHtpV69Stub.so        # host stub for V69
   lib\aarch64-android\libQnnSystem.so            # context-binary loader
   lib\hexagon-v69\libQnnHtpV69Skel.so            # cDSP-side V69 skel
   lib\hexagon-v69\libQnnHtpV69.so                # cDSP-side V69 runtime
   examples\Genie\genie-t2t-run\                  # LLM text-to-text runner
   examples\Genie\configs\llama3-3b\              # closest config to Qwen3-4B
   examples\QNN\SampleApp\                        # C++ integration template
```

AI Hub artifact (need to re-download — sandbox /tmp where it landed is
gone): `compile_job j5wm6m34g` → 172 KB QNN context binary. The
`backends/qnn_aihub/v69_workflow.py` script supports `--check <job_id>`
to re-pull from an existing job.

S22U device: connected (R5CT22445JA), known-good FastRPC stack
(`/data/local/tmp/sp22u/S22U` runs cleanly).

## Concrete on-device run (three terminal commands)

```powershell
# 1. Pull the validated .bin from AI Hub
cd D:\F\shannon-prime-repos\shannon-prime\backends\qnn_aihub
python v69_workflow.py --check j5wm6m34g
# Produces: <somewhere>\v69_sp_v69_baseline.bin (172 KB)

# 2. Stage the device-side bundle
$dev = '/data/local/tmp/sp_qnn'
$qairt = 'C:\Qualcomm\AIStack\QAIRT\2.45.40.260406'
adb shell "mkdir -p $dev"
adb push $qairt\bin\aarch64-android\qnn-net-run               $dev/
adb push $qairt\lib\aarch64-android\libQnnHtp.so              $dev/
adb push $qairt\lib\aarch64-android\libQnnHtpV69Stub.so       $dev/
adb push $qairt\lib\aarch64-android\libQnnSystem.so           $dev/
adb push $qairt\lib\hexagon-v69\libQnnHtpV69Skel.so           $dev/
adb push $qairt\lib\hexagon-v69\libQnnHtpV69.so               $dev/
adb push <bin_path>\v69_sp_v69_baseline.bin                   $dev/
# A 1×256 fp32 input tensor for the MatMul (host-generated):
python -c "import numpy as np; np.zeros((1,256), dtype=np.float32).tofile('input.raw')"
adb push input.raw                                             $dev/
echo "input.raw" > input_list.txt
adb push input_list.txt                                        $dev/

# 3. Run on the cDSP via qnn-net-run, 1000 inferences with profiling
adb shell "cd $dev; chmod 755 qnn-net-run; \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context v69_sp_v69_baseline.bin \
    --input_list input_list.txt \
    --num_inferences 1000 \
    --profiling_level basic \
    --use_mmap \
    --output_dir out"
```

Expected outcome: ~81 µs/inference matching the AI Hub lab number,
confirming the V69 HTP execution path works on *our* device with *our*
toolchain — no AI Hub round-trip required.

## Parallel track: freethedsp shim baked in

Concurrently with Phase 2, `backends/freethedsp/` adapts geohot's
`freethedsp` (https://github.com/geohot/freethedsp) for our S22U so
that ANY signed-PD-gated cDSP API "just works" without per-call-site
plumbing. Phasing:

  - **D.1** discover.so dumps `fastrpc_shell_3` from `init->mem` during
    a real FastRPC session. Lets us derive the device-specific
    PATCH_ADDR offline.
  - **D.2** `tools/find_is_test_enabled.py` runs hexagon-llvm-objdump
    on the dump, locates `is_test_enabled`, emits the constants to
    bake into `freethedsp_s22u.c`.
  - **D.3** Build libfreethedsp.so, smoke-test against the existing
    Mode D Stage 1 probe (commit 9e6ce90). Expected outcome:
    `halide_hexagon_dma_allocate_engine` returns success instead of
    qurt_exit.
  - **D.4** Wire LD_PRELOAD into `build-example.ps1 -Run` and
    `backends/hexagon/scaffold/build.cmd`. Default-loaded, opt-in-active
    via `SP_FREETHEDSP=1` env var.

This Phase-2-and-D-in-parallel posture means: even if we discover that
QNN-on-device hits an unexpected permission wall, freethedsp is already
in place to bypass it. Conversely, if QNN runs cleanly without the
shim (the most likely outcome), we still have freethedsp ready for
Mode D Stage 1+ rerun and any future signed-PD experiment.

## What this unblocks

1. **Phase 2.1**: pin the V69 HTP latency baseline locally so subsequent
   transformer-block compiles can be benchmarked end-to-end without
   AI Hub. ~30 min.
2. **Phase 2.2**: compile a transformer attention block (not a matmul)
   via AI Hub for Qwen3-4B-equivalent shape. Re-run `v69_workflow.py`
   with a more realistic ONNX. ~1 hour.
3. **Phase 2.3**: integrate QNN execute() as the matmul kernel in
   shannon-prime-llama. Use `examples\QNN\SampleApp\` as the C++
   integration template. SP decompresses K-rows in the FastRPC bridge
   that already ships, then hands them to QNN's HTP-resident graph.
   ~1-2 days.
4. **Phase 2.4**: end-to-end Qwen3-4B-Coder run on V69 HTP with
   SP-compressed KV cache. The architectural goal of the V69 pivot.

## Risks/unknowns to surface in Phase 2.1 itself

- The .bin from `compile_job j5wm6m34g` was compiled targeting V69 HTP
  with `--target_runtime qnn_context_binary` — should be device-portable,
  but worth verifying that it loads on our local cDSP without AI Hub's
  signing wrap.
- `qnn-net-run` may require `libxdsprpc.so` or other infra libs not in
  our push list. Standard fallback: copy missing libs from /vendor/lib64
  if shell-readable. Run with `--log_level=debug` if first attempt fails.
- We may hit a similar "unsigned context binary rejected" issue. If so,
  same `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` trick
  applies — `qnn-net-run` should call this internally but if not, we
  patch our own runner.

## When Phase 2.1 succeeds → next-session entry point

Move on to Phase 2.2 (transformer attention block compile). The
v69_workflow.py becomes a parametric ONNX builder; only the
`build_tiny_matmul_onnx()` function needs to grow into a
`build_attention_block_onnx(seq_len, d_model, n_heads)`.

After 2.2 numbers: decide whether to go straight into 2.3 integration,
or first do a w4a16 quantization pass through AI Hub's QAIRT toolchain
to get the realistic deployment perf number.
