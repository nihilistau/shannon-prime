# Mode D Stage 1 build notes — what I tried, what's blocked

## TL;DR

Scaffold is committed and copied into the SDK example tree at
`C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\Examples\standalone\device\apps\sp_dma_raw\`.
A parallel CMake-based skeleton sits at
`C:\Qualcomm\Hexagon_IDE\sp_dma_raw\` mirroring the validated S22U
working setup. **Neither has been built yet** because two distinct
toolchain issues block the easy paths.

## What I tried

### Attempt 1 — The SDK's Linux-flavored Makefile via `gow` (Gnu on Windows)

The `dma_raw_blur_rw_async` example I copied from is meant to build
via `make` from the SDK's `tools/utils/gow-0.8.0/bin` shell tools.
After `setup_sdk_env.cmd`, `make` and `bash` are on PATH and the
include chain (Makefile.common → halide_config.make → Makefile.rules)
loads cleanly.

**Blocker**: `make` errors with
`No rule to make target '...bin/print.o'` even though the literal
rule `$(BIN)/print.o: dsp/print.cpp` exists in `Makefile.rules` and
is visible in the make-rule database (`make -p`). Cause is mixed
`/` and `\` in the expanded paths — gow's `make` treats them as
different strings, and the rule DB has backslashes for the part
before `/../../` (because `CURR_DIR=$(shell pwd)` returns
`C:\…` form on cmd) but the target make ends up searching for has
forward slashes (because they came from a `run-%` pattern expansion
that normalized).

This reproduces on the **unmodified** upstream `dma_raw_blur_rw_async`
example. Confirmed by overriding CURR_DIR with forward slashes
explicitly — same failure. It's a real Windows port bug in the SDK's
example Makefile, not something specific to our probe.

**Estimate to fix**: 2-4 hours of Makefile plumbing; would likely
need to patch CURR_DIR canonicalization OR rewrite Makefile.rules
to be path-agnostic. Not worth doing here when alternatives exist.

### Attempt 2 — CMake S22U-pattern adaptation

`C:\Qualcomm\Hexagon_IDE\S22U\` is the user's validated working
FastRPC + cDSP build pattern (commit-clean, produces `libS22U_skel.so`
+ Android exe, runs end-to-end on the S22U via adb push). I mirrored
its directory shape to `C:\Qualcomm\Hexagon_IDE\sp_dma_raw\` with
`build.cmd` adapted via `sed`-style rename.

**What's unfinished**:
1. CMakeLists.txt needs to be adapted to:
   - Build skel from `sp_dma_raw.idl` (qaic invocation works in S22U;
     same flow applies)
   - Compile `src_dsp/sp_dma_raw_imp.c` (port of our `dsp/sp_dma_raw_run.cpp`
     with FastRPC-naming conventions: `sp_dma_raw_run` → `sp_dma_raw_run`
     stays the same as it's already in IDL form)
   - Link in the pre-built Halide kernel `.o` (we have a validated
     Windows pipeline for producing this — see commits e938fa8,
     `backends/halide/Makefile`)
   - Link the **Halide Hexagon DMA runtime library** — this is the
     piece S22U doesn't currently use, and we'd need to figure out
     where the `halide_hexagon_dma_*` symbols live.
     Likely `$(HALIDE_ROOT)/lib/v69/libhalide_hexagon_dma.a` or similar.

2. The DSP impl file structure needs minor renames:
   - `sp_dma_raw_run.cpp` → `src_dsp/sp_dma_raw_imp.c` (or stay .cpp
     with CMakeLists.txt OS_TYPE branch using add_library with C++)
   - Open/close handlers (`sp_dma_raw_open`/`sp_dma_raw_close`) need
     to be added per the S22U pattern (just `malloc(1)` / `free`).

3. Host main needs to be packaged as `src_app/sp_dma_raw_main.c` plus
   `src_app/sp_dma_raw_ext.c/.h` per the S22U pattern (the ext layer
   is what selects the FastRPC domain and calls the IDL-generated
   stub).

**Estimate**: 4-6 hours of careful CMake + FastRPC plumbing. Higher
ceiling than Attempt 1 because Halide DMA runtime linking is
unfamiliar territory.

### Attempt 3 (recommended) — Skip Halide, use `dmaWrapper.h` directly

The Halide-DMA path's value-add is *scheduling* (vectorize, prefetch,
async) — the underlying primitives are SDK calls into
`dmaWrapper.h` / `dma_def.h`. For a **probe** answering "does
eDmaFmt_RawData accept non-camera bytes," we don't need Halide as
a scheduler. We can:

1. Reuse the existing shannon-prime-hexagon FastRPC + cDSP scaffold
   (working, validated, ships every day).
2. Add a single new IDL method `sp_hex_probe_dma_raw` that:
   - Takes packed bytes via rpcmem (already proven to work).
   - Allocates an `eDmaFmt_RawData` frame via `getFrameSize`.
   - Calls `dma_descriptor_init` + `dma_descriptor_*_user_buffer` +
     `dma_engine_start`.
   - Reads back to host, returns timing.
3. The DSP code then just answers: did the descriptor init accept
   non-camera bytes? Did the engine run? Did the read complete?

This **directly** answers Stage 1 without any Halide build chain.
Halide scheduling becomes a downstream optimization once Stage 1
proves the silicon path is reachable.

**Estimate**: 2-3 hours, all inside the validated build path.
Same FastRPC scaffold that runs `compress_f32_batch` today.

## Recommended next step

Pivot to Attempt 3. Concrete plan:

1. Add `sp_hex_probe_dma_raw` method to `shannon_prime_hexagon.idl`.
2. Implement DSP-side using `dmaWrapper.h` calls directly (no Halide).
3. Implement bridge wrapper in `shannon_prime_hexagon.c`.
4. Wire into a new `tests/test_probe_dma_raw.c` host harness or
   extend an existing test target.
5. Build via existing build path (validated): `cmake --build`.
6. Push libsp_hex_skel.so to S22U via the existing `/data/local/tmp/sp22u`
   workflow.
7. Run, capture FARF logs, document pass/fail.

The Halide-as-kernel piece becomes a Stage 1.5 follow-up *after*
Stage 1 binary answers the silicon question.

## Files committed so far

- `backends/halide/probe/sp_dma_raw/halide/sp_dma_raw_generator.cpp` —
  2D Halide generator, builds in isolation via the validated cl.exe
  + Halide.lib pipeline, produces a Hexagon `.o`. Reusable for
  Stage 1.5.
- `backends/halide/probe/sp_dma_raw/dsp/sp_dma_raw_run.cpp` —
  Halide-runtime-using DSP impl. Blocked on the build chain above
  but the code is correct relative to the upstream `dma_raw_blur_rw_async`
  template.
- `backends/halide/probe/sp_dma_raw/host/main.cpp` — host harness
  with packing + reference verify. Reusable for Attempt 3 with
  minor renames (remove the buffer.h dependency, or keep it via
  ION allocation).
- `backends/halide/probe/sp_dma_raw/rpc/sp_dma_raw.idl` — IDL.
  Directly portable to Attempt 3 (just change `interface` name and
  add `: remote_handle64`).
- `backends/halide/probe/sp_dma_raw/Makefile`, `README.md` —
  Reference / aspirational build instructions for the SDK example
  path. Will be deprecated when Attempt 3 is wired.
