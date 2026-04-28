# Backend: Qualcomm Hexagon DSP

This document is the implementation roadmap for the Hexagon DSP backend. The header (`backends/hexagon/shannon_prime_hexagon.h`) declares the API surface; the .c file is a scaffolding stub that returns "unavailable" so bridges fall back cleanly until the kernels exist.

Target hardware: Snapdragon 8 Gen 1+ (Hexagon V69 with HVX). Primary validation device: Samsung Galaxy S22 Ultra. SDK requirement: Qualcomm Hexagon SDK 5.x with `toolv87` and DSP_ARCH=v69.

---

## What Hexagon brings to Shannon-Prime

The Hexagon DSP is a third compute resource on Snapdragon SoCs alongside the ARM CPU cores and the Adreno GPU. Three things make it interesting for SP:

- **Always-on, low-power.** The DSP runs at ~500mW even under load. Compression / decompression of KV cache vectors can happen continuously without affecting CPU/GPU thermal budget. Important for sustained-load deployments where the phone otherwise thermal-throttles.

- **HVX vector instructions.** 1024-bit SIMD with int8/int16/fp16 pipes. The banded quantize/dequantize loops are exactly what HVX is designed for: load a vector of int8 values, apply scale, write fp16. Theoretical throughput on V69: 128 int8 ops/cycle.

- **Shared physical memory with CPU/GPU.** The DSP, ARM, and Adreno all access the same DRAM through the SoC's SMMU. Round-trips don't need PCIe-class transfers; they happen through page tables. The "stack and pop" memory pattern from earlier phone work is implemented natively in the SoC's memory subsystem.

What it doesn't bring on V69: HMX matrix instructions (those are V73+, Snapdragon 8 Gen 2+). The VHT2 butterfly will run on HVX scalar+vector ops on V69, then can pivot to HMX matmul on later hardware.

---

## Where the SDK pieces live

Standard install layout for Hexagon SDK 5.5.6.0 (Windows):

```
C:\Qualcomm\Hexagon_SDK\5.5.6.0\
  setup_sdk_env.cmd            <- source from a fresh cmd shell to set
                                  HEXAGON_SDK_ROOT, DEFAULT_DSP_ARCH,
                                  PATH, etc. Use this as the entry point
                                  for any build session.
  build/                       <- CMake/make machinery for FastRPC stubs
  examples/                    <- canonical sample projects
    calculator/                <- "hello world" of FastRPC (CPU<->DSP IPC)
    qhl_hvx/                   <- HVX intrinsics demo (the closest analog
                                  to what SP needs)
    multithreading/            <- DSP thread management
  incs/                        <- Hexagon-specific headers (HAP_*, etc.)
  ipc/                         <- FastRPC plumbing
  libs/                        <- Qualcomm-supplied DSP libraries
  tools/
    HEXAGON_Tools/8.7.06/      <- the actual compiler (hexagon-clang) +
                                  HVX/HMX intrinsics headers
    android-ndk-r25c/          <- ARM-side build chain
    qhcg/                      <- Hexagon Graph Compiler (high-level)
    cmake-3.22.2-windows-x86_64/  <- bundled CMake (no need to install
                                  separately)
    wrapperTools/              <- qaic, etc.
```

Auxiliary Qualcomm tooling that does NOT need to be installed for SP work, in case it's already on the system:

- `C:\Program Files (x86)\Qualcomm\QPM3` - Qualcomm Package Manager. Just the installer; not needed at build/run time.
- `C:\Program Files\Qualcomm\QUTS` - Qualcomm Universal Trace System. Useful for performance profiling later, not for getting started.
- `%LOCALAPPDATA%\Qualcomm\PCAT` - PC Configuration / Analysis Tool. Not relevant.

If any of those are missing, ignore. They're orthogonal to building DSP code.

---

## Setup checklist (one-time)

1. **Open a fresh `cmd` (not PowerShell)** and run:
   ```
   "C:\Qualcomm\Hexagon_SDK\5.5.6.0\setup_sdk_env.cmd"
   ```
   This sets `HEXAGON_SDK_ROOT`, prepends the Hexagon CMake to PATH, and points to the v87 compiler tree.

2. **Verify the toolchain** is reachable:
   ```
   hexagon-clang --version
   ```
   Should print something like `clang version X.X.X (Hexagon ToolsX.Y.Z)`. If "command not found" the env setup didn't take.

3. **Set DSP arch for V69** (S22 Ultra):
   ```
   set DEFAULT_DSP_ARCH=v69
   ```
   The default in `setup_sdk_env.cmd` is `v65`, which won't have all the HVX intrinsics SP wants. Override before building.

4. **Build the calculator example** to verify FastRPC works end-to-end:
   ```
   cd %HEXAGON_SDK_ROOT%\examples\calculator
   make tree V=hexagon_Release_dynamic_toolv87_v69
   ```
   This builds the DSP-side .so and the ARM-side .so. If both produce binaries without errors, the SDK is functional.

5. **For phone deployment**, you'll also need:
   - Android Studio (for the host-side app shell)
   - `adb` on PATH (Android Debug Bridge)
   - S22 Ultra in USB Debugging mode
   - Optional: a Qualcomm signature for the DSP binary if running outside dev mode (the SDK ships a default dev signature in `tools/elfsigner/`)

---

## SP-specific architecture

Three layers compose end-to-end:

```
+---------------------+    +---------------------+    +---------------------+
| ARM Host (Android)  |    | FastRPC IPC         |    | Hexagon DSP V69     |
|---------------------|    |---------------------|    |---------------------|
| llama.cpp + bridge  |    | qaic-generated stub |    | qaic-generated skel |
| sp_hexagon_init()   |--->| sp_hexagon_stub.c   |--->| sp_hexagon_skel.c   |
| sp_hexagon_alloc()  |    |                     |    |                     |
| sp_hexagon_         |    | rpcmem_alloc()      |    | shared phys page    |
|   round_trip_k() ---|--->| FastRPC dispatch    |--->| HVX kernel:         |
|                     |    |                     |    |  promote fp16->fp32 |
|                     |    |                     |    |  VHT2 butterfly     |
|                     |    |                     |    |  Mobius reorder     |
|                     |    |                     |    |  band quantize/     |
|                     |    |                     |    |   dequantize        |
|                     |    |                     |    |  inverse VHT2       |
|                     |    |                     |    |  demote fp32->fp16  |
| out_fp16 in shared  |<---| return value        |<---| write back to       |
| mem - no copy       |    |                     |    | shared region       |
+---------------------+    +---------------------+    +---------------------+
```

The IDL file (`shannon_prime_hexagon.idl`, not yet written) declares the FastRPC interface. `qaic` consumes it and generates `sp_hexagon_stub.c` (linked into the Android app) and `sp_hexagon_skel.c` (linked into the DSP .so). Each function call goes through the stub, marshals arguments via FastRPC, executes the skel on the DSP, and unmarshals the return.

---

## Compute mapping

Which SP operations live where:

| Operation | Where it runs |
|---|---|
| `sp_band_quantize` (encode) | Hexagon HVX scalar + vector |
| `sp_band_dequantize` (decode) | Hexagon HVX scalar + vector |
| `sp_band_dequantize_partial` | Hexagon HVX, with band-count parameter |
| `sp_vht2_forward_f32` (transform) | Hexagon HVX butterfly on V69; HMX matmul on V73+ |
| `sp_mobius_reorder` / `_unreorder` | Hexagon scalar (small, cache-friendly) |
| `sp_shadow_cache_*` (allocator + dispatch) | ARM host, calls into Hexagon |
| `sp_shadow_cache_load_partial` (disk IO) | ARM host, only the dequantize step on Hexagon |

The dispatch glue stays on ARM because UFS / file system access is host-side. The Hexagon side only sees compressed band buffers in shared memory once the host has read them off disk.

---

## Implementation milestones

### M0: Scaffolding (this commit)

- `shannon_prime_hexagon.h` with the API surface
- `shannon_prime_hexagon.c` stub returning -1 / NULL
- This roadmap doc
- README.md in the backend dir pointing at this doc

What works: nothing runtime; the SP build stays green and other backends still function.

### M1: Hello-world FastRPC round-trip

- Write `shannon_prime_hexagon.idl` declaring a single function:
  `int hexagon_echo(in uint8_t* buf, in size_t n, out uint8_t* result);`
- Run `qaic` to generate stubs.
- DSP side: copy input to output. No HVX. Just verify the IPC works.
- ARM side: `sp_hexagon_init` opens the FastRPC session and exposes the
  echo function via a private API.
- Smoke test: push to S22 Ultra via adb, run, verify output matches.

What works: FastRPC IPC is verified, signing flow is verified, build pipeline produces working binaries.

### M2: Single-vector round-trip

- IDL gains `hexagon_round_trip_k(in *uint16_t in, out *uint16_t out, ...)`.
- DSP side implements the full pipeline IN SCALAR HVX (no vectorisation):
  fp16->fp32 promote, VHT2 butterfly, Mobius reorder, band quantize +
  dequantize, Mobius unreorder, VHT2 inverse, fp32->fp16 demote.
- Compare the output against the CPU reference path (Adreno backend's
  scalar fallback) and assert correlation > 0.99 across 100 random inputs.

What works: SP round-trip is functionally correct on Hexagon. No
performance claims yet - scalar HVX is slow; the win comes in M3.

### M3: HVX-vectorised kernel

- Refactor M2's scalar code to use HVX intrinsics from
  `hvx_hexagon_protos.h`. The inner loops of band_quantize / dequantize
  vectorise cleanly: 128 int8 lanes per HVX vector, scale broadcast,
  multiply-accumulate.
- VHT2 butterfly (1/sqrt(2) scaling) runs as paired HVX
  add/subtract + multiply-by-constant.
- Bench: round-trip throughput in vectors-per-second on the S22 Ultra
  vs the ARM NEON Adreno backend. Target: >= 2x ARM NEON.

What works: full Hexagon backend, runtime-selectable via
`SHANNON_PRIME_BACKEND=hexagon` env var (or auto-detected when DSP is
present and HVX is available).

### M4: Async / overlapped IO with phase 3

- Hook the Hexagon backend into the partial-load path
  (`sp_shadow_cache_load_partial`). Disk reads happen on the ARM side;
  the Hexagon DSP receives band 0 first, processes it, then band 1
  while the next attention query is being computed.
- Pair with phase 3 attention short-circuit: most queries terminate
  after band 0 + 1, the rest walk further down the band stack.

This is the milestone where the architectural win actually shows up
in user-perceived latency.

---

## Validation plan

End-to-end smoke test for each milestone:

1. Build on the dev machine with the SDK set up.
2. `adb push` the `.so` files to the S22 Ultra.
3. Run a tiny integration test from a host Android app or via
   `adb shell` if signing allows.
4. Compare output against the CPU reference within tolerance.
5. Record cycle counts via `HEXAGON_REG_TIMER` and report
   in `archive/eval/bench/hexagon-<milestone>-<date>.csv`.

Per-milestone, the smoke test should be small enough to fit in a
single FastRPC call (microseconds-class). Performance comparisons go
through the existing bench harnesses (`bench_disk_partial`, the
speculative-decoding bench) once the backend is dispatchable from the
bridge.

---

## Open questions

These come up during M1-M3 and need decisions:

- **Signing for production builds.** Dev builds use the included
  testsig; shipping requires either Qualcomm signing or a custom-key
  approach if the user has access to a manufacturer-line signing
  workflow. Out of scope for the math/kernel work but blocks public
  distribution.

- **Vendor library reuse.** Qualcomm ships `qhl` (Qualcomm HVX
  Library) with BLAS-style kernels. SP's banded quantize is too
  specialised to map onto qhl directly, but the inverse VHT2
  butterfly might benefit from qhl's batched fp16 multiply
  routines. Worth measuring once M3 lands.

- **Multithreading on the DSP.** V69 has 6 hardware threads. The
  obvious mapping: one thread per layer, batching across heads. Not
  required for correctness; pure throughput optimisation.

- **Memory budget per FastRPC call.** Shared memory is constrained
  (typically a few MB). For long-context inference, one layer's worth
  of K (8 heads x 2048 positions x 76 bytes ~= 1.2 MB) fits, but
  multi-layer batches need streaming. Design decision: per-call
  granularity = single layer, all heads + positions.

---

## See also

- `BACKEND-ADRENO.md` - the existing ARM/NEON backend that the
  Hexagon work composes with on the same Snapdragon SoC.
- `DISK-TIER-ARCHITECTURE.md` - end-to-end storage tier story; the
  Hexagon DSP is the compute side that pairs with the storage tiering.
- `PHASE-3-ATTENTION-DESIGN.md` - where the partial-band-read primitive
  meets attention; M4 of this roadmap is what actually wires it up.
- `position_is_arithmetic_v8.md` - VHT2 + Mobius math underlying what
  the HVX kernels are computing.
