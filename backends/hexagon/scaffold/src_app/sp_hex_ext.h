// Shannon-Prime VHT2 - Hexagon DSP FastRPC scaffold (ARM-side header).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Forked from the Hexagon SDK 5.5.6.0 S22U sample. Qualcomm copyright on the
// pattern; SP-specific code is AGPLv3.

#ifndef SP_HEX_EXT_H
#define SP_HEX_EXT_H

#include "AEEStdDef.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Top-level smoke-test driver. Allocates rpcmem, opens a FastRPC session
// to the cDSP, runs round_trip_f32 on a deterministic input, and reports
// the worst-case fp32 error. Returns 0 on full success.
int sp_hex_process(int domain, int head_dim, bool isUnsignedPD_Enabled);

// Second smoke-test path: drives the engine-side hexagon API
// (sp_hexagon_init / sp_hexagon_round_trip_k / sp_hexagon_free) instead
// of the qaic IDL directly. Validates that the engine's host-side
// FastRPC shim works end-to-end on the phone before we wire it into
// llama-cpp-sp's bridge.
int sp_hex_engine_smoke(int head_dim);

// Cycle bench: runs the VHT2 forward butterfly through both the scalar
// reference path and the HVX-vectorised path on the cDSP, prints
// per-call pcycles + speedup ratio across a sweep of head_dim values.
// Uses sp_hex_vht2_bench (FastRPC) under the hood with HAP_perf_get_pcycles
// timing.
int sp_hex_run_bench_sweep(void);

// Disk-tier proof: scalar host-side quantize fills an rpcmem-backed
// packed-bands buffer, sp_hexagon_band_dequantize_partial processes via
// FastRPC zero-copy, output compared to scalar dequantize reference.
// Demonstrates the rpcmem → DSP path that disk I/O (fread into rpcmem)
// would feed into.
int sp_hex_disk_tier_proof(int head_dim);

// Per-element validation harness for the compress_f32 / decompress_f32
// IDL pair. Drives a deterministic input through (a) the DSP-side
// compress + decompress round-trip and (b) the host-side scalar
// reference (sp_band_quantize → sp_band_dequantize on already-VHT2'd
// coeffs), then compares per-element worst-abs.
//
// This is the validation lesson from the 2026-04-29 V69 IEEE-HVX
// debugging episode: round-trip RMS alone is not sufficient — two
// paths can produce the same RMS while differing wildly in their
// intermediate values. A per-element comparator is required before
// any new IDL method goes on the bridge hot path.
int sp_hex_compress_decompress_validate(int head_dim);

#ifdef __cplusplus
}
#endif

#endif // SP_HEX_EXT_H
