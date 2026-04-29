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

#ifdef __cplusplus
}
#endif

#endif // SP_HEX_EXT_H
