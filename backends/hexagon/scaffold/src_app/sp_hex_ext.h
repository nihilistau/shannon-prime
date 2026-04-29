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

#ifdef __cplusplus
}
#endif

#endif // SP_HEX_EXT_H
