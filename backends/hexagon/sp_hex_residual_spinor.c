// Shannon-Prime — Residual quantize + SU(2) spinor phase (Strike 12).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// HVX pipeline (DSP side):
//   1. vsub_qf32(actual, predicted) -> sf  per chunk        (5 HVX vectors)
//   2. vmax-tree-reduce of |residual| -> scalar amax         (5 vmax + 5 vror)
//   3. Per-lane scalar tail: mag_q3 + phase + dense pack    (140 iterations)
//
// vsub uses the qfloat path because V69's fp32 vsub returns qf32 natively;
// we round-trip through Q6_Vsf_equals_Vqf32 to recover an sf-format vector
// that we can sign-bit-strip with a plain bitwise AND (Q6_Vsf_vabs is broken
// on V69 per the Strike 7 lesson).

#include "sp_hex_residual_spinor.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__hexagon__)
#  include <hexagon_protos.h>
#  include <hexagon_types.h>
#  define SP_HEX_HVX_AVAILABLE 1
#else
#  define SP_HEX_HVX_AVAILABLE 0
#endif

// ----------------------------------------------------------------------------
// Shared scalar-tail helpers — pack/unpack 3-bit + 1-bit into the 71-byte
// blob.  Same code for HVX and host reference paths so any kernel-correctness
// divergence is impossible to attribute to the pack step.
// ----------------------------------------------------------------------------
static void sp_hex_pack_residual(const float *residual_sf,
                                  int n_lanes,
                                  float amax,
                                  uint8_t *packed /* 71 bytes */) {
    memset(packed, 0, SP_HEX_RESIDUAL_TOTAL_BYTES);
    if (amax <= 0.0f) return;  // all-zero residuals; packed stays zeroed.

    const float scale = 7.0f / amax;
    for (int i = 0; i < n_lanes; ++i) {
        float r = residual_sf[i];
        int phase = (r < 0.0f) ? 1 : 0;
        float ar = (r < 0.0f) ? -r : r;
        /* Round half-up (matching the Q15 hier_predict convention so the
         * downstream decoder uses the same rounding rule everywhere). */
        int mag = (int)(ar * scale + 0.5f);
        if (mag > 7) mag = 7;
        if (mag < 0) mag = 0;

        /* Dense 3-bit magnitude pack — may straddle a byte boundary. */
        const int bit_idx  = i * 3;
        const int byte_idx = bit_idx >> 3;
        const int shift    = bit_idx & 7;
        const uint32_t cell = (uint32_t)mag << shift;
        packed[byte_idx]     |= (uint8_t)(cell & 0xFF);
        packed[byte_idx + 1] |= (uint8_t)((cell >> 8) & 0xFF);

        /* 1-bit phase pack into bytes 53..70. */
        const int p_byte = SP_HEX_RESIDUAL_MAG_BYTES + (i >> 3);
        const int p_bit  = i & 7;
        packed[p_byte] |= (uint8_t)(phase << p_bit);
    }
}

// ============================================================================
// HVX kernel — vsub + vmax-tree + scalar pack.
// ============================================================================

int sp_hex_residual_spinor_hvx(const float *actual,
                                const float *predicted,
                                int n_lanes,
                                int n_padded,
                                uint8_t *packed,
                                int packed_capacity,
                                float *amax_out) {
    if (!actual || !predicted || !packed || !amax_out) return -1;
    if (n_lanes != SP_HEX_RESIDUAL_LANES_MAX ||
        n_padded != SP_HEX_RESIDUAL_PAD) return -1;
    if (packed_capacity < SP_HEX_RESIDUAL_TOTAL_BYTES) return -1;

    /* Stack scratch for the residual in sf-format. */
    float residual[SP_HEX_RESIDUAL_PAD] __attribute__((aligned(128)));
    memset(residual, 0, sizeof(residual));

    float amax = 0.0f;

#if SP_HEX_HVX_AVAILABLE
    const HVX_Vector *act_v = (const HVX_Vector *)actual;
    const HVX_Vector *prd_v = (const HVX_Vector *)predicted;
    HVX_Vector *res_v       = (HVX_Vector *)residual;
    const int n_vec = SP_HEX_RESIDUAL_PAD / 32;   /* 5 */

    /* Step 1: vsub via qfloat round-trip. */
    for (int c = 0; c < n_vec; ++c) {
        HVX_Vector v_qf32 = Q6_Vqf32_vsub_VsfVsf(act_v[c], prd_v[c]);
        res_v[c]          = Q6_Vsf_equals_Vqf32(v_qf32);
    }

    /* Step 2: |.| via sign-bit strip + vmax-tree reduce. */
    const HVX_Vector v_signmask = Q6_V_vsplat_R((int32_t)0x7FFFFFFF);
    HVX_Vector v_max = Q6_V_vzero();
    for (int c = 0; c < n_vec; ++c) {
        HVX_Vector v_abs = Q6_V_vand_VV(res_v[c], v_signmask);
        v_max = Q6_Vsf_vmax_VsfVsf(v_max, v_abs);
    }
    /* Cross-lane reduce: 32 i32 lanes (= 32 fp32) → 1 scalar.
     * Halving rotates: 64-byte (16 lanes) → 32 (8) → 16 (4) → 8 (2) → 4 (1). */
    HVX_Vector v_r;
    v_r = Q6_V_vror_VR(v_max, 64); v_max = Q6_Vsf_vmax_VsfVsf(v_max, v_r);
    v_r = Q6_V_vror_VR(v_max, 32); v_max = Q6_Vsf_vmax_VsfVsf(v_max, v_r);
    v_r = Q6_V_vror_VR(v_max, 16); v_max = Q6_Vsf_vmax_VsfVsf(v_max, v_r);
    v_r = Q6_V_vror_VR(v_max,  8); v_max = Q6_Vsf_vmax_VsfVsf(v_max, v_r);
    v_r = Q6_V_vror_VR(v_max,  4); v_max = Q6_Vsf_vmax_VsfVsf(v_max, v_r);
    /* Lane 0 of v_max now holds the global amax. */
    int32_t amax_bits = Q6_R_vextract_VR(v_max, 0);
    memcpy(&amax, &amax_bits, sizeof(amax));
#else
    /* Off-target host fallback path: scalar vsub + amax. */
    for (int i = 0; i < n_padded; ++i) {
        residual[i] = (i < n_lanes) ? (actual[i] - predicted[i]) : 0.0f;
        float a = residual[i] < 0.0f ? -residual[i] : residual[i];
        if (a > amax) amax = a;
    }
#endif

    *amax_out = amax;

    /* Step 3: scalar tail — Q3 mag + phase + dense pack. */
    sp_hex_pack_residual(residual, n_lanes, amax, packed);
    return 0;
}

// ============================================================================
// Host-side scalar reference — IDENTICAL math, no HVX.
// ============================================================================

int sp_hex_residual_spinor_ref(const float *actual,
                                const float *predicted,
                                int n_lanes,
                                int n_padded,
                                uint8_t *packed,
                                int packed_capacity,
                                float *amax_out) {
    if (!actual || !predicted || !packed || !amax_out) return -1;
    if (n_lanes <= 0 || n_padded < n_lanes) return -1;
    if (packed_capacity < SP_HEX_RESIDUAL_TOTAL_BYTES) return -1;

    float *residual = (float *)calloc((size_t)n_padded, sizeof(float));
    if (!residual) return -1;

    float amax = 0.0f;
    for (int i = 0; i < n_lanes; ++i) {
        residual[i] = actual[i] - predicted[i];
        float a = residual[i] < 0.0f ? -residual[i] : residual[i];
        if (a > amax) amax = a;
    }
    *amax_out = amax;
    sp_hex_pack_residual(residual, n_lanes, amax, packed);
    free(residual);
    return 0;
}
